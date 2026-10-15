"""D-123 공통 진입: 입력 확인 → 고정 조건 → Qwen → 결과 좌표 분석 → 사람 검토 대기.
원본 머리 합성은 없다. 자동 특징 관찰과 최종 품질 승인을 분리한다.
"""
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from genai_lab.character_features import DISTINCTIVE, TAG_ITEMS, extract_features, write_features
from genai_lab.character_verification import measure_result_scores, retry_reasons, verify_features
from genai_lab.free_generation import (HIDDEN_NOTES, INTENT, SEEDS, FreeGenerationRejected, check_request, distinctive_notes, features_sentence, free_conditions,
                                       garment_tag_changes, part_counts, request_conflicts, request_observations, safety_observation,
                                       staging_text)
from genai_lab.outfit_edit import SIZE, canvas, sha, write_json
from genai_lab.outfit_edit_pipeline import segment_canvas, detect_joints

from genai_lab.qwen_free_generation import (FreeGenerationWorkflow, make_free_request, prepare_free_spec,
                                           run_free_generation_edit)


OUTCOME_SCHEMA = "free_generation_outcome/1"     # 반환 형식(D-133, docs/FREE_GENERATION_API.md). 키를 바꾸면 버전을 올린다


def validate_inputs(base_path, base_kind, features, gender, body_parts, request_text, confirmed, adult_confirmed, *, snapshot_directory=None):
    """원본과 승인 편집본을 구분하고 이미지 바이트·승인 근거를 확인한다."""
    if confirmed is not True or adult_confirmed is not True:
        raise FreeGenerationRejected("confirmation_required", "성인 캐릭터와 특징표·연출 범위를 확인해 주세요.")
    if base_kind not in ("original", "edited"):
        raise FreeGenerationRejected("invalid_input", "기준은 original 또는 edited여야 합니다.")
    if gender not in ("male", "female", "unspecified") or body_parts is None:
        raise FreeGenerationRejected("invalid_input", "성별과 부위를 확인해 주세요(부위 없음은 빈 목록).")
    if len(set(body_parts)) != len(body_parts) or not set(body_parts) <= {"ears", "tail", "wings"}:
        raise FreeGenerationRejected("invalid_input", "부위 값 오류")
    if not isinstance(features, dict) or set(features) - {DISTINCTIVE} != set(TAG_ITEMS):
        raise FreeGenerationRejected("invalid_features", "확정 특징표의 모든 태그 항목이 필요합니다(모르면 빈 목록).")
    distinctive_notes(features.get(DISTINCTIVE))      # 고유 형태 메모 검사(D-126 ④)
    from genai_lab.character_features import features_text, parse_features_text
    if any(v is not None and (not isinstance(v, (list, tuple)) or any(not isinstance(x, str) for x in v)) for v in features.values()):
        raise FreeGenerationRejected("invalid_features", "특징표 값은 태그 문자열 목록(없음은 빈 목록, 모름은 None)이어야 합니다.")
    normalized = parse_features_text(features_text({k: dict(values=v) for k, v in features.items()}))
    import re
    for key, values in normalized.items():
        if key == DISTINCTIVE:
            continue
        for tag in values or ():
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", tag) or not check_request(tag.replace("_", " "))["ok"]:
                raise FreeGenerationRejected("invalid_features", "특징표에는 외형 태그만 쓸 수 있습니다.")
    checked = check_request(request_text)
    if not checked["ok"]:
        raise FreeGenerationRejected(checked["code"], checked["reason"] + " " + ", ".join(checked["matched"]),
                                     detail=dict(codes=checked["codes"], matched=checked["matched"]))
    path = Path(base_path).resolve()
    data = path.read_bytes()
    import hashlib
    digest = hashlib.sha256(data).hexdigest()
    approval = None
    if base_kind == "edited":
        sidecar = path.with_suffix(".review.json")
        approval = json.loads(sidecar.read_text(encoding="utf-8"))
        checks = approval.get("checks") or {}
        if (approval.get("mode") != "outfit_edit" or approval.get("reviewer") != "user"
                or approval.get("sha256") != digest
                or not all(checks.get(k) is True for k in ("character", "garment", "exposure", "original_outfit_gone"))):
            raise FreeGenerationRejected("approval_required", "같은 이미지에 대한 사용자 의상 편집 승인이 필요합니다.")
        approval = dict(path=str(sidecar), sha256=sha(sidecar), record=approval)
    with Image.open(io.BytesIO(data)) as image:
        prepared = canvas(image)
    snapshot = None
    if snapshot_directory is not None:
        snapshot = Path(snapshot_directory) / ("source" + path.suffix.lower())
        snapshot.write_bytes(data)  # 확인한 동일 바이트를 보존, 원본을 다시 읽어 바뀐 파일을 복사하지 않는다.
    return prepared, dict(path=str(path), snapshot=str(snapshot) if snapshot else None, sha256=digest, kind=base_kind, approval=approval,
                          proves_outfit_edit=False, source="user_confirmed"), checked


HEAD_WINDOW = (3.0, 1.5, 1.0)     # 얼굴 상자 기준 창: 폭 = 얼굴 폭×3, 위 = 얼굴 높이×1.5, 아래 = 얼굴 높이×1.0 (D-126 ③)
WIDE_REFERENCE = 0.35             # 크롭이 캔버스의 이 비율을 넘으면 넓은 참조(지시문이 head close-up이라고 부르지 않음)


def head_box(parts, size):
    """얼굴 기준 창 안의 머리 영역 상자. 긴 머리카락이 몸까지 내려가도 머리 둘레만 자른다. 얼굴이 없으면 None."""
    width, height = size
    face = np.logical_or.reduce([parts[k] for k in ("face", "eyes") if k in parts]) if any(k in parts for k in ("face", "eyes")) else None
    if face is None or not face.any():
        return None
    ys, xs = np.where(face)
    fw, fh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    cx = (xs.min() + xs.max()) / 2
    across, above, below = HEAD_WINDOW
    window = (max(0, int(cx - fw * across / 2)), max(0, int(ys.min() - fh * above)),
              min(width, int(cx + fw * across / 2) + 1), min(height, int(ys.max() + fh * below) + 1))
    keys = ("hair", "face", "eyes", "nose", "mouth", "headwear", "eyewear", "earwear", "ears")
    head = np.logical_or.reduce([parts[k] for k in keys if k in parts])
    inside = np.zeros_like(head)
    inside[window[1]:window[3], window[0]:window[2]] = head[window[1]:window[3], window[0]:window[2]]
    ys, xs = np.where(inside)
    return (max(0, int(xs.min()) - 12), max(0, int(ys.min()) - 12), min(width, int(xs.max()) + 13), min(height, int(ys.max()) + 13))


def prepare_reference(image, parts, directory):
    """같은 캔버스의 얼굴·머리 영역을 자른 참조. 얼굴 기준 창 → (얼굴 없음) 머리 전체 상자 → (머리 없음) 전체 캔버스."""
    box, status = head_box(parts, image.size), "derived_head_crop"
    if box is None:
        keys = ("hair", "face", "eyes", "nose", "mouth", "headwear", "eyewear", "earwear")
        found = [parts[k] for k in keys if k in parts]
        mask = np.logical_or.reduce(found) if found else np.zeros((image.height, image.width), bool)
        ys, xs = np.where(mask)
        box, status = (((max(0, int(xs.min())-12), max(0, int(ys.min())-12), min(image.width, int(xs.max())+13),
                         min(image.height, int(ys.max())+13)), "hair_box_fallback") if len(xs)
                       else ((0, 0, image.width, image.height), "whole_canvas_fallback"))
    ratio = (box[2] - box[0]) * (box[3] - box[1]) / (image.width * image.height)
    scope = "wide" if ratio > WIDE_REFERENCE or status != "derived_head_crop" else "head"   # 얼굴을 못 찾았으면 머리 확대라고 부르지 않는다
    path = Path(directory) / "head-reference.png"
    image.crop(box).save(path)
    return path, dict(path=str(path), sha256=sha(path), box=list(box), source="automatic_analysis",
                      parent_sha256=sha(Path(directory) / "canvas.png"), status=status, scope=scope, area_ratio=round(ratio, 3),
                      note="기준과 같은 이미지에서 파생한 시각 참조. 동일성을 보장하지 않는다.")


def observe_result(reference, raw, directory, runtime, *, cancelled, popen, skip=None, base_scores=None, request_text="",
                   skeleton=None):
    """Qwen 종료 후 생성 결과 자체를 분할한다. 기준 마스크를 결과에 적용하지 않는다.
    skip: 요청과 충돌해 재지 않는 특징(request_conflicts), base_scores: 의상 태그 차이 관찰용 기준 태그."""
    directory = Path(directory)
    directory.mkdir()
    with Image.open(raw) as image:
        image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS).save(directory / "canvas.png")
    path = directory / "canvas.png"
    prepared = dict(canvas=str(path), canvas_sha256=sha(path))
    errors = {}
    parts = {}
    seg = None
    try:
        seg = segment_canvas(prepared, directory / "parts", runtime, cancelled=cancelled, popen=popen)
        with np.load(seg["parts"], allow_pickle=False) as data:
            parts = {k: data[k].astype(bool) for k in data.files}
        if any(v.shape != (SIZE[1], SIZE[0]) for v in parts.values()):
            raise ValueError("결과 분할 좌표가 결과 캔버스와 다릅니다.")
    except Exception as error:
        if cancelled():
            raise
        errors["segmentation"] = str(error)
        parts = {}
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    scores = measure_result_scores(rgb, runtime)
    verification = (verify_features(reference, scores, rgb, parts, skip or {}) if scores is not None
                    else dict(items={}, summary={}, retry=[], status="unmeasured", reason="tagger unavailable"))
    observed = safety_observation(scores)
    safety = dict(status="not_reviewed", judge="user", reason="자동 안전 검사 통과를 보장하지 않는다", review=observed["review"])
    # 노출·금지 외형 태그가 강하게 관찰되면 승인할 수 없는 결과로 둔다. 나이 관련 태그는 사람 확인 근거로만 남긴다.
    if observed["blocked"]:
        safety = dict(status="violated", judge="automatic_analysis", evidence=observed["blocked"], review=observed["review"])
    pose_match = dict(status="unmeasurable", reason="결과 관절 인식 실패") if skeleton else None
    try:
        joints = detect_joints(rgb, directory / "joints", runtime, cancelled=cancelled)
        if skeleton:      # D-131: 결과 관절의 팔·다리 각도를 골격과 비교(관찰, 재시도 근거 아님)
            from genai_lab.pose_guidance import compare_pose
            pose_match = compare_pose(skeleton, joints)
        limbs = dict(status="unmeasurable", joints=joints,
                     reason="출력 좌표에서만 인식. 미검출은 가림·인식 오류와 소실을 구분하지 못하므로 사람 검토")
    except Exception as error:
        if cancelled():
            raise
        limbs = dict(status="unmeasurable", reason=str(error))
    result = dict(image=str(path), image_sha256=sha(path), coordinate_basis="generated_output", pose_match=pose_match,
                  segmentation=seg, verification=verification, body_part_counts=part_counts(parts), limbs=limbs, safety=safety,
                  garment_tags=garment_tag_changes(base_scores, scores), request_tags=request_observations(request_text, scores),
                  errors=errors)
    write_json(directory / "observation.json", result)
    return result


def hair_restore_candidate(features, raw, analysis_directory, observation):
    """머리색이 바뀌었거나 확인 필요로 관찰되면 색감만 기준 분포로 되돌린 보정 후보를 따로 만든다(D-130, 선택 마무리).
    원 결과는 그대로 두고, 사용자가 결과 목록에서 고를 때만 쓴다. 만들지 않으면 None."""
    import cv2
    from genai_lab.character_features import hair_mask_without_background
    from genai_lab.hair_color_restore import restore_hair_color
    hair = (observation["verification"].get("items") or {}).get("hair_color") or {}
    if not (hair.get("status") in ("changed", "added") or hair.get("review")):
        return None
    reference = ((features.get("items") or {}).get("hair_color") or {}).get("ab_quantiles")
    parts_path = (observation.get("segmentation") or {}).get("parts")
    if not reference or not parts_path:
        return dict(status="unmeasurable", reason="기준 분포 또는 결과 분할 없음")
    with Image.open(raw) as source:
        image = np.asarray(source.convert("RGB"))
    with np.load(parts_path, allow_pickle=False) as data:
        if "hair" not in data.files:
            return dict(status="unmeasurable", reason="결과 머리 영역 없음")
        mask = cv2.resize(data["hair"].astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    restored, report = restore_hair_color(image, hair_mask_without_background(image, mask), reference)
    if restored is None:
        return report
    path = Path(analysis_directory) / "hair-restored.png"
    Image.fromarray(restored).save(path)
    return dict(report, path=str(path), sha256=sha(path), source=str(raw), source_sha256=sha(raw),
                trigger=dict(status=hair.get("status"), review=hair.get("review")))


def condition_judgements(conditions, observation, number, raw):
    """자동 판정은 특징별 관찰에 한정. 의상·동일성·연출 전체를 추정하지 않는다."""
    target = dict(attempt=number, image=str(raw), sha256=sha(raw))
    rows = [dict(key=c["key"], status="not_reviewed", judge="human_pending", target=target,
                 evidence=[], reason="의상 세부·동일성·연출 반영은 사람 검토") for c in conditions]
    # 자동 관찰은 '충족'을 말하지 않는다(D-127). 유지 = 위반 관찰 없음, 여유가 적으면 확인 필요. 충족은 사람 판정만.
    status_map = {"kept": "not_violated", "changed": "violated", "missing": "unmeasurable", "added": "violated", "unknown": "unmeasurable"}
    for key, value in observation["verification"].get("items", {}).items():
        status = "needs_review" if value.get("status") == "kept" and value.get("review") else status_map.get(value.get("status"), "unmeasurable")
        rows.append(dict(key="feature."+key, status=status, judge="automatic_analysis", target=target, evidence=value,
                         note="태그·색 관찰은 최종 동일성 판정이 아니다. 부위 개수/연결은 확인 불가"))
    match = observation.get("pose_match")
    if match:
        rows.append(dict(key="pose.skeleton_match", judge="automatic_analysis", target=target, evidence=match,
                         status={"match": "not_violated", "mismatch": "needs_review"}.get(match["status"], "unmeasurable"),
                         note="골격 대비 팔·다리 각도 관찰. 다리·팔 개수 이상은 사람 확인"))
    for item in observation.get("request_tags") or ():
        rows.append(dict(key="request."+item["key"], judge="automatic_analysis", target=target, evidence=item,
                         status={"observed": "not_violated", "not_observed": "needs_review"}.get(item["status"], "unmeasurable"),
                         note="요청 연출의 태그 관찰. 반영 판정은 사람"))
    rows.append(dict(key="safety", **observation["safety"], target=target))
    return rows


def prepare_skeleton(image, staging, directory, runtime, cancelled):
    """지원 자세면 기준 그림 관절로 비율을 맞춘 골격을 그려 저장(D-131). 반환 (경로, 관절 기록, 기록) 또는 (None, None, 기록)."""
    from genai_lab.pose_guidance import classify_pose, draw_skeleton, retarget
    pose_key = classify_pose(staging)
    if pose_key is None:
        return None, None, dict(status="unsupported", reason="지원 자세 목록 밖 — 글 지시만(결과 보장 없음)", staging=staging)
    canvas = np.asarray(image)
    try:
        joints = detect_joints(canvas, directory / "base-joints", runtime, cancelled=cancelled)
        joint_error = None
    except Exception as error:
        if cancelled():
            raise
        joints, joint_error = [], str(error)[:200]
    records, info = retarget(pose_key, joints, image.size)
    path = directory / "inputs" / "pose-skeleton.png"
    Image.fromarray(draw_skeleton(records, image.size)).save(path)
    return path, records, dict(status="guided", pose_key=pose_key, path=str(path), sha256=sha(path), retarget=info,
                                base_joint_error=joint_error, source="code_template+automatic_analysis")


def prepare_generation_inputs(base_path, character_features, request_text, *, base_kind, gender, body_parts,
                              confirmed, adult_confirmed, directory, runtime, cancelled, progress, popen, trace, persist, selection_seed,
                              pose_guidance=False):
    """확인된 입력의 같은 바이트·특징·조건·파생 참조를 실행 폴더에 고정하고 Qwen 명세를 반환한다."""
    inputs = directory / "inputs"
    inputs.mkdir()
    image, base, checked = validate_inputs(base_path, base_kind, character_features, gender, body_parts,
                                           request_text, confirmed, adult_confirmed, snapshot_directory=inputs)
    image.save(inputs / "canvas.png")
    prepared = dict(canvas=str(inputs / "canvas.png"), canvas_sha256=sha(inputs / "canvas.png"))
    trace["input_facts"] = dict(base=base, gender=dict(value=gender, source="user_confirmed"),
        body_parts=dict(value=list(body_parts), source="user_confirmed"),
        confirmed_features=dict(value=character_features, source="user_confirmed"),
        request=dict(value=request_text, source="user_confirmed", validation=checked),
        adult=dict(value=True, source="user_confirmed"), policy=dict(source="common_policy", contract="D-123"))
    persist()
    progress("상태: 기준 이미지 특징 측정 중")
    scores = measure_result_scores(np.asarray(image), runtime) or {}
    observed = safety_observation(scores)
    blocked = observed["blocked"]
    trace["input_facts"]["automatic_tags"] = dict(source="automatic_analysis", scores=scores, blocked=blocked, review=observed["review"])
    persist()
    if blocked:
        raise FreeGenerationRejected("unsafe_input", "기준 이미지에서 금지된 외형·노출 태그가 관찰됐습니다: " + ", ".join(blocked),
                                     detail=dict(tags=blocked))
    segmented = segment_canvas(prepared, directory / "base-parts", runtime, cancelled=cancelled, popen=popen)
    with np.load(segmented["parts"], allow_pickle=False) as data:
        parts = {k: data[k].astype(bool) for k in data.files}
    if any(v.shape != (SIZE[1], SIZE[0]) for v in parts.values()):
        raise ValueError("기준 분할 좌표 불일치")
    features = extract_features(np.asarray(image), parts, scores, gender=gender, body_parts=body_parts,
                                confirmed=character_features)
    for key, values in character_features.items():
        if values is None:          # 모름(확인 불가). '없음'(빈 목록)은 사용자가 확정한 값이므로 그대로 둔다(검토 5번)
            features["items"][key].update(state="unknown", reason="사용자가 '모름'으로 둔 항목")
    features["unknown"] = [k for k, v in features["items"].items() if v["state"] in ("unknown", "weak") and k != DISTINCTIVE]
    write_features(directory / "character-features", features)
    reference, crop_record = prepare_reference(image, parts, inputs)
    trace["input_facts"].update(character_features=features, base_segmentation=segmented, derived_reference=crop_record)
    staging = staging_text(request_text, selection_seed)        # 빈 요청이면 고른 고정 자세로 가림을 판단(D-128)
    conflicts = request_conflicts(staging)
    sentence = features_sentence(features, hidden=conflicts)
    skeleton_path, skeleton_records, skeleton_record = (prepare_skeleton(image, staging, directory, runtime, cancelled) if pose_guidance
                                                        else (None, None, dict(status="off")))
    trace["input_facts"]["pose_guidance"] = skeleton_record
    assembled = free_conditions(sentence, gender, body_parts, request_text, selection_seed, base_kind,
                                reference_scope=crop_record["scope"], skeleton=skeleton_path is not None)
    assembled["skeleton"] = skeleton_records
    if pose_guidance and skeleton_path is None:
        assembled["excluded"].append(dict(key="pose_skeleton", reason=skeleton_record["reason"]))
    assembled["request_conflicts"] = conflicts
    assembled["staging_text"] = staging
    if conflicts:
        assembled["excluded"].append(dict(key="feature_text." + "+".join(sorted(conflicts)), reason="요청이 가리는 특징은 지시문에서 보이게 유지하라고 하지 않는다(D-127). 결과 확인 체크에는 남는다",
                                          detail=conflicts))
    assembled["base_scores"] = scores
    notes = distinctive_notes(character_features.get(DISTINCTIVE))
    assembled["distinctive_review"] = notes["review"]
    import re
    prompted = [n for n in notes["prompt"] if not any(re.search(HIDDEN_NOTES[k], n, re.I) for k in conflicts if k in HIDDEN_NOTES)]
    if len(prompted) < len(notes["review"]):
        assembled["excluded"].append(dict(key="distinctive", reason="영어가 아닌 메모·요청이 가리는 특징의 메모는 지시문에 넣지 않고 결과 확인 창에서만 확인",
                                          notes=[n for n in notes["review"] if n not in prompted]))
    assembled["excluded"].extend(dict(key="feature."+key, reason="미확정/확인 불가 특징은 프롬프트 값에서 제외") for key in features["unknown"])
    recorded = {k: v for k, v in assembled.items() if k != "base_scores"}      # 기준 태그는 input_facts.automatic_tags에 이미 있다
    trace.update(conditions=assembled["conditions"], excluded=assembled["excluded"], direction=recorded)
    write_json(inputs / "conditions.json", dict(input_facts=trace["input_facts"], **recorded))
    spec = prepare_free_spec(prepared["canvas"], reference, directory=directory / "qwen-inputs", request_text=request_text,
                             gender=gender, body_parts=body_parts, features_text=sentence, base_kind=base_kind, reference_box=crop_record["box"], selection_seed=selection_seed,
                             reference_scope=crop_record["scope"], skeleton_path=skeleton_path, skeleton_joints=skeleton_records or (),
                             pose_key=skeleton_record.get("pose_key", ""))
    return features, assembled, spec


def run_free_generation(base_path, character_features, request_text="", *, base_kind, gender, body_parts,
                        confirmed, adult_confirmed, settings, runtime, directory, cancelled=lambda: False,
                        progress=lambda _: None, popen=subprocess.Popen, selection_seed=None, pose_guidance=False):
    """GUI와 향후 API가 사용할 함수. 실패는 trace를 보존하고 예외, 성공은 사람 검토 대기 결과 반환."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    trace = dict(schema="free_generation_trace/1", intent=INTENT, status="validating", attempts=[],
                 judgement=dict(quality_pass=None, same_character="not_reviewed"), human_judgements=[])
    start = time.perf_counter()
    def persist():
        trace["elapsed_seconds"] = round(time.perf_counter()-start, 3)
        write_json(directory / "trace.json", trace)
    persist()
    try:
        from genai_lab.qwen_preservation import json_sha
        source_root = Path(__file__).resolve().parents[1]
        code_files = ("free_generation", "qwen_free_generation", "free_generation_pipeline", "qwen_pose_worker", "qwen_pose_edit", "qwen_pose_settings", "qwen_outfit_edit", "outfit_edit", "outfit_edit_pipeline", "character_features", "character_verification", "pose_guidance", "hair_color_restore")
        trace["execution_contract"] = dict(settings=settings.record(), settings_sha256=json_sha(settings.record()),
            code_sha256={name: sha(source_root / "genai_lab" / (name+".py")) for name in code_files},
            contract_sha256=sha(source_root / "docs/FREE_GENERATION_CONTRACT.md"),
            memory_policy=dict(reserved_limit_gib=6.5, vae_tiling=True, text_encoder_lm_head="cpu"))
        persist()
        if cancelled():
            raise RuntimeError("사용자 취소")
        if selection_seed is None:
            import secrets
            selection_seed = secrets.randbelow(2**31) if not request_text.strip() else SEEDS[0]
        if type(selection_seed) is not int or not 0 <= selection_seed < 2**63:
            raise FreeGenerationRejected("invalid_input", "연출 선택 seed 형식 오류")
        features, assembled, spec = prepare_generation_inputs(base_path, character_features, request_text,
            base_kind=base_kind, gender=gender, body_parts=body_parts, confirmed=confirmed, adult_confirmed=adult_confirmed,
            directory=directory, runtime=runtime, cancelled=cancelled, progress=progress, popen=popen, trace=trace, persist=persist, selection_seed=selection_seed,
            pose_guidance=pose_guidance)
        for number, seed in enumerate(SEEDS, 1):
            if cancelled():
                raise RuntimeError("사용자 취소")
            suffix = "" if number == 1 else f"-{number}"
            qwen = directory / f"qwen{suffix}"
            request = make_free_request(spec, settings=settings, seed=seed)
            attempt = dict(number=number, seed=seed, selection_seed=spec.selection_seed, status="generating",
                           request=request, actual_images=[dict(path=spec.image_path, sha256=spec.image_sha256),
                                                         dict(path=spec.reference_path, sha256=spec.reference_sha256),
                                                         *([dict(path=spec.skeleton_path, sha256=spec.skeleton_sha256)] if spec.skeleton_path else [])])
            trace["attempts"].append(attempt)
            trace["status"] = "generating"
            persist()
            progress(f"상태: 자유 생성 {number}/{len(SEEDS)} · 약 30분")
            run_free_generation_edit(request, qwen, FreeGenerationWorkflow(spec), cancelled=cancelled, popen=popen,
                                     on_progress=lambda data: progress(f"상태: 자유 생성 · {len(data.get('steps', []))}/40"))
            raw = qwen / "raw.png"
            if (qwen / "run.json").exists():
                attempt["worker_record"] = json.loads((qwen / "run.json").read_text(encoding="utf-8"))
            progress("상태: 생성 결과 자체의 부위·특징 확인 중")
            trace["status"] = "verifying"
            persist()
            observation = observe_result(features, raw, directory / f"result-analysis{suffix}", runtime,
                                         cancelled=cancelled, popen=popen, skip=assembled["request_conflicts"],
                                         base_scores=assembled["base_scores"], request_text=assembled["staging_text"],
                                         skeleton=assembled["skeleton"])
            judgement = condition_judgements(assembled["conditions"], observation, number, raw)
            restore = hair_restore_candidate(features, raw, directory / f"result-analysis{suffix}", observation)
            attempt.update(status="completed", raw=str(raw), raw_sha256=sha(raw), observation=observation, judgements=judgement,
                           hair_restore=restore)
            persist()
            # 확신 있는 위반만 재시도한다(D-126 ⑤). 경계값·엇갈림·가림·요청 충돌·의상 태그 차이는 사람 확인으로 남긴다.
            reasons = retry_reasons(observation["verification"])
            unsafe = observation["safety"]["status"] == "violated"
            attempt["retry_reason"] = ("안전 태그 차단" if unsafe else ("확신 있는 특징 위반: " + ", ".join(reasons) if reasons else None))
            attempt["retry_items"] = reasons
            attempt["violation_score"] = 1000 if unsafe else len(reasons)
            if not attempt["retry_reason"]:
                break
        # 두 시도 모두 위반이면 위반이 적은 시도를 먼저 보여 준다(같으면 앞 시도). 어느 쪽을 쓸지는 사람이 고른다(candidates).
        chosen = min(trace["attempts"], key=lambda a: a["violation_score"]) if attempt["retry_reason"] else attempt
        raw, observation, judgement = Path(chosen["raw"]), chosen["observation"], chosen["judgements"]
        trace["selected_attempt"] = chosen["number"]
        blocked = observation["safety"]["status"] == "violated"
        trace["status"] = "blocked" if blocked else "awaiting_user_review"
        outcome = dict(schema=OUTCOME_SCHEMA, status=trace["status"], directory=str(directory), raw=str(raw), result=str(raw), result_sha256=sha(raw),
                       composited=False, base_kind=base_kind, attempts=trace["attempts"],
                       conditions=str(directory / "inputs" / "conditions.json"), trace=str(directory / "trace.json"),
                       condition_judgements=judgement, automatic_features=observation["verification"], quality_pass=None,
                       selected_attempt=chosen["number"],
                       candidates=[c for a in trace["attempts"] if a.get("status") == "completed" for c in (
                                       dict(attempt=a["number"], variant="raw", seed=a["seed"], raw=a["raw"], sha256=a["raw_sha256"],
                                            violation_score=a["violation_score"], retry_items=a["retry_items"],
                                            blocked=a["observation"]["safety"]["status"] == "violated",
                                            automatic_features=a["observation"]["verification"].get("summary", {}),
                                            garment_tags=a["observation"]["garment_tags"]),
                                       *([dict(attempt=a["number"], variant="hair_restored", seed=a["seed"], raw=a["hair_restore"]["path"],
                                               sha256=a["hair_restore"]["sha256"], violation_score=a["violation_score"],
                                               retry_items=a["retry_items"], blocked=a["observation"]["safety"]["status"] == "violated",
                                               automatic_features=a["observation"]["verification"].get("summary", {}),
                                               garment_tags=a["observation"]["garment_tags"], hair_restore=a["hair_restore"])]
                                         if (a.get("hair_restore") or {}).get("status") == "restored" else []))],
                       distinctive_review=assembled["distinctive_review"],
                       note="자동 검사는 동일성·최종 품질 승인이 아니다. 원본 기준 성공은 의상 교체 해결 증거가 아니다.")
        write_json(directory / "outcome.json", outcome)
        persist()
        return outcome
    except Exception as error:
        trace.update(status="cancelled" if cancelled() else ("rejected" if isinstance(error, FreeGenerationRejected) else "failed"),
                     error=str(error), rejection=error.record() if isinstance(error, FreeGenerationRejected) else None)
        if trace["attempts"]:
            trace["attempts"][-1].update(status=trace["status"], error=str(error))
            run_record = qwen / "run.json"
            if run_record.exists():
                trace["attempts"][-1]["worker_record"] = json.loads(run_record.read_text(encoding="utf-8"))
        persist()
        raise
