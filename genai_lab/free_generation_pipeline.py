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

from genai_lab.character_features import TAG_ITEMS, extract_features, write_features
from genai_lab.character_verification import measure_result_scores, verify_features
from genai_lab.free_generation import INTENT, SEEDS, check_request, features_sentence, free_conditions, part_counts
from genai_lab.outfit_edit import SIZE, canvas, sha, write_json
from genai_lab.outfit_edit_pipeline import segment_canvas, detect_joints
SAFETY_TAGS = ("nude", "nipples", "genitals", "loli", "shota")

from genai_lab.qwen_free_generation import (FreeGenerationWorkflow, make_free_request, prepare_free_spec,
                                           run_free_generation_edit)


def validate_inputs(base_path, base_kind, features, gender, body_parts, request_text, confirmed, adult_confirmed, *, snapshot_directory=None):
    """원본과 승인 편집본을 구분하고 이미지 바이트·승인 근거를 확인한다."""
    if confirmed is not True or adult_confirmed is not True:
        raise ValueError("성인 캐릭터와 특징표·연출 범위를 확인해 주세요.")
    if base_kind not in ("original", "edited"):
        raise ValueError("기준은 original 또는 edited여야 합니다.")
    if gender not in ("male", "female", "unspecified") or body_parts is None:
        raise ValueError("성별과 부위를 확인해 주세요(부위 없음은 빈 목록).")
    if len(set(body_parts)) != len(body_parts) or not set(body_parts) <= {"ears", "tail", "wings"}:
        raise ValueError("부위 값 오류")
    if not isinstance(features, dict) or set(features) != set(TAG_ITEMS):
        raise ValueError("확정 특징표의 모든 태그 항목이 필요합니다(모르면 빈 목록).")
    from genai_lab.character_features import features_text, parse_features_text
    if any(not isinstance(v, (list, tuple)) or any(not isinstance(x, str) for x in v) for v in features.values()):
        raise ValueError("특징표 값은 태그 문자열 목록이어야 합니다.")
    normalized = parse_features_text(features_text({k: dict(values=v) for k, v in features.items()}))
    import re
    for values in normalized.values():
        for tag in values:
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", tag) or not check_request(tag.replace("_", " "))["ok"]:
                raise ValueError("특징표에는 외형 태그만 쓸 수 있습니다.")
    checked = check_request(request_text)
    if not checked["ok"]:
        raise ValueError(checked["reason"] + " " + ", ".join(checked["matched"]))
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
            raise ValueError("같은 이미지에 대한 사용자 의상 편집 승인이 필요합니다.")
        approval = dict(path=str(sidecar), sha256=sha(sidecar), record=approval)
    with Image.open(io.BytesIO(data)) as image:
        prepared = canvas(image)
    snapshot = None
    if snapshot_directory is not None:
        snapshot = Path(snapshot_directory) / ("source" + path.suffix.lower())
        snapshot.write_bytes(data)  # 확인한 동일 바이트를 보존, 원본을 다시 읽어 바뀐 파일을 복사하지 않는다.
    return prepared, dict(path=str(path), snapshot=str(snapshot) if snapshot else None, sha256=digest, kind=base_kind, approval=approval,
                          proves_outfit_edit=False, source="user_confirmed"), checked


def prepare_reference(image, parts, directory):
    """같은 캔버스의 얼굴·머리 영역을 자른 참조. 위치 추출 실패는 전체 캔버스 참조로 명시한다."""
    keys = ("hair", "face", "eyes", "nose", "mouth", "headwear", "eyewear", "earwear")
    found = [parts[k] for k in keys if k in parts]
    mask = np.logical_or.reduce(found) if found else np.zeros((image.height, image.width), bool)
    ys, xs = np.where(mask)
    box = (max(0, int(xs.min())-12), max(0, int(ys.min())-12), min(image.width, int(xs.max())+13),
           min(image.height, int(ys.max())+13)) if len(xs) else (0, 0, image.width, image.height)
    path = Path(directory) / "head-reference.png"
    image.crop(box).save(path)
    return path, dict(path=str(path), sha256=sha(path), box=list(box), source="automatic_analysis",
                      parent_sha256=sha(Path(directory) / "canvas.png"),
                      status="derived_head_crop" if len(xs) else "whole_canvas_fallback",
                      note="기준과 같은 이미지에서 파생한 시각 참조. 동일성을 보장하지 않는다.")


def observe_result(reference, raw, directory, runtime, *, cancelled, popen):
    """Qwen 종료 후 생성 결과 자체를 분할한다. 기준 마스크를 결과에 적용하지 않는다."""
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
    verification = (verify_features(reference, scores, rgb, parts) if scores is not None
                    else dict(items={}, summary={}, status="unmeasured", reason="tagger unavailable"))
    safety = dict(status="not_reviewed", judge="user", reason="자동 안전 검사 통과를 보장하지 않는다")
    # 기존 금지 외형/노출 태그가 강하게 관찰되면 승인할 수 없는 결과로 둔다. 약한 값은 사람 확인.
    blocked = [k for k, v in (scores or {}).items() if k in SAFETY_TAGS and v >= .5]
    if blocked:
        safety = dict(status="violated", judge="automatic_analysis", evidence=blocked)
    try:
        joints = detect_joints(rgb, directory / "joints", runtime, cancelled=cancelled)
        limbs = dict(status="unmeasurable", joints=joints,
                     reason="출력 좌표에서만 인식. 미검출은 가림·인식 오류와 소실을 구분하지 못하므로 사람 검토")
    except Exception as error:
        if cancelled():
            raise
        limbs = dict(status="unmeasurable", reason=str(error))
    result = dict(image=str(path), image_sha256=sha(path), coordinate_basis="generated_output",
                  segmentation=seg, verification=verification, body_part_counts=part_counts(parts), limbs=limbs, safety=safety, errors=errors)
    write_json(directory / "observation.json", result)
    return result


def condition_judgements(conditions, observation, number, raw):
    """자동 판정은 특징별 관찰에 한정. 의상·동일성·연출 전체를 추정하지 않는다."""
    target = dict(attempt=number, image=str(raw), sha256=sha(raw))
    rows = [dict(key=c["key"], status="not_reviewed", judge="human_pending", target=target,
                 evidence=[], reason="의상 세부·동일성·연출 반영은 사람 검토") for c in conditions]
    status_map = {"kept": "fulfilled", "changed": "violated", "missing": "unmeasurable", "added": "violated", "unknown": "unmeasurable"}
    for key, value in observation["verification"].get("items", {}).items():
        rows.append(dict(key="feature."+key, status=status_map.get(value.get("status"), "unmeasurable"),
                         judge="automatic_analysis", target=target, evidence=value,
                         note="태그·색 관찰은 최종 동일성 판정이 아니다. 부위 개수/연결은 확인 불가"))
    rows.append(dict(key="safety", **observation["safety"], target=target))
    return rows


def prepare_generation_inputs(base_path, character_features, request_text, *, base_kind, gender, body_parts,
                              confirmed, adult_confirmed, directory, runtime, cancelled, progress, popen, trace, persist, selection_seed):
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
    blocked = [k for k, v in scores.items() if k in SAFETY_TAGS and v >= .5]
    trace["input_facts"]["automatic_tags"] = dict(source="automatic_analysis", scores=scores, blocked=blocked)
    persist()
    if blocked:
        raise ValueError("기준 이미지에서 금지된 외형·노출 태그가 관찰됐습니다: " + ", ".join(blocked))
    segmented = segment_canvas(prepared, directory / "base-parts", runtime, cancelled=cancelled, popen=popen)
    with np.load(segmented["parts"], allow_pickle=False) as data:
        parts = {k: data[k].astype(bool) for k in data.files}
    if any(v.shape != (SIZE[1], SIZE[0]) for v in parts.values()):
        raise ValueError("기준 분할 좌표 불일치")
    features = extract_features(np.asarray(image), parts, scores, gender=gender, body_parts=body_parts,
                                confirmed=character_features)
    for key, values in character_features.items():
        if not values:
            features["items"][key].update(state="unknown", reason="빈 특징값은 없음/가림을 구분할 수 없다")
    features["unknown"] = [k for k, v in features["items"].items() if v["state"] in ("unknown", "weak")]
    write_features(directory / "character-features", features)
    reference, crop_record = prepare_reference(image, parts, inputs)
    trace["input_facts"].update(character_features=features, base_segmentation=segmented, derived_reference=crop_record)
    assembled = free_conditions(features_sentence(features), gender, body_parts, request_text, selection_seed, base_kind)
    assembled["excluded"].extend(dict(key="feature."+key, reason="미확정/확인 불가 특징은 프롬프트 값에서 제외") for key in features["unknown"])
    trace.update(conditions=assembled["conditions"], excluded=assembled["excluded"], direction=assembled)
    write_json(inputs / "conditions.json", dict(input_facts=trace["input_facts"], **assembled))
    spec = prepare_free_spec(prepared["canvas"], reference, directory=directory / "qwen-inputs", request_text=request_text,
                             gender=gender, body_parts=body_parts, features_text=features_sentence(features), base_kind=base_kind, reference_box=crop_record["box"], selection_seed=selection_seed)
    return features, assembled, spec


def run_free_generation(base_path, character_features, request_text="", *, base_kind, gender, body_parts,
                        confirmed, adult_confirmed, settings, runtime, directory, cancelled=lambda: False,
                        progress=lambda _: None, popen=subprocess.Popen, selection_seed=None):
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
        code_files = ("free_generation", "qwen_free_generation", "free_generation_pipeline", "qwen_pose_worker", "qwen_pose_edit", "qwen_pose_settings", "qwen_outfit_edit", "outfit_edit", "outfit_edit_pipeline", "character_features", "character_verification")
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
            raise ValueError("연출 선택 seed 형식 오류")
        features, assembled, spec = prepare_generation_inputs(base_path, character_features, request_text,
            base_kind=base_kind, gender=gender, body_parts=body_parts, confirmed=confirmed, adult_confirmed=adult_confirmed,
            directory=directory, runtime=runtime, cancelled=cancelled, progress=progress, popen=popen, trace=trace, persist=persist, selection_seed=selection_seed)
        for number, seed in enumerate(SEEDS, 1):
            if cancelled():
                raise RuntimeError("사용자 취소")
            suffix = "" if number == 1 else f"-{number}"
            qwen = directory / f"qwen{suffix}"
            request = make_free_request(spec, settings=settings, seed=seed)
            attempt = dict(number=number, seed=seed, selection_seed=spec.selection_seed, status="generating",
                           request=request, actual_images=[dict(path=spec.image_path, sha256=spec.image_sha256),
                                                         dict(path=spec.reference_path, sha256=spec.reference_sha256)])
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
                                         cancelled=cancelled, popen=popen)
            judgement = condition_judgements(assembled["conditions"], observation, number, raw)
            attempt.update(status="completed", raw=str(raw), raw_sha256=sha(raw), observation=observation, judgements=judgement)
            persist()
            violations = observation["verification"].get("summary", {})
            retry = bool(violations.get("changed") or violations.get("added") or observation["safety"]["status"] == "violated")
            attempt["retry_reason"] = "자동 특징 위반 관찰" if retry else None
            if not retry:
                break
        blocked = observation["safety"]["status"] == "violated"
        trace["status"] = "blocked" if blocked else "awaiting_user_review"
        outcome = dict(status=trace["status"], directory=str(directory), raw=str(raw), result=str(raw), result_sha256=sha(raw),
                       composited=False, base_kind=base_kind, attempts=trace["attempts"],
                       conditions=str(directory / "inputs" / "conditions.json"), trace=str(directory / "trace.json"),
                       condition_judgements=judgement, automatic_features=observation["verification"], quality_pass=None,
                       note="자동 검사는 동일성·최종 품질 승인이 아니다. 원본 기준 성공은 의상 교체 해결 증거가 아니다.")
        write_json(directory / "outcome.json", outcome)
        persist()
        return outcome
    except Exception as error:
        trace.update(status="cancelled" if cancelled() else "failed", error=str(error))
        if trace["attempts"]:
            trace["attempts"][-1].update(status=trace["status"], error=str(error))
            run_record = qwen / "run.json"
            if run_record.exists():
                trace["attempts"][-1]["worker_record"] = json.loads(run_record.read_text(encoding="utf-8"))
        persist()
        raise
