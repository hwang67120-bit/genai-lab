"""모드 B 작업 프로세스(STYLE_MODE_APP_DESIGN.md S4). 요청 폴더의 request.json만 읽고 결과를 같은 폴더에 쓴다.

prepare (CPU): 태그·안전 검사(옷 그림 포함) → 배경 분리·분할 → 특징 문장·원본 부위 마스크·원본 손 개수 → prepare.json
generate (GPU): FLUX.2-klein-4B 편집(D-210), NF4 + CPU 내려 두기, 상한 6.5GiB, 후보 3장 → 색 맞춤·손 개수 → generation.json
combine (CPU): 부위별 고르기(D-214) — 얼굴·머리 후보 + 몸·옷 후보 → combine-result.json
관절(pose.json)은 이 프로세스 전에 제품 관절 분석(studio_analysis pose)이 만든다.
예전 SDXL 경로(RealVisXL·IP-Adapter·스타일 LoRA, 얼굴 다듬기·다듬기)는 D-216에서 지웠다.
사용: python style_redraw_worker.py <prepare|generate|combine> <directory>
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genai_lab.style_redraw_anatomy import assess_hands, attach_arm_evidence, gate_correction

MEMORY_CAP_GIB = 6.5
HAND_MODEL = "models--deepghs--anime_hand_detection"
ANIME_SEG = "models--skytnt--anime-seg/snapshots/493cb60893f47441b26ec4fb9a306bce9e342982/isnetis.onnx"
JOINT_SCORE = 0.3
TAG_THRESHOLD = 0.5
MULTI_VIEW_TAGS = ("multiple_views", "reference_sheet", "turnaround")


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def joints_from_pose(record):
    """제품 관절 기록(joint_name·x·y·confidence_score·detected) → {이름: [x, y]}. 믿을 만한 관절만(0.3 이상)."""
    return {r["joint_name"]: [float(r["x"]), float(r["y"])] for r in record.get("joints", [])
            if r.get("detected") and float(r.get("confidence_score", 0.0)) >= JOINT_SCORE}


def support_warnings(scores, joints):
    """지원 범위 경고(설계 7절). 시작을 막지 않고 알린다. 확실할 때만(태그 0.5 이상·관절 없음) 낸다."""
    from genai_lab.style_redraw_inputs import EXTRA_HAIR_STYLE_TAGS
    scores = scores or {}
    def hit(*tags):
        return any(float(scores.get(t, 0.0)) >= TAG_THRESHOLD for t in tags)
    found = []
    if not any(name in joints for name in ("right_knee", "left_knee", "right_ankle", "left_ankle")):
        found.append(dict(code="upper_body_only", message="다리가 보이지 않는 그림이라 무릎 위까지만 그립니다. 아래옷 색·모양이 달라질 수 있습니다."))
    if hit(*EXTRA_HAIR_STYLE_TAGS):
        found.append(dict(code="rare_hairstyle", message="드문 머리 모양입니다. 머리 모양이 흔한 형태로 바뀔 수 있습니다."))
    if hit("heterochromia"):
        found.append(dict(code="heterochromia", message="양쪽 눈 색이 다릅니다. 눈 색이 한 가지로 될 수 있습니다."))
    if hit("sitting", "squatting", "kneeling"):
        found.append(dict(code="sitting", message="앉거나 웅크린 자세입니다. 옷 색 읽기가 틀릴 수 있습니다."))
    return found


def prepare(directory):
    from PIL import Image
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.free_generation import safety_observation
    from genai_lab.style_redraw_inputs import StyleRedrawInputError, face_box
    req = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    started = time.time()
    canvas_path = directory / "character.png"
    canvas = Image.open(canvas_path).convert("RGB")
    rgb = np.asarray(canvas)
    pose = json.loads((directory / "pose.json").read_text(encoding="utf-8"))
    def refuse(code, message, **extra):
        save(directory / "prepare.json", dict(state="refused", code=code, message=message, **extra))
    if pose.get("person_count") != 1:
        return refuse("person_count", "캐릭터가 한 명 보이는 그림을 넣어 주세요.", person_count=pose.get("person_count"))
    joints = joints_from_pose(pose)
    try:
        face = face_box(pose.get("face_points", ()), pose.get("face_scores", ()), canvas.size)
    except StyleRedrawInputError as error:
        return refuse("face", str(error))
    settings = ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)
    with WdTagSession(settings) as tagger:
        scores = dict(tagger.analyze(canvas).raw_general_scores)
    if any(float(scores.get(tag, 0.0)) >= TAG_THRESHOLD for tag in MULTI_VIEW_TAGS):
        # 앞·뒷모습이 한 장에 있는 설정화는 관절 분석이 1명으로 세기도 한다(D-226 미도리야 그림)
        return refuse("multiple_views", "한 사람이 한 번만 그려진 그림을 넣어 주세요(설정화·여러 시점 그림은 쓸 수 없습니다).")
    safety = safety_observation(scores)
    if safety["blocked"] or safety["review"]:
        # 성인 캐릭터만 다룬다. 어려 보이는 신호가 있으면 자동 진행하지 않는다.
        return refuse("safety", "이 그림은 그림체 다시 그리기를 진행할 수 없습니다.", safety=safety)
    prepare_flux(directory, req, canvas, rgb, scores, safety, joints, face, started)


def person_mask(rgb, req):
    """사람 영역(anime-seg)."""
    from PIL import Image
    from types import SimpleNamespace
    from genai_lab.proportion_foreground import AnimeForeground
    model = Path(req["model_cache"]) / ANIME_SEG
    fg = AnimeForeground(SimpleNamespace(foreground_model=model, foreground_sha256=sha(model)))
    try:
        return fg.alpha(Image.fromarray(rgb)) > 127
    finally:
        fg.close()


def person_and_parts(rgb, req):
    """사람 영역(anime-seg)과 부위 분할(See-through). 색 맞춤용 네 부위 마스크를 돌려준다."""
    from genai_lab.head_paste_semantics import HeadSegmenter
    from genai_lab.head_paste_worker import figure_parts
    from genai_lab.style_redraw_flux import part_masks
    person = person_mask(rgb, req)
    seg = HeadSegmenter(Path(req["head_semantic_repo"]), Path(req["head_semantic_checkpoint"]))
    try:
        parts = figure_parts(seg, rgb)
    finally:
        seg.close()
    return parts, part_masks(rgb, parts, person)


def hand_detections(images, req):
    """각 이미지의 검출 위치와 신뢰도를 반환한다. 검출 실패는 None이다."""
    cache = Path(req.get("hand_cache", ""))
    if not (cache / HAND_MODEL).is_dir():
        return [None] * len(images)
    # 생성 모델이 GPU에 있는 동안에도 검출은 CPU만 사용한다. 검사 실패는 미확인으로 남긴다.
    previous = os.environ.get("ONNX_MODE")
    os.environ["ONNX_MODE"] = "cpu"
    try:
        import huggingface_hub.constants as hub
        hub.HF_HUB_CACHE = str(cache)
        from PIL import Image
        from imgutils.detect import detect_hands
        return [detect_hands(Image.fromarray(np.asarray(im))) for im in images]
    except Exception:
        import logging
        logging.getLogger(__name__).exception("손 검출 실패: 정상으로 처리하지 않고 미확인으로 남깁니다.")
        return [None] * len(images)
    finally:
        if previous is None:
            os.environ.pop("ONNX_MODE", None)
        else:
            os.environ["ONNX_MODE"] = previous


def count_hands(images, req):
    """입력 준비와 기존 호출자는 검출 수만 사용한다."""
    return [len(boxes) if boxes is not None else None for boxes in hand_detections(images, req)]


def locate_excess_detections(image, boxes, req):
    """검출 상자가 인물 마스크와 겹치는지 기록해 배경 오검출 조사에 쓴다."""
    records = [dict(box=[int(v) for v in box], confidence=float(score)) for box, _, score in boxes]
    try:
        mask = person_mask(np.asarray(image), req)
        if not mask.any():
            return dict(detections=records, hand_location="unassessed")
        height, width = mask.shape
        for record in records:
            x0, y0, x1, y1 = record["box"]
            area = mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
            if x1 <= 0 or y1 <= 0 or not area.size:
                return dict(detections=records, hand_location="unassessed")
            record["foreground_fraction"] = float(area.mean())
    except Exception:
        import logging
        logging.getLogger(__name__).exception("손 위치와 인물 마스크 대조 실패")
        return dict(detections=records, hand_location="unassessed")
    overlap = sum(r["foreground_fraction"] > 0 for r in records)
    location = "excess_overlaps_foreground" if overlap > 2 else "outside_foreground_detected"
    return dict(detections=records, foreground_box_count=overlap, hand_location=location)


def observe_arm_poses(directory, images, req, phase):
    """기존 관절 전용 환경에서 CPU 검사한다. 실패 기록을 남기고 생성 결과를 지우지 않는다."""
    from PIL import Image
    from genai_lab.child_env import child_environment
    from genai_lab.style_redraw_pipeline import run_process
    folder = directory / "anatomy-pose" / f"{phase}-{time.time_ns()}"
    folder.mkdir(parents=True, exist_ok=True)
    failure = lambda message: [dict(state="unassessed", error=message) for _ in images]
    interpreter, models = req.get("pose_python"), req.get("pose_models")
    if not interpreter or not models:
        records = failure("관절 검사 환경 경로가 없습니다.")
        save(folder / "result.json", dict(records=records))
        return records
    paths = []
    for i, image in enumerate(images):
        path = folder / f"input-{i}.png"
        Image.fromarray(np.asarray(image)).save(path)
        paths.append(str(path.resolve()))
    request_path, output_path = folder / "request.json", folder / "result.json"
    save(request_path, dict(models=models, images=paths, input_sha256=[sha(Path(p)) for p in paths]))
    # 검사마다 새 폴더를 써 이전 실패 기록과 입력을 보존한다.
    env = child_environment()
    env.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        command = [str(interpreter), "-B", "-X", "utf8", str(Path(__file__).with_name("style_redraw_pose_evidence.py")),
                   str(request_path.resolve()), str(output_path.resolve())]
        code = run_process(command, folder / "worker.log", 120, env=env, stage="anatomy_pose_cpu")
        save(folder / "exit.json", dict(exit_code=code))
        result = json.loads(output_path.read_text(encoding="utf-8")) if code == 0 else {}
        records = result.get("records", [])
        if len(records) != len(images):
            raise ValueError(f"관절 검사 결과 수 불일치: {len(records)}/{len(images)}, exit={code}")
        return records
    except Exception as error:
        records = failure(f"{type(error).__name__}: {error}")
        save(folder / "failure.json", dict(records=records))
        return records


def inspect_anatomy(directory, images, req, phase, *, previous=None):
    """손 수와 관절 연결 근거를 교차 확인한다. 검토 허용은 인체 합격이나 자동 보정 허용이 아니다."""
    from genai_lab.style_redraw_arm_evidence import assess_arm_evidence, apply_rejection_review_guard
    if previous is not None and len(previous) != len(images):
        raise ValueError("보정 전후 인체 기록 수 불일치")
    poses = observe_arm_poses(directory, images, req, phase)
    hands = hand_detections(images, req)
    records = []
    for image, boxes, pose in zip(images, hands, poses):
        record = assess_hands(len(boxes) if boxes is not None else None)
        if record["state"] == "excess_hands_suspected":
            record.update(locate_excess_detections(image, boxes, req))
        height, width = np.asarray(image).shape[:2]
        evidence = assess_arm_evidence(pose, None if boxes is None else [box for box, _, _ in boxes], width, height)
        evidence = apply_rejection_review_guard(evidence, req.get("user_rejection"))
        record = attach_arm_evidence(record, evidence)
        record["detections"] = [dict(box=list(map(float, box)), confidence=float(score)) for box, _, score in boxes or []]
        record["image_size"] = [width, height]
        if previous is not None:
            from genai_lab.style_redraw_correction_evidence import guard_correction_anatomy
            record = guard_correction_anatomy(previous[len(records)], record, (width, height))
        records.append(record)
    save(directory / f"anatomy-{phase}.json", records)
    return records


def prepare_flux(directory, req, canvas, rgb, scores, safety, joints, face, started):
    """FLUX 경로 준비(D-210): 특징 문장·원본 부위 마스크·원본 손 개수. 그림체 참조·의상 참조·입력 묶음은 만들지 않는다."""
    from genai_lab.character_features import extract_features
    from genai_lab.studio_controller import guessed_gender
    from genai_lab.style_redraw_flux import flux_prompt, framing, style_lora_weight
    from genai_lab.style_redraw_inputs import trait_rows
    garment_path = directory / "garment.png"
    with_garment = garment_path.is_file()
    if with_garment:
        # 옷 그림도 같은 안전 검사(성인 캐릭터만, 어려 보이는 신호와 노출이 겹치면 막음)
        from PIL import Image
        from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
        from genai_lab.free_generation import safety_observation
        with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)) as tagger, \
                Image.open(garment_path) as garment:
            garment_scores = dict(tagger.analyze(garment.convert("RGB")).raw_general_scores)
            garment_safety = safety_observation(garment_scores)
        if garment_safety["blocked"] or garment_safety["review"]:
            save(directory / "prepare.json", dict(state="refused", code="garment_safety",
                                                  message="이 옷 그림은 사용할 수 없습니다.", safety=garment_safety))
            return
    from genai_lab.identity_anchors import anchors, body_part_kinds
    from genai_lab.studio_controller import guessed_body_parts
    parts, masks = person_and_parts(rgb, req)
    guessed = guessed_gender(dict(character_scores=scores))
    confirmed = req.get("confirmed_gender")
    features = extract_features(rgb, parts, scores, gender=confirmed or "unknown")
    traits = trait_rows(features, scores, confirmed)
    # 동물 귀·꼬리·날개: 사람이 확정한 값(캐릭터 기억 등)이 있으면 그것, 없으면 태그 자동 추정(사용자 화면과 같은 규칙, D-221)
    confirmed_parts = req.get("confirmed_body_parts")
    body_parts = tuple(confirmed_parts) if confirmed_parts is not None else guessed_body_parts(dict(character_scores=scores))
    kinds = body_part_kinds(scores, body_parts)     # 귀·꼬리·날개 종류를 문장에(D-236)
    shot = framing(scores)          # 원본이 보여 준 만큼만 그린다(D-247): 전신·허벅지 위·상반신
    from genai_lab.style_redraw_inputs import garment_words
    worn = garment_words(garment_scores) if with_garment else []       # 옷 그림의 옷 낱말을 문장에(D-249)
    from genai_lab.style_redraw_prompt import compact_flux_prompt
    prompt_plan = compact_flux_prompt(traits, with_garment, body_parts, req.get("scene"), kinds, shot, confirmed or guessed, worn)
    np.savez_compressed(directory / "original-masks.npz", **masks)
    hands = count_hands([rgb], req)[0]
    save(directory / "prepare.json", dict(
        state="ready", engine="flux", with_garment=with_garment, seconds=round(time.time() - started, 1), gender=confirmed,
        guessed_gender=guessed, traits=traits, body_parts=list(body_parts), identity_anchors=anchors(scores),
        body_parts_source="confirmed" if confirmed_parts is not None else "automatic_tags", body_part_kinds=kinds,
        style_lora_weight=style_lora_weight(scores),         # 털 수인이면 그림체를 약하게(D-245)
        framing=shot, garment_words=worn,
        scene=req.get("scene"), prompt=prompt_plan["prompt"], prompt_sections=prompt_plan["sections"],
        prompt_format_version=prompt_plan["format_version"],
        legacy_prompt=flux_prompt(traits, with_garment, body_parts, req.get("scene"), kinds, shot, confirmed or guessed, worn),
        face_box=face,
        original_hands=hands,
        warnings=support_warnings(scores, joints), safety=safety, joints_used=sorted(joints),
        note="특징 문장·성별은 자동 측정이며 사람 확인 전이다. "
             + ("자세·장면은 장면 문장을 따른다." if req.get("scene") else "자세·구도는 원본을 따른다.")))
    prepared = json.loads((directory / "prepare.json").read_text(encoding="utf-8"))
    audit = prompt_delivery(directory, req, prepared["prompt"], "prepare")
    prepared["prompt_delivery"] = {key: audit[key] for key in ("state", "token_count", "limit", "truncated")}
    if audit["truncated"]:
        prepared.update(state="refused", code="prompt_truncated",
                        message=f"생성 지시가 입력 한도를 넘었습니다({audit['token_count']}/{audit['limit']}). "
                                "누락된 지시로 생성하지 않도록 중단했습니다. 문장 구성 정비가 필요합니다.")
    save(directory / "prepare.json", prepared)


def prompt_delivery(directory, req, prompt, phase, tokenizer=None):
    """로컬 토크나이저로 전달을 검사하고 증거를 보존한다. 모델 가중치는 읽지 않는다."""
    from genai_lab.flux_prompt_delivery import plan_flux_prompt
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(Path(req["flux_model"]) / "tokenizer"), local_files_only=True)
    audit = plan_flux_prompt(tokenizer, prompt)
    audit.update(phase=phase, prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    save(directory / f"prompt-delivery-{phase}.json", audit)
    return audit


def require_initial_prompt(directory, req):
    """이전 준비 기록으로 직접 실행하더라도 문장 잘림은 GPU 초기화 전에 중단한다."""
    prepared = json.loads((directory / "prepare.json").read_text(encoding="utf-8"))
    audit = prompt_delivery(directory, req, prepared["prompt"], "generate")
    if audit["truncated"]:
        raise ValueError(f"생성 지시가 입력 한도를 넘었습니다({audit['token_count']}/{audit['limit']}). "
                         "일부 지시를 버리고 생성하지 않습니다. 문장 구성 정비가 필요합니다.")
    return prepared


def generate_flux(directory, req):
    """FLUX.2-klein 편집으로 후보 3장(D-210) → 자동 고치기(D-241) → 부위별 색 맞춤 → 손 개수. 원래 생성 그림(raw)도 남긴다.
    자동 고치기: 원본 태그 중 후보 과반이 잃은 몸·입은 것 태그가 있으면 '빠진 특징' 문장을 붙여 같은 seed로 한 번 더 그리고, seed마다 덜 잃은
    쪽을 후보로 쓴다(identity_autofix, 사람이 고른 항목 없음). 바뀐 후보의 첫 그림은 result-raw-seed*-r0.png로 남긴다.
    얼굴 다듬기(D-243): 고른 후보마다 머리를 잘라 원래 얼굴에서 출발해 뒤쪽 단계만 다시 그려 붙인다(face_detail). 다듬기 전 그림은
    result-raw-seed*-preface.png."""
    prepared = require_initial_prompt(directory, req)
    import gc
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import (GUIDANCE, IDENTITY_PARTS, IDENTITY_SHIFT_CAP, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, PARTS, SEEDS, STEPS,
                                             FURRY_TAGS, STYLE_LORA_WEIGHT, body_part_tags, colour_match, missing_body_parts)
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.identity_anchors import compare as compare_anchors
    from genai_lab.identity_autofix import PRESENT, fixed_prompt, garment_fix_sentence, kinds_table, lost_tags, remaining
    from genai_lab.style_redraw_quality import build_contract, choose_correction, correction_regressions, diagnose_candidate
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    started = time.time()
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(str(req["flux_model"]), torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    # 그림체 LoRA(D-236): 4비트 본체 위에 붙인다. CPU 오프로드 전에 붙여야 한다(시험 S27과 같은 순서).
    lora = Path(req["flux_style_lora"])
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    weight = prepared.get("style_lora_weight", STYLE_LORA_WEIGHT)
    pipe.set_adapters(["style"], adapter_weights=[weight])
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    loaded = round(time.time() - started, 1)
    reference = Image.open(directory / "character.png").convert("RGB")
    with_garment = bool(prepared.get("with_garment"))
    references = [reference] + ([Image.open(directory / "garment.png").convert("RGB")] if with_garment else [])
    torch.cuda.reset_peak_memory_stats()

    feedback = req.get("user_rejection")
    candidate_seeds = (feedback["seed"],) if feedback else SEEDS

    def draw(prompt, phase, suffix, seeds=candidate_seeds):
        audit = prompt_delivery(directory, req, prompt, phase, pipe.tokenizer)
        if audit["truncated"]:
            raise ValueError("생성 지시가 입력 한도를 넘었습니다. 잘린 문장으로 생성하지 않습니다.")
        from genai_lab.flux_prompt_delivery import build_flux_condition
        condition, condition_record = build_flux_condition(pipe, prompt, MAX_SEQUENCE_LENGTH)
        save(directory / f"prompt-condition-{phase}.json", condition_record)
        drawn = []
        for number, seed in enumerate(seeds):
            save(directory / "progress.json", dict(candidate=number + 1, candidates=len(candidate_seeds), step=0, total=STEPS, phase=phase))
            t0 = time.time()
            image = pipe(image=references, width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1],
                         num_inference_steps=STEPS, guidance_scale=GUIDANCE, **condition,
                         generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
            image.save(directory / f"result-raw-seed{seed}{suffix}.png")
            drawn.append((seed, np.asarray(image), round(time.time() - t0, 1)))
        return drawn

    settings = ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)
    with WdTagSession(settings) as tagger:
        original_scores = dict(tagger.analyze(reference).raw_general_scores)
        garment_scores = dict(tagger.analyze(references[1]).raw_general_scores) if with_garment else None
    contract = build_contract(original_scores, garment_scores, kinds_table(), with_garment=with_garment,
                              framing=prepared.get("framing", ""))
    save(directory / "quality-contract.json", contract)
    target = contract["target_scores"]
    raws = draw(prepared["prompt"], "generate", "")
    anatomy = inspect_anatomy(directory, [r for _, r, _ in raws], req, "initial")
    with WdTagSession(settings) as tagger:
        result_scores = [dict(tagger.analyze(Image.fromarray(r)).raw_general_scores) for _, r, _ in raws]
    save(directory / "quality-scores-initial.json", result_scores)
    lost = lost_tags(target, result_scores)
    asked = [tag for tag in lost if tag in contract["identity"] or not with_garment]
    worn_missing = [tag for tag in lost if with_garment and tag in contract["garment"]]
    watch = list(dict.fromkeys(asked + worn_missing))
    autofix = dict(lost=lost, asked=asked, garment_missing=worn_missing, rounds=0, picks=[0] * len(candidate_seeds))
    if watch:
        prompt = fixed_prompt(prepared["prompt"], asked)
        if worn_missing:
            anchor = "Keep the same face,"
            sentence = garment_fix_sentence(worn_missing)
            prompt = prompt.replace(anchor, sentence + anchor, 1) if anchor in prompt else prompt + " " + sentence.strip()
        audit = prompt_delivery(directory, req, prompt, "autofix", pipe.tokenizer)
        if audit["truncated"]:
            autofix.update(blocked_reason="prompt_truncated", attempted_prompt=prompt)
        else:
            again = draw(prompt, "autofix", "-fix")
            with WdTagSession(settings) as tagger:
                again_scores = [dict(tagger.analyze(Image.fromarray(r)).raw_general_scores) for _, r, _ in again]
            save(directory / "quality-scores-autofix.json", again_scores)
            decisions = [choose_correction(contract, before, after, watch)
                         for before, after in zip(result_scores, again_scores)]
            proposed = inspect_anatomy(directory, [r for _, r, _ in again], req, "autofix", previous=anatomy)
            decisions = [gate_correction(d, a) for d, a in zip(decisions, proposed)]
            picks = [int(decision["use_correction"]) for decision in decisions]
            for i, pick in enumerate(picks):
                seed = candidate_seeds[i]
                if pick == 1:
                    (directory / f"result-raw-seed{seed}.png").replace(directory / f"result-raw-seed{seed}-r0.png")
                    (directory / f"result-raw-seed{seed}-fix.png").replace(directory / f"result-raw-seed{seed}.png")
                    raws[i] = (seed, again[i][1], round(raws[i][2] + again[i][2], 1))
                    result_scores[i] = again_scores[i]
                    anatomy[i] = proposed[i]
            autofix.update(prompt=prompt, picks=picks, rounds=1, decisions=decisions)
    autofix["still_lost"] = [remaining(target, scores, watch) for scores in result_scores]
    # 털 수인 마지막 장치(D-246): 세기 0.3 + 자동 고치기 뒤에도 털 수인 태그를 잃은 후보는 그림체 LoRA를 끄고 같은 seed로 한 번 더 그려,
    # 털 수인이 돌아오면 그것을 쓴다(그 후보는 그림체가 빠진다). 흑백 선화 털 토끼(F1)가 장면에 따라 사람 얼굴이 됨.
    weights = [weight] * len(candidate_seeds)
    furry_watch = [t for t in FURRY_TAGS if original_scores.get(t, 0.0) >= PRESENT]
    if weight < STYLE_LORA_WEIGHT and furry_watch:
        lost_furry = [i for i, s in enumerate(result_scores) if remaining(original_scores, s, furry_watch)]
        if lost_furry:
            pipe.set_adapters(["style"], adapter_weights=[0.0])
            plain = draw(autofix.get("prompt", prepared["prompt"]), "furry", "-nostyle", [candidate_seeds[i] for i in lost_furry])
            pipe.set_adapters(["style"], adapter_weights=[weight])
            with WdTagSession(settings) as tagger:
                plain_scores = [dict(tagger.analyze(Image.fromarray(r)).raw_general_scores) for _, r, _ in plain]
            kept, furry_decisions = [], []
            save(directory / "quality-scores-furry.json", plain_scores)
            for i, (seed, rgb, seconds), scores in zip(lost_furry, plain, plain_scores):
                changed = correction_regressions(contract, result_scores[i], scores)
                restored = not remaining(original_scores, scores, furry_watch)
                furry_decisions.append(dict(seed=seed, furry_restored=restored, regressions=changed))
                proposed = inspect_anatomy(directory, [rgb], req, f"furry-seed{seed}", previous=[anatomy[i]])[0]
                furry_decisions[-1]["anatomy"] = proposed
                if not restored or any(changed.values()) or not proposed.get("automatic_correction_eligible", False):
                    continue
                (directory / f"result-raw-seed{seed}.png").replace(directory / f"result-raw-seed{seed}-styled.png")
                (directory / f"result-raw-seed{seed}-nostyle.png").replace(directory / f"result-raw-seed{seed}.png")
                raws[i] = (seed, rgb, round(raws[i][2] + seconds, 1))
                result_scores[i] = scores
                anatomy[i] = proposed
                weights[i] = 0.0
                kept.append(seed)
            autofix.update(furry_retry=dict(seeds=[candidate_seeds[i] for i in lost_furry], used=kept, decisions=furry_decisions))
    faces = []
    if req.get("head_cache"):
        from genai_lab.face_detail import detail_face, PROMPT as FACE_PROMPT
        from genai_lab.style_redraw_flux import STYLE_LORA_TRIGGER
        face_audit = prompt_delivery(directory, req, f"{STYLE_LORA_TRIGGER} {FACE_PROMPT}".strip(), "face", pipe.tokenizer)
        for i, (seed, raw, seconds) in enumerate(raws):
            save(directory / "progress.json", dict(candidate=i + 1, candidates=len(candidate_seeds), step=0, total=STEPS, phase="face"))
            t0 = time.time()
            if not anatomy[i].get("automatic_correction_eligible", False):
                detailed, info = raw, dict(pasted=False, reason=anatomy[i]["state"])
            elif face_audit["truncated"]:
                detailed, info = raw, dict(pasted=False, reason="prompt_truncated")
            else:
                detailed, info = detail_face(pipe, raw, reference, req["head_cache"], STYLE_LORA_TRIGGER, steps=STEPS, guidance=GUIDANCE,
                                             max_sequence_length=MAX_SEQUENCE_LENGTH)
            info["seconds"] = round(time.time() - t0, 1)
            faces.append((info, detailed))
        with WdTagSession(settings) as tagger:             # 다듬기가 원본 특징(눈 색 등)을 잃게 했으면 되돌린다
            for i, (info, detailed) in enumerate(faces):
                if not info.get("pasted"):
                    continue
                after = dict(tagger.analyze(Image.fromarray(detailed)).raw_general_scores)
                changed = correction_regressions(contract, result_scores[i], after)
                info["revalidation"] = changed
                info["proposed_scores"] = after
                proposed = inspect_anatomy(directory, [detailed], req, f"face-seed{raws[i][0]}", previous=[anatomy[i]])[0]
                info["anatomy"] = proposed
                if any(changed.values()) or not proposed.get("automatic_correction_eligible", False):
                    Image.fromarray(detailed).save(directory / f"result-seed{raws[i][0]}-face-rejected.png")
                    info.update(pasted=False, reason="다듬기 후 특징·의상·오염 또는 손 과다 검사에서 거절됨", changed=changed)
                    continue
                seed, _, seconds = raws[i]
                (directory / f"result-raw-seed{seed}.png").replace(directory / f"result-raw-seed{seed}-preface.png")
                Image.fromarray(detailed).save(directory / f"result-raw-seed{seed}.png")
                raws[i] = (seed, detailed, round(seconds + info["seconds"], 1))
                result_scores[i] = after
                anatomy[i] = proposed
        faces = [info for info, _ in faces]
    peak = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()
    original = np.asarray(reference)
    original_masks = dict(np.load(directory / "original-masks.npz"))
    candidates = []
    for (seed, raw, seconds), raw_anatomy, scores, pick, face, cand_weight in zip(raws, anatomy, result_scores, autofix["picks"],
                                                                             faces or [None] * len(raws), weights):
        _, masks = person_and_parts(raw, req) if raw_anatomy.get("automatic_correction_eligible", False) else ({}, {})
        # 옷을 바꿨으면 원본에 맞출 부위는 피부·머리뿐(옷은 옷 그림 것), 아니면 네 부위 모두
        scene = bool(prepared.get("scene"))         # 새 장면이면 밝기는 조명 몫(D-225)
        fixed, shifts = raw, {}
        colour_attempted = raw_anatomy.get("automatic_correction_eligible", False)
        if colour_attempted:
            if with_garment:
                fixed, shifts = colour_match(raw, original, masks, original_masks, parts=IDENTITY_PARTS,
                                              cap=IDENTITY_SHIFT_CAP, scene_lighting=scene)
            else:
                fixed, shifts = colour_match(raw, original, masks, original_masks, parts=PARTS, scene_lighting=scene)
        with WdTagSession(settings) as tagger:
            final_scores = dict(tagger.analyze(Image.fromarray(fixed)).raw_general_scores)
        colour_check = correction_regressions(contract, scores, final_scores)
        final_anatomy = inspect_anatomy(directory, [fixed], req, f"colour-seed{seed}", previous=[raw_anatomy])[0] if raw_anatomy.get("automatic_correction_eligible", False) else raw_anatomy
        colour_check["anatomy_rejected"] = colour_attempted and not final_anatomy.get("automatic_correction_eligible", False)
        colour_applied = colour_attempted and not any(colour_check.values())
        proposed_scores = final_scores
        if colour_attempted and not colour_applied:
            Image.fromarray(fixed).save(directory / f"result-seed{seed}-colour-rejected.png")
            fixed, final_scores = raw, scores
            final_anatomy = raw_anatomy
        scores = final_scores
        diagnostic = diagnose_candidate(contract, scores, style_weight=cand_weight)
        save(directory / f"quality-seed{seed}.json", dict(scores=scores, diagnostic=diagnostic,
                                                        colour_revalidation=colour_check, colour_applied=colour_applied,
                                                        colour_attempted=colour_attempted,
                                                        colour_skip_reason=None if colour_attempted else raw_anatomy["state"],
                                                        proposed_scores=proposed_scores))
        path = directory / f"result-seed{seed}.png"
        Image.fromarray(fixed).save(path)
        candidates.append(dict(seed=seed, file=path.name, raw_file=f"result-raw-seed{seed}.png", sha256=sha(path), seconds=seconds,
                               colour_shift=shifts, colour_applied=colour_applied, quality_diagnostic=diagnostic, hands=final_anatomy["hands"], anatomy=final_anatomy,
                               extra_hands=final_anatomy["state"] == "excess_hands_suspected",
                               missing_body_parts=missing_body_parts(prepared.get("body_parts"), scores),
                               body_part_tags=body_part_tags(scores),
                               identity_anchors=compare_anchors(prepared.get("identity_anchors") or [], scores),
                               autofix=dict(round=pick, asked=watch if pick else [], blocked_reason=autofix.get("blocked_reason"),
                                            still_lost=remaining(target, scores, watch)),
                               face_detail=face, style_lora_weight=cand_weight))
    save(directory / "generation.json", dict(
        state="generated_unreviewed", engine="flux", style_lora=dict(file=str(lora), weight=weight),
        with_garment=with_garment, load_seconds=loaded, candidates=candidates, review="사용자가 후보 중 1장을 고른다", autofix=autofix,
        peak_reserved_gib=peak, memory_cap_gib=MEMORY_CAP_GIB, prompt=prepared["prompt"],
        model=dict(name="black-forest-labs/FLUX.2-klein-4B", path=str(req["flux_model"]), license="Apache-2.0")))


def generate(directory):
    generate_flux(directory, json.loads((directory / "request.json").read_text(encoding="utf-8")))


def combine(directory):
    """부위별 고르기(D-214, CPU). combine-request.json의 두 후보 → 얼굴·머리 후보 + 몸·옷 후보 → combine-result.json."""
    from PIL import Image
    from genai_lab.head_paste_semantics import HeadSegmenter
    from genai_lab.head_paste_worker import figure_parts
    from genai_lab.style_part_combine import combine_parts, head_mask
    req = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    ask = json.loads((directory / "combine-request.json").read_text(encoding="utf-8"))
    from genai_lab.style_redraw_anatomy import require_reviewable
    generation = json.loads((directory / "generation.json").read_text(encoding="utf-8"))
    by_file = {c["file"]: c for c in generation.get("candidates", [])}
    pictures = {}
    for role in ("face", "body"):
        path = directory / ask[f"{role}_candidate"]
        if path.parent != directory or not path.name.startswith("result-seed"):
            raise ValueError("고를 그림은 이 실행 폴더의 후보 그림이어야 합니다.")
        if sha(path) != ask[f"{role}_sha256"]:
            raise ValueError("고른 후보 그림이 바뀌었습니다.")
        require_reviewable(by_file.get(path.name, {}))
        pictures[role] = np.asarray(Image.open(path).convert("RGB"))
    started = time.time()
    seg = HeadSegmenter(Path(req["head_semantic_repo"]), Path(req["head_semantic_checkpoint"]))
    try:
        heads = {role: head_mask(figure_parts(seg, rgb), rgb.shape[:2]) for role, rgb in pictures.items()}
    finally:
        seg.close()
    out, info = combine_parts(pictures["face"], pictures["body"], heads["face"], heads["body"],
                              face_person=person_mask(pictures["face"], req))
    name = f"combined-face{ask['face_seed']}-body{ask['body_seed']}.png"
    Image.fromarray(out).save(directory / name)
    anatomy = inspect_anatomy(directory, [out], req, "combine")[0]
    save(directory / "combine-result.json", dict(
        hands=anatomy["hands"], anatomy=anatomy,
        state="combined_unreviewed", face_candidate=ask["face_candidate"], body_candidate=ask["body_candidate"], file=name,
        sha256=sha(directory / name), seconds=round(time.time() - started, 1), **info))


if __name__ == "__main__":
    stage, folder = sys.argv[1], Path(sys.argv[2])
    try:
        {"prepare": prepare, "generate": generate, "combine": combine}[stage](folder)
    except Exception as error:
        save(folder / f"{stage}-error.json", dict(stage=stage, error_type=type(error).__name__, message=str(error)[:500]))
        raise
