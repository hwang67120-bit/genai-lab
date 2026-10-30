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

MEMORY_CAP_GIB = 6.5
HAND_MODEL = "models--deepghs--anime_hand_detection"
ANIME_SEG = "models--skytnt--anime-seg/snapshots/493cb60893f47441b26ec4fb9a306bce9e342982/isnetis.onnx"
JOINT_SCORE = 0.3
TAG_THRESHOLD = 0.5


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


def count_hands(images, req):
    """손 검출(deepghs anime_hand_detection, 로컬 캐시만). 모델이 없으면 [None, ...]."""
    cache = Path(req.get("hand_cache", ""))
    if not (cache / HAND_MODEL).is_dir():
        return [None] * len(images)
    # huggingface_hub는 가져올 때 캐시 위치를 읽으므로 환경 변수 대신 상수를 바꾼다(오프라인, 이 폴더만).
    import huggingface_hub.constants as hub
    hub.HF_HUB_CACHE = str(cache)
    from PIL import Image
    from imgutils.detect import detect_hands
    try:
        return [len(detect_hands(Image.fromarray(np.asarray(im)))) for im in images]
    except Exception:                               # 손 검출은 보조 표시 — 실패해도 생성은 계속한다
        return [None] * len(images)


def prepare_flux(directory, req, canvas, rgb, scores, safety, joints, face, started):
    """FLUX 경로 준비(D-210): 특징 문장·원본 부위 마스크·원본 손 개수. 그림체 참조·의상 참조·입력 묶음은 만들지 않는다."""
    from genai_lab.character_features import extract_features
    from genai_lab.studio_controller import guessed_gender
    from genai_lab.style_redraw_flux import flux_prompt
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
            garment_safety = safety_observation(dict(tagger.analyze(garment.convert("RGB")).raw_general_scores))
        if garment_safety["blocked"] or garment_safety["review"]:
            save(directory / "prepare.json", dict(state="refused", code="garment_safety",
                                                  message="이 옷 그림은 사용할 수 없습니다.", safety=garment_safety))
            return
    parts, masks = person_and_parts(rgb, req)
    guessed = guessed_gender(dict(character_scores=scores))
    confirmed = req.get("confirmed_gender")
    features = extract_features(rgb, parts, scores, gender=confirmed or "unknown")
    traits = trait_rows(features, scores, confirmed)
    np.savez_compressed(directory / "original-masks.npz", **masks)
    hands = count_hands([rgb], req)[0]
    save(directory / "prepare.json", dict(
        state="ready", engine="flux", with_garment=with_garment, seconds=round(time.time() - started, 1), gender=confirmed,
        guessed_gender=guessed, traits=traits, prompt=flux_prompt(traits, with_garment), face_box=face, original_hands=hands,
        warnings=support_warnings(scores, joints), safety=safety, joints_used=sorted(joints),
        note="특징 문장·성별은 자동 측정이며 사람 확인 전이다. 자세·구도는 원본을 따른다."))


def generate_flux(directory, req):
    """FLUX.2-klein 편집으로 후보 3장(D-210) → 부위별 색 맞춤 → 손 개수. 원래 생성 그림(raw)도 남긴다."""
    import gc
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import (GUIDANCE, IDENTITY_PARTS, IDENTITY_SHIFT_CAP, MAX_SEQUENCE_LENGTH, PARTS, SEEDS, STEPS,
                                             colour_match, hand_flag)
    prepared = json.loads((directory / "prepare.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    started = time.time()
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(str(req["flux_model"]), torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    loaded = round(time.time() - started, 1)
    reference = Image.open(directory / "character.png").convert("RGB")
    with_garment = bool(prepared.get("with_garment"))
    references = [reference] + ([Image.open(directory / "garment.png").convert("RGB")] if with_garment else [])
    torch.cuda.reset_peak_memory_stats()
    raws = []
    for number, seed in enumerate(SEEDS):
        save(directory / "progress.json", dict(candidate=number + 1, candidates=len(SEEDS), step=0, total=STEPS))
        t0 = time.time()
        image = pipe(image=references, prompt=prepared["prompt"], width=reference.width, height=reference.height,
                     num_inference_steps=STEPS, guidance_scale=GUIDANCE, max_sequence_length=MAX_SEQUENCE_LENGTH,
                     generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
        image.save(directory / f"result-raw-seed{seed}.png")
        raws.append((seed, np.asarray(image), round(time.time() - t0, 1)))
    peak = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()
    original = np.asarray(reference)
    original_masks = dict(np.load(directory / "original-masks.npz"))
    hands = count_hands([r for _, r, _ in raws], req)
    candidates = []
    for (seed, raw, seconds), count in zip(raws, hands):
        _, masks = person_and_parts(raw, req)
        # 옷을 바꿨으면 원본에 맞출 부위는 피부·머리뿐(옷은 옷 그림 것), 아니면 네 부위 모두
        fixed, shifts = (colour_match(raw, original, masks, original_masks, parts=IDENTITY_PARTS, cap=IDENTITY_SHIFT_CAP)
                         if with_garment else colour_match(raw, original, masks, original_masks, parts=PARTS))
        path = directory / f"result-seed{seed}.png"
        Image.fromarray(fixed).save(path)
        candidates.append(dict(seed=seed, file=path.name, raw_file=f"result-raw-seed{seed}.png", sha256=sha(path), seconds=seconds,
                               colour_shift=shifts, hands=count, extra_hands=hand_flag(prepared.get("original_hands"), count)))
    save(directory / "generation.json", dict(
        state="generated_unreviewed", engine="flux", with_garment=with_garment, load_seconds=loaded, candidates=candidates, review="사용자가 후보 중 1장을 고른다",
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
    pictures = {}
    for role in ("face", "body"):
        path = directory / ask[f"{role}_candidate"]
        if path.parent != directory or not path.name.startswith("result-seed"):
            raise ValueError("고를 그림은 이 실행 폴더의 후보 그림이어야 합니다.")
        if sha(path) != ask[f"{role}_sha256"]:
            raise ValueError("고른 후보 그림이 바뀌었습니다.")
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
    save(directory / "combine-result.json", dict(
        state="combined_unreviewed", face_candidate=ask["face_candidate"], body_candidate=ask["body_candidate"], file=name,
        sha256=sha(directory / name), seconds=round(time.time() - started, 1), **info))


if __name__ == "__main__":
    stage, folder = sys.argv[1], Path(sys.argv[2])
    try:
        {"prepare": prepare, "generate": generate, "combine": combine}[stage](folder)
    except Exception as error:
        save(folder / f"{stage}-error.json", dict(stage=stage, error_type=type(error).__name__, message=str(error)[:500]))
        raise
