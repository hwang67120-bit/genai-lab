"""모드 B 작업 프로세스(STYLE_MODE_APP_DESIGN.md S4). 요청 폴더의 request.json만 읽고 결과를 같은 폴더에 쓴다.

prepare (CPU): 태그·안전 검사 → 배경 분리·분할 → 의상 참조·그림체 참조 색 맞춤 → 특징 문장 → 입력 묶음(inputs/) → prepare.json
generate (GPU): SDXL + 자세 ControlNet + IP-Adapter 3개, fp8 저장 + CPU 내려 두기, 상한 6.5GiB, 1장 → result.png, generation.json
관절(pose.json)은 이 프로세스 전에 제품 관절 분석(studio_analysis pose)이 만든다.
사용: python style_redraw_worker.py <prepare|generate> <directory>
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
ENGINE_DEFAULT = "flux"                 # D-210: FLUX.2-klein 편집. "sdxl_lora" = 이전 RealVisXL + 스타일 LoRA + IP-Adapter 경로
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
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from genai_lab.character_features import extract_features
    from genai_lab.character_style_inputs import prepare_character_style_bundle
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.free_generation import safety_observation
    from genai_lab.studio_controller import guessed_gender
    from genai_lab.style_redraw_inputs import (HAIR_SCALE, STYLE_PLAN, build_annotations, face_box, fitted_pose, framing_for,
                                               hair_reference, measured_proportions, outfit_references, trait_rows,
                                               StyleRedrawInputError)
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
    if req.get("engine", ENGINE_DEFAULT) == "flux":
        return prepare_flux(directory, req, canvas, rgb, scores, safety, joints, face, started)
    from genai_lab.proportion_foreground import AnimeForeground
    from genai_lab.head_paste_semantics import HeadSegmenter
    from genai_lab.head_paste_worker import figure_parts
    from genai_lab.style_reference_colors import recolor_reference
    reference = Image.open(req["style_reference"]).convert("RGB")
    reference_joints = json.loads(Path(req["style_reference_pose"]).read_text(encoding="utf-8"))["joints"]
    segmenter_model = Path(req["model_cache"]) / ANIME_SEG
    fg = AnimeForeground(SimpleNamespace(foreground_model=segmenter_model, foreground_sha256=sha(segmenter_model)))
    try:
        reference_alpha = fg.alpha(reference).astype(np.float32) / 255.0
        source_alpha = fg.alpha(canvas).astype(np.float32) / 255.0
    finally:
        fg.close()
    # 생성 자세: 정면 서 있는 자세를 원본 비율로(D-201). 참조 적용 영역도 이 자세에서 잡는다.
    proportions, proportion_source = measured_proportions(joints, face)
    framing = framing_for(joints)                   # 발목이 없으면 무릎 위까지만(D-202)
    pose_joints, pose_boxes = fitted_pose(proportions, framing=framing)
    hips = [joints[k][1] for k in ("right_hip", "left_hip") if k in joints]
    try:
        pieces = outfit_references(canvas, source_alpha, joints.get("neck", [0, canvas.height * 0.22])[1],
                                   sum(hips) / len(hips) if hips else None, pose_boxes)
    except StyleRedrawInputError as error:
        return refuse("outfit", str(error))
    outfit_paths = {}
    for name, piece, _crop, _box in pieces:
        outfit_paths[name] = directory / f"outfit-{name}.png"
        piece.save(outfit_paths[name])
    seg = HeadSegmenter(Path(req["head_semantic_repo"]), Path(req["head_semantic_checkpoint"]))
    try:
        reference_rgb = np.asarray(reference)
        source_parts = figure_parts(seg, rgb)
        styled, colour_record = recolor_reference(reference_rgb, figure_parts(seg, reference_rgb), rgb, source_parts, STYLE_PLAN,
                                                  reference_alpha, reference_joints=reference_joints, source_joints=joints)
    finally:
        seg.close()
    Image.fromarray(styled).save(directory / "style-reference.png")
    try:
        hair, hair_crop = hair_reference(canvas, source_alpha, source_parts["hair"], source_parts["face"])
    except StyleRedrawInputError as error:
        return refuse("hair", str(error))
    outfit_paths["hair"] = directory / "outfit-hair.png"
    hair.save(outfit_paths["hair"])
    pieces.append(("hair", hair, hair_crop, list(pose_boxes["head"]), HAIR_SCALE))
    # 자동 성별 추정은 기록만 한다. 문장에는 사람이 확정한 성별(request의 confirmed_gender)만 쓴다(D-198).
    guessed = guessed_gender(dict(character_scores=scores))
    confirmed = req.get("confirmed_gender")
    features = extract_features(rgb, source_parts, scores, gender=confirmed or "unknown")
    traits = trait_rows(features, scores, confirmed)
    annotations = build_annotations(sha(canvas_path), sha(directory / "style-reference.png"),
                                    [(p[0], sha(outfit_paths[p[0]]), *p[3:]) for p in pieces],
                                    face, traits, confirmed, pose_joints, pose_boxes["face"], framing)
    save(directory / "annotations.json", annotations)
    prepare_character_style_bundle(canvas_path, directory / "style-reference.png", annotations, directory / "inputs", outfit_paths)
    save(directory / "colour-record.json", colour_record)
    save(directory / "prepare.json", dict(
        state="ready", seconds=round(time.time() - started, 1), gender=confirmed, guessed_gender=guessed, traits=traits, face_box=face,
        outfits={p[0]: dict(crop=p[2], box=p[3]) for p in pieces}, proportions=proportions, proportion_source=proportion_source,
        framing=framing,
        warnings=support_warnings(scores, joints), safety=safety,
        colour={k: v["state"] for k, v in colour_record.items()}, joints_used=sorted(joints),
        note="특징 문장·성별은 자동 측정이며 사람 확인 전이다."))


def person_and_parts(rgb, req):
    """사람 영역(anime-seg)과 부위 분할(See-through). 색 맞춤용 네 부위 마스크를 돌려준다."""
    from PIL import Image
    from types import SimpleNamespace
    from genai_lab.proportion_foreground import AnimeForeground
    from genai_lab.head_paste_semantics import HeadSegmenter
    from genai_lab.head_paste_worker import figure_parts
    from genai_lab.style_redraw_flux import part_masks
    model = Path(req["model_cache"]) / ANIME_SEG
    fg = AnimeForeground(SimpleNamespace(foreground_model=model, foreground_sha256=sha(model)))
    try:
        person = fg.alpha(Image.fromarray(rgb)) > 127
    finally:
        fg.close()
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
    parts, masks = person_and_parts(rgb, req)
    guessed = guessed_gender(dict(character_scores=scores))
    confirmed = req.get("confirmed_gender")
    features = extract_features(rgb, parts, scores, gender=confirmed or "unknown")
    traits = trait_rows(features, scores, confirmed)
    np.savez_compressed(directory / "original-masks.npz", **masks)
    hands = count_hands([rgb], req)[0]
    save(directory / "prepare.json", dict(
        state="ready", engine="flux", seconds=round(time.time() - started, 1), gender=confirmed, guessed_gender=guessed,
        traits=traits, prompt=flux_prompt(traits), face_box=face, original_hands=hands,
        warnings=support_warnings(scores, joints), safety=safety, joints_used=sorted(joints),
        note="특징 문장·성별은 자동 측정이며 사람 확인 전이다. 자세·구도는 원본을 따른다."))


def generate_flux(directory, req):
    """FLUX.2-klein 편집으로 후보 3장(D-210) → 부위별 색 맞춤 → 손 개수. 원래 생성 그림(raw)도 남긴다."""
    import gc
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, SEEDS, STEPS, colour_match, hand_flag
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
    torch.cuda.reset_peak_memory_stats()
    raws = []
    for number, seed in enumerate(SEEDS):
        save(directory / "progress.json", dict(candidate=number + 1, candidates=len(SEEDS), step=0, total=STEPS))
        t0 = time.time()
        image = pipe(image=[reference], prompt=prepared["prompt"], width=reference.width, height=reference.height,
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
        fixed, shifts = colour_match(raw, original, masks, original_masks)
        path = directory / f"result-seed{seed}.png"
        Image.fromarray(fixed).save(path)
        candidates.append(dict(seed=seed, file=path.name, raw_file=f"result-raw-seed{seed}.png", sha256=sha(path), seconds=seconds,
                               colour_shift=shifts, hands=count, extra_hands=hand_flag(prepared.get("original_hands"), count)))
    save(directory / "generation.json", dict(
        state="generated_unreviewed", engine="flux", load_seconds=loaded, candidates=candidates, review="사용자가 후보 중 1장을 고른다",
        peak_reserved_gib=peak, memory_cap_gib=MEMORY_CAP_GIB, prompt=prepared["prompt"],
        model=dict(name="black-forest-labs/FLUX.2-klein-4B", path=str(req["flux_model"]), license="Apache-2.0")))


def load_pipeline(models, outfit_count=1, loras=None, style_adapter=True):
    """loras: {이름: LoRA 폴더}. fp8 보관 전에 얹고, LoRA 층은 fp8로 바꾸지 않는다(바꾸면 LoRA 계산이 fp8에서 멈춘다, D-204).
    style_adapter=False: 그림체 참조 어댑터를 불러오지 않는다(그림체는 스타일 LoRA)."""
    import torch
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionXLControlNetPipeline
    controlnet = ControlNetModel.from_pretrained(str(models / "extra/pose"), torch_dtype=torch.float16, local_files_only=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(str(models / "base/realvis"), controlnet=controlnet,
                                                               torch_dtype=torch.float16, variant="fp16", local_files_only=True,
                                                               add_watermarker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++",
                                                             use_karras_sigmas=True)
    # 순서 고정: 그림체(plus) · 얼굴(plus-face) · 의상 참조마다(plus). 이미지 인코더(ViT-H)는 배포 전 결정 대상(D-197).
    pipe.load_ip_adapter(str(models / "extra/ip"), subfolder="sdxl_models",
                         weight_name=(["ip-adapter-plus_sdxl_vit-h.safetensors"] if style_adapter else [])
                         + ["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
                         + ["ip-adapter-plus_sdxl_vit-h.safetensors"] * outfit_count,
                         image_encoder_folder="models/image_encoder", local_files_only=True)
    for name, folder in (loras or {}).items():
        # 오프라인(HF_HUB_OFFLINE=1)에서는 파일 이름을 직접 줘야 한다(diffusers가 이름을 추측하지 않음).
        pipe.load_lora_weights(str(folder), weight_name="pytorch_lora_weights.safetensors", adapter_name=name)
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN
    # LoRA가 없으면 예전과 같은 기본값(건너뛸 층 지정 없음). 있으면 기본 목록 + LoRA 층.
    extra = dict(skip_modules_pattern=tuple(DEFAULT_SKIP_MODULES_PATTERN) + ("lora",)) if loras else {}
    for module in (pipe.unet, pipe.controlnet):
        module.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16, **extra)
    pipe.enable_model_cpu_offload()
    return pipe


def generate(directory):
    import torch
    from genai_lab.character_style_conditioning import ConditioningOptions, build_conditioning_call, dispatch_conditioning_once
    from genai_lab.character_style_inputs import outfit_roles
    from genai_lab.style_redraw_inputs import SCALES, SEEDS, STYLE_LORA_WEIGHT
    req = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if req.get("engine", ENGINE_DEFAULT) == "flux":
        return generate_flux(directory, req)
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    manifest = json.loads((directory / "inputs" / "manifest.json").read_text(encoding="utf-8"))
    started = time.time()
    # 그림체는 스타일 LoRA(D-204), 그림체 참조 어댑터는 불러오지 않는다.
    lora = Path(req["style_lora"])
    pipe = load_pipeline(Path(req["style_models"]), len(outfit_roles(manifest["profile"])), loras={"style": lora}, style_adapter=False)
    pipe.set_adapters(["style"], adapter_weights=[STYLE_LORA_WEIGHT])
    loaded = round(time.time() - started, 1)
    torch.cuda.reset_peak_memory_stats()
    candidates = []
    for number, seed in enumerate(SEEDS):
        # 후보마다 같은 입력, seed만 다르다(D-199). 호출은 한 번만 쓰므로 후보마다 새로 조립한다.
        call = build_conditioning_call(directory / "inputs", ConditioningOptions(**SCALES, seed=seed, style_visual=False))
        total = call.kwargs["num_inference_steps"]
        def progress(pipeline, step, timestep, kwargs, number=number, total=total):
            save(directory / "progress.json", dict(candidate=number + 1, candidates=len(SEEDS), step=step + 1, total=total))
            return kwargs
        t0 = time.time()
        with torch.inference_mode():
            image, evidence = dispatch_conditioning_once(pipe, call, execution_authorized=True, callback=progress)
        path = directory / f"result-seed{seed}.png"
        image.save(path)
        candidates.append(dict(seed=seed, file=path.name, sha256=sha(path), seconds=round(time.time() - t0, 1), evidence=evidence))
    save(directory / "generation.json", dict(
        state="generated_unreviewed", load_seconds=loaded, candidates=candidates, review="사용자가 후보 중 1장을 고른다",
        style=dict(kind="lora", path=str(lora), weight=STYLE_LORA_WEIGHT,
                   sha256=sha(lora / "pytorch_lora_weights.safetensors") if (lora / "pytorch_lora_weights.safetensors").is_file() else None),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2), memory_cap_gib=MEMORY_CAP_GIB))


def refine(directory):
    """고른 후보 1장 다듬기(D-203). refine-request.json의 후보 파일·seed를 읽어 refined-seed<n>.png와 refine-result.json을 쓴다."""
    import torch
    from PIL import Image
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionXLControlNetImg2ImgPipeline
    from diffusers.image_processor import IPAdapterMaskProcessor
    from genai_lab.character_style_inputs import read_verified_bundle
    from genai_lab.style_redraw_refine import (REFINE_FACE_SCALE, REFINE_STEPS, REFINE_STRENGTH, REFINE_STRUCTURE_SCALE,
                                               REFINE_STYLE_SCALE, colour_back, refine_prompts, style_mask_without_garments)
    req = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    ask = json.loads((directory / "refine-request.json").read_text(encoding="utf-8"))
    chosen_path = directory / ask["candidate"]
    if chosen_path.parent != directory or not chosen_path.name.startswith("result-seed"):
        raise ValueError("다듬을 후보는 이 실행 폴더의 후보 그림이어야 합니다.")
    if sha(chosen_path) != ask["candidate_sha256"]:
        raise ValueError("고른 후보 그림이 바뀌었습니다.")
    manifest, images = read_verified_bundle(directory / "inputs")
    try:
        profile, canvas = manifest["profile"], tuple(manifest["canvas"])
        torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        models = Path(req["style_models"])
        started = time.time()
        controlnet = ControlNetModel.from_pretrained(str(models / "extra/pose"), torch_dtype=torch.float16, local_files_only=True)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(str(models / "base/realvis"), controlnet=controlnet,
                                                                          torch_dtype=torch.float16, variant="fp16", local_files_only=True,
                                                                          add_watermarker=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++",
                                                                 use_karras_sigmas=True)
        pipe.load_ip_adapter(str(models / "extra/ip"), subfolder="sdxl_models",
                             weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"],
                             image_encoder_folder="models/image_encoder", local_files_only=True)
        for module in (pipe.unet, pipe.controlnet):
            module.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        loaded = round(time.time() - started, 1)
        processor = IPAdapterMaskProcessor()
        style_mask = processor.preprocess(style_mask_without_garments(profile, canvas), height=canvas[1], width=canvas[0])
        face_mask = processor.preprocess(images["identity_mask"], height=canvas[1], width=canvas[0])
        positive, negative = refine_prompts(profile)
        chosen = Image.open(chosen_path).convert("RGB")
        target = Image.open(req["style_reference"]).convert("RGB")      # 깎지 않은 그림체 참조
        pipe.set_ip_adapter_scale([{"up": {"block_0": [0.0, REFINE_STYLE_SCALE, 0.0]}}, REFINE_FACE_SCALE])
        def progress(pipeline, step, timestep, kwargs):
            save(directory / "progress.json", dict(stage="refine", step=step + 1, total=int(REFINE_STEPS * REFINE_STRENGTH)))
            return kwargs
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            raw = pipe(prompt=positive, prompt_2=positive, negative_prompt=negative, image=chosen, control_image=images["pose"].copy(),
                       strength=REFINE_STRENGTH, ip_adapter_image=[target, images["identity"].copy()],
                       cross_attention_kwargs=dict(ip_adapter_masks=[style_mask, face_mask]), num_inference_steps=REFINE_STEPS,
                       guidance_scale=5.0, controlnet_conditioning_scale=REFINE_STRUCTURE_SCALE,
                       generator=torch.Generator(device="cpu").manual_seed(int(ask["seed"])),
                       callback_on_step_end=progress).images[0]
    finally:
        for image in images.values():
            image.close()
    seed = int(ask["seed"])
    raw.save(directory / f"refined-raw-seed{seed}.png")
    Image.fromarray(colour_back(np.asarray(chosen), np.asarray(raw))).save(directory / f"refined-seed{seed}.png")
    save(directory / "refine-result.json", dict(
        state="refined_unreviewed", candidate=ask["candidate"], seed=seed, file=f"refined-seed{seed}.png",
        raw_file=f"refined-raw-seed{seed}.png", sha256=sha(directory / f"refined-seed{seed}.png"), load_seconds=loaded,
        refine_seconds=round(time.time() - t0, 1), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
        settings=dict(strength=REFINE_STRENGTH, steps=REFINE_STEPS, style_scale=REFINE_STYLE_SCALE, face_scale=REFINE_FACE_SCALE,
                      style_off_on_garments=True, colour_back=True), prompt=positive, negative_prompt=negative))


def face(directory):
    """고른 후보 1장 얼굴 다듬기(D-206). face-request.json의 후보를 읽어 face-detail-seed<n>.png와 face-result.json을 쓴다."""
    import torch
    from PIL import Image
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN
    from genai_lab.style_redraw_face import (FACE_LORA_WEIGHT, FACE_REF_SCALE, FACE_SIZE, FACE_STEPS, FACE_STRENGTH, eye_colour_back,
                                             face_crop_box, face_prompts, paste_face)
    req = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    ask = json.loads((directory / "face-request.json").read_text(encoding="utf-8"))
    chosen_path = directory / ask["candidate"]
    if chosen_path.parent != directory or not chosen_path.name.startswith("result-seed"):
        raise ValueError("다듬을 후보는 이 실행 폴더의 후보 그림이어야 합니다.")
    if sha(chosen_path) != ask["candidate_sha256"]:
        raise ValueError("고른 후보 그림이 바뀌었습니다.")
    profile = json.loads((directory / "annotations.json").read_text(encoding="utf-8"))
    chosen = Image.open(chosen_path).convert("RGB")
    box = face_crop_box(profile.get("identity_mask_box", profile["face_box"]), chosen.size)
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    models, started = Path(req["style_models"]), time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(str(models / "base/realvis"), torch_dtype=torch.float16, variant="fp16",
                                                            local_files_only=True, add_watermarker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
    pipe.load_ip_adapter(str(models / "extra/ip"), subfolder="sdxl_models", weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"],
                         image_encoder_folder="models/image_encoder", local_files_only=True)
    pipe.load_lora_weights(str(req["style_lora"]), weight_name="pytorch_lora_weights.safetensors", adapter_name="style")
    pipe.unet.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16,
                                       skip_modules_pattern=tuple(DEFAULT_SKIP_MODULES_PATTERN) + ("lora",))
    pipe.enable_model_cpu_offload()
    pipe.set_adapters(["style"], adapter_weights=[FACE_LORA_WEIGHT])
    pipe.set_ip_adapter_scale(FACE_REF_SCALE)
    loaded = round(time.time() - started, 1)
    positive, negative = face_prompts(profile)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        redrawn = pipe(prompt=positive, prompt_2=positive, negative_prompt=negative,
                       image=chosen.crop(tuple(box)).resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS), strength=FACE_STRENGTH,
                       ip_adapter_image=Image.open(directory / "inputs" / "identity-face.png").convert("RGB"),
                       num_inference_steps=FACE_STEPS, guidance_scale=5.0,
                       generator=torch.Generator(device="cpu").manual_seed(int(ask["seed"]))).images[0]
    peak = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    del pipe
    torch.cuda.empty_cache()
    merged = paste_face(chosen, redrawn, box)
    from genai_lab.head_paste_semantics import HeadSegmenter
    from genai_lab.head_paste_worker import figure_parts
    seg = HeadSegmenter(Path(req["head_semantic_repo"]), Path(req["head_semantic_checkpoint"]))
    try:
        parts = figure_parts(seg, np.asarray(merged))
    finally:
        seg.close()
    final, eye_px = eye_colour_back(np.asarray(merged), np.asarray(chosen), parts["eyes"], parts.get("face"))
    seed = int(ask["seed"])
    merged.save(directory / f"face-detail-raw-seed{seed}.png")
    Image.fromarray(final).save(directory / f"face-detail-seed{seed}.png")
    save(directory / "face-result.json", dict(
        state="face_detailed_unreviewed", candidate=ask["candidate"], seed=seed, file=f"face-detail-seed{seed}.png",
        raw_file=f"face-detail-raw-seed{seed}.png", sha256=sha(directory / f"face-detail-seed{seed}.png"), box=box,
        eye_pixels_recoloured=eye_px, load_seconds=loaded, face_seconds=round(time.time() - t0, 1), peak_reserved_gib=peak,
        settings=dict(strength=FACE_STRENGTH, steps=FACE_STEPS, lora_weight=FACE_LORA_WEIGHT, face_ref_scale=FACE_REF_SCALE,
                      size=FACE_SIZE), prompt=positive, negative_prompt=negative))


if __name__ == "__main__":
    stage, folder = sys.argv[1], Path(sys.argv[2])
    try:
        {"prepare": prepare, "generate": generate, "refine": refine, "face": face}[stage](folder)
    except Exception as error:
        save(folder / f"{stage}-error.json", dict(stage=stage, error_type=type(error).__name__, message=str(error)[:500]))
        raise
