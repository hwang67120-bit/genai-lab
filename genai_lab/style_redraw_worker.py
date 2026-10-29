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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MEMORY_CAP_GIB = 6.5
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


def load_pipeline(models, outfit_count=1):
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
                         weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
                         + ["ip-adapter-plus_sdxl_vit-h.safetensors"] * outfit_count,
                         image_encoder_folder="models/image_encoder", local_files_only=True)
    for module in (pipe.unet, pipe.controlnet):
        module.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    return pipe


def generate(directory):
    import torch
    from genai_lab.character_style_conditioning import ConditioningOptions, build_conditioning_call, dispatch_conditioning_once
    from genai_lab.character_style_inputs import outfit_roles
    from genai_lab.style_redraw_inputs import SCALES, SEEDS
    req = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    manifest = json.loads((directory / "inputs" / "manifest.json").read_text(encoding="utf-8"))
    started = time.time()
    pipe = load_pipeline(Path(req["style_models"]), len(outfit_roles(manifest["profile"])))
    loaded = round(time.time() - started, 1)
    torch.cuda.reset_peak_memory_stats()
    candidates = []
    for number, seed in enumerate(SEEDS):
        # 후보마다 같은 입력, seed만 다르다(D-199). 호출은 한 번만 쓰므로 후보마다 새로 조립한다.
        call = build_conditioning_call(directory / "inputs", ConditioningOptions(**SCALES, seed=seed))
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
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2), memory_cap_gib=MEMORY_CAP_GIB))


if __name__ == "__main__":
    stage, folder = sys.argv[1], Path(sys.argv[2])
    try:
        {"prepare": prepare, "generate": generate}[stage](folder)
    except Exception as error:
        save(folder / f"{stage}-error.json", dict(stage=stage, error_type=type(error).__name__, message=str(error)[:500]))
        raise
