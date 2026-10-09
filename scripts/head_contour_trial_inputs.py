"""R6 자세·머리 스케치 시험의 고정 입력이다. CPU만 사용한다."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path

from genai_lab.onepass_generation import OnePassInputs, read_inputs, validate_local_models
from genai_lab.onepass_generation_settings import OnePassGenerationSettings
from genai_lab.onepass_pose import InputDecision
from genai_lab.onepass_prompt import OnePassPrompt, EncoderChunkPlan, PromptChunk

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "outputs/head-contour-adapter-design-20261007"
BASE_DIR = ROOT / "outputs/body-proportion-target-ab-20261007"
POSE_DIR = ROOT / "outputs/body-proportion-control-20261007/cpu-01/R6"
SKETCH_ID = "TencentARC/t2i-adapter-sketch-sdxl-1.0"
REVISION = "cc3c4e3362296c6825c370b83838306723ece983"
MODEL_HASHES = {
    "config.json": "2f5ed7aee869a54b61f0d377147eb1cdf4aceffeb0e28b0589116415380de89f",
    "diffusion_pytorch_model.fp16.safetensors": "9f88c53368d53bec6b70fc9d797ec12b88a512d2a7e9068d11fbf621917d9fdd",
}
CONTROL_SHA = "2d83e8afa7891350d97d01d9956e9708a8d5ec006fa738bf20cc0361472dfb95"
FACE_SHA = "9b6415076df8dd2cb35f8b9cb8dd877c050ed5ef386e01336f4e3a1f1d46d6ad"
CONTOUR_SHA = "dedfc13657050ade10a3616797252ba53cc9f103fd48de01e619b06c53f065ab"
BASE_SHAS = {
    209212003: "116a279a4d2a7bbb3a1491560f87262356d216ef9555cb8a225befe6a33e135e",
    209212001: "526a6bcd2b94cc60595d9f613fdafdc1df43cc1a09d8ac3803470169a24adc81",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def restore_request(plan):
    """모든 프롬프트 토큰과 입력 항목을 유지하며 저장된 BASE 요청을 복원한다."""
    data = dict(plan["inputs"]["BASE"])
    prompt = dict(data.pop("prompt"))
    encoders = []
    for saved in prompt.pop("encoders"):
        values = dict(saved)
        for polarity in ("positive", "negative"):
            values[polarity] = tuple(PromptChunk(c["kind"], c["text"], tuple(c["token_ids"]))
                                     for c in values[polarity])
        encoders.append(EncoderChunkPlan(**values))
    data["prompt"] = OnePassPrompt(encoders=tuple(encoders), **prompt)
    for name in ("control_file", "face_file"):
        data[name] = Path(data[name]) if data[name] else None
    if data["input_decision"] is not None:
        d = data["input_decision"]
        data["input_decision"] = InputDecision(d["status"], tuple(d["rejects"]),
                                               tuple(d["warnings"]), tuple(d["choices"]))
    inputs = OnePassInputs(**data)
    settings_data = dict(plan["settings"])
    for name in ("model_root", "adapter_root", "ip_root"):
        settings_data[name] = Path(settings_data[name])
    settings = OnePassGenerationSettings(**settings_data)
    require(canonical(asdict(inputs)) == canonical(plan["inputs"]["BASE"]), "BASE changed during decoding")
    return inputs, settings


def verify_locks(folder):
    locked = {}
    for line in (folder / "sha256_inputs.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()), "Lock path escapes design folder")
        require(sha(path) == digest, "Locked input changed: " + name)
        locked[name] = digest
    require("head_draft/contour_4px.png" in locked, "Contour lock missing")
    return locked


def verify_download(record_path):
    """로드 전에 오래된 리비전·내용 불일치·불완전 다운로드를 거부한다."""
    record = read(record_path)
    require(record["repo"] == SKETCH_ID and record["revision"] == REVISION, "Sketch revision mismatch")
    files = record["files"]
    require(set(files) == set(MODEL_HASHES), "Unexpected sketch file manifest")
    roots = set()
    for name, digest in MODEL_HASHES.items():
        item = files[name]
        path = Path(item["path"])
        require(path.name == name, "Sketch filename mismatch")
        require(item["match"] is True and item["sha256"] == digest, "Download did not pass SHA validation")
        require(sha(path) == digest, "Sketch content SHA mismatch")
        require(path.stat().st_size == item["size"] == item["remote_size"], "Sketch size mismatch")
        if name.endswith("safetensors"):
            require(item["remote_lfs_sha256"] == digest, "Remote weight SHA mismatch")
        roots.add(path.parent.resolve())
    require(len(roots) == 1, "Sketch files have different roots")
    return roots.pop(), record


def check_geometry(contour_path, mask_path, joints, size=(736, 1232)):
    """이미 정규화된 픽셀을 검사한다. 크기 변경·재작성·다른 마스크 추정은 없다."""
    import cv2
    import numpy as np
    from PIL import Image
    with Image.open(contour_path) as im:
        require(im.size == size and im.mode == "RGB", "Contour must be locked RGB canvas")
        pixels = np.asarray(im)
    require(np.isin(pixels, [0, 255]).all(), "Contour is not binary")
    require((pixels[..., 0] == pixels[..., 1]).all() and
            (pixels[..., 0] == pixels[..., 2]).all(), "Contour is not monochrome")
    with Image.open(mask_path) as im:
        require(im.size == size, "Head mask canvas mismatch")
        mask = np.asarray(im.convert("L"))
    require(np.isin(mask, [0, 255]).all(), "Head mask is not binary")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    require(len(contours) == 1, "Expected one confirmed head boundary")
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    white = pixels[..., 0] > 0
    require(white.any(), "Empty contour")
    distance = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    require(float(distance[white].max()) <= 2.001, "Contour pixels exceed boundary tolerance 2px")
    reverse = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    require(float(reverse[boundary > 0].max()) <= 2.001, "Contour boundary coverage incomplete")
    expected = np.zeros_like(pixels)
    cv2.drawContours(expected, contours, -1, (255, 255, 255), 4)
    require(np.array_equal(expected, pixels), "Contour is not the locked 4px outline")
    points = {j["joint_name"]: j for j in joints}
    nose, neck = points["nose"], points["neck"]
    for point in (nose, neck):
        require(point["detected"] and point["confidence_score"] >= .3, "Nose/neck invalid")
        require(0 <= point["x"] < size[0] and 0 <= point["y"] < size[1], "Nose/neck outside canvas")
    require(mask[round(nose["y"]), round(nose["x"])] > 0, "Nose outside head")
    ys, xs = np.where(mask > 0)
    bottom, height = int(ys.max()), int(ys.max() - ys.min() + 1)
    gap = neck["y"] - bottom
    # 정렬 진단의 허용 범위다. 새로 추정한 인체 측정값이 아니다.
    require(xs.min() <= neck["x"] <= xs.max() and 0 <= gap <= .25 * height,
            "Neck is not just below confirmed head (0..25% head-box height)")
    return {"white_pixels": int(white.sum()), "boundary_max_distance_px": float(distance[white].max()),
            "neck_gap_px": gap, "neck_gap_limit_px": .25 * height,
            "canvas": list(size), "coordinate_transform": "none; locked normalized canvas",
            "nose": nose, "neck": neck, "anatomical_ratio_verified": False}


def model_manifest(settings, sketch_record):
    pose_files = {name: {"path": str(settings.adapter_root / name), "sha256": sha(settings.adapter_root / name)}
                  for name in ("config.json", "diffusion_pytorch_model.safetensors")}
    return [
        {"id": "TencentARC/t2i-adapter-openpose-sdxl-1.0", "path": str(settings.adapter_root),
         "revision": None, "revision_note": "flat existing cache; files identified by SHA", "files": pose_files},
        {"id": SKETCH_ID, "path": str(Path(sketch_record["files"]["config.json"]["path"]).parent),
         "revision": sketch_record["revision"], "files": sketch_record["files"]},
    ]


def preflight():
    """모델 추론 없이 변경 불가 근거와 로컬 파일을 검증한다."""
    locks = verify_locks(DESIGN)
    require(sha(BASE_DIR / "plan.json") == (BASE_DIR / "plan.sha256").read_text().strip(), "BASE plan changed")
    plan = read(BASE_DIR / "plan.json")
    inputs, settings = restore_request(plan)
    require((settings.width, settings.height, settings.steps, settings.guidance_scale,
             settings.adapter_scale, settings.adapter_steps, settings.ip_start, settings.ip_scale,
             settings.max_reserved_gib) == (736, 1232, 28, 5.5, 1.2, 11, 11, .9, 6.5), "Trial settings changed")
    require(inputs.ip_early == .5 and inputs.pose_mode == "with_pose", "Baseline face/pose settings changed")
    require(inputs.control_sha256 == CONTROL_SHA and inputs.face_sha256 == FACE_SHA, "Baseline references changed")
    require("body_proportion_condition" not in inputs.prompt.rules, "Numeric ratio condition is forbidden")
    for image in read_inputs(inputs, settings):
        image.close()
    validate_local_models(settings)
    confirmed = read(DESIGN / "head_draft/head_confirmed.json")
    require(confirmed["status"] == "user_confirmed" and confirmed["thickness_locked"] == 4, "Head not confirmed")
    require(sha(DESIGN / "head_draft/contour_4px.png") == CONTOUR_SHA, "Wrong contour")
    require(sha(POSE_DIR / "normalized.png") == confirmed["normalized_sha256"], "Normalized image changed")
    require(sha(POSE_DIR / "control.png") == confirmed["control_sha256"] == CONTROL_SHA, "Different pose canvas")
    observations = read(POSE_DIR / "observations.json")
    second = observations["observations"][1]
    require(second["size"] == [736, 1232], "Second-pass pose uses different coordinates")
    geometry = check_geometry(DESIGN / "head_draft/contour_4px.png", DESIGN / "head_draft/head_mask_draft.png", second["joints"])
    sketch_root, download = verify_download(DESIGN / "download.json")
    pose_config, sketch_config = read(settings.adapter_root / "config.json"), read(sketch_root / "config.json")
    for key in ("adapter_type", "channels", "downscale_factor", "in_channels", "num_res_blocks"):
        require(pose_config[key] == sketch_config[key], "Adapter structure mismatch: " + key)
    for seed, digest in BASE_SHAS.items():
        prior = BASE_DIR / "generation" / f"BASE_{seed}"
        record = read(prior / "run.json")
        require(record["valid"] and record["raw_sha256"] == sha(prior / "raw.png") == digest, "Baseline raw invalid")
        require(canonical(record["prompt"]) == canonical(asdict(inputs.prompt)), "Baseline prompt mismatch")
    protected = [BASE_DIR / "plan.json", DESIGN / "sha256_inputs.txt", DESIGN / "download.json",
                 DESIGN / "codex_runner_request.md", POSE_DIR / "observations.json", Path(__file__),
                 ROOT / "scripts/head_contour_adapter_trial.py"]
    protected += sorted((ROOT / "genai_lab").glob("*.py"))
    distribution = importlib.metadata.distribution("diffusers")
    protected += [Path(distribution.locate_file(name)) for name in (
        "diffusers/models/adapter.py",
        "diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py")]
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "diffusers", "transformers", "accelerate", "Pillow", "numpy")}
    report = {"status": "cpu_preflight_passed_gpu_not_run", "versions": versions, "geometry": geometry, "locks": locks,
              "models": model_manifest(settings, download), "settings": asdict(settings),
              "base_inputs": asdict(inputs), "sketch_root": str(sketch_root),
              "strengths": [1.2, .9], "applied_steps": list(range(11)),
              "control_shas": [CONTROL_SHA, CONTOUR_SHA], "face_sha256": FACE_SHA,
              "baseline_shas": BASE_SHAS, "protected": {str(p): sha(p) for p in protected}}
    return inputs, settings, report
