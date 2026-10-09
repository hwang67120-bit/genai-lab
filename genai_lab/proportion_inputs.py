"""Reviewed head pixels and frozen sketch rules; no hidden coordinate transform."""
from dataclasses import dataclass, asdict
import hashlib
import io
import json
from pathlib import Path, PurePath

import cv2
import numpy as np
from PIL import Image

FOREGROUND_SHA = "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99"
SKETCH_ID = "TencentARC/t2i-adapter-sketch-sdxl-1.0"
SKETCH_REVISION = "cc3c4e3362296c6825c370b83838306723ece983"
SKETCH_HASHES = {
    "config.json": "2f5ed7aee869a54b61f0d377147eb1cdf4aceffeb0e28b0589116415380de89f",
    "diffusion_pytorch_model.fp16.safetensors": "9f88c53368d53bec6b70fc9d797ec12b88a512d2a7e9068d11fbf621917d9fdd",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_value(value):
    def path_only(item):
        if isinstance(item, PurePath):
            return str(item)
        raise TypeError(f"JSON에 기록할 수 없는 값: {type(item).__name__}")
    return json.loads(json.dumps(value, default=path_only, ensure_ascii=False))


@dataclass(frozen=True)
class HeadOutline:
    # Already on the pose canvas. Hair silhouette is included; this is not an anatomical head ratio.
    normalized_file: Path
    normalized_sha256: str
    control_sha256: str
    mask_file: Path
    mask_sha256: str
    contour_file: Path
    contour_sha256: str
    nose: tuple[float, float]
    neck: tuple[float, float]
    confirmed: bool = False


@dataclass(frozen=True)
class ProportionOptions:
    enabled: bool = False
    head: HeadOutline | None = None
    sketch_root: Path | None = None
    foreground_model: Path | None = None
    foreground_sha256: str | None = None
    sketch_revision: str = SKETCH_REVISION
    outline_source: str = "base"
    head_lines: str = "none"
    head_lines_file: Path | None = None
    head_lines_sha256: str | None = None
    head_lines_confirmed: bool = False

    def __post_init__(self):
        require(self.head_lines in ("none","hair","all"), "머리 선은 none, hair, all 중 하나여야 합니다.")
        require(type(self.head_lines_confirmed) is bool,"머리 선 확인 여부는 bool입니다.")
        require(type(self.enabled) is bool, "비율 생성 사용 여부는 bool입니다.")
        require(self.outline_source in ("base", "original"), "윤곽 출처는 base 또는 original이어야 합니다.")


def checked_image(path, digest, size, mode):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, f"입력 SHA 불일치: {path}")
    with Image.open(io.BytesIO(data)) as image:
        require(image.size == size and image.mode == mode, f"입력 크기·모드 불일치: {path}")
        return np.asarray(image).copy()


def draw_head_contour(mask):
    require(np.isin(mask, [0, 255]).all(), "머리 마스크는 0/255만 허용합니다.")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    require(len(contours) == 1, "확인할 머리 외곽선은 한 덩어리여야 합니다.")
    canvas = np.zeros((*mask.shape, 3), np.uint8)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 4)
    return canvas


def validate_head(head, inputs, size):
    require(isinstance(head, HeadOutline) and head.confirmed is True, "사용자가 확인한 머리 윤곽이 필요합니다.")
    require(head.control_sha256 == inputs.control_sha256, "머리 윤곽과 골격의 좌표 기준이 다릅니다.")
    checked_image(head.normalized_file, head.normalized_sha256, size, "RGB")
    mask = checked_image(head.mask_file, head.mask_sha256, size, "L")
    contour = checked_image(head.contour_file, head.contour_sha256, size, "RGB")
    require(np.array_equal(draw_head_contour(mask), contour), "머리 윤곽은 확인한 마스크의 4px 선이어야 합니다.")
    for x, y in (head.nose, head.neck):
        require(np.isfinite([x, y]).all() and 0 <= round(x) < size[0] and 0 <= round(y) < size[1],
                "코·목 좌표가 캔버스 밖입니다.")
    require(mask[round(head.nose[1]), round(head.nose[0])] > 0, "코 좌표가 머리 밖입니다.")
    ys, xs = np.where(mask > 0)
    gap = head.neck[1] - int(ys.max())
    require(xs.min() <= head.neck[0] <= xs.max() and 0 <= gap <= .25 * (ys.max() - ys.min() + 1),
            "목 좌표가 확인한 머리 바로 아래에 있지 않습니다.")
    return mask, contour


def prepare_head_outline(normalized_file, control_file, mask_file, directory, *, nose, neck):
    """Preview a supplied head mask on the existing pose canvas. Never auto-confirm it."""
    normalized_file, control_file, mask_file = map(Path, (normalized_file, control_file, mask_file))
    with Image.open(normalized_file) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with Image.open(control_file) as image:
        require(image.size == (rgb.shape[1], rgb.shape[0]), "골격과 정규화 이미지 크기가 다릅니다.")
    with Image.open(mask_file) as image:
        require(image.mode == "L" and image.size == (rgb.shape[1], rgb.shape[0]), "머리 마스크 크기·모드 불일치")
        mask = np.asarray(image).copy()
    contour = draw_head_contour(mask)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    contour_file = directory / "head-contour.png"
    Image.fromarray(contour).save(contour_file)
    rgb[contour[..., 0] > 0] = (255, 0, 0)
    Image.fromarray(rgb).save(directory / "preview.png")
    return HeadOutline(normalized_file, sha(normalized_file), sha(control_file), mask_file, sha(mask_file),
                       contour_file, sha(contour_file), tuple(nose), tuple(neck))


def validate_models(settings, options):
    from genai_lab.onepass_generation import validate_local_models
    validate_local_models(settings)
    require(options.sketch_root is not None and options.sketch_revision == SKETCH_REVISION,
            "검증된 스케치 revision과 로컬 경로가 필요합니다.")
    sketch = Path(options.sketch_root)
    for name, digest in SKETCH_HASHES.items():
        require(sha(sketch / name) == digest, "스케치 모델 SHA 불일치: " + name)
    require(options.foreground_model is not None and options.foreground_sha256 is not None,
            "로컬 isnet-anime 경로와 SHA가 필요합니다.")
    require(options.foreground_sha256 == FOREGROUND_SHA and sha(options.foreground_model) == FOREGROUND_SHA,
            "검증된 isnet-anime SHA 불일치")
    pose_config = json.loads((settings.adapter_root / "config.json").read_text())
    sketch_config = json.loads((sketch / "config.json").read_text())
    for key in ("adapter_type", "channels", "downscale_factor", "in_channels", "num_res_blocks"):
        require(pose_config[key] == sketch_config[key], "두 어댑터 구조 불일치: " + key)
    pose = {name: sha(settings.adapter_root / name)
            for name in ("config.json", "diffusion_pytorch_model.safetensors")}
    return {"pose": {"id": "TencentARC/t2i-adapter-openpose-sdxl-1.0", "path": str(settings.adapter_root), "sha256": pose},
            "sketch": {"id": SKETCH_ID, "revision": SKETCH_REVISION, "path": str(sketch), "sha256": dict(SKETCH_HASHES)},
            "foreground": {"id": "skytnt/anime-seg", "file": str(options.foreground_model),
                           "sha256": options.foreground_sha256, "provider": "CPUExecutionProvider"}}


def validate_proportion_request(inputs, settings, options):
    require(options.enabled is True, "2단계 생성은 명시적으로 켜야 합니다.")
    require(inputs.pose_mode == "with_pose", "2단계 비율 생성에는 확인된 원본 골격이 필요합니다.")
    require((settings.width, settings.height, settings.steps, settings.guidance_scale,
             settings.adapter_scale, settings.adapter_steps, settings.ip_start, settings.ip_scale,
             settings.max_reserved_gib) == (736, 1232, 28, 5.5, 1.2, 11, 11, .9, 6.5),
            "검증된 2단계 생성 설정과 다릅니다.")
    mask, contour = validate_head(options.head, inputs, (settings.width, settings.height))
    from genai_lab.onepass_generation import read_inputs
    for image in read_inputs(inputs, settings):
        image.close()
    from genai_lab.head_lines import checked_lines
    checked_lines(options,(settings.width,settings.height))
    models = validate_models(settings, options)
    return mask, contour, models


def make_sketch(alpha, head_mask, head_contour):
    """Frozen gpu-05 rule: largest component, 4px outline, 6px head erasure."""
    require(alpha.shape == head_mask.shape and alpha.dtype == np.uint8, "알파 크기·자료형 불일치")
    mask = (alpha.astype(np.float32) / 255 > .5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    require(count > 1, "몸 외곽을 찾지 못했습니다. 2단계를 실행하지 않습니다.")
    mask = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    body = np.zeros_like(mask)
    cv2.drawContours(body, contours, -1, 255, 4)
    erase = cv2.dilate((head_mask > 0).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    body[erase] = 0
    output = np.maximum(body, head_contour[..., 0])
    return np.repeat(output[..., None], 3, axis=2)


def white_background(rgb, alpha):
    require(rgb.shape[:2] == alpha.shape and alpha.dtype == np.uint8 and alpha.max() > 0,
            "흰 배경 합성용 알파가 비어 있거나 크기가 다릅니다.")
    a = alpha.astype(np.float32) / 255
    output = rgb.astype(np.float32) * a[..., None] + 255.0 * (1 - a[..., None])
    return np.clip(output + .5, 0, 255).astype(np.uint8)
