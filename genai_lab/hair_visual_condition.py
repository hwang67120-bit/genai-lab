"""Build an isolated hair-only visual condition without mutating other parts."""

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from PIL import Image

from genai_lab.reference_regions import read_binary_mask


@dataclass(frozen=True)
class HairVisualConditionSettings:
    enabled: bool = False
    reference_scale: float = 0.60
    padding_ratio: float = 0.15
    neutral_background: tuple[int, int, int] = (128, 128, 128)


def resolve_hair_visual_condition(config):
    raw = config.get("reference_analysis", {}).get(
        "hair_visual_reference", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("헤어 시각 참조 설정은 객체여야 합니다.")
    allowed = HairVisualConditionSettings.__dataclass_fields__
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(f"알 수 없는 헤어 시각 참조 설정: {sorted(unknown)}")
    values = dict(raw)
    if "neutral_background" in values:
        values["neutral_background"] = tuple(values["neutral_background"])
    settings = HairVisualConditionSettings(**values)
    if (not math.isfinite(settings.reference_scale)
            or not 0 <= settings.reference_scale <= 1):
        raise ValueError("헤어 시각 참조 강도는 0~1이어야 합니다.")
    if (not math.isfinite(settings.padding_ratio)
            or not 0 <= settings.padding_ratio <= .25):
        raise ValueError("헤어 시각 참조 여백은 0~0.25여야 합니다.")
    if (len(settings.neutral_background) != 3
            or any(type(value) is not int or not 0 <= value <= 255
                   for value in settings.neutral_background)):
        raise ValueError("헤어 시각 참조 중성 배경은 RGB 정수 3개여야 합니다.")
    return settings


def _image_hash(image):
    digest = hashlib.sha256(str((image.mode, image.size)).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _neutral_hair_crop(source, selected, settings):
    hard = Image.fromarray(selected.astype(np.uint8) * 255)
    try:
        box = hard.getbbox()
        if box is None:
            raise ValueError("헤어 시각 참조 영역이 비었습니다.")
        with source.convert("RGB") as rgb:
            background = Image.new(
                "RGB", source.size, settings.neutral_background)
            try:
                isolated = Image.composite(rgb, background, hard)
                try:
                    crop = isolated.crop(box)
                finally:
                    isolated.close()
            finally:
                background.close()
        side = max(crop.size)
        side += 2 * round(side * settings.padding_ratio)
        result = Image.new("RGB", (side, side), settings.neutral_background)
        result.paste(crop, ((side - crop.width) // 2,
                            (side - crop.height) // 2))
        crop.close()
        return result
    finally:
        hard.close()


def hair_identity_separation_available(identity_mask, hair_mask) -> bool:
    if identity_mask.size != hair_mask.size:
        return False
    identity = read_binary_mask(identity_mask, identity_mask.size)
    hair = read_binary_mask(hair_mask, identity_mask.size)
    return bool((identity & ~hair).any())

def prepare_hair_visual_condition(source, identity_mask, garment_mask,
                                  hair_mask, extra_regions, settings):
    """Return disjoint face/hair conditions and an immutable audit record."""
    size = source.size
    identity = read_binary_mask(identity_mask, size)
    garment = read_binary_mask(garment_mask, size)
    hair = read_binary_mask(hair_mask, size)
    if not hair.any():
        raise ValueError("헤어 시각 참조 마스크가 비었습니다.")
    overlap_by_part = {"garment": int((hair & garment).sum())}
    for name, region in extra_regions:
        selected = read_binary_mask(region, size)
        overlap_by_part[name] = int((hair & selected).sum())
    conflicts = {name: count for name, count in overlap_by_part.items()
                 if count}
    if conflicts:
        raise ValueError(
            f"헤어 시각 참조가 다른 부위 조건과 겹칩니다: {conflicts}")
    face_only = identity & ~hair
    if not face_only.any():
        raise ValueError("헤어 분리 후 얼굴 전용 identity 영역이 비었습니다.")
    face_mask = Image.fromarray(face_only.astype(np.uint8) * 255)
    hair_reference = _neutral_hair_crop(source, hair, settings)
    record = {
        "version": "returned_hair_mask_condition_v1",
        "status": "prepared",
        "reference_scale": settings.reference_scale,
        "neutral_background": list(settings.neutral_background),
        "padding_ratio": settings.padding_ratio,
        "hair_pixels": int(hair.sum()),
        "hair_mask_source": "hair_mask_refinement_return",
        "identity_pixels_before": int(identity.sum()),
        "identity_pixels_after": int(face_only.sum()),
        "identity_hair_overlap_removed": int((identity & hair).sum()),
        "hair_outside_identity_pixels": int((hair & ~identity).sum()),
        "overlap_by_part": overlap_by_part,
        "source_identity_mask_sha256": _image_hash(identity_mask),
        "face_only_mask_sha256": _image_hash(face_mask),
        "source_hair_mask_sha256": _image_hash(hair_mask),
        "hair_reference_sha256": _image_hash(hair_reference),
        "other_conditions_mutated": False,
    }
    return face_mask, hair_reference, record

