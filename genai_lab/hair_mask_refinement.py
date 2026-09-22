"""얼굴, 사람 귀, 동물 귀, 머리 장식을 침범한 헤어 마스크를 복구한다."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from genai_lab.reference_regions import read_binary_mask


@dataclass(frozen=True)
class HairMaskRefinementSettings:
    enabled: bool
    minimum_hair_pixels: int
    minimum_retained_ratio: float
    maximum_missing_ratio: float
    protection_padding_pixels: int
    hole_inspection_radius: int
    retry_close_radii: tuple[int, ...]


@dataclass
class HairMaskResult:
    status: str
    mask: Image.Image | None
    reason: str | None
    attempts: tuple[dict[str, Any], ...]
    repaired: bool = False

    def record(self):
        return {
            "version": "protected_hair_mask_refinement_v2",
            "status": self.status,
            "reason": self.reason,
            "repaired": self.repaired,
            "mask_returned": self.mask is not None,
            "attempts": list(self.attempts),
            "failure_policy": "unresolved_without_generation_condition",
        }


def resolve_hair_mask_refinement(config):
    raw = config.get("reference_analysis", {}).get("hair_mask_refinement", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("헤어 마스크 정밀화 설정은 객체여야 합니다.")
    enabled = bool(raw.get("enabled", False))
    minimum_pixels = raw.get("minimum_hair_pixels", 256)
    retained_ratio = float(raw.get("minimum_retained_ratio", .55))
    missing_ratio = float(raw.get("maximum_missing_ratio", .04))
    padding = raw.get("protection_padding_pixels", 2)
    inspection_radius = raw.get("hole_inspection_radius", 4)
    radii = tuple(raw.get("retry_close_radii", (0, 2, 4)))
    if type(minimum_pixels) is not int or minimum_pixels < 1:
        raise ValueError("최소 헤어 픽셀 수는 양의 정수여야 합니다.")
    if not math.isfinite(retained_ratio) or not 0 < retained_ratio <= 1:
        raise ValueError("헤어 유지 비율은 0 초과 1 이하여야 합니다.")
    if not math.isfinite(missing_ratio) or not 0 <= missing_ratio <= 1:
        raise ValueError("헤어 결손 비율은 0~1이어야 합니다.")
    if type(padding) is not int or not 0 <= padding <= 16:
        raise ValueError("헤어 보호 영역 여백은 0~16px이어야 합니다.")
    if type(inspection_radius) is not int or not 1 <= inspection_radius <= 16:
        raise ValueError("헤어 결손 검사 반경은 1~16px이어야 합니다.")
    if (not radii or len(radii) > 5 or tuple(sorted(set(radii))) != radii
            or any(type(radius) is not int or not 0 <= radius <= 16 for radius in radii)):
        raise ValueError("헤어 재시도 반경은 중복 없는 오름차순 0~16 목록이어야 합니다.")
    return HairMaskRefinementSettings(
        enabled, minimum_pixels, retained_ratio, missing_ratio,
        padding, inspection_radius, radii,
    )


def _image_mask(selected):
    return Image.fromarray(selected.astype(np.uint8) * 255, mode="L")


def _dilate(selected, radius):
    if not radius:
        return selected.copy()
    with _image_mask(selected) as image:
        expanded = image.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    try:
        return np.asarray(expanded).copy() >= 128
    finally:
        expanded.close()


def _close(selected, radius):
    if not radius:
        return selected.copy()
    kernel = radius * 2 + 1
    with _image_mask(selected) as image:
        expanded = image.filter(ImageFilter.MaxFilter(kernel))
        try:
            closed = expanded.filter(ImageFilter.MinFilter(kernel))
        finally:
            expanded.close()
    try:
        return np.asarray(closed).copy() >= 128
    finally:
        closed.close()


def _overlap_ratio(selected, other):
    area = int(selected.sum())
    return float((selected & other).sum() / area) if area else 0.0


def refine_hair_mask(
    raw_hair_mask,
    head_mask,
    face_mask,
    animal_ears_mask,
    settings,
    *,
    human_ears_mask=None,
    hair_accessory_mask=None,
):
    """복구 가능한 혼입은 제거하고, 모든 시도 실패 시에도 결과 객체를 반환한다."""
    size = raw_hair_mask.size
    raw = read_binary_mask(raw_hair_mask, size)
    head = read_binary_mask(head_mask, size)
    face = read_binary_mask(face_mask, size)
    human_ears = (np.zeros_like(raw) if human_ears_mask is None
                  else read_binary_mask(human_ears_mask, size))
    animal_ears = (np.zeros_like(raw) if animal_ears_mask is None
                   else read_binary_mask(animal_ears_mask, size))
    ears = human_ears | animal_ears
    accessory = (np.zeros_like(raw) if hair_accessory_mask is None
                 else read_binary_mask(hair_accessory_mask, size))
    raw_pixels = int(raw.sum())
    if not settings.enabled:
        return HairMaskResult(
            "disabled", raw_hair_mask.copy(), "refinement_disabled", (), False)
    protected = face | ears | accessory
    expanded_protected = _dilate(protected, settings.protection_padding_pixels)
    allowed = head & ~expanded_protected
    base = raw & allowed
    raw_face_overlap = _overlap_ratio(raw, face)
    raw_ear_overlap = _overlap_ratio(raw, ears)
    raw_human_ear_overlap = _overlap_ratio(raw, human_ears)
    raw_animal_ear_overlap = _overlap_ratio(raw, animal_ears)
    raw_accessory_overlap = _overlap_ratio(raw, accessory)
    raw_outside_head = _overlap_ratio(raw, ~head)
    repaired = bool(raw_face_overlap or raw_ear_overlap
                    or raw_accessory_overlap or raw_outside_head)
    attempts = []
    for index, radius in enumerate(settings.retry_close_radii, start=1):
        candidate = _close(base, radius) & allowed
        pixels = int(candidate.sum())
        retained_ratio = float(pixels / raw_pixels) if raw_pixels else 0.0
        envelope = _close(candidate, settings.hole_inspection_radius) & allowed
        missing_pixels = int((envelope & ~candidate).sum())
        missing_ratio = float(missing_pixels / max(int(envelope.sum()), 1))
        face_overlap = _overlap_ratio(candidate, face)
        ear_overlap = _overlap_ratio(candidate, ears)
        human_ear_overlap = _overlap_ratio(candidate, human_ears)
        animal_ear_overlap = _overlap_ratio(candidate, animal_ears)
        accessory_overlap = _overlap_ratio(candidate, accessory)
        outside_head = _overlap_ratio(candidate, ~head)
        accepted = (
            pixels >= settings.minimum_hair_pixels
            and retained_ratio >= settings.minimum_retained_ratio
            and missing_ratio <= settings.maximum_missing_ratio
            and face_overlap == 0.0
            and human_ear_overlap == 0.0
            and animal_ear_overlap == 0.0
            and accessory_overlap == 0.0
            and outside_head == 0.0
        )
        attempts.append({
            "attempt": index,
            "close_radius": radius,
            "raw_hair_pixels": raw_pixels,
            "hair_pixels": pixels,
            "retained_ratio": retained_ratio,
            "missing_pixels": missing_pixels,
            "missing_ratio": missing_ratio,
            "raw_face_overlap": raw_face_overlap,
            "raw_ear_overlap": raw_ear_overlap,
            "raw_human_ear_overlap": raw_human_ear_overlap,
            "raw_animal_ear_overlap": raw_animal_ear_overlap,
            "raw_accessory_overlap": raw_accessory_overlap,
            "raw_outside_head": raw_outside_head,
            "face_overlap_after": face_overlap,
            "ear_overlap_after": ear_overlap,
            "human_ear_overlap_after": human_ear_overlap,
            "animal_ear_overlap_after": animal_ear_overlap,
            "accessory_overlap_after": accessory_overlap,
            "outside_head_after": outside_head,
            "accepted": accepted,
        })
        if accepted:
            return HairMaskResult(
                "repaired_and_accepted" if repaired or radius else "accepted",
                _image_mask(candidate),
                "protected_regions_removed" if repaired else None,
                tuple(attempts),
                repaired or bool(radius),
            )
    return HairMaskResult(
        "unresolved", None, "hair_mask_refinement_exhausted",
        tuple(attempts), repaired,
    )
