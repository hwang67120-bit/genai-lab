"""사용자 보정과 자동 보정으로 기존 의상 마스크를 안전하게 다듬는다."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from genai_lab.removal_diagnostics import trace_removal


@dataclass(frozen=True)
class LocalMaskRepairCandidate:
    """아직 적용하지 않은 작은 마스크 누락 후보."""

    mask: Image.Image
    candidate_pixel_count: int
    candidate_component_count: int
    closing_radius_pixels: int
    maximum_component_area_pixels: int

    def close(self) -> None:
        self.mask.close()


@dataclass(frozen=True)
class AutoMaskRepairResult:
    """독립 의상 힌트로 보정한 하드 마스크와 합성용 소프트 마스크."""

    original_mask: Image.Image
    inpaint_mask: Image.Image
    composite_mask: Image.Image
    added_mask: Image.Image
    removed_mask: Image.Image
    original_pixel_count: int
    repaired_pixel_count: int
    added_pixel_count: int
    removed_pixel_count: int
    closing_radius_pixels: int
    feather_radius_pixels: int
    hard_protected_overlap_pixel_count: int
    soft_protected_overlap_pixel_count: int
    hard_outside_foreground_pixel_count: int
    soft_outside_foreground_pixel_count: int

    def close(self) -> None:
        for image in (
            self.original_mask,
            self.inpaint_mask,
            self.composite_mask,
            self.added_mask,
            self.removed_mask,
        ):
            image.close()


def _to_binary(image: Image.Image) -> np.ndarray:
    with image.convert("L") as gray:
        return np.asarray(gray, dtype=np.uint8).copy() >= 128


def _binary_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def _create_protected_feather_mask(
    hard_mask: np.ndarray,
    clothing_hint: np.ndarray,
    allowed: np.ndarray,
    *,
    feather_radius_pixels: int,
) -> np.ndarray:
    """하드 마스크를 늘리지 않고 근거리 의상 힌트에만 소프트 알파를 둔다."""
    outside_hard = (~hard_mask).astype(np.uint8)
    distance_from_hard = cv2.distanceTransform(outside_hard, cv2.DIST_L2, 5)
    alpha = np.zeros(hard_mask.shape, dtype=np.float32)
    alpha[hard_mask] = 1.0
    feather_band = (
        ~hard_mask & clothing_hint & allowed
        & (distance_from_hard > 0)
        & (distance_from_hard <= feather_radius_pixels)
    )
    alpha[feather_band] = (
        1.0 - distance_from_hard[feather_band] / (feather_radius_pixels + 1.0)
    )
    alpha[~allowed] = 0.0
    return np.rint(alpha * 255.0).astype(np.uint8)


@trace_removal("automatic_mask_repair")
def repair_sam_mask_automatically(
    sam_mask: Image.Image,
    independent_clothing_hint: Image.Image,
    protection_mask: Image.Image,
    character_foreground_mask: Image.Image,
    *,
    closing_radius_pixels: int = 1,
    feather_radius_pixels: int = 2,
) -> AutoMaskRepairResult:
    """보호·외곽 0px 계약 안에서 닫기 후보와 합성 페더를 만든다."""
    if len({
        sam_mask.size, independent_clothing_hint.size,
        protection_mask.size, character_foreground_mask.size,
    }) != 1:
        raise ValueError("의상·힌트·보호·외곽 마스크 크기가 다릅니다.")
    if not 1 <= closing_radius_pixels <= 3:
        raise ValueError("자동 닫기 반경은 1~3픽셀이어야 합니다.")
    if not 1 <= feather_radius_pixels <= 8:
        raise ValueError("합성 페더 반경은 1~8픽셀이어야 합니다.")

    clothing = _to_binary(sam_mask)
    clothing_hint = _to_binary(independent_clothing_hint)
    protected = _to_binary(protection_mask)
    foreground = _to_binary(character_foreground_mask)
    allowed = foreground & ~protected
    safe_original = clothing & allowed
    kernel_size = closing_radius_pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
    )
    closed = cv2.morphologyEx(
        safe_original.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE, kernel, iterations=1,
    ) >= 128
    safe_additions = closed & ~safe_original & clothing_hint & allowed
    repaired_hard = (safe_original | safe_additions) & allowed
    if not np.any(repaired_hard):
        raise ValueError("자동 보정 뒤 Inpaint 마스크가 비었습니다.")

    composite_alpha = _create_protected_feather_mask(
        repaired_hard, clothing_hint, allowed,
        feather_radius_pixels=feather_radius_pixels,
    )
    composite_support = composite_alpha > 0
    added = repaired_hard & ~clothing
    removed = clothing & ~repaired_hard
    measurements = {
        "hard_protected": int(np.count_nonzero(repaired_hard & protected)),
        "soft_protected": int(np.count_nonzero(composite_support & protected)),
        "hard_outside": int(np.count_nonzero(repaired_hard & ~foreground)),
        "soft_outside": int(np.count_nonzero(composite_support & ~foreground)),
    }
    if any(measurements.values()):
        raise ValueError(
            "자동 보정 안전 계약 위반: "
            f"하드 보호={measurements['hard_protected']:,}px, "
            f"소프트 보호={measurements['soft_protected']:,}px, "
            f"하드 외곽={measurements['hard_outside']:,}px, "
            f"소프트 외곽={measurements['soft_outside']:,}px"
        )

    return AutoMaskRepairResult(
        original_mask=_binary_image(clothing),
        inpaint_mask=_binary_image(repaired_hard),
        composite_mask=Image.fromarray(composite_alpha, mode="L"),
        added_mask=_binary_image(added),
        removed_mask=_binary_image(removed),
        original_pixel_count=int(np.count_nonzero(clothing)),
        repaired_pixel_count=int(np.count_nonzero(repaired_hard)),
        added_pixel_count=int(np.count_nonzero(added)),
        removed_pixel_count=int(np.count_nonzero(removed)),
        closing_radius_pixels=closing_radius_pixels,
        feather_radius_pixels=feather_radius_pixels,
        hard_protected_overlap_pixel_count=measurements["hard_protected"],
        soft_protected_overlap_pixel_count=measurements["soft_protected"],
        hard_outside_foreground_pixel_count=measurements["hard_outside"],
        soft_outside_foreground_pixel_count=measurements["soft_outside"],
    )


def create_region_mask(
    image_size: tuple[int, int],
    boxes_xyxy: tuple[tuple[int, int, int, int], ...],
) -> Image.Image:
    """승인된 이미지 좌표 상자를 같은 캔버스의 이진 마스크로 바꾼다."""
    width, height = image_size
    if width < 1 or height < 1 or not boxes_xyxy:
        raise ValueError("보정 영역은 1개 이상이어야 합니다.")
    region = np.zeros((height, width), dtype=np.uint8)
    for left, top, right, bottom in boxes_xyxy:
        x1, x2 = sorted((max(0, left), min(width, right)))
        y1, y2 = sorted((max(0, top), min(height, bottom)))
        if x1 >= x2 or y1 >= y2:
            raise ValueError("보정 영역 좌표가 이미지 안에서 비어 있습니다.")
        region[y1:y2, x1:x2] = 255
    return Image.fromarray(region, mode="L")


def propose_local_mask_repair(
    clothing_mask: Image.Image,
    selected_region_mask: Image.Image,
    protection_mask: Image.Image,
    *,
    closing_radius_pixels: int = 2,
    maximum_component_area_pixels: int = 64,
) -> LocalMaskRepairCandidate:
    """작은 닫기 차이 중 선택 영역 안이며 보호와 무관한 조각만 제안한다."""
    if (
        clothing_mask.size != selected_region_mask.size
        or clothing_mask.size != protection_mask.size
    ):
        raise ValueError("의상·선택 영역·보호 마스크 크기가 다릅니다.")
    if not 1 <= closing_radius_pixels <= 4:
        raise ValueError("작은 누락 보정 반경은 1~4픽셀이어야 합니다.")
    if not 1 <= maximum_component_area_pixels <= 4096:
        raise ValueError("누락 후보 조각 면적은 1~4096픽셀이어야 합니다.")

    with (
        clothing_mask.convert("L") as clothing_image,
        selected_region_mask.convert("L") as region_image,
        protection_mask.convert("L") as protection_image,
    ):
        clothing = np.asarray(clothing_image, dtype=np.uint8) >= 128
        selected_region = np.asarray(region_image, dtype=np.uint8) >= 128
        protected = np.asarray(protection_image, dtype=np.uint8) >= 128

    kernel_size = closing_radius_pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    closed = cv2.morphologyEx(
        clothing.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        kernel,
    )
    # 닫기가 의상 외곽을 확장한 픽셀은 후보가 아니다. 원래 배경의
    # 연결요소 중 이미지 테두리와 연결되지 않은 내부 구멍만 허용한다.
    _, background_labels = cv2.connectedComponents(
        (~clothing).astype(np.uint8),
        connectivity=8,
    )
    boundary_labels = np.unique(np.concatenate((
        background_labels[0, :],
        background_labels[-1, :],
        background_labels[:, 0],
        background_labels[:, -1],
    )))
    enclosed_background = (
        ~clothing
        & ~np.isin(background_labels, boundary_labels)
    )
    gaps = (closed >= 128) & enclosed_background
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        gaps.astype(np.uint8),
        connectivity=8,
    )

    allowed = selected_region & ~protected
    proposal = np.zeros(clothing.shape, dtype=np.uint8)
    accepted_component_count = 0
    for component_label in range(1, component_count):
        component = labels == component_label
        area = int(statistics[component_label, cv2.CC_STAT_AREA])
        if (
            area <= maximum_component_area_pixels
            and np.all(allowed[component])
        ):
            proposal[component] = 255
            accepted_component_count += 1

    return LocalMaskRepairCandidate(
        mask=Image.fromarray(proposal, mode="L"),
        candidate_pixel_count=int(np.count_nonzero(proposal)),
        candidate_component_count=accepted_component_count,
        closing_radius_pixels=closing_radius_pixels,
        maximum_component_area_pixels=maximum_component_area_pixels,
    )


def merge_approved_mask_repair(
    clothing_mask: Image.Image,
    approved_repair_mask: Image.Image,
    protection_mask: Image.Image,
) -> Image.Image:
    """사용자가 승인한 보정만 합치며 보호 충돌은 오류로 중단한다."""
    if (
        clothing_mask.size != approved_repair_mask.size
        or clothing_mask.size != protection_mask.size
    ):
        raise ValueError("의상·승인 보정·보호 마스크 크기가 다릅니다.")
    with (
        clothing_mask.convert("L") as clothing_image,
        approved_repair_mask.convert("L") as repair_image,
        protection_mask.convert("L") as protection_image,
    ):
        clothing = np.asarray(clothing_image, dtype=np.uint8) >= 128
        repair = np.asarray(repair_image, dtype=np.uint8) >= 128
        protected = np.asarray(protection_image, dtype=np.uint8) >= 128
    conflicts = int(np.count_nonzero(repair & protected))
    if conflicts:
        raise ValueError(f"승인 보정과 보호 영역이 {conflicts:,}px 겹칩니다.")
    return Image.fromarray((clothing | repair).astype(np.uint8) * 255, mode="L")
