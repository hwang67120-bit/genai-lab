"""승인된 TPS 의상에서 외곽선·내부 디테일 조건 이미지를 만든다."""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from genai_lab.garment_warp_review import GarmentWarpApprovedInput


@dataclass(frozen=True)
class GarmentLineartSettings:
    """Lineart 추출에 사용하는 공개 수치."""

    alpha_threshold: int = 128
    canny_lower_threshold: int = 50
    canny_upper_threshold: int = 150
    gaussian_kernel_size: int = 3
    outer_boundary_radius: int = 1
    internal_erosion_radius: int = 1
    minimum_edge_pixels: int = 1


@dataclass(frozen=True)
class GarmentLineartReviewCandidate:
    """사용자 승인 전 Lineart 중간 자료와 측정값."""

    white_background_garment_preview: Image.Image
    outer_boundary_mask: Image.Image
    internal_detail_mask: Image.Image
    combined_edge_mask: Image.Image
    control_image: Image.Image
    overlay_preview: Image.Image
    canvas_size: tuple[int, int]
    visible_alpha_pixels: int
    alpha_consistency_mismatch_pixels: int
    raw_outer_boundary_pixels: int
    raw_internal_detail_pixels: int
    overlapping_edge_pixels: int
    raw_total_edge_pixels: int
    raw_edge_pixels_outside_approved_mask: int
    protected_edge_pixels_outside_approved_mask: int
    total_edge_pixels: int
    edge_density_percent: float
    elapsed_seconds: float
    settings: GarmentLineartSettings
    automatic_save_count: int = 0

    def close(self) -> None:
        """후보가 소유한 미리보기 6개를 해제한다."""
        self.white_background_garment_preview.close()
        self.outer_boundary_mask.close()
        self.internal_detail_mask.close()
        self.combined_edge_mask.close()
        self.control_image.close()
        self.overlay_preview.close()


@dataclass(frozen=True)
class GarmentLineartApprovedInput:
    """사용자가 승인해 다음 Inpaint 단계로 넘길 Lineart 입력."""

    control_image: Image.Image
    edge_mask: Image.Image
    canvas_size: tuple[int, int]
    total_edge_pixels: int
    edge_density_percent: float

    def close(self) -> None:
        """승인 입력 이미지 2개를 해제한다."""
        self.control_image.close()
        self.edge_mask.close()


class GarmentLineartError(ValueError):
    """Lineart 후보 또는 승인 입력을 만들 수 없는 오류."""


def create_garment_lineart_review(
    approved_warp: GarmentWarpApprovedInput,
    approved_change_mask: Image.Image,
    settings: GarmentLineartSettings | None = None,
) -> GarmentLineartReviewCandidate:
    """외곽선과 내부 디테일을 분리해 검토 후보를 만든다."""
    started_at = perf_counter()
    resolved_settings = settings or GarmentLineartSettings()
    _validate_settings(resolved_settings)
    _validate_sizes(approved_warp, approved_change_mask)

    garment_rgba = approved_warp.warped_rgba.convert("RGBA")
    separate_alpha_image = approved_warp.alpha_mask.convert("L")
    approved_mask_image = approved_change_mask.convert("L")
    try:
        garment_array = np.asarray(garment_rgba, dtype=np.uint8)
        separate_alpha = np.asarray(separate_alpha_image, dtype=np.uint8)
        approved_alpha = np.asarray(approved_mask_image, dtype=np.uint8)
    finally:
        garment_rgba.close()
        separate_alpha_image.close()
        approved_mask_image.close()

    rgba_alpha = garment_array[:, :, 3]
    mismatch_pixels = int(np.count_nonzero(rgba_alpha != separate_alpha))
    if mismatch_pixels != 0:
        raise GarmentLineartError(
            "TPS RGBA 알파와 승인 알파 마스크가 다릅니다: "
            f"{mismatch_pixels}px"
        )

    garment_binary = rgba_alpha >= resolved_settings.alpha_threshold
    approved_binary = approved_alpha >= resolved_settings.alpha_threshold
    visible_alpha_pixels = int(np.count_nonzero(garment_binary))
    if visible_alpha_pixels == 0:
        raise GarmentLineartError("Lineart를 추출할 의상 알파 픽셀이 0개입니다.")

    white_composite = _composite_rgba_on_white(garment_array)
    gray = cv2.cvtColor(white_composite, cv2.COLOR_RGB2GRAY)
    if resolved_settings.gaussian_kernel_size > 1:
        gray = cv2.GaussianBlur(
            gray,
            (
                resolved_settings.gaussian_kernel_size,
                resolved_settings.gaussian_kernel_size,
            ),
            0,
        )

    binary_u8 = garment_binary.astype(np.uint8) * 255
    boundary_kernel = _ellipse_kernel(resolved_settings.outer_boundary_radius)
    raw_outer = cv2.morphologyEx(
        binary_u8,
        cv2.MORPH_GRADIENT,
        boundary_kernel,
    ) > 0

    raw_canny = cv2.Canny(
        gray,
        resolved_settings.canny_lower_threshold,
        resolved_settings.canny_upper_threshold,
        L2gradient=True,
    ) > 0
    if resolved_settings.internal_erosion_radius > 0:
        interior = cv2.erode(
            binary_u8,
            _ellipse_kernel(resolved_settings.internal_erosion_radius),
            iterations=1,
        ) > 0
    else:
        interior = garment_binary
    raw_internal = raw_canny & interior

    raw_combined = raw_outer | raw_internal
    protected_outer = raw_outer & approved_binary
    protected_internal = raw_internal & approved_binary
    protected_combined = raw_combined & approved_binary
    outside_raw = raw_combined & ~approved_binary
    outside_protected = protected_combined & ~approved_binary

    outer_u8 = protected_outer.astype(np.uint8) * 255
    internal_u8 = protected_internal.astype(np.uint8) * 255
    combined_u8 = protected_combined.astype(np.uint8) * 255
    control_rgb = np.repeat(combined_u8[:, :, None], 3, axis=2)
    overlay_rgb = white_composite.copy()
    overlay_rgb[protected_combined] = (255, 0, 0)

    total_edge_pixels = int(np.count_nonzero(protected_combined))
    return GarmentLineartReviewCandidate(
        white_background_garment_preview=Image.fromarray(white_composite),
        outer_boundary_mask=Image.fromarray(outer_u8),
        internal_detail_mask=Image.fromarray(internal_u8),
        combined_edge_mask=Image.fromarray(combined_u8),
        control_image=Image.fromarray(control_rgb),
        overlay_preview=Image.fromarray(overlay_rgb),
        canvas_size=approved_warp.warped_rgba.size,
        visible_alpha_pixels=visible_alpha_pixels,
        alpha_consistency_mismatch_pixels=mismatch_pixels,
        raw_outer_boundary_pixels=int(np.count_nonzero(raw_outer)),
        raw_internal_detail_pixels=int(np.count_nonzero(raw_internal)),
        overlapping_edge_pixels=int(np.count_nonzero(raw_outer & raw_internal)),
        raw_total_edge_pixels=int(np.count_nonzero(raw_combined)),
        raw_edge_pixels_outside_approved_mask=int(np.count_nonzero(outside_raw)),
        protected_edge_pixels_outside_approved_mask=int(
            np.count_nonzero(outside_protected)
        ),
        total_edge_pixels=total_edge_pixels,
        edge_density_percent=(
            total_edge_pixels / visible_alpha_pixels * 100.0
        ),
        elapsed_seconds=perf_counter() - started_at,
        settings=resolved_settings,
    )


def approve_garment_lineart_review(
    review_candidate: GarmentLineartReviewCandidate,
) -> GarmentLineartApprovedInput:
    """명시적 사용자 승인 뒤 Lineart 복사본만 반환한다."""
    if review_candidate.alpha_consistency_mismatch_pixels != 0:
        raise GarmentLineartError("알파 일치 실패 후보는 승인할 수 없습니다.")
    if review_candidate.protected_edge_pixels_outside_approved_mask != 0:
        raise GarmentLineartError(
            "승인 변경 영역 밖 최종 선 픽셀이 0개가 아닙니다."
        )
    if review_candidate.total_edge_pixels < review_candidate.settings.minimum_edge_pixels:
        raise GarmentLineartError(
            "최종 선 픽셀이 승인 최소값보다 작습니다: "
            f"{review_candidate.total_edge_pixels}px < "
            f"{review_candidate.settings.minimum_edge_pixels}px"
        )
    if review_candidate.automatic_save_count != 0:
        raise GarmentLineartError("Lineart 검토 단계 자동 저장 수가 0개가 아닙니다.")
    return GarmentLineartApprovedInput(
        control_image=review_candidate.control_image.copy(),
        edge_mask=review_candidate.combined_edge_mask.copy(),
        canvas_size=review_candidate.canvas_size,
        total_edge_pixels=review_candidate.total_edge_pixels,
        edge_density_percent=review_candidate.edge_density_percent,
    )


def _validate_settings(settings: GarmentLineartSettings) -> None:
    if not 1 <= settings.alpha_threshold <= 255:
        raise GarmentLineartError("알파 임계값은 1~255여야 합니다.")
    if not 0 <= settings.canny_lower_threshold < settings.canny_upper_threshold <= 255:
        raise GarmentLineartError(
            "Canny 임계값은 0 <= lower < upper <= 255여야 합니다."
        )
    if settings.gaussian_kernel_size < 1 or settings.gaussian_kernel_size % 2 == 0:
        raise GarmentLineartError("Gaussian 커널은 1 이상의 홀수여야 합니다.")
    if settings.outer_boundary_radius < 1:
        raise GarmentLineartError("외곽선 반경은 1px 이상이어야 합니다.")
    if settings.internal_erosion_radius < 0:
        raise GarmentLineartError("내부 침식 반경은 0px 이상이어야 합니다.")
    if settings.minimum_edge_pixels < 1:
        raise GarmentLineartError("최소 선 픽셀은 1px 이상이어야 합니다.")


def _validate_sizes(
    approved_warp: GarmentWarpApprovedInput,
    approved_change_mask: Image.Image,
) -> None:
    expected_size = approved_warp.warped_rgba.size
    if approved_warp.alpha_mask.size != expected_size:
        raise GarmentLineartError(
            "TPS RGBA와 알파 마스크 크기가 다릅니다: "
            f"{expected_size} != {approved_warp.alpha_mask.size}"
        )
    if approved_change_mask.size != expected_size:
        raise GarmentLineartError(
            "TPS RGBA와 승인 변경 마스크 크기가 다릅니다: "
            f"{expected_size} != {approved_change_mask.size}"
        )


def _ellipse_kernel(radius: int) -> np.ndarray:
    diameter = radius * 2 + 1
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (diameter, diameter),
    )


def _composite_rgba_on_white(rgba: np.ndarray) -> np.ndarray:
    rgb_u16 = rgba[:, :, :3].astype(np.uint16)
    alpha_u16 = rgba[:, :, 3:4].astype(np.uint16)
    composite = (
        rgb_u16 * alpha_u16
        + 255 * (255 - alpha_u16)
        + 127
    ) // 255
    return composite.astype(np.uint8)
