"""조각별 TPS 워핑 결과를 합성하고 사용자 승인 후보로 만든다."""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from genai_lab.character_target_landmarks import (
    CharacterTargetLandmarkResult,
    find_nearest_mask_span,
)
from genai_lab.garment_component_matching import (
    GarmentComponentMatchResult,
    GarmentTargetSlot,
)
from genai_lab.garment_landmarks import (
    GarmentComponentLandmarks,
    GarmentMaskLandmarkResult,
)
from genai_lab.garment_warp import (
    GarmentTpsWarpRequest,
    warp_garment_rgba_tps,
)


TARGET_SLOT_INTERVALS: dict[
    GarmentTargetSlot,
    tuple[float, float, float],
] = {
    GarmentTargetSlot.UPPER_BODY: (0.00, 0.25, 0.50),
    GarmentTargetSlot.LOWER_BODY: (0.50, 0.75, 1.00),
    GarmentTargetSlot.FULL_BODY: (0.00, 0.50, 1.00),
    GarmentTargetSlot.IMAGE_LEFT_FOOT: (0.78, 0.89, 1.00),
    GarmentTargetSlot.IMAGE_RIGHT_FOOT: (0.78, 0.89, 1.00),
    GarmentTargetSlot.FOOTWEAR_PAIR: (0.78, 0.89, 1.00),
}


@dataclass(frozen=True)
class GarmentWarpReviewSettings:
    """조각 분리·TPS·승인 마스크 보호 수치."""

    mask_alpha_threshold: int = 128
    hard_alpha_threshold: int = 128
    source_soft_margin_pixels: int = 2
    tps_regularization: float = 0.0
    maximum_component_count: int = 8


@dataclass(frozen=True)
class GarmentComponentWarpPreview:
    """조각 1개의 TPS 입력점·결과와 픽셀 수치."""

    source_component_index: int
    target_slot: GarmentTargetSlot
    source_points_xy: np.ndarray
    target_points_xy: np.ndarray
    warped_rgba: Image.Image
    warped_alpha_mask: Image.Image
    source_alpha_pixels: int
    warped_alpha_pixels: int
    alpha_pixel_change: int
    source_foreign_hard_pixels_excluded: int

    def close(self) -> None:
        """조각별 워핑 이미지 2개를 해제한다."""
        self.warped_rgba.close()
        self.warped_alpha_mask.close()


@dataclass(frozen=True)
class GarmentWarpReviewCandidate:
    """사용자 승인 전 조각별·통합 TPS 미리보기."""

    component_previews: tuple[GarmentComponentWarpPreview, ...]
    raw_combined_rgba: Image.Image
    outside_approved_mask: Image.Image
    protected_combined_rgba: Image.Image
    protected_alpha_mask: Image.Image
    overlay_preview: Image.Image
    approved_mask_preview: Image.Image
    source_scale_xy: tuple[float, float]
    source_padding_ltrb: tuple[int, int, int, int]
    component_count: int
    ambiguous_component_count: int
    shared_target_slot_component_count: int
    component_hard_overlap_pixels: int
    source_foreign_hard_pixels_excluded: int
    outside_soft_alpha_pixels: int
    outside_hard_alpha_pixels: int
    removed_outside_alpha_pixels: int
    protected_outside_alpha_pixels: int
    elapsed_seconds: float
    automatic_save_count: int = 0

    def close(self) -> None:
        """검토 후보가 소유한 조각별·통합 이미지 전부를 해제한다."""
        for preview in self.component_previews:
            preview.close()
        self.raw_combined_rgba.close()
        self.outside_approved_mask.close()
        self.protected_combined_rgba.close()
        self.protected_alpha_mask.close()
        self.overlay_preview.close()
        self.approved_mask_preview.close()


@dataclass(frozen=True)
class GarmentWarpApprovedInput:
    """사용자가 승인한 보호 후 TPS 의상 입력."""

    warped_rgba: Image.Image
    alpha_mask: Image.Image
    component_count: int
    component_hard_overlap_pixels: int
    outside_soft_alpha_pixels_before_protection: int
    outside_hard_alpha_pixels_before_protection: int
    protected_outside_alpha_pixels: int
    source_scale_xy: tuple[float, float]
    source_padding_ltrb: tuple[int, int, int, int]

    def close(self) -> None:
        """승인 TPS 이미지 2개를 해제한다."""
        self.warped_rgba.close()
        self.alpha_mask.close()


class GarmentWarpReviewError(ValueError):
    """조각별 TPS 검토 후보 또는 승인 입력을 만들 수 없는 오류."""


def create_garment_tps_warp_review(
    garment_rgba_source: Image.Image,
    base_character_image: Image.Image,
    approved_change_mask: Image.Image,
    source_landmarks: GarmentMaskLandmarkResult,
    target_landmarks: CharacterTargetLandmarkResult,
    component_matches: GarmentComponentMatchResult,
    settings: GarmentWarpReviewSettings | None = None,
) -> GarmentWarpReviewCandidate:
    """조각별 TPS 결과와 승인 영역 보호 전후 미리보기를 만든다."""
    started_at = perf_counter()
    resolved_settings = settings or GarmentWarpReviewSettings()
    _validate_settings(resolved_settings)
    _validate_inputs(
        garment_rgba_source=garment_rgba_source,
        base_character_image=base_character_image,
        approved_change_mask=approved_change_mask,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
        component_matches=component_matches,
        maximum_component_count=resolved_settings.maximum_component_count,
    )

    target_size = target_landmarks.target_canvas_size
    target_mask_l = approved_change_mask.convert("L")
    source_rgba = garment_rgba_source.convert("RGBA")
    try:
        target_alpha = np.asarray(target_mask_l, dtype=np.uint8)
        target_binary = target_alpha >= resolved_settings.mask_alpha_threshold
        source_array = np.asarray(source_rgba, dtype=np.uint8)
    finally:
        target_mask_l.close()
        source_rgba.close()

    fit_geometry = _calculate_fit_geometry(
        source_size=source_landmarks.canvas_size,
        target_size=target_size,
    )
    component_region_masks, excluded_foreign_hard_pixels = (
        _build_component_region_masks(
            source_array=source_array,
            components=source_landmarks.components,
            alpha_threshold=resolved_settings.mask_alpha_threshold,
            margin_pixels=resolved_settings.source_soft_margin_pixels,
        )
    )
    components_by_index = {
        component.component_index: component
        for component in source_landmarks.components
    }
    previews: list[GarmentComponentWarpPreview] = []
    raw_combined = Image.new("RGBA", target_size, (0, 0, 0, 0))
    component_hard_overlap_pixels = 0
    try:
        for proposal in component_matches.proposals:
            component = components_by_index[proposal.source_component_index]
            isolated_component = _create_isolated_component_canvas(
                source_array=source_array,
                component_region_mask=component_region_masks[
                    component.component_index
                ],
            )
            fitted_component = _fit_rgba_component_to_canvas(
                isolated_component,
                target_size,
                fit_geometry,
            )
            isolated_component.close()
            source_points = _transform_source_points(
                component.points_xy,
                fit_geometry,
            )
            target_points = _create_slot_target_points(
                target_landmarks=target_landmarks,
                target_binary_mask=target_binary,
                target_slot=proposal.target_slot,
            )
            warp_result = warp_garment_rgba_tps(
                GarmentTpsWarpRequest(
                    garment_rgba_canvas=fitted_component,
                    source_points_xy=source_points,
                    target_points_xy=target_points,
                    canvas_size=target_size,
                    regularization=resolved_settings.tps_regularization,
                )
            )
            fitted_component.close()
            preview = GarmentComponentWarpPreview(
                source_component_index=proposal.source_component_index,
                target_slot=proposal.target_slot,
                source_points_xy=source_points,
                target_points_xy=target_points,
                warped_rgba=warp_result.warped_rgba,
                warped_alpha_mask=warp_result.warped_alpha_mask,
                source_alpha_pixels=warp_result.source_alpha_pixels,
                warped_alpha_pixels=warp_result.warped_alpha_pixels,
                alpha_pixel_change=warp_result.alpha_pixel_change,
                source_foreign_hard_pixels_excluded=(
                    excluded_foreign_hard_pixels[component.component_index]
                ),
            )
            existing_alpha = np.asarray(
                raw_combined.getchannel("A"),
                dtype=np.uint8,
            )
            next_alpha = np.asarray(
                preview.warped_alpha_mask,
                dtype=np.uint8,
            )
            component_hard_overlap_pixels += int(
                np.count_nonzero(
                    (existing_alpha >= resolved_settings.hard_alpha_threshold)
                    & (next_alpha >= resolved_settings.hard_alpha_threshold)
                )
            )
            next_combined = Image.alpha_composite(
                raw_combined,
                preview.warped_rgba,
            )
            raw_combined.close()
            raw_combined = next_combined
            previews.append(preview)

        review_images = _create_protected_review_images(
            raw_combined_rgba=raw_combined,
            base_character_image=base_character_image,
            target_binary_mask=target_binary,
            hard_alpha_threshold=resolved_settings.hard_alpha_threshold,
        )
    except Exception:
        raw_combined.close()
        for preview in previews:
            preview.close()
        raise

    return GarmentWarpReviewCandidate(
        component_previews=tuple(previews),
        raw_combined_rgba=raw_combined,
        outside_approved_mask=review_images["outside_mask"],
        protected_combined_rgba=review_images["protected_rgba"],
        protected_alpha_mask=review_images["protected_alpha"],
        overlay_preview=review_images["overlay"],
        approved_mask_preview=Image.fromarray(
            (target_binary.astype(np.uint8) * 255),
            mode="L",
        ),
        source_scale_xy=(fit_geometry[0], fit_geometry[1]),
        source_padding_ltrb=(
            fit_geometry[2],
            fit_geometry[3],
            fit_geometry[4],
            fit_geometry[5],
        ),
        component_count=len(previews),
        ambiguous_component_count=component_matches.ambiguous_component_count,
        shared_target_slot_component_count=(
            component_matches.shared_target_slot_component_count
        ),
        component_hard_overlap_pixels=component_hard_overlap_pixels,
        source_foreign_hard_pixels_excluded=sum(
            excluded_foreign_hard_pixels.values()
        ),
        outside_soft_alpha_pixels=review_images["outside_soft_pixels"],
        outside_hard_alpha_pixels=review_images["outside_hard_pixels"],
        removed_outside_alpha_pixels=review_images["removed_outside_pixels"],
        protected_outside_alpha_pixels=review_images["protected_outside_pixels"],
        elapsed_seconds=perf_counter() - started_at,
    )


def approve_garment_tps_warp_review(
    review_candidate: GarmentWarpReviewCandidate,
) -> GarmentWarpApprovedInput:
    """사용자 승인 호출 시 보호 후 TPS 이미지 복사본만 반환한다."""
    if review_candidate.component_count < 1:
        raise GarmentWarpReviewError("승인할 TPS 의상 조각이 0개입니다.")
    if review_candidate.protected_outside_alpha_pixels != 0:
        raise GarmentWarpReviewError(
            "보호 후 승인 마스크 밖 알파 픽셀이 0개가 아닙니다: "
            f"{review_candidate.protected_outside_alpha_pixels}px"
        )
    if review_candidate.automatic_save_count != 0:
        raise GarmentWarpReviewError(
            "TPS 검토 단계에서 자동 저장 수가 0개가 아닙니다."
        )
    return GarmentWarpApprovedInput(
        warped_rgba=review_candidate.protected_combined_rgba.copy(),
        alpha_mask=review_candidate.protected_alpha_mask.copy(),
        component_count=review_candidate.component_count,
        component_hard_overlap_pixels=(
            review_candidate.component_hard_overlap_pixels
        ),
        outside_soft_alpha_pixels_before_protection=(
            review_candidate.outside_soft_alpha_pixels
        ),
        outside_hard_alpha_pixels_before_protection=(
            review_candidate.outside_hard_alpha_pixels
        ),
        protected_outside_alpha_pixels=(
            review_candidate.protected_outside_alpha_pixels
        ),
        source_scale_xy=review_candidate.source_scale_xy,
        source_padding_ltrb=review_candidate.source_padding_ltrb,
    )


def _create_slot_target_points(
    target_landmarks: CharacterTargetLandmarkResult,
    target_binary_mask: np.ndarray,
    target_slot: GarmentTargetSlot,
) -> np.ndarray:
    intervals = TARGET_SLOT_INTERVALS[target_slot]
    points: list[tuple[float, float]] = []
    for vertical_ratio in intervals:
        expected_left, expected_right = _interpolate_target_span(
            target_landmarks.points_xy,
            vertical_ratio,
        )
        target_y = (expected_left[1] + expected_right[1]) / 2.0
        selected_y, selected_left, selected_right, _ = find_nearest_mask_span(
            binary_mask=target_binary_mask,
            target_y=target_y,
            expected_left=expected_left[0],
            expected_right=expected_right[0],
            search_radius=target_landmarks.row_search_radius_pixels,
        )
        if target_slot in {
            GarmentTargetSlot.IMAGE_LEFT_FOOT,
            GarmentTargetSlot.IMAGE_RIGHT_FOOT,
        }:
            midpoint = (selected_left + selected_right) // 2
            if target_slot == GarmentTargetSlot.IMAGE_LEFT_FOOT:
                selected_right = midpoint
            else:
                selected_left = midpoint + 1
        if selected_left >= selected_right:
            raise GarmentWarpReviewError(
                "목표 슬롯의 좌우 폭이 2px 미만입니다: "
                f"슬롯={target_slot.value}, y={selected_y}px"
            )
        points.extend(
            (
                (float(selected_left), float(selected_y)),
                (float(selected_right), float(selected_y)),
            )
        )
    points_array = np.asarray(points, dtype=np.float32)
    if np.unique(points_array, axis=0).shape[0] != 6:
        raise GarmentWarpReviewError(
            f"목표 슬롯 {target_slot.value}의 6점 중 중복 좌표가 있습니다."
        )
    if float(cv2.contourArea(cv2.convexHull(points_array))) <= 0.0:
        raise GarmentWarpReviewError(
            f"목표 슬롯 {target_slot.value}의 볼록 껍질 면적이 0입니다."
        )
    return points_array


def _interpolate_target_span(
    target_points_xy: np.ndarray,
    ratio: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.asarray(target_points_xy, dtype=np.float32)
    if points.shape != (6, 2):
        raise GarmentWarpReviewError(
            f"캐릭터 목표점은 6×2여야 합니다: shape={points.shape}"
        )
    if ratio <= 0.5:
        start_index, end_index = 0, 2
        local_ratio = ratio / 0.5
    else:
        start_index, end_index = 2, 4
        local_ratio = (ratio - 0.5) / 0.5
    left = points[start_index] + (
        points[end_index] - points[start_index]
    ) * local_ratio
    right = points[start_index + 1] + (
        points[end_index + 1] - points[start_index + 1]
    ) * local_ratio
    return (
        (float(left[0]), float(left[1])),
        (float(right[0]), float(right[1])),
    )


def _create_isolated_component_canvas(
    source_array: np.ndarray,
    component_region_mask: np.ndarray,
) -> Image.Image:
    isolated = np.zeros_like(source_array)
    isolated[component_region_mask] = source_array[component_region_mask]
    return Image.fromarray(isolated, mode="RGBA")


def _build_component_region_masks(
    source_array: np.ndarray,
    components: tuple[GarmentComponentLandmarks, ...],
    alpha_threshold: int,
    margin_pixels: int,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    binary = (source_array[:, :, 3] >= alpha_threshold).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    masks: dict[int, np.ndarray] = {}
    excluded_foreign_pixels: dict[int, int] = {}
    if margin_pixels > 0:
        kernel_size = margin_pixels * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
    else:
        kernel = None
    for component in components:
        matching_labels = [
            label
            for label in range(1, label_count)
            if (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            == component.bbox_xywh
            and int(stats[label, cv2.CC_STAT_AREA])
            == component.foreground_pixel_count
        ]
        if len(matching_labels) != 1:
            raise GarmentWarpReviewError(
                "의상 조각 좌표와 원본 알파 연결요소를 1:1로 대조하지 못했습니다: "
                f"조각={component.component_index}, 후보={len(matching_labels)}개"
            )
        label = matching_labels[0]
        hard_region = (labels == label).astype(np.uint8)
        expanded_region = (
            cv2.dilate(hard_region, kernel, iterations=1)
            if kernel is not None
            else hard_region
        )
        other_hard_components = (labels != 0) & (labels != label)
        excluded_foreign_pixels[component.component_index] = int(
            np.count_nonzero((expanded_region > 0) & other_hard_components)
        )
        region_mask = (expanded_region > 0) & ~other_hard_components
        masks[component.component_index] = region_mask
    return masks, excluded_foreign_pixels


def _calculate_fit_geometry(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, float, int, int, int, int, int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    padding_left = (target_width - resized_width) // 2
    padding_top = (target_height - resized_height) // 2
    padding_right = target_width - resized_width - padding_left
    padding_bottom = target_height - resized_height - padding_top
    return (
        resized_width / source_width,
        resized_height / source_height,
        padding_left,
        padding_top,
        padding_right,
        padding_bottom,
        resized_width,
        resized_height,
    )


def _fit_rgba_component_to_canvas(
    component_rgba: Image.Image,
    target_size: tuple[int, int],
    fit_geometry: tuple[float, float, int, int, int, int, int, int],
) -> Image.Image:
    _, _, padding_left, padding_top, _, _, resized_width, resized_height = (
        fit_geometry
    )
    source = np.asarray(component_rgba, dtype=np.uint8)
    alpha = source[:, :, 3].astype(np.float32)
    premultiplied = source[:, :, :3].astype(np.float32) * (
        alpha[:, :, None] / 255.0
    )
    resized_alpha = cv2.resize(
        alpha,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resized_premultiplied = cv2.resize(
        premultiplied,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resized_alpha = np.clip(resized_alpha, 0.0, 255.0)
    resized_alpha_u8 = np.rint(resized_alpha).astype(np.uint8)
    restored_rgb = np.zeros_like(resized_premultiplied)
    visible = resized_alpha_u8 > 0
    restored_rgb[visible] = (
        resized_premultiplied[visible]
        * 255.0
        / resized_alpha[visible, None]
    )
    resized_rgba = np.dstack(
        (
            np.rint(np.clip(restored_rgb, 0.0, 255.0)),
            resized_alpha_u8,
        )
    ).astype(np.uint8)
    target_width, target_height = target_size
    canvas = np.zeros((target_height, target_width, 4), dtype=np.uint8)
    canvas[
        padding_top:padding_top + resized_height,
        padding_left:padding_left + resized_width,
    ] = resized_rgba
    return Image.fromarray(canvas, mode="RGBA")


def _transform_source_points(
    source_points_xy: np.ndarray,
    fit_geometry: tuple[float, float, int, int, int, int, int, int],
) -> np.ndarray:
    scale_x, scale_y, padding_left, padding_top, _, _, _, _ = fit_geometry
    points = np.asarray(source_points_xy, dtype=np.float32).copy()
    points[:, 0] = points[:, 0] * scale_x + padding_left
    points[:, 1] = points[:, 1] * scale_y + padding_top
    return points


def _create_protected_review_images(
    raw_combined_rgba: Image.Image,
    base_character_image: Image.Image,
    target_binary_mask: np.ndarray,
    hard_alpha_threshold: int,
) -> dict[str, Image.Image | int]:
    raw_array = np.asarray(raw_combined_rgba, dtype=np.uint8)
    raw_alpha = raw_array[:, :, 3]
    outside = ~target_binary_mask
    outside_soft = outside & (raw_alpha > 0) & (raw_alpha < hard_alpha_threshold)
    outside_hard = outside & (raw_alpha >= hard_alpha_threshold)
    outside_any = outside & (raw_alpha > 0)
    outside_alpha = np.where(outside, raw_alpha, 0).astype(np.uint8)

    protected_array = raw_array.copy()
    protected_array[outside, :3] = 0
    protected_array[outside, 3] = 0
    protected_rgba = Image.fromarray(protected_array, mode="RGBA")
    protected_alpha = Image.fromarray(protected_array[:, :, 3], mode="L")
    protected_outside_pixels = int(
        np.count_nonzero(protected_array[:, :, 3][outside] > 0)
    )
    base_rgba = base_character_image.convert("RGBA")
    try:
        overlay = Image.alpha_composite(base_rgba, protected_rgba)
    finally:
        base_rgba.close()
    overlay_rgb = overlay.convert("RGB")
    overlay.close()
    return {
        "outside_mask": Image.fromarray(outside_alpha, mode="L"),
        "protected_rgba": protected_rgba,
        "protected_alpha": protected_alpha,
        "overlay": overlay_rgb,
        "outside_soft_pixels": int(np.count_nonzero(outside_soft)),
        "outside_hard_pixels": int(np.count_nonzero(outside_hard)),
        "removed_outside_pixels": int(np.count_nonzero(outside_any)),
        "protected_outside_pixels": protected_outside_pixels,
    }


def _validate_inputs(
    garment_rgba_source: Image.Image,
    base_character_image: Image.Image,
    approved_change_mask: Image.Image,
    source_landmarks: GarmentMaskLandmarkResult,
    target_landmarks: CharacterTargetLandmarkResult,
    component_matches: GarmentComponentMatchResult,
    maximum_component_count: int,
) -> None:
    if garment_rgba_source.size != source_landmarks.canvas_size:
        raise GarmentWarpReviewError(
            "참조 의상과 원본 좌표 캔버스 크기가 다릅니다: "
            f"의상={garment_rgba_source.size}, "
            f"좌표={source_landmarks.canvas_size}"
        )
    target_size = target_landmarks.target_canvas_size
    if base_character_image.size != target_size:
        raise GarmentWarpReviewError(
            "기준 캐릭터와 목표 좌표 캔버스 크기가 다릅니다: "
            f"캐릭터={base_character_image.size}, 목표={target_size}"
        )
    if approved_change_mask.size != target_size:
        raise GarmentWarpReviewError(
            "승인 변경 마스크와 목표 좌표 캔버스 크기가 다릅니다: "
            f"마스크={approved_change_mask.size}, 목표={target_size}"
        )
    if component_matches.source_component_count != len(
        source_landmarks.components
    ):
        raise GarmentWarpReviewError(
            "조각 대응 원본 수와 의상 좌표 조각 수가 다릅니다."
        )
    if component_matches.proposal_count != len(component_matches.proposals):
        raise GarmentWarpReviewError(
            "조각 대응 제안 수 기록과 실제 제안 수가 다릅니다."
        )
    if not 1 <= component_matches.proposal_count <= maximum_component_count:
        raise GarmentWarpReviewError(
            "TPS 조각 수가 허용 범위를 벗어났습니다: "
            f"{component_matches.proposal_count}개, 최대={maximum_component_count}개"
        )
    component_indices = {
        component.component_index for component in source_landmarks.components
    }
    proposal_indices = [
        proposal.source_component_index
        for proposal in component_matches.proposals
    ]
    if set(proposal_indices) != component_indices or len(proposal_indices) != len(
        set(proposal_indices)
    ):
        raise GarmentWarpReviewError(
            "의상 조각과 목표 슬롯 제안이 1:1로 대응하지 않습니다."
        )


def _validate_settings(settings: GarmentWarpReviewSettings) -> None:
    if not 1 <= settings.mask_alpha_threshold <= 255:
        raise GarmentWarpReviewError("승인 마스크 임계값은 1~255여야 합니다.")
    if not 1 <= settings.hard_alpha_threshold <= 255:
        raise GarmentWarpReviewError("하드 알파 임계값은 1~255여야 합니다.")
    if settings.source_soft_margin_pixels < 0:
        raise GarmentWarpReviewError("소프트 알파 여백은 0px 이상이어야 합니다.")
    if settings.tps_regularization < 0:
        raise GarmentWarpReviewError("TPS 정규화 값은 0 이상이어야 합니다.")
    if settings.maximum_component_count < 1:
        raise GarmentWarpReviewError("최대 TPS 조각 수는 1개 이상이어야 합니다.")
