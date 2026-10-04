"""승인 의상 알파 마스크에서 TPS 검토용 기하 대응점을 추출한다."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


LANDMARK_ROLES = (
    "upper_left",
    "upper_right",
    "middle_left",
    "middle_right",
    "lower_left",
    "lower_right",
)


@dataclass(frozen=True)
class GarmentMaskLandmarkSettings:
    """알파 이분화와 3개 수평 탐색 밴드의 수치 계약."""

    alpha_threshold: int = 128
    vertical_ratios: tuple[float, float, float] = (0.10, 0.50, 0.90)
    search_band_ratio: float = 0.03
    minimum_component_pixels: int = 16


@dataclass(frozen=True)
class GarmentComponentLandmarks:
    """연결된 의상 조각 1개의 상·중·하단 좌우 6점."""

    component_index: int
    foreground_pixel_count: int
    bbox_xywh: tuple[int, int, int, int]
    points_xy: np.ndarray
    roles: tuple[str, ...]
    sampled_rows_y: tuple[int, int, int]
    geometry_coverage_score: float


@dataclass(frozen=True)
class GarmentMaskLandmarkResult:
    """노이즈 제거와 연결요소별 좌표 추출 수치."""

    canvas_size: tuple[int, int]
    components: tuple[GarmentComponentLandmarks, ...]
    source_foreground_pixel_count: int
    retained_foreground_pixel_count: int
    discarded_noise_pixel_count: int
    source_component_count: int
    retained_component_count: int
    discarded_component_count: int
    extraction_method: str = "mask_geometry_v1"

    def single_component_tps_points(self) -> np.ndarray:
        """단일 의상 조각일 때만 TPS N×2 좌표 복사본을 반환한다."""
        if self.retained_component_count != 1:
            raise GarmentMaskLandmarkError(
                "TPS 자동 연결은 유효 의상 조각이 정확히 1개여야 합니다: "
                f"{self.retained_component_count}개"
            )
        return self.components[0].points_xy.copy()


class GarmentMaskLandmarkError(ValueError):
    """마스크 기하 좌표를 신뢰할 수 없어 다음 단계를 차단한 오류."""


def extract_garment_mask_landmarks(
    alpha_mask: Image.Image,
    settings: GarmentMaskLandmarkSettings | None = None,
) -> GarmentMaskLandmarkResult:
    """연결요소별 상·중·하단 좌우 6점을 결정론적으로 추출한다.

    이 좌표는 신체 관절이 아니라 승인된 마스크의 기하 경계다. 여러 의상
    조각은 하나로 합치지 않고 각각 반환해 잘못된 TPS 자동 연결을 막는다.
    """
    resolved_settings = settings or GarmentMaskLandmarkSettings()
    _validate_settings(resolved_settings)
    if alpha_mask.width < 2 or alpha_mask.height < 2:
        raise GarmentMaskLandmarkError(
            f"알파 마스크는 가로·세로 2px 이상이어야 합니다: {alpha_mask.size}"
        )

    mask_l = alpha_mask.convert("L")
    try:
        alpha_array = np.asarray(mask_l, dtype=np.uint8)
    finally:
        mask_l.close()
    binary = (alpha_array >= resolved_settings.alpha_threshold).astype(np.uint8)
    source_foreground_pixel_count = int(np.count_nonzero(binary))
    if source_foreground_pixel_count == 0:
        raise GarmentMaskLandmarkError(
            "알파 임계값 이상인 의상 픽셀이 0개입니다: "
            f"임계값={resolved_settings.alpha_threshold}"
        )

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    source_component_count = label_count - 1
    retained_labels = [
        label
        for label in range(1, label_count)
        if int(stats[label, cv2.CC_STAT_AREA])
        >= resolved_settings.minimum_component_pixels
    ]
    if not retained_labels:
        raise GarmentMaskLandmarkError(
            "최소 면적을 만족한 의상 조각이 0개입니다: "
            f"최소={resolved_settings.minimum_component_pixels}px, "
            f"탐지={source_component_count}개"
        )

    component_records: list[GarmentComponentLandmarks] = []
    retained_foreground_pixel_count = 0
    for label in retained_labels:
        area = int(stats[label, cv2.CC_STAT_AREA])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        points, rows, score = _extract_component_points(
            component_mask=labels == label,
            bbox_xywh=(left, top, width, height),
            vertical_ratios=resolved_settings.vertical_ratios,
            search_band_ratio=resolved_settings.search_band_ratio,
        )
        retained_foreground_pixel_count += area
        component_records.append(
            GarmentComponentLandmarks(
                component_index=0,
                foreground_pixel_count=area,
                bbox_xywh=(left, top, width, height),
                points_xy=points,
                roles=LANDMARK_ROLES,
                sampled_rows_y=rows,
                geometry_coverage_score=score,
            )
        )

    component_records.sort(
        key=lambda component: (
            component.bbox_xywh[1],
            component.bbox_xywh[0],
            -component.foreground_pixel_count,
        )
    )
    components = tuple(
        GarmentComponentLandmarks(
            component_index=index,
            foreground_pixel_count=component.foreground_pixel_count,
            bbox_xywh=component.bbox_xywh,
            points_xy=component.points_xy,
            roles=component.roles,
            sampled_rows_y=component.sampled_rows_y,
            geometry_coverage_score=component.geometry_coverage_score,
        )
        for index, component in enumerate(component_records)
    )
    discarded_component_count = source_component_count - len(components)
    discarded_noise_pixel_count = (
        source_foreground_pixel_count - retained_foreground_pixel_count
    )
    return GarmentMaskLandmarkResult(
        canvas_size=alpha_mask.size,
        components=components,
        source_foreground_pixel_count=source_foreground_pixel_count,
        retained_foreground_pixel_count=retained_foreground_pixel_count,
        discarded_noise_pixel_count=discarded_noise_pixel_count,
        source_component_count=source_component_count,
        retained_component_count=len(components),
        discarded_component_count=discarded_component_count,
    )


def _extract_component_points(
    component_mask: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    vertical_ratios: tuple[float, float, float],
    search_band_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    left, top, width, height = bbox_xywh
    if width < 2 or height < 3:
        raise GarmentMaskLandmarkError(
            "의상 조각의 외접 영역이 6점 추출에 너무 작습니다: "
            f"bbox={bbox_xywh}"
        )
    band_radius = max(1, int(round(height * search_band_ratio)))
    points: list[tuple[float, float]] = []
    sampled_rows: list[int] = []
    span_scores: list[float] = []
    bottom = top + height - 1

    for ratio in vertical_ratios:
        target_y = int(round(top + (height - 1) * ratio))
        start_y = max(top, target_y - band_radius)
        end_y = min(bottom, target_y + band_radius)
        candidates: list[tuple[int, int, int, int]] = []
        for y in range(start_y, end_y + 1):
            xs = np.flatnonzero(component_mask[y])
            if xs.size >= 2:
                span = int(xs[-1] - xs[0])
                candidates.append((span, -abs(y - target_y), -y, y))
        if not candidates:
            raise GarmentMaskLandmarkError(
                "탐색 밴드에서 좌우 경계 2점을 찾지 못했습니다: "
                f"목표Y={target_y}px, 범위={start_y}~{end_y}px, "
                f"bbox={bbox_xywh}"
            )
        _, _, _, selected_y = max(candidates)
        selected_xs = np.flatnonzero(component_mask[selected_y])
        selected_left = int(selected_xs[0])
        selected_right = int(selected_xs[-1])
        points.extend(
            (
                (float(selected_left), float(selected_y)),
                (float(selected_right), float(selected_y)),
            )
        )
        sampled_rows.append(selected_y)
        span_scores.append((selected_right - selected_left + 1) / width)

    points_array = np.asarray(points, dtype=np.float32)
    if np.unique(points_array, axis=0).shape[0] != 6:
        raise GarmentMaskLandmarkError("추출된 기하 대응점 6개 중 중복 좌표가 있습니다.")
    return (
        points_array,
        (sampled_rows[0], sampled_rows[1], sampled_rows[2]),
        float(np.mean(span_scores)),
    )


def _validate_settings(settings: GarmentMaskLandmarkSettings) -> None:
    if not 1 <= settings.alpha_threshold <= 255:
        raise GarmentMaskLandmarkError(
            "알파 임계값은 1~255 범위여야 합니다: "
            f"{settings.alpha_threshold}"
        )
    ratios = settings.vertical_ratios
    if len(ratios) != 3 or not all(0.0 < ratio < 1.0 for ratio in ratios):
        raise GarmentMaskLandmarkError(
            f"수직 비율은 0~1 사이 값 3개여야 합니다: {ratios}"
        )
    if not ratios[0] < ratios[1] < ratios[2]:
        raise GarmentMaskLandmarkError(
            f"수직 비율 3개는 오름차순이어야 합니다: {ratios}"
        )
    if not 0.0 < settings.search_band_ratio <= 0.50:
        raise GarmentMaskLandmarkError(
            "탐색 밴드 비율은 0 초과 0.50 이하여야 합니다: "
            f"{settings.search_band_ratio}"
        )
    if settings.minimum_component_pixels < 1:
        raise GarmentMaskLandmarkError(
            "최소 연결요소 면적은 1px 이상이어야 합니다: "
            f"{settings.minimum_component_pixels}px"
        )
