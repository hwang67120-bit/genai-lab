"""복수 의상 조각을 캐릭터 목표 영역에 배치할 검토 제안을 만든다."""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from genai_lab.clothing import ClothingCategory
from genai_lab.garment_landmarks import (
    GarmentComponentLandmarks,
    GarmentMaskLandmarkResult,
)


class GarmentTargetSlot(str, Enum):
    """의상 조각이 향할 캐릭터의 의미 영역."""

    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"
    FULL_BODY = "full_body"
    IMAGE_LEFT_FOOT = "image_left_foot"
    IMAGE_RIGHT_FOOT = "image_right_foot"
    FOOTWEAR_PAIR = "footwear_pair"


@dataclass(frozen=True)
class GarmentComponentMatchingSettings:
    """전신 의상 조각 분류 경계의 수치 계약."""

    upper_body_end_ratio: float = 0.45
    footwear_start_ratio: float = 0.78
    full_body_minimum_height_ratio: float = 0.55
    center_footwear_left_ratio: float = 0.45
    center_footwear_right_ratio: float = 0.55
    minimum_rule_fit_score: float = 0.35


@dataclass(frozen=True)
class GarmentComponentMatchProposal:
    """사용자 승인 전 의상 조각 1개의 목표 영역 제안."""

    source_component_index: int
    target_slot: GarmentTargetSlot
    assignment_basis: str
    source_bbox_xywh: tuple[int, int, int, int]
    normalized_center_xy: tuple[float, float]
    normalized_height_ratio: float
    foreground_pixel_count: int
    rule_fit_score: float
    ambiguous: bool
    review_reasons: tuple[str, ...]


@dataclass(frozen=True)
class GarmentComponentMatchResult:
    """전체 조각 대응 제안과 자동 실행 차단 상태."""

    clothing_category: ClothingCategory
    proposals: tuple[GarmentComponentMatchProposal, ...]
    overall_bbox_xywh: tuple[int, int, int, int]
    source_component_count: int
    proposal_count: int
    ambiguous_component_count: int
    shared_target_slot_component_count: int
    requires_user_approval: bool
    automatic_warp_allowed: bool
    classification_method: str = "category_geometry_slots_v1"


class GarmentComponentMatchingError(ValueError):
    """복수 의상 조각 대응 제안을 안전하게 만들 수 없는 오류."""


def propose_garment_component_matches(
    source_landmarks: GarmentMaskLandmarkResult,
    clothing_category: ClothingCategory,
    settings: GarmentComponentMatchingSettings | None = None,
) -> GarmentComponentMatchResult:
    """사용자 카테고리와 정규화 기하로 조각별 목표 슬롯을 제안한다."""
    resolved_settings = settings or GarmentComponentMatchingSettings()
    _validate_settings(resolved_settings)
    if clothing_category not in {
        ClothingCategory.TOP,
        ClothingCategory.BOTTOM,
        ClothingCategory.DRESS,
        ClothingCategory.FULL_BODY_OUTFIT,
    }:
        raise GarmentComponentMatchingError(
            "복수 조각 대응이 지원하지 않는 의상 종류입니다: "
            f"{clothing_category.value}"
        )
    _validate_source_landmarks(source_landmarks)
    overall_bbox = _union_bbox(
        tuple(component.bbox_xywh for component in source_landmarks.components)
    )

    initial_proposals = tuple(
        _propose_component_match(
            component=component,
            overall_bbox_xywh=overall_bbox,
            clothing_category=clothing_category,
            settings=resolved_settings,
        )
        for component in source_landmarks.components
    )
    slot_counts = Counter(proposal.target_slot for proposal in initial_proposals)
    proposals = tuple(
        _append_shared_slot_review_reason(proposal, slot_counts)
        for proposal in initial_proposals
    )
    return GarmentComponentMatchResult(
        clothing_category=clothing_category,
        proposals=proposals,
        overall_bbox_xywh=overall_bbox,
        source_component_count=source_landmarks.retained_component_count,
        proposal_count=len(proposals),
        ambiguous_component_count=sum(
            1 for proposal in proposals if proposal.ambiguous
        ),
        shared_target_slot_component_count=sum(
            1 for proposal in proposals if slot_counts[proposal.target_slot] > 1
        ),
        requires_user_approval=True,
        automatic_warp_allowed=False,
    )


def _propose_component_match(
    component: GarmentComponentLandmarks,
    overall_bbox_xywh: tuple[int, int, int, int],
    clothing_category: ClothingCategory,
    settings: GarmentComponentMatchingSettings,
) -> GarmentComponentMatchProposal:
    center_x, center_y, height_ratio = _normalize_component_geometry(
        component.bbox_xywh,
        overall_bbox_xywh,
    )
    if clothing_category == ClothingCategory.TOP:
        slot = GarmentTargetSlot.UPPER_BODY
        basis = "user_category_top"
        score = 1.0
        reasons: tuple[str, ...] = ()
    elif clothing_category == ClothingCategory.BOTTOM:
        slot = GarmentTargetSlot.LOWER_BODY
        basis = "user_category_bottom"
        score = 1.0
        reasons = ()
    elif clothing_category == ClothingCategory.DRESS:
        slot = GarmentTargetSlot.FULL_BODY
        basis = "user_category_dress"
        score = 1.0
        reasons = ()
    else:
        slot, basis, score, reasons = _classify_full_body_component(
            normalized_center_x=center_x,
            normalized_center_y=center_y,
            normalized_height_ratio=height_ratio,
            settings=settings,
        )

    reason_list = list(reasons)
    if score < settings.minimum_rule_fit_score:
        reason_list.append(
            "rule_fit_below_"
            f"{settings.minimum_rule_fit_score:.2f}"
        )
    ambiguous = bool(reason_list)
    return GarmentComponentMatchProposal(
        source_component_index=component.component_index,
        target_slot=slot,
        assignment_basis=basis,
        source_bbox_xywh=component.bbox_xywh,
        normalized_center_xy=(center_x, center_y),
        normalized_height_ratio=height_ratio,
        foreground_pixel_count=component.foreground_pixel_count,
        rule_fit_score=score,
        ambiguous=ambiguous,
        review_reasons=tuple(reason_list),
    )


def _classify_full_body_component(
    normalized_center_x: float,
    normalized_center_y: float,
    normalized_height_ratio: float,
    settings: GarmentComponentMatchingSettings,
) -> tuple[GarmentTargetSlot, str, float, tuple[str, ...]]:
    if normalized_height_ratio >= settings.full_body_minimum_height_ratio:
        return (
            GarmentTargetSlot.FULL_BODY,
            "geometry_vertical_span",
            normalized_height_ratio,
            (),
        )
    if normalized_center_y < settings.upper_body_end_ratio:
        score = _band_fit_score(
            normalized_center_y,
            0.0,
            settings.upper_body_end_ratio,
        )
        return GarmentTargetSlot.UPPER_BODY, "geometry_upper_band", score, ()
    if normalized_center_y < settings.footwear_start_ratio:
        score = _band_fit_score(
            normalized_center_y,
            settings.upper_body_end_ratio,
            settings.footwear_start_ratio,
        )
        return GarmentTargetSlot.LOWER_BODY, "geometry_lower_band", score, ()

    if normalized_center_x < settings.center_footwear_left_ratio:
        score = min(1.0, max(0.0, (0.5 - normalized_center_x) / 0.5))
        return (
            GarmentTargetSlot.IMAGE_LEFT_FOOT,
            "geometry_footwear_x",
            score,
            (),
        )
    if normalized_center_x > settings.center_footwear_right_ratio:
        score = min(1.0, max(0.0, (normalized_center_x - 0.5) / 0.5))
        return (
            GarmentTargetSlot.IMAGE_RIGHT_FOOT,
            "geometry_footwear_x",
            score,
            (),
        )
    score = 1.0 - min(
        1.0,
        abs(normalized_center_x - 0.5)
        / max(0.001, settings.center_footwear_right_ratio - 0.5),
    )
    return (
        GarmentTargetSlot.FOOTWEAR_PAIR,
        "geometry_centered_footwear",
        score,
        ("footwear_left_right_unresolved",),
    )


def _normalize_component_geometry(
    component_bbox_xywh: tuple[int, int, int, int],
    overall_bbox_xywh: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    component_x, component_y, component_width, component_height = (
        component_bbox_xywh
    )
    overall_x, overall_y, overall_width, overall_height = overall_bbox_xywh
    x_denominator = max(1, overall_width - 1)
    y_denominator = max(1, overall_height - 1)
    center_x = (
        component_x + (component_width - 1) / 2.0 - overall_x
    ) / x_denominator
    center_y = (
        component_y + (component_height - 1) / 2.0 - overall_y
    ) / y_denominator
    height_ratio = (component_height - 1) / y_denominator
    return (
        float(min(1.0, max(0.0, center_x))),
        float(min(1.0, max(0.0, center_y))),
        float(min(1.0, max(0.0, height_ratio))),
    )


def _band_fit_score(value: float, start: float, end: float) -> float:
    center = (start + end) / 2.0
    half_width = max(0.001, (end - start) / 2.0)
    return float(min(1.0, max(0.0, 1.0 - abs(value - center) / half_width)))


def _append_shared_slot_review_reason(
    proposal: GarmentComponentMatchProposal,
    slot_counts: Counter[GarmentTargetSlot],
) -> GarmentComponentMatchProposal:
    if slot_counts[proposal.target_slot] <= 1:
        return proposal
    reasons = proposal.review_reasons + ("shared_target_slot",)
    return GarmentComponentMatchProposal(
        source_component_index=proposal.source_component_index,
        target_slot=proposal.target_slot,
        assignment_basis=proposal.assignment_basis,
        source_bbox_xywh=proposal.source_bbox_xywh,
        normalized_center_xy=proposal.normalized_center_xy,
        normalized_height_ratio=proposal.normalized_height_ratio,
        foreground_pixel_count=proposal.foreground_pixel_count,
        rule_fit_score=proposal.rule_fit_score,
        ambiguous=True,
        review_reasons=reasons,
    )


def _union_bbox(
    boxes_xywh: tuple[tuple[int, int, int, int], ...],
) -> tuple[int, int, int, int]:
    left = min(box[0] for box in boxes_xywh)
    top = min(box[1] for box in boxes_xywh)
    right = max(box[0] + box[2] for box in boxes_xywh)
    bottom = max(box[1] + box[3] for box in boxes_xywh)
    return left, top, right - left, bottom - top


def _validate_source_landmarks(
    source_landmarks: GarmentMaskLandmarkResult,
) -> None:
    components = source_landmarks.components
    if not components:
        raise GarmentComponentMatchingError("대응할 유효 의상 조각이 0개입니다.")
    if source_landmarks.retained_component_count != len(components):
        raise GarmentComponentMatchingError(
            "유지 연결요소 수와 의상 조각 자료 수가 다릅니다: "
            f"기록={source_landmarks.retained_component_count}, "
            f"자료={len(components)}"
        )
    width, height = source_landmarks.canvas_size
    if width < 1 or height < 1:
        raise GarmentComponentMatchingError(
            f"의상 캔버스 크기가 유효하지 않습니다: {source_landmarks.canvas_size}"
        )
    indices = [component.component_index for component in components]
    if len(indices) != len(set(indices)):
        raise GarmentComponentMatchingError("의상 조각 인덱스가 중복되었습니다.")
    for component in components:
        x, y, component_width, component_height = component.bbox_xywh
        if (
            component_width < 1
            or component_height < 1
            or x < 0
            or y < 0
            or x + component_width > width
            or y + component_height > height
        ):
            raise GarmentComponentMatchingError(
                "의상 조각 외접 영역이 캔버스 밖에 있습니다: "
                f"조각={component.component_index}, bbox={component.bbox_xywh}, "
                f"캔버스={source_landmarks.canvas_size}"
            )
        points = np.asarray(component.points_xy, dtype=np.float32)
        if points.shape != (6, 2) or not np.isfinite(points).all():
            raise GarmentComponentMatchingError(
                "의상 조각 좌표는 유한한 6×2 배열이어야 합니다: "
                f"조각={component.component_index}, shape={points.shape}"
            )


def _validate_settings(settings: GarmentComponentMatchingSettings) -> None:
    values = (
        settings.upper_body_end_ratio,
        settings.footwear_start_ratio,
        settings.full_body_minimum_height_ratio,
        settings.center_footwear_left_ratio,
        settings.center_footwear_right_ratio,
        settings.minimum_rule_fit_score,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise GarmentComponentMatchingError(
            f"조각 대응 비율은 모두 0.0~1.0 범위여야 합니다: {values}"
        )
    if not settings.upper_body_end_ratio < settings.footwear_start_ratio:
        raise GarmentComponentMatchingError(
            "상체 종료 비율은 신발 시작 비율보다 작아야 합니다."
        )
    if not (
        settings.center_footwear_left_ratio
        < settings.center_footwear_right_ratio
    ):
        raise GarmentComponentMatchingError(
            "중앙 신발 X구간의 왼쪽 비율은 오른쪽보다 작아야 합니다."
        )
