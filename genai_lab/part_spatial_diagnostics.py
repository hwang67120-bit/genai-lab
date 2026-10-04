"""Observe-only spatial diagnostics for detected character parts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping

import numpy as np


DIAGNOSTIC_VERSION = "part_spatial_diagnostics_v1"
HEAD_PARTS = ("animal_ears", "hair_accessory")


@dataclass(frozen=True)
class PartSpatialDiagnosticPolicy:
    """Thresholds for diagnostics that never change accepted masks."""

    head_fallback_bottom_ratio: float = 0.55
    head_context_expansion_ratio: float = 0.15
    minimum_head_overlap_ratio: float = 0.50
    tail_fallback_left_ratio: float = 0.15
    tail_fallback_top_ratio: float = 0.45
    tail_fallback_right_ratio: float = 0.85
    tail_fallback_bottom_ratio: float = 0.78
    tail_context_expansion_ratio: float = 0.10
    minimum_tail_anchor_overlap_ratio: float = 0.02
    identical_minimum_smaller_overlap_ratio: float = 0.90
    identical_minimum_iou: float = 0.60

    def __post_init__(self) -> None:
        unit_values = (
            self.head_fallback_bottom_ratio,
            self.head_context_expansion_ratio,
            self.minimum_head_overlap_ratio,
            self.tail_fallback_left_ratio,
            self.tail_fallback_top_ratio,
            self.tail_fallback_right_ratio,
            self.tail_fallback_bottom_ratio,
            self.tail_context_expansion_ratio,
            self.minimum_tail_anchor_overlap_ratio,
            self.identical_minimum_smaller_overlap_ratio,
            self.identical_minimum_iou,
        )
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in unit_values):
            raise ValueError("부위 공간 진단 임계값은 0~1이어야 합니다.")
        if self.tail_fallback_left_ratio >= self.tail_fallback_right_ratio:
            raise ValueError("꼬리 연결 영역의 좌우 비율 순서가 올바르지 않습니다.")
        if self.tail_fallback_top_ratio >= self.tail_fallback_bottom_ratio:
            raise ValueError("꼬리 연결 영역의 상하 비율 순서가 올바르지 않습니다.")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "PartSpatialDiagnosticPolicy":
        raw = dict(value or {})
        raw.pop("enabled", None)
        raw.pop("mode", None)
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "등록되지 않은 부위 공간 진단 설정입니다: "
                + ", ".join(sorted(unknown))
            )
        return cls(**{key: float(item) for key, item in raw.items()})

    def snapshot(self) -> dict[str, float]:
        return asdict(self)


def _validated_mask(
    value: np.ndarray | None,
    size: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=bool)
    expected = (size[1], size[0])
    if mask.shape != expected:
        raise ValueError(f"{name} 마스크 크기가 원본 좌표와 다릅니다.")
    return mask


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _box_mask(
    size: tuple[int, int],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    width, height = size
    left, top, right, bottom = box
    left = max(0, min(width, int(left)))
    right = max(left, min(width, int(right)))
    top = max(0, min(height, int(top)))
    bottom = max(top, min(height, int(bottom)))
    result = np.zeros((height, width), dtype=bool)
    result[top:bottom, left:right] = True
    return result


def _expanded_box(
    bounds: tuple[int, int, int, int],
    size: tuple[int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    width, height = size
    pad_x = max(1, int(round((right - left) * ratio)))
    pad_y = max(1, int(round((bottom - top) * ratio)))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def _head_roi(
    size: tuple[int, int],
    hair_context: np.ndarray | None,
    policy: PartSpatialDiagnosticPolicy,
) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
    if hair_context is not None:
        bounds = _mask_bounds(hair_context)
        if bounds is not None:
            box = _expanded_box(
                bounds,
                size,
                policy.head_context_expansion_ratio,
            )
            return _box_mask(size, box), "hair_context_bounds", box
    width, height = size
    box = (
        0,
        0,
        width,
        max(1, int(round(height * policy.head_fallback_bottom_ratio))),
    )
    return _box_mask(size, box), "normalized_image_fallback", box


def locate_head_roi(
    size: tuple[int, int],
    hair_context: np.ndarray | None,
    policy: PartSpatialDiagnosticPolicy | None = None,
) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
    """Return the same padded head ROI used by observe-only diagnostics."""

    return _head_roi(
        size,
        hair_context,
        policy or PartSpatialDiagnosticPolicy(),
    )


def _tail_anchor_roi(
    size: tuple[int, int],
    garment_context: np.ndarray | None,
    policy: PartSpatialDiagnosticPolicy,
) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
    if garment_context is not None:
        bounds = _mask_bounds(garment_context)
        if bounds is not None:
            left, top, right, bottom = bounds
            garment_height = max(1, bottom - top)
            anchor = (
                left,
                top + int(round(garment_height * 0.35)),
                right,
                top + int(round(garment_height * 0.85)),
            )
            box = _expanded_box(
                anchor,
                size,
                policy.tail_context_expansion_ratio,
            )
            return _box_mask(size, box), "garment_context_lower_torso", box
    width, height = size
    box = (
        int(round(width * policy.tail_fallback_left_ratio)),
        int(round(height * policy.tail_fallback_top_ratio)),
        int(round(width * policy.tail_fallback_right_ratio)),
        int(round(height * policy.tail_fallback_bottom_ratio)),
    )
    return _box_mask(size, box), "normalized_body_fallback", box


def _roi_check(
    mask: np.ndarray | None,
    roi: np.ndarray,
    *,
    minimum_overlap_ratio: float,
    passing_status: str,
    failing_status: str,
) -> dict[str, Any]:
    if mask is None or not mask.any():
        return {
            "status": "not_observed",
            "candidate_pixels": 0,
            "overlap_pixels": 0,
            "candidate_overlap_ratio": None,
            "review_required": False,
        }
    candidate_pixels = int(np.count_nonzero(mask))
    overlap_pixels = int(np.count_nonzero(mask & roi))
    ratio = float(overlap_pixels / candidate_pixels)
    passed = ratio >= minimum_overlap_ratio
    return {
        "status": passing_status if passed else failing_status,
        "candidate_pixels": candidate_pixels,
        "overlap_pixels": overlap_pixels,
        "candidate_overlap_ratio": ratio,
        "minimum_overlap_ratio": minimum_overlap_ratio,
        "review_required": not passed,
    }


def _pair_diagnostic(
    first: str,
    second: str,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    policy: PartSpatialDiagnosticPolicy,
) -> dict[str, Any]:
    first_pixels = int(np.count_nonzero(first_mask))
    second_pixels = int(np.count_nonzero(second_mask))
    shared_pixels = int(np.count_nonzero(first_mask & second_mask))
    union_pixels = int(np.count_nonzero(first_mask | second_mask))
    smaller = min(first_pixels, second_pixels)
    smaller_ratio = float(shared_pixels / smaller) if smaller else None
    iou = float(shared_pixels / union_pixels) if union_pixels else None
    same_mask = bool(
        smaller_ratio is not None
        and iou is not None
        and smaller_ratio
        >= policy.identical_minimum_smaller_overlap_ratio
        and iou >= policy.identical_minimum_iou
    )
    if same_mask:
        status = "same_mask_cross_class_conflict"
    elif shared_pixels:
        status = "overlapping_distinct_candidates"
    else:
        status = "separate_candidates"
    return {
        "parts": [first, second],
        "status": status,
        "first_pixels": first_pixels,
        "second_pixels": second_pixels,
        "overlap_pixels": shared_pixels,
        "smaller_part_overlap_ratio": smaller_ratio,
        "mask_iou": iou,
        "same_mask_conflict": same_mask,
        "review_required": same_mask,
    }


def diagnose_part_spatial_consistency(
    *,
    size: tuple[int, int],
    part_masks: Mapping[str, np.ndarray],
    context_masks: Mapping[str, np.ndarray] | None = None,
    policy: PartSpatialDiagnosticPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Record geometry conflicts without changing predictions or masks."""

    active_policy = policy or PartSpatialDiagnosticPolicy()
    masks = {
        name: _validated_mask(mask, size, name)
        for name, mask in part_masks.items()
    }
    context = context_masks or {}
    hair_context = _validated_mask(
        context.get("hair"),
        size,
        "hair_context",
    )
    garment_context = _validated_mask(
        context.get("garment"),
        size,
        "garment_context",
    )
    head_roi, head_source, head_box = locate_head_roi(
        size,
        hair_context,
        active_policy,
    )
    tail_roi, tail_source, tail_box = _tail_anchor_roi(
        size,
        garment_context,
        active_policy,
    )

    head_checks = {
        name: _roi_check(
            masks.get(name),
            head_roi,
            minimum_overlap_ratio=active_policy.minimum_head_overlap_ratio,
            passing_status="inside_head_roi",
            failing_status="outside_head_roi",
        )
        for name in HEAD_PARTS
    }
    tail_check = _roi_check(
        masks.get("tail"),
        tail_roi,
        minimum_overlap_ratio=active_policy.minimum_tail_anchor_overlap_ratio,
        passing_status="body_anchor_contact",
        failing_status="outside_body_anchor",
    )

    pair_diagnostics = [
        _pair_diagnostic(
            first,
            second,
            masks[first],
            masks[second],
            active_policy,
        )
        for first, second in combinations(sorted(masks), 2)
        if masks[first] is not None
        and masks[second] is not None
        and masks[first].any()
        and masks[second].any()
    ]
    conflicts = [
        item for item in pair_diagnostics
        if item["same_mask_conflict"]
    ]
    review_required = bool(
        any(item["review_required"] for item in head_checks.values())
        or tail_check["review_required"]
        or conflicts
    )
    report = {
        "version": DIAGNOSTIC_VERSION,
        "mode": "observe_only",
        "policy": active_policy.snapshot(),
        "head_roi": {
            "source": head_source,
            "box": list(head_box),
        },
        "tail_body_anchor_roi": {
            "source": tail_source,
            "box": list(tail_box),
        },
        "head_checks": head_checks,
        "tail_connection_check": tail_check,
        "cross_class_pairs": pair_diagnostics,
        "same_mask_cross_class_conflicts": conflicts,
        "review_required": review_required,
        "predictions_modified": False,
        "masks_modified": False,
        "automatic_conditioning_applied": False,
        "generation_policy_changed": False,
    }
    return report, {
        "head_roi": head_roi,
        "tail_body_anchor_roi": tail_roi,
    }
