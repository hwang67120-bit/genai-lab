"""Conservative spatial checks before scoring and after local correction."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CandidateStructureGateSettings:
    enabled: bool = False
    background_color_distance: float = 22.0
    minimum_primary_height_ratio: float = 0.42
    minimum_primary_bbox_area_ratio: float = 0.055
    maximum_tall_component_aspect_ratio: float = 0.38
    maximum_tall_component_solidity: float = 0.97
    minimum_component_area_ratio: float = 0.001
    overhead_component_area_ratio: float = 0.10
    overhead_component_width_ratio: float = 0.24
    minimum_overhead_gap_ratio: float = 0.04
    upper_band_fraction: float = 0.32
    maximum_upper_width_ratio: float = 2.50
    minimum_upper_canvas_width_ratio: float = 0.48

    def __post_init__(self) -> None:
        ratios = (
            self.minimum_primary_height_ratio,
            self.minimum_primary_bbox_area_ratio,
            self.maximum_tall_component_aspect_ratio,
            self.maximum_tall_component_solidity,
            self.minimum_component_area_ratio,
            self.overhead_component_area_ratio,
            self.overhead_component_width_ratio,
            self.upper_band_fraction,
            self.minimum_upper_canvas_width_ratio,
        )
        if self.background_color_distance <= 0:
            raise ValueError("구조 게이트 배경 색상 거리는 0보다 커야 합니다.")
        if any(value <= 0 or value > 1 for value in ratios):
            raise ValueError("구조 게이트 비율은 0보다 크고 1 이하여야 합니다.")
        if self.maximum_upper_width_ratio <= 1:
            raise ValueError("상단 폭 이상 임계값은 1보다 커야 합니다.")

    def record(self) -> dict[str, Any]:
        return {
            "version": "candidate_structure_gate_v3",
            **self.__dict__,
            "policy": "reject_spatial_artifacts_before_semantic_fallback_score",
            "pose_policy": "pose_invariant_no_front_or_standing_requirement",
        }


def resolve_candidate_structure_gate(
        config: Mapping[str, Any]) -> CandidateStructureGateSettings:
    raw = config.get("candidate_structure_gate", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("후보 구조 게이트 설정은 객체여야 합니다.")
    allowed = {field.name for field in fields(CandidateStructureGateSettings)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"알 수 없는 후보 구조 게이트 설정: {sorted(unknown)}")
    return CandidateStructureGateSettings(**dict(raw))


def _foreground_mask(image: Image.Image, distance: float) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    border_width = max(1, int(round(min(height, width) * 0.02)))
    border = np.concatenate((
        rgb[:border_width].reshape(-1, 3),
        rgb[-border_width:].reshape(-1, 3),
        rgb[:, :border_width].reshape(-1, 3),
        rgb[:, -border_width:].reshape(-1, 3),
    ))
    background = np.median(border.astype(np.float32), axis=0)
    delta = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    mask = np.where(delta >= distance, 255, 0).astype(np.uint8)
    kernel_size = max(3, int(round(min(height, width) * 0.006)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _component_records(mask: np.ndarray, minimum_area: int) -> list[dict[str, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    records = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area >= minimum_area:
            records.append({
                "label": label, "x": x, "y": y, "width": width,
                "height": height, "area": area,
            })
    return records


def evaluate_candidate_structure(
        image: Image.Image,
        settings: CandidateStructureGateSettings) -> dict[str, Any]:
    if not settings.enabled:
        return {**settings.record(), "status": "DISABLED", "action": "pass",
                "violations": [], "checks": {}}

    mask = _foreground_mask(image, settings.background_color_distance)
    height, width = mask.shape
    canvas_area = height * width
    minimum_area = max(4, int(round(
        canvas_area * settings.minimum_component_area_ratio)))
    components = _component_records(mask, minimum_area)
    if not components:
        return {
            **settings.record(), "status": "REVIEW", "action": "review",
            "violations": [], "unresolved": ["foreground_not_detected"],
            "checks": {}, "foreground_ratio": 0.0,
        }

    # Prefer a tall, centrally placed component. This avoids treating a wide
    # detached prop above the head as the character itself.
    center_x = width / 2.0
    primary = max(components, key=lambda item: (
        item["height"] / height,
        int(item["x"] <= center_x <= item["x"] + item["width"]),
        item["area"],
    ))
    primary_height_ratio = primary["height"] / height
    primary_bbox_area_ratio = (
        primary["width"] * primary["height"] / canvas_area)
    violations: list[str] = []
    checks: dict[str, Any] = {}

    too_small = (
        primary_height_ratio < settings.minimum_primary_height_ratio
        or primary_bbox_area_ratio < settings.minimum_primary_bbox_area_ratio
    )
    checks["character_occupancy"] = {
        "status": "FAIL" if too_small else "PASS",
        "primary_height_ratio": primary_height_ratio,
        "primary_bbox_area_ratio": primary_bbox_area_ratio,
    }
    if too_small:
        violations.append("character_occupancy_too_small")

    overhead = []
    pose_ambiguous_overhead = []
    for component in components:
        if component is primary:
            continue
        component_bottom = component["y"] + component["height"]
        above_primary_center = (
            component_bottom
            <= primary["y"] + primary["height"] * 0.45)
        large_enough = (
            component["area"] >= primary["area"]
            * settings.overhead_component_area_ratio
            and component["width"] / width
            >= settings.overhead_component_width_ratio
        )
        vertical_gap_ratio = max(0, primary["y"] - component_bottom) / height
        if above_primary_center and large_enough:
            item = {**component, "vertical_gap_ratio": vertical_gap_ratio}
            if vertical_gap_ratio >= settings.minimum_overhead_gap_ratio:
                overhead.append(item)
            else:
                pose_ambiguous_overhead.append(item)
    checks["detached_overhead_object"] = {
        "status": (
            "FAIL" if overhead else
            "UNRESOLVED" if pose_ambiguous_overhead else "PASS"
        ),
        "components": overhead,
        "pose_ambiguous_components": pose_ambiguous_overhead,
        "minimum_overhead_gap_ratio": settings.minimum_overhead_gap_ratio,
    }
    if overhead:
        violations.append("large_detached_object_above_character")

    primary_mask = np.zeros_like(mask)
    # connectedComponents labels are deliberately recomputed so the report
    # remains independent of label-array lifetime and component filtering.
    _, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    primary_mask[labels == primary["label"]] = 255
    contours, _ = cv2.findContours(
        primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    primary_contour = max(contours, key=cv2.contourArea)
    convex_hull = cv2.convexHull(primary_contour)
    convex_hull_area = max(float(cv2.contourArea(convex_hull)), 1.0)
    primary_solidity = primary["area"] / convex_hull_area
    primary_aspect_ratio = primary["width"] / max(primary["height"], 1)
    tall_convex_enclosure = (
        primary_aspect_ratio <= settings.maximum_tall_component_aspect_ratio
        and primary_solidity >= settings.maximum_tall_component_solidity
    )
    checks["primary_component_shape"] = {
        "status": "FAIL" if tall_convex_enclosure else "PASS",
        "width_height_ratio": primary_aspect_ratio,
        "solidity": primary_solidity,
    }
    if tall_convex_enclosure:
        violations.append("tall_convex_enclosure_not_character")

    y0, y1 = primary["y"], primary["y"] + primary["height"]
    split = min(y1, y0 + max(1, int(round(
        primary["height"] * settings.upper_band_fraction))))
    upper_widths = np.count_nonzero(primary_mask[y0:split], axis=1)
    lower_widths = np.count_nonzero(primary_mask[split:y1], axis=1)
    upper_width = int(upper_widths.max()) if upper_widths.size else 0
    nonzero_lower = lower_widths[lower_widths > 0]
    lower_width = float(np.median(nonzero_lower)) if nonzero_lower.size else 1.0
    upper_width_ratio = upper_width / max(lower_width, 1.0)
    upper_width_anomaly = (
        upper_width / width >= settings.minimum_upper_canvas_width_ratio
        and upper_width_ratio >= settings.maximum_upper_width_ratio
    )
    checks["abnormal_upper_width"] = {
        "status": "FAIL" if upper_width_anomaly else "PASS",
        "upper_canvas_width_ratio": upper_width / width,
        "upper_to_lower_width_ratio": upper_width_ratio,
    }
    if upper_width_anomaly:
        violations.append("abnormally_wide_object_above_body")

    checks["feet_body_connection"] = {
        "status": "NOT_EVALUATED", "reason": "body_keypoint_parser_required"}
    checks["garment_body_overlap"] = {
        "status": "NOT_EVALUATED", "reason": "garment_parser_required"}
    # Border-color segmentation cannot establish character occupancy when its
    # chosen component bridges the entire canvas. A large prop/background can
    # otherwise masquerade as one large, valid character. Fail closed without
    # claiming that this connected component is a detected person.
    ambiguous_foreground = primary['x'] == 0 and primary['width'] == width
    unresolved = []
    if pose_ambiguous_overhead:
        unresolved.append('overhead_component_near_primary_pose_ambiguous')
    if ambiguous_foreground:
        unresolved.append('foreground_bridges_horizontal_borders')
    checks['foreground_separation'] = {
        'status': 'UNRESOLVED' if ambiguous_foreground else 'PASS',
        'reason': unresolved[0] if unresolved else None,
    }
    action = 'reject' if violations else 'review' if unresolved else 'pass'
    return {
        **settings.record(), "status": action.upper(), "action": action,
        "violations": violations, "unresolved": unresolved, "checks": checks,
        "foreground_ratio": float(np.count_nonzero(mask) / canvas_area),
        "component_count": len(components), "primary_component": primary,
    }


