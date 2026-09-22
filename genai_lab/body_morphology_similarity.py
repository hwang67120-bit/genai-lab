"""Diagnostic-only comparison between a sealed body vector and output silhouette."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image

from genai_lab.body_morphology import (
    BodyMorphologyVector,
    COMPONENT_NAMES,
)
from genai_lab.candidate_structure_gate import (
    _component_records,
    _foreground_mask,
)


OBSERVABLE_COMPONENTS = (
    "shoulder_width",
    "pelvis_width",
    "waist_hip_ratio",
)
CALIBRATION_RANGES = {
    "shoulder_width": (0.12, 0.42),
    "pelvis_width": (0.10, 0.36),
    "waist_hip_ratio": (0.55, 1.10),
}
BAND_FRACTIONS = {
    "shoulder_width": 0.30,
    "waist_width": 0.47,
    "pelvis_width": 0.56,
}


@dataclass(frozen=True)
class BodyMorphologySimilaritySettings:
    enabled: bool = False
    background_color_distance: float = 22.0
    minimum_observable_components: int = 3
    band_half_height_ratio: float = 0.02

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("체형 유사도 enabled는 bool이어야 합니다.")
        if (
            not isinstance(self.background_color_distance, (int, float))
            or isinstance(self.background_color_distance, bool)
            or not math.isfinite(float(self.background_color_distance))
            or self.background_color_distance <= 0
        ):
            raise ValueError("체형 유사도 배경 색상 거리는 0보다 큰 유한한 수여야 합니다.")
        if (
            type(self.minimum_observable_components) is not int
            or not 1 <= self.minimum_observable_components <= len(OBSERVABLE_COMPONENTS)
        ):
            raise ValueError("체형 유사도 최소 관측 항목 수가 잘못됐습니다.")
        if (
            not isinstance(self.band_half_height_ratio, (int, float))
            or isinstance(self.band_half_height_ratio, bool)
            or not 0 < float(self.band_half_height_ratio) <= .10
        ):
            raise ValueError("체형 유사도 폭 측정 띠 비율은 0보다 크고 0.10 이하여야 합니다.")

    def record(self) -> dict[str, Any]:
        return {
            "version": "body_morphology_similarity_v1",
            **self.__dict__,
            "policy": "diagnostic_only_never_blocks_return_or_retry",
            "measurement_method": "output_foreground_silhouette_proxy_v1",
        }


def resolve_body_morphology_similarity(
    config: Mapping[str, Any],
) -> BodyMorphologySimilaritySettings:
    raw = config.get("body_morphology_similarity", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("체형 유사도 설정은 객체여야 합니다.")
    allowed = {item.name for item in fields(BodyMorphologySimilaritySettings)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"알 수 없는 체형 유사도 설정: {sorted(unknown)}")
    return BodyMorphologySimilaritySettings(**dict(raw))


def _unresolved(
    settings: BodyMorphologySimilaritySettings,
    target: BodyMorphologyVector,
    reason: str,
) -> dict[str, Any]:
    return {
        **settings.record(),
        "status": "UNRESOLVED",
        "reason": reason,
        "target": target.record(),
        "observed": {name: None for name in COMPONENT_NAMES},
        "component_similarity": {name: None for name in COMPONENT_NAMES},
        "observable_components": [],
        "unobservable_components": list(COMPONENT_NAMES),
        "observable_component_count": 0,
        "total_component_count": len(COMPONENT_NAMES),
        "coverage_percentage": 0.0,
        "overall_similarity": None,
        "overall_similarity_percentage": None,
        "percentage_is_probability": False,
        "blocks_return": False,
        "used_for_retry": False,
    }


def _normalize(value: float, minimum: float, maximum: float) -> float:
    return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))


def _band_width_ratio(
    mask: np.ndarray,
    primary: Mapping[str, int],
    fraction: float,
    half_ratio: float,
) -> float | None:
    x = int(primary["x"])
    y = int(primary["y"])
    width = int(primary["width"])
    height = int(primary["height"])
    center = y + int(round(height * fraction))
    half = max(1, int(round(height * half_ratio)))
    y0 = max(y, center - half)
    y1 = min(mask.shape[0], y + height, center + half + 1)
    rows = mask[y0:y1, x:x + width] > 0
    if rows.size == 0:
        return None
    widths = np.count_nonzero(rows, axis=1)
    widths = widths[widths > 0]
    if widths.size == 0:
        return None
    return float(np.median(widths) / max(height, 1))


def evaluate_body_morphology_similarity(
    image: Image.Image,
    target: BodyMorphologyVector,
    settings: BodyMorphologySimilaritySettings,
    *,
    framing_type: str = "full_body",
    structure_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a partial silhouette indicator without making a quality decision."""
    if not settings.enabled:
        return {
            **settings.record(),
            "status": "DISABLED",
            "blocks_return": False,
            "used_for_retry": False,
        }
    if not isinstance(target, BodyMorphologyVector):
        raise ValueError("체형 유사도 대상 벡터 타입이 올바르지 않습니다.")
    frame = getattr(framing_type, "value", framing_type)
    if frame != "full_body":
        return _unresolved(settings, target, "full_body_frame_required")
    if (
        structure_report
        and structure_report.get("action") in {"reject", "review"}
    ):
        return _unresolved(
            settings,
            target,
            "structure_gate_not_passed",
        )

    mask = _foreground_mask(image, settings.background_color_distance)
    height, width = mask.shape
    minimum_area = max(4, int(round(height * width * .001)))
    components = _component_records(mask, minimum_area)
    if not components:
        return _unresolved(settings, target, "foreground_not_detected")
    center_x = width / 2.0
    primary = max(
        components,
        key=lambda item: (
            item["height"] / height,
            int(item["x"] <= center_x <= item["x"] + item["width"]),
            item["area"],
        ),
    )
    if primary["x"] == 0 and primary["width"] == width:
        return _unresolved(
            settings,
            target,
            "foreground_bridges_horizontal_borders",
        )

    _, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    primary_mask = np.where(labels == primary["label"], 255, 0).astype(np.uint8)
    shoulder_raw = _band_width_ratio(
        primary_mask,
        primary,
        BAND_FRACTIONS["shoulder_width"],
        settings.band_half_height_ratio,
    )
    waist_raw = _band_width_ratio(
        primary_mask,
        primary,
        BAND_FRACTIONS["waist_width"],
        settings.band_half_height_ratio,
    )
    pelvis_raw = _band_width_ratio(
        primary_mask,
        primary,
        BAND_FRACTIONS["pelvis_width"],
        settings.band_half_height_ratio,
    )
    raw_measurements = {
        "shoulder_width_to_height": shoulder_raw,
        "waist_width_to_height": waist_raw,
        "pelvis_width_to_height": pelvis_raw,
        "waist_to_pelvis_width": (
            waist_raw / pelvis_raw
            if waist_raw is not None and pelvis_raw not in {None, 0.0}
            else None
        ),
    }
    observed = {name: None for name in COMPONENT_NAMES}
    if shoulder_raw is not None:
        observed["shoulder_width"] = _normalize(
            shoulder_raw, *CALIBRATION_RANGES["shoulder_width"])
    if pelvis_raw is not None:
        observed["pelvis_width"] = _normalize(
            pelvis_raw, *CALIBRATION_RANGES["pelvis_width"])
    if raw_measurements["waist_to_pelvis_width"] is not None:
        observed["waist_hip_ratio"] = _normalize(
            raw_measurements["waist_to_pelvis_width"],
            *CALIBRATION_RANGES["waist_hip_ratio"],
        )

    target_values = target.values()
    component_similarity = {
        name: (
            round(1.0 - abs(target_values[name] - observed[name]), 6)
            if observed[name] is not None
            else None
        )
        for name in COMPONENT_NAMES
    }
    available = [
        name for name in OBSERVABLE_COMPONENTS
        if component_similarity[name] is not None
    ]
    if len(available) < settings.minimum_observable_components:
        report = _unresolved(settings, target, "insufficient_observable_components")
        report["observed"] = observed
        report["component_similarity"] = component_similarity
        report["observable_components"] = available
        report["unobservable_components"] = [
            name for name in COMPONENT_NAMES if name not in available
        ]
        report["observable_component_count"] = len(available)
        report["coverage_percentage"] = round(
            len(available) / len(COMPONENT_NAMES) * 100.0, 2)
        report["raw_measurements"] = raw_measurements
        return report

    overall = sum(component_similarity[name] for name in available) / len(available)
    return {
        **settings.record(),
        "status": "RECORDED",
        "reason": None,
        "target": target.record(),
        "observed": observed,
        "component_similarity": component_similarity,
        "component_similarity_percentages": {
            name: (
                round(component_similarity[name] * 100.0, 2)
                if component_similarity[name] is not None
                else None
            )
            for name in COMPONENT_NAMES
        },
        "observable_components": available,
        "unobservable_components": [
            name for name in COMPONENT_NAMES if name not in available
        ],
        "observable_component_count": len(available),
        "total_component_count": len(COMPONENT_NAMES),
        "coverage_percentage": round(
            len(available) / len(COMPONENT_NAMES) * 100.0, 2),
        "overall_similarity": round(overall, 6),
        "overall_similarity_percentage": round(overall * 100.0, 2),
        "percentage_is_probability": False,
        "blocks_return": False,
        "used_for_retry": False,
        "raw_measurements": raw_measurements,
        "primary_component": dict(primary),
        "limitations": [
            "silhouette_width_is_affected_by_pose_clothing_hair_and_animal_parts",
            "chest_muscle_body_fat_jaw_height_and_limb_thickness_not_measured",
        ],
    }
