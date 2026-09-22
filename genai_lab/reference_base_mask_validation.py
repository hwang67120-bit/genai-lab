"""Cross-check between source-reference and independently parsed Base masks.

Source-coordinate masks are resized only for measurement. The report can block
automatic progression to native refinement, but never becomes a generation mask.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from PIL import Image


VERSION = "reference_base_mask_cross_validation_v2"
STABLE_PARTS = ("face", "hair", "identity", "foreground")
DIAGNOSTIC_PARTS = ("garment",)

DEFAULT_RULES = {
    "face": {
        "minimum_iou": 0.25, "maximum_centroid_shift": 0.10,
        "minimum_area_ratio": 0.35, "maximum_area_ratio": 2.85,
        "minimum_pixels": 64, "minimum_canvas_area_ratio": 0.0005,
    },
    "hair": {
        "minimum_iou": 0.30, "maximum_centroid_shift": 0.12,
        "minimum_area_ratio": 0.35, "maximum_area_ratio": 2.85,
        "minimum_pixels": 128, "minimum_canvas_area_ratio": 0.0015,
    },
    "identity": {
        "minimum_iou": 0.45, "maximum_centroid_shift": 0.08,
        "minimum_area_ratio": 0.55, "maximum_area_ratio": 1.80,
        "minimum_pixels": 128, "minimum_canvas_area_ratio": 0.0015,
    },
    "foreground": {
        "minimum_iou": 0.45, "maximum_centroid_shift": 0.08,
        "minimum_area_ratio": 0.55, "maximum_area_ratio": 1.80,
        "minimum_pixels": 256, "minimum_canvas_area_ratio": 0.01,
    },
    "garment": {
        "minimum_iou": 0.20, "maximum_centroid_shift": 0.15,
        "minimum_area_ratio": 0.25, "maximum_area_ratio": 4.00,
        "minimum_pixels": 128, "minimum_canvas_area_ratio": 0.001,
    },
}


def _binary(mask: Image.Image, size: tuple[int, int]) -> tuple[np.ndarray, str]:
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be a PIL image")
    alignment = "same_canvas"
    if mask.size != size:
        alignment = "normalized_canvas_resize"
        with mask.resize(size, Image.Resampling.NEAREST) as resized:
            values = np.asarray(resized.convert("L"), dtype=np.uint8)
    else:
        values = np.asarray(mask.convert("L"), dtype=np.uint8)
    return values >= 128, alignment


def _centroid(binary: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return None
    height, width = binary.shape
    return (
        float(xs.mean() / max(width - 1, 1)),
        float(ys.mean() / max(height - 1, 1)),
    )


def _box(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _part_metrics(
    source: np.ndarray,
    base: np.ndarray,
    rules: Mapping[str, float],
) -> dict[str, Any]:
    source_pixels = int(source.sum())
    base_pixels = int(base.sum())
    canvas_pixels = int(source.size)
    source_canvas_ratio = source_pixels / canvas_pixels
    base_canvas_ratio = base_pixels / canvas_pixels
    minimum_pixels = int(rules["minimum_pixels"])
    minimum_canvas_ratio = float(rules["minimum_canvas_area_ratio"])
    failures = []
    if source_pixels < minimum_pixels or source_canvas_ratio < minimum_canvas_ratio:
        failures.append("source_mask_below_measurement_minimum")
    if base_pixels < minimum_pixels or base_canvas_ratio < minimum_canvas_ratio:
        failures.append("base_mask_below_measurement_minimum")
    result = {
        "source_pixels": source_pixels,
        "base_pixels": base_pixels,
        "canvas_pixels": canvas_pixels,
        "source_canvas_area_ratio": float(source_canvas_ratio),
        "base_canvas_area_ratio": float(base_canvas_ratio),
        "measurement_eligible": not failures,
        "measurement_failure_reasons": failures,
        "comparable": not failures,
    }
    if failures:
        return result

    intersection = int(np.logical_and(source, base).sum())
    union = int(np.logical_or(source, base).sum())
    source_centroid = _centroid(source)
    base_centroid = _centroid(base)
    assert source_centroid is not None and base_centroid is not None
    centroid_shift = math.dist(source_centroid, base_centroid) / math.sqrt(2.0)
    source_box = _box(source)
    base_box = _box(base)
    assert source_box is not None and base_box is not None
    source_box_size = (
        source_box[2] - source_box[0], source_box[3] - source_box[1])
    base_box_size = (base_box[2] - base_box[0], base_box[3] - base_box[1])
    result.update({
        "intersection_pixels": intersection,
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (source_pixels + base_pixels)),
        "source_containment": float(intersection / source_pixels),
        "base_containment": float(intersection / base_pixels),
        "base_to_source_area_ratio": float(base_pixels / source_pixels),
        "source_centroid": list(source_centroid),
        "base_centroid": list(base_centroid),
        "normalized_centroid_shift": float(centroid_shift),
        "source_bbox": list(source_box),
        "base_bbox": list(base_box),
        "base_to_source_bbox_width_ratio": float(
            base_box_size[0] / max(source_box_size[0], 1)),
        "base_to_source_bbox_height_ratio": float(
            base_box_size[1] / max(source_box_size[1], 1)),
    })
    return result


def _resolved_rules(settings: Mapping[str, Any] | None) -> dict[str, dict[str, float]]:
    overrides = settings if isinstance(settings, Mapping) else {}
    result: dict[str, dict[str, float]] = {}
    for name, defaults in DEFAULT_RULES.items():
        part = overrides.get(name, {})
        if not isinstance(part, Mapping):
            part = {}
        values = {key: float(part.get(key, value)) for key, value in defaults.items()}
        if (
            not 0 <= values["minimum_iou"] <= 1
            or not 0 <= values["maximum_centroid_shift"] <= 1
            or values["minimum_area_ratio"] <= 0
            or values["maximum_area_ratio"] < values["minimum_area_ratio"]
            or values["minimum_pixels"] < 1
            or not 0 <= values["minimum_canvas_area_ratio"] <= 1
            or not all(math.isfinite(value) for value in values.values())
        ):
            raise ValueError(f"invalid reference/Base mask rules: {name}")
        result[name] = values
    return result


def _classify(
    metrics: Mapping[str, Any],
    rules: Mapping[str, float],
) -> tuple[str, list[str]]:
    if not metrics.get("measurement_eligible"):
        return "not_measurable", list(metrics.get("measurement_failure_reasons", ()))
    reasons = []
    if float(metrics["iou"]) < rules["minimum_iou"]:
        reasons.append("low_canvas_iou")
    if float(metrics["normalized_centroid_shift"]) > rules["maximum_centroid_shift"]:
        reasons.append("centroid_shift")
    ratio = float(metrics["base_to_source_area_ratio"])
    if not rules["minimum_area_ratio"] <= ratio <= rules["maximum_area_ratio"]:
        reasons.append("area_ratio_change")
    return ("review" if reasons else "consistent"), reasons


def validate_reference_base_masks(
    source_masks: Mapping[str, Image.Image | None],
    base_masks: Mapping[str, Image.Image | None],
    *,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure independent masks and decide whether A7 may run automatically."""
    raw_settings = settings if isinstance(settings, Mapping) else {}
    block_on_review = raw_settings.get("block_on_review", True)
    if type(block_on_review) is not bool:
        raise ValueError("reference_base_mask_validation.block_on_review must be boolean")
    rules = _resolved_rules(raw_settings)
    output_size = next(
        (mask.size for mask in base_masks.values() if isinstance(mask, Image.Image)),
        None,
    )
    if output_size is None:
        return {
            "version": VERSION,
            "status": "not_measurable",
            "reason": "base_masks_unavailable",
            "parts": {},
            "blocking": True,
            "progression": "blocked",
            "blocking_reasons": ["base_masks_unavailable"],
            "generation_masks_modified": False,
        }

    parts: dict[str, Any] = {}
    for name in (*STABLE_PARTS, *DIAGNOSTIC_PARTS):
        source_mask = source_masks.get(name)
        base_mask = base_masks.get(name)
        role = "stable_character" if name in STABLE_PARTS else "expected_edit"
        if not isinstance(source_mask, Image.Image) or not isinstance(base_mask, Image.Image):
            status = "not_measurable"
            reasons = ["mask_unavailable"]
            parts[name] = {
                "role": role,
                "status": status,
                "reasons": reasons,
                "blocks_generation": name in STABLE_PARTS,
            }
            continue
        source, alignment = _binary(source_mask, output_size)
        base, _ = _binary(base_mask, output_size)
        metrics = _part_metrics(source, base, rules[name])
        status, reasons = _classify(metrics, rules[name])
        parts[name] = {
            "role": role,
            "status": status,
            "reasons": reasons,
            "alignment": alignment,
            "metrics": metrics,
            "rules": dict(rules[name]),
            "blocks_generation": (
                name in STABLE_PARTS
                and (
                    status == "not_measurable"
                    or (status == "review" and block_on_review)
                )
            ),
        }

    stable_statuses = [parts[name]["status"] for name in STABLE_PARTS]
    if "not_measurable" in stable_statuses:
        overall = "not_measurable"
    elif "review" in stable_statuses:
        overall = "review_required"
    else:
        overall = "consistent"
    blocking_reasons = [
        f"{name}:{reason}"
        for name in STABLE_PARTS
        if parts[name].get("blocks_generation")
        for reason in (parts[name].get("reasons") or [parts[name]["status"]])
    ]
    blocking = bool(blocking_reasons)
    return {
        "version": VERSION,
        "status": overall,
        "mode": "automatic_progression_gate",
        "coordinate_policy": (
            "source masks are resized only for measurement; Base masks are "
            "independently redetected in Base coordinates"
        ),
        "parts": parts,
        "stable_parts": list(STABLE_PARTS),
        "expected_edit_parts": list(DIAGNOSTIC_PARTS),
        "blocking": blocking,
        "progression": "blocked" if blocking else "eligible",
        "blocking_reasons": blocking_reasons,
        "block_on_review": block_on_review,
        "generation_masks_modified": False,
        "source_masks_reused_for_generation": False,
        "semantic_accuracy_verified": False,
        "rule_calibration": "conservative_defaults_not_dataset_calibrated",
    }
