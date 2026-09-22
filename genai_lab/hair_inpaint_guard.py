"""Pixel and feature gates for isolated hair inpaint promotion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class HairBoundaryGuardSettings:
    enabled: bool
    boundary_width_pixels: int
    feather_radius_pixels: int
    diagnostic_channel_difference: int
    maximum_raw_outside_changed_ratio: float
    seam_mean_difference_max: float
    seam_p95_difference_max: float
    seam_changed_ratio_max: float
    require_exact_protected_pixels: bool
    require_exact_outside_pixels: bool

    def record(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class HairPromotionGateSettings:
    enabled: bool
    minimum_color_similarity: float
    minimum_shape_iou: float
    maximum_length_change_ratio: float
    maximum_area_change_ratio: float

    def record(self):
        return dict(self.__dict__)


def resolve_hair_boundary_guard(raw):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("헤어 경계 보호 설정은 객체여야 합니다.")
    settings = HairBoundaryGuardSettings(
        enabled=bool(raw.get("enabled", True)),
        boundary_width_pixels=int(raw.get("boundary_width_pixels", 8)),
        feather_radius_pixels=int(raw.get("feather_radius_pixels", 4)),
        diagnostic_channel_difference=int(
            raw.get("diagnostic_channel_difference", 3)),
        maximum_raw_outside_changed_ratio=float(
            raw.get("maximum_raw_outside_changed_ratio", 1.0)),
        seam_mean_difference_max=float(
            raw.get("seam_mean_difference_max", 12.0)),
        seam_p95_difference_max=float(
            raw.get("seam_p95_difference_max", 32.0)),
        seam_changed_ratio_max=float(
            raw.get("seam_changed_ratio_max", .35)),
        require_exact_protected_pixels=bool(
            raw.get("require_exact_protected_pixels", True)),
        require_exact_outside_pixels=bool(
            raw.get("require_exact_outside_pixels", True)),
    )
    if (settings.boundary_width_pixels < 0
            or settings.feather_radius_pixels < 0
            or not 0 <= settings.diagnostic_channel_difference <= 255
            or not 0 <= settings.maximum_raw_outside_changed_ratio <= 1
            or not 0 <= settings.seam_changed_ratio_max <= 1
            or not 0 <= settings.seam_mean_difference_max <= 255
            or not 0 <= settings.seam_p95_difference_max <= 255):
        raise ValueError("헤어 경계 보호 설정값이 올바르지 않습니다.")
    return settings


def resolve_hair_promotion_gate(raw):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("헤어 승격 게이트 설정은 객체여야 합니다.")
    settings = HairPromotionGateSettings(
        enabled=bool(raw.get("enabled", True)),
        minimum_color_similarity=float(
            raw.get("minimum_color_similarity", .80)),
        minimum_shape_iou=float(raw.get("minimum_shape_iou", .65)),
        maximum_length_change_ratio=float(
            raw.get("maximum_length_change_ratio", .20)),
        maximum_area_change_ratio=float(
            raw.get("maximum_area_change_ratio", .25)),
    )
    ratios = (
        settings.minimum_color_similarity,
        settings.minimum_shape_iou,
        settings.maximum_length_change_ratio,
        settings.maximum_area_change_ratio,
    )
    if any(not np.isfinite(value) or not 0 <= value <= 1
           for value in ratios):
        raise ValueError("헤어 승격 게이트 설정값이 올바르지 않습니다.")
    return settings


def _binary(mask, size):
    if mask.size != size:
        raise ValueError("헤어 경계 보호 마스크 크기가 이미지와 다릅니다.")
    with mask.convert("L") as converted:
        return np.asarray(converted, dtype=np.uint8) >= 128


def _dilate(region, radius):
    if radius <= 0:
        return region.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(region.astype(np.uint8), kernel, iterations=1) > 0


def _difference_metrics(base, proposed, region, threshold):
    pixel_count = int(np.count_nonzero(region))
    if pixel_count == 0:
        return {
            "pixel_count": 0,
            "mean_difference": 0.0,
            "p95_difference": 0.0,
            "changed_pixel_count": 0,
            "changed_ratio": 0.0,
        }
    difference = np.max(
        np.abs(base.astype(np.int16) - proposed.astype(np.int16)), axis=2)
    selected = difference[region]
    changed = selected > threshold
    return {
        "pixel_count": pixel_count,
        "mean_difference": round(float(np.mean(selected)), 4),
        "p95_difference": round(float(np.percentile(selected, 95)), 4),
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_ratio": round(float(np.mean(changed)), 6),
    }


def guarded_hair_composite(
        generated_image, proposed, repair_mask, protected_mask, settings):
    """Composite a hair proposal while preserving every protected pixel."""
    if generated_image.size != proposed.size:
        raise ValueError("헤어 보정 전후 이미지 크기가 다릅니다.")
    size = generated_image.size
    with generated_image.convert("RGB") as base_rgb:
        base = np.asarray(base_rgb, dtype=np.uint8).copy()
    with proposed.convert("RGB") as proposed_rgb:
        candidate = np.asarray(proposed_rgb, dtype=np.uint8).copy()
    repair = _binary(repair_mask, size)
    protected = _binary(protected_mask, size)
    clean = repair & ~protected
    allowed = clean.copy()
    outside = ~allowed
    boundary = clean & _dilate(~clean, settings.boundary_width_pixels)

    report = {
        "version": "guarded_hair_composite_v1",
        "status": "running",
        "reason": None,
        "clean_pixels": int(np.count_nonzero(clean)),
        "boundary_pixels": int(np.count_nonzero(boundary)),
        "protected_pixels": int(np.count_nonzero(protected)),
        "raw_protected": _difference_metrics(
            base, candidate, protected,
            settings.diagnostic_channel_difference),
        "raw_outside": _difference_metrics(
            base, candidate, outside,
            settings.diagnostic_channel_difference),
        "seam": _difference_metrics(
            base, candidate, boundary,
            settings.diagnostic_channel_difference),
    }
    if not clean.any():
        report.update(status="rejected", reason="empty_clean_hair_region")
        return None, report
    seam = report["seam"]
    if (seam["mean_difference"] > settings.seam_mean_difference_max
            or seam["p95_difference"] > settings.seam_p95_difference_max
            or seam["changed_ratio"] > settings.seam_changed_ratio_max):
        report.update(status="rejected", reason="seam_threshold_exceeded")
        return None, report
    if (report["raw_outside"]["changed_ratio"]
            > settings.maximum_raw_outside_changed_ratio):
        report.update(status="rejected", reason="raw_outside_change_exceeded")
        return None, report

    with Image.fromarray(clean.astype(np.uint8) * 255, mode="L") as alpha_image:
        if settings.feather_radius_pixels:
            feathered = alpha_image.filter(ImageFilter.GaussianBlur(
                radius=settings.feather_radius_pixels))
        else:
            feathered = alpha_image.copy()
    try:
        alpha = np.asarray(feathered, dtype=np.uint8).copy()
    finally:
        feathered.close()
    alpha[outside | protected] = 0
    with Image.fromarray(alpha, mode="L") as alpha_mask:
        corrected = Image.composite(proposed, generated_image, alpha_mask)

    with corrected.convert("RGB") as corrected_rgb:
        corrected_array = np.asarray(corrected_rgb, dtype=np.uint8).copy()
    restore = outside | protected
    corrected_array[restore] = base[restore]
    corrected.close()
    corrected = Image.fromarray(corrected_array, mode="RGB")

    protected_changed = int(np.count_nonzero(np.any(
        corrected_array[protected] != base[protected], axis=1))) \
        if protected.any() else 0
    outside_changed = int(np.count_nonzero(np.any(
        corrected_array[outside] != base[outside], axis=1))) \
        if outside.any() else 0
    report["final_protected_changed_pixels"] = protected_changed
    report["final_outside_changed_pixels"] = outside_changed
    if settings.require_exact_protected_pixels and protected_changed:
        corrected.close()
        report.update(status="rejected", reason="protected_pixel_changed")
        return None, report
    if settings.require_exact_outside_pixels and outside_changed:
        corrected.close()
        report.update(status="rejected", reason="outside_pixel_changed")
        return None, report
    report["status"] = "passed"
    return corrected, report


def _normalized_lab_histogram(image, mask):
    with image.convert("RGB") as rgb:
        values = np.asarray(rgb, dtype=np.uint8).copy()
    selected = _binary(mask, image.size).astype(np.uint8) * 255
    if not np.any(selected):
        return None
    lab = cv2.cvtColor(values, cv2.COLOR_RGB2Lab)
    histogram = cv2.calcHist(
        [lab], [1, 2], selected, [32, 32], [0, 256, 0, 256])
    total = float(np.sum(histogram))
    if total <= 0:
        return None
    return histogram / total


def _relative_geometry(hair_mask, face_mask):
    hair = _binary(hair_mask, hair_mask.size)
    face = _binary(face_mask, hair_mask.size)
    hair_y = np.flatnonzero(np.any(hair, axis=1))
    face_y = np.flatnonzero(np.any(face, axis=1))
    if hair_y.size == 0 or face_y.size == 0:
        return None
    hair_height = int(hair_y[-1] - hair_y[0] + 1)
    face_height = int(face_y[-1] - face_y[0] + 1)
    face_area = max(1, int(np.count_nonzero(face)))
    return {
        "relative_length": hair_height / max(1, face_height),
        "relative_area": int(np.count_nonzero(hair)) / face_area,
    }


def evaluate_hair_promotion(
        source_image, source_mask, before_image, before_hair_mask,
        before_face_mask, corrected_image, after_hair_mask, after_face_mask,
        settings):
    """Evaluate color plus same-coordinate pre/post hair geometry."""
    report = {
        "version": "hair_promotion_gate_v1",
        "status": "disabled" if not settings.enabled else "running",
        "reason": None,
    }
    if not settings.enabled:
        report["passed"] = True
        return report
    source_histogram = _normalized_lab_histogram(source_image, source_mask)
    corrected_histogram = _normalized_lab_histogram(
        corrected_image, after_hair_mask)
    before_geometry = _relative_geometry(before_hair_mask, before_face_mask)
    after_geometry = _relative_geometry(after_hair_mask, after_face_mask)
    if (source_histogram is None or corrected_histogram is None
            or before_geometry is None or after_geometry is None):
        report.update(
            status="unresolved", passed=False,
            reason="hair_promotion_measurement_unavailable")
        return report

    color_similarity = float(cv2.compareHist(
        source_histogram, corrected_histogram, cv2.HISTCMP_CORREL))
    before_hair = _binary(before_hair_mask, before_image.size)
    after_hair = _binary(after_hair_mask, before_image.size)
    union = int(np.count_nonzero(before_hair | after_hair))
    shape_iou = (int(np.count_nonzero(before_hair & after_hair)) / union
                 if union else 0.0)
    length_change = abs(
        after_geometry["relative_length"]
        - before_geometry["relative_length"]
    ) / max(before_geometry["relative_length"], 1e-6)
    area_change = abs(
        after_geometry["relative_area"]
        - before_geometry["relative_area"]
    ) / max(before_geometry["relative_area"], 1e-6)
    checks = {
        "color": color_similarity >= settings.minimum_color_similarity,
        "shape": shape_iou >= settings.minimum_shape_iou,
        "length": length_change <= settings.maximum_length_change_ratio,
        "area": area_change <= settings.maximum_area_change_ratio,
    }
    passed = all(checks.values())
    report.update(
        status="passed" if passed else "rejected",
        passed=passed,
        reason=None if passed else "hair_feature_threshold_exceeded",
        metrics={
            "color_similarity": round(color_similarity, 6),
            "shape_iou": round(shape_iou, 6),
            "length_change_ratio": round(length_change, 6),
            "area_change_ratio": round(area_change, 6),
        },
        checks=checks,
    )
    return report
