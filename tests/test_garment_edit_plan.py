import json

import numpy as np
import pytest
from PIL import Image

from genai_lab.garment_edit_plan import (
    build_garment_edit_plan,
    project_target_garment_coverage,
)
from genai_lab.output_coordinate_control import measure_outside_rgb_change


def mask(values):
    return Image.fromarray(values.astype(np.uint8) * 255, mode="L")


def test_long_sleeve_target_coverage_adds_exposed_generated_base_arms():
    foreground = np.zeros((160, 100), dtype=bool)
    foreground[20:150, 40:60] = True
    foreground[45:120, 20:40] = True
    foreground[45:120, 60:80] = True
    source = np.zeros_like(foreground)
    source[45:90, 42:58] = True
    with mask(source) as source_image, mask(foreground) as foreground_image:
        result = project_target_garment_coverage(
            source_image,
            foreground_image,
            ("blue_jacket", "long_sleeves"),
            growth_pixels=4,
        )
    try:
        target = np.asarray(result.mask) == 255
        assert not source[70, 25]
        assert target[70, 25]
        assert result.record["coordinate_source"] == "generated_base"
        assert result.record["status"] == "REVIEW"
    finally:
        result.close()


def test_skirt_target_coverage_has_bounded_growth_outside_current_silhouette():
    foreground = np.zeros((160, 100), dtype=bool)
    foreground[10:150, 42:58] = True
    source = np.zeros_like(foreground)
    source[70:95, 44:56] = True
    with mask(source) as source_image, mask(foreground) as foreground_image:
        result = project_target_garment_coverage(
            source_image,
            foreground_image,
            ("mini_skirt",),
            growth_pixels=6,
        )
    try:
        target = np.asarray(result.mask) == 255
        growth = np.asarray(result.growth_envelope) == 255
        assert np.any(growth & ~foreground)
        assert np.all(growth <= target)
        assert result.record["growth_pixels_count"] > 0
    finally:
        result.close()


def test_unknown_garment_semantics_stay_unresolved_without_guessing():
    foreground = np.zeros((64, 64), dtype=bool)
    foreground[4:60, 20:44] = True
    source = np.zeros_like(foreground)
    source[20:40, 24:40] = True
    with mask(source) as source_image, mask(foreground) as foreground_image:
        result = project_target_garment_coverage(
            source_image, foreground_image, ("novel_unknown_design",)
        )
    try:
        assert result.record["status"] == "UNRESOLVED"
        assert not np.asarray(result.mask).any()
    finally:
        result.close()


def test_hard_soft_plan_removes_protection_and_never_leaks_soft_values():
    size = (64, 64)
    base = Image.new("RGB", size, "white")
    source = np.zeros((64, 64), dtype=bool)
    source[16:48, 20:44] = True
    target = source.copy()
    target[12:52, 16:48] = True
    foreground = np.zeros_like(source)
    foreground[4:60, 8:56] = True
    hard = np.zeros_like(source)
    hard[12:20, 28:36] = True
    conditional = np.zeros_like(source)
    conditional[38:46, 40:48] = True
    boundary = np.zeros_like(source)
    boundary[20:22, 16:48] = True
    images = [
        mask(source), mask(target), mask(hard), mask(foreground),
        mask(boundary), mask(conditional),
    ]
    try:
        plan = build_garment_edit_plan(
            base,
            images[0],
            images[1],
            images[2],
            images[3],
            soft_boundary_protection=images[4],
            conditional_protection=images[5],
            feather_radius=4,
        )
        try:
            hard_values = np.asarray(plan.hard_edit_domain) == 255
            soft_values = np.asarray(plan.soft_guidance)
            assert not np.any(hard_values & hard)
            assert not np.any(hard_values & conditional)
            assert not np.any(soft_values[~hard_values])
            assert plan.metrics["soft_outside_hard_pixels"] == 0
            assert plan.record["status"] == "REVIEW"
            assert plan.metrics["protection_conflict_pixels"] > 0
        finally:
            plan.close()
    finally:
        base.close()
        for image in images:
            image.close()


def test_mask_plan_rejects_reference_coordinate_size_mismatch():
    base = Image.new("RGB", (64, 64))
    good = Image.new("L", (64, 64), 255)
    wrong = Image.new("L", (32, 32), 255)
    try:
        with pytest.raises(ValueError, match="generated Base coordinates"):
            build_garment_edit_plan(base, wrong, good, good, good)
    finally:
        base.close()
        good.close()
        wrong.close()


def test_outside_rgb_change_is_measured_without_restoring_pixels():
    original = Image.new("RGB", (16, 16), "white")
    proposed = Image.new("RGB", (16, 16), "black")
    hard = np.zeros((16, 16), dtype=bool)
    hard[4:12, 4:12] = True
    hard_image = mask(hard)
    try:
        report = measure_outside_rgb_change(original, proposed, hard_image)
        assert report["outside_changed_pixels"] == 192
        assert report["outside_changed_ratio"] == 1.0
        assert report["outside_pixel_restoration_applied"] is False
        assert proposed.getpixel((0, 0)) == (0, 0, 0)
    finally:
        original.close()
        proposed.close()
        hard_image.close()

