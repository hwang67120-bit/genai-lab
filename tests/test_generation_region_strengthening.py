"""작은 부위 생성 영역 보강 검사. 이미지 품질 검사가 아니다."""
import cv2
import numpy as np
from PIL import Image
import pytest

from genai_lab.generation_region_strengthening import strengthen_generation_region
from genai_lab.reference_conditions import source_part_conditions
from genai_lab.regional_reference import DetectedPart, crop_reference


def ear_mask():
    values = np.zeros((1344, 768), dtype=np.uint8)
    cv2.ellipse(values, (340, 155), (31, 22), 0, 0, 360, 255, -1)
    cv2.ellipse(values, (450, 155), (31, 22), 0, 0, 360, 255, -1)
    return Image.fromarray(values)


def policy(**overrides):
    ears = {
        "strong_threshold": .10,
        "middle_queries": 252,
        "minimum_middle_cells": 2,
        "coarse_queries": 66,
        "minimum_coarse_cells": 1,
        "maximum_area_growth": 2.0,
        "maximum_radius_ratio": .02,
    }
    ears.update(overrides)
    return {"enabled": True, "parts": {"ears": ears}}


def test_only_generation_copy_is_strengthened_and_audited():
    source = ear_mask()
    before = source.tobytes()
    output, audit = strengthen_generation_region("ears", source, policy())
    try:
        assert source.tobytes() == before
        assert audit["status"] in ("already_sufficient", "strengthened")
        assert audit["source_mask_preserved"] is True
        assert audit["reference_rgb_changed"] is False
        assert audit["scale_changed"] is False
        assert audit["area_growth"] <= 2.0
        assert audit["attention_cells"]["252"]["strong_cells"] >= 2
        assert audit["attention_cells"]["66"]["strong_cells"] >= 1
        assert np.count_nonzero(np.asarray(output)) >= np.count_nonzero(np.asarray(source))
    finally:
        source.close()
        output.close()


def test_unconfigured_tail_is_byte_identical():
    source = ear_mask()
    output, audit = strengthen_generation_region("tail", source, policy())
    try:
        assert output.tobytes() == source.tobytes()
        assert audit["status"] == "disabled_or_not_configured"
    finally:
        source.close()
        output.close()


def test_returned_hair_mask_blocks_only_ear_dilation_growth():
    source = ear_mask()
    source_before = source.tobytes()
    blocked_values = np.zeros((1344, 768), dtype=np.uint8)
    blocked_values[120:205, 300:500] = 255
    blocked_values[np.asarray(source) == 255] = 0
    blocked = Image.fromarray(blocked_values)
    output, audit = strengthen_generation_region(
        "ears", source, policy(), blocked_mask=blocked)
    try:
        assert not np.any(
            (np.asarray(output) == 255) & (np.asarray(blocked) == 255))
        assert audit["blocked_mask_applied"] is True
        assert audit["status"] in ("strengthened", "blocked_fallback_source")
        assert audit["blocked_pixels_removed"] > 0
        assert source.tobytes() == source_before
    finally:
        source.close()
        blocked.close()
        output.close()


def test_impossible_growth_is_blocked_instead_of_overexpanding():
    source = ear_mask()
    with pytest.raises(ValueError, match="안전 한도"):
        strengthen_generation_region(
            "ears", source,
            policy(minimum_middle_cells=100, maximum_area_growth=1.01,
                   maximum_radius_ratio=.001))
    source.close()


def test_source_rgb_crop_and_detected_mask_are_not_changed():
    source_mask = ear_mask()
    detected = DetectedPart("ears", source_mask, None, .35)
    with Image.new("RGB", source_mask.size, (80, 110, 190)) as source:
        with crop_reference(source, source_mask) as expected:
            audit = {}
            conditions = source_part_conditions(
                source, [detected], policy(), audit)
            with conditions[0].image() as actual:
                assert actual.size == expected.size
                assert actual.tobytes() == expected.tobytes()
    try:
        assert detected.source_mask is source_mask
        assert audit["ears"]["scale_changed"] is False
        assert conditions[0].scale == .35
        with conditions[0].region() as region:
            assert np.count_nonzero(np.asarray(region)) >= np.count_nonzero(np.asarray(source_mask))
    finally:
        detected.close()
