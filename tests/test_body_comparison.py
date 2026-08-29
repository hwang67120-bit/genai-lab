"""신체 비교 단계의 수치 계산과 마스크 보호 규칙만 빠르게 검증한다."""

import numpy as np
from PIL import Image
import pytest

from genai_lab.body_comparison import (
    CharacterBodyComparisonError,
    calculate_mask_expansion_radius,
    create_human_agnostic_image_candidate,
    refine_character_clothing_change_mask,
)


@pytest.mark.parametrize(
    ("image_size", "expected_pixels"),
    (
        ((512, 512), 5),
        ((768, 768), 8),
        ((768, 1344), 13),
        ((2048, 2048), 15),
    ),
)
def test_calculate_mask_expansion_radius_is_limited_to_5_through_15_pixels(
    image_size: tuple[int, int],
    expected_pixels: int,
) -> None:
    assert calculate_mask_expansion_radius(image_size) == expected_pixels


def test_refine_mask_closes_hole_expands_edge_and_protects_body() -> None:
    raw_mask_array = np.zeros((64, 64), dtype=np.uint8)
    raw_mask_array[20:44, 20:44] = 255
    raw_mask_array[31:33, 31:33] = 0
    protection_mask_array = np.zeros((64, 64), dtype=np.uint8)
    protection_mask_array[18:24, 18:24] = 255

    raw_mask = Image.fromarray(raw_mask_array, mode="L")
    protection_mask = Image.fromarray(protection_mask_array, mode="L")
    refinement = refine_character_clothing_change_mask(
        raw_clothing_mask=raw_mask,
        identity_protection_mask=protection_mask,
        expansion_radius_pixels=5,
        closing_radius_pixels=2,
    )
    try:
        assert refinement.closed_mask.getpixel((31, 31)) == 255
        assert refinement.expanded_mask.getpixel((16, 32)) == 255
        assert refinement.safe_change_mask.getpixel((21, 21)) == 0
        assert refinement.attempted_protected_overlap_pixels > 0
        assert 0.0 < refinement.safe_change_percent < 100.0
    finally:
        refinement.close()
        raw_mask.close()
        protection_mask.close()


def test_refine_mask_rejects_expansion_outside_fixed_range() -> None:
    raw_mask = Image.new("L", (32, 32), 255)
    protection_mask = Image.new("L", (32, 32), 0)
    try:
        with pytest.raises(CharacterBodyComparisonError, match="5~15픽셀"):
            refine_character_clothing_change_mask(
                raw_clothing_mask=raw_mask,
                identity_protection_mask=protection_mask,
                expansion_radius_pixels=4,
            )
    finally:
        raw_mask.close()
        protection_mask.close()


def test_create_human_agnostic_image_neutralizes_only_approved_mask() -> None:
    source_image = Image.new("RGB", (4, 4), (10, 20, 30))
    raw_mask_array = np.zeros((4, 4), dtype=np.uint8)
    raw_mask_array[1:3, 1:3] = 255
    erasure_mask_array = raw_mask_array.copy()
    erasure_mask_array[2, 2] = 0
    raw_mask = Image.fromarray(raw_mask_array, mode="L")
    erasure_mask = Image.fromarray(erasure_mask_array, mode="L")

    candidate = create_human_agnostic_image_candidate(
        source_image=source_image,
        clothing_erasure_mask=erasure_mask,
        raw_clothing_mask=raw_mask,
        neutral_rgb=(127, 127, 127),
    )
    try:
        assert candidate.neutralized_image.getpixel((1, 1)) == (127, 127, 127)
        assert candidate.neutralized_image.getpixel((0, 0)) == (10, 20, 30)
        assert candidate.neutralized_image.getpixel((2, 2)) == (10, 20, 30)
        assert candidate.neutralized_pixel_count == 3
        assert candidate.neutralized_percent == pytest.approx(18.75)
        assert candidate.raw_mask_pixel_count == 4
        assert candidate.raw_mask_coverage_percent == pytest.approx(75.0)
        assert candidate.changed_pixel_count_outside_mask == 0
    finally:
        candidate.close()
        source_image.close()
        raw_mask.close()
        erasure_mask.close()


def test_create_human_agnostic_image_rejects_different_sizes() -> None:
    source_image = Image.new("RGB", (4, 4), (0, 0, 0))
    erasure_mask = Image.new("L", (3, 4), 255)
    raw_mask = Image.new("L", (4, 4), 255)
    try:
        with pytest.raises(CharacterBodyComparisonError, match="크기가 다릅니다"):
            create_human_agnostic_image_candidate(
                source_image=source_image,
                clothing_erasure_mask=erasure_mask,
                raw_clothing_mask=raw_mask,
            )
    finally:
        source_image.close()
        erasure_mask.close()
        raw_mask.close()


def test_create_human_agnostic_image_rejects_empty_erasure_mask() -> None:
    source_image = Image.new("RGB", (4, 4), (0, 0, 0))
    erasure_mask = Image.new("L", (4, 4), 0)
    raw_mask = Image.new("L", (4, 4), 255)
    try:
        with pytest.raises(CharacterBodyComparisonError, match="0개"):
            create_human_agnostic_image_candidate(
                source_image=source_image,
                clothing_erasure_mask=erasure_mask,
                raw_clothing_mask=raw_mask,
            )
    finally:
        source_image.close()
        erasure_mask.close()
        raw_mask.close()
