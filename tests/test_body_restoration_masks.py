import numpy as np
from PIL import Image
import pytest

from genai_lab.body_restoration_masks import (
    GarmentThickness,
    prepare_body_restoration_mask_roles,
)


def create_safe_masks(size=(40, 40)):
    clothing = Image.new("L", size, 0)
    clothing.paste(255, (10, 8, 30, 32))
    protected = Image.new("L", size, 0)
    foreground = Image.new("L", size, 255)
    return clothing, protected, foreground


@pytest.mark.parametrize("thickness", ["thin", "normal", "unknown"])
def test_non_thick_garment_preserves_removal_and_has_no_inner_hint(thickness):
    clothing, protected, foreground = create_safe_masks()
    result = prepare_body_restoration_mask_roles(
        clothing, protected, foreground, thickness,
    )
    try:
        assert result.inner_body_hint is None
        assert result.contraction_applied is False
        assert result.inset_pixels == 0
        assert np.array_equal(
            np.asarray(result.clothing_removal_mask),
            np.asarray(clothing),
        )
    finally:
        result.close()
        clothing.close()
        protected.close()
        foreground.close()


def test_thick_garment_creates_separate_subset_without_shrinking_removal():
    clothing, protected, foreground = create_safe_masks()
    result = prepare_body_restoration_mask_roles(
        clothing,
        protected,
        foreground,
        GarmentThickness.THICK,
        erosion_ratio=0.20,
    )
    try:
        removal = np.asarray(result.clothing_removal_mask) >= 128
        inner = np.asarray(result.inner_body_hint) >= 128
        original = np.asarray(clothing) >= 128
        assert result.contraction_applied is True
        assert result.inset_pixels == 2
        assert np.array_equal(removal, original)
        assert np.all(~inner | removal)
        assert 0 < result.inner_hint_pixel_count < result.removal_pixel_count
    finally:
        result.close()
        clothing.close()
        protected.close()
        foreground.close()


def test_thickness_parser_is_explicit_and_rejects_unknown_values():
    assert GarmentThickness.parse(" THICK ") is GarmentThickness.THICK
    with pytest.raises(ValueError, match="지원하지 않는 의상 두께"):
        GarmentThickness.parse("winter-ish")


def test_removal_mask_protection_conflict_is_rejected_not_silently_trimmed():
    clothing, protected, foreground = create_safe_masks()
    protected.putpixel((12, 12), 255)
    try:
        with pytest.raises(ValueError, match="보호 영역=1px"):
            prepare_body_restoration_mask_roles(
                clothing, protected, foreground, "thick",
            )
    finally:
        clothing.close()
        protected.close()
        foreground.close()


def test_removal_mask_outside_foreground_is_rejected():
    clothing, protected, foreground = create_safe_masks()
    foreground.putpixel((12, 12), 0)
    try:
        with pytest.raises(ValueError, match="캐릭터 외곽 밖=1px"):
            prepare_body_restoration_mask_roles(
                clothing, protected, foreground, "normal",
            )
    finally:
        clothing.close()
        protected.close()
        foreground.close()


@pytest.mark.parametrize("ratio", [0.0, -0.1, 0.51])
def test_erosion_ratio_is_bounded(ratio):
    clothing, protected, foreground = create_safe_masks()
    try:
        with pytest.raises(ValueError, match="수축 비율"):
            prepare_body_restoration_mask_roles(
                clothing,
                protected,
                foreground,
                "thick",
                erosion_ratio=ratio,
            )
    finally:
        clothing.close()
        protected.close()
        foreground.close()
