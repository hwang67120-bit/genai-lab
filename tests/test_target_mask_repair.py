import numpy as np
from PIL import Image
import pytest

from genai_lab.target_mask_repair import (
    create_region_mask,
    merge_approved_mask_repair,
    propose_local_mask_repair,
    repair_sam_mask_automatically,
)


def create_masks():
    clothing = Image.new("L", (16, 16), 0)
    clothing.paste(255, (2, 2, 14, 14))
    clothing.putpixel((7, 7), 0)
    region = Image.new("L", (16, 16), 255)
    protection = Image.new("L", (16, 16), 0)
    return clothing, region, protection


def test_small_gap_is_only_proposed_until_explicit_merge():
    clothing, region, protection = create_masks()
    candidate = propose_local_mask_repair(clothing, region, protection)
    try:
        assert clothing.getpixel((7, 7)) == 0
        assert candidate.mask.getpixel((7, 7)) == 255
        assert candidate.candidate_pixel_count == 1
        merged = merge_approved_mask_repair(
            clothing, candidate.mask, protection,
        )
        try:
            assert merged.getpixel((7, 7)) == 255
        finally:
            merged.close()
    finally:
        candidate.close()
        clothing.close()
        region.close()
        protection.close()


def test_gap_outside_selected_region_is_not_proposed():
    clothing, region, protection = create_masks()
    region.paste(0, (0, 0, 16, 16))
    region.paste(255, (2, 2, 6, 6))
    candidate = propose_local_mask_repair(clothing, region, protection)
    try:
        assert candidate.candidate_pixel_count == 0
        assert candidate.mask.getpixel((7, 7)) == 0
    finally:
        candidate.close()
        clothing.close()
        region.close()
        protection.close()


def test_closing_never_expands_the_outer_garment_boundary():
    clothing, region, protection = create_masks()
    candidate = propose_local_mask_repair(clothing, region, protection)
    try:
        original = np.asarray(clothing) >= 128
        proposal = np.asarray(candidate.mask) >= 128
        exterior = np.zeros(original.shape, dtype=bool)
        exterior[:2, :] = True
        exterior[-2:, :] = True
        exterior[:, :2] = True
        exterior[:, -2:] = True
        assert not np.any(proposal & exterior)
    finally:
        candidate.close()
        clothing.close()
        region.close()
        protection.close()


def test_protected_gap_is_not_silently_added():
    clothing, region, protection = create_masks()
    protection.putpixel((7, 7), 255)
    candidate = propose_local_mask_repair(clothing, region, protection)
    try:
        assert candidate.candidate_pixel_count == 0
    finally:
        candidate.close()
        clothing.close()
        region.close()
        protection.close()


def test_component_area_limit_filters_candidate():
    clothing, region, protection = create_masks()
    clothing.putpixel((7, 8), 0)
    candidate = propose_local_mask_repair(
        clothing, region, protection,
        maximum_component_area_pixels=1,
    )
    try:
        assert candidate.candidate_pixel_count == 0
    finally:
        candidate.close()
        clothing.close()
        region.close()
        protection.close()


def test_merge_rejects_protection_conflict():
    clothing, region, protection = create_masks()
    repair = Image.new("L", (16, 16), 0)
    repair.putpixel((7, 7), 255)
    protection.putpixel((7, 7), 255)
    try:
        with pytest.raises(ValueError, match="1px"):
            merge_approved_mask_repair(clothing, repair, protection)
    finally:
        repair.close()
        clothing.close()
        region.close()
        protection.close()


def test_region_boxes_use_image_coordinates_and_clip_edges():
    region = create_region_mask(
        (10, 8),
        ((2, 3, 5, 6), (-2, -3, 2, 2)),
    )
    try:
        array = np.asarray(region)
        assert int(np.count_nonzero(array)) == 13
        assert region.getpixel((4, 5)) == 255
        assert region.getpixel((5, 5)) == 0
    finally:
        region.close()


def test_automatic_repair_closes_supported_hole_without_hard_dilation():
    clothing = Image.new("L", (20, 20), 0)
    clothing.paste(255, (4, 4, 16, 16))
    clothing.putpixel((10, 10), 0)
    hint = Image.new("L", (20, 20), 0)
    hint.paste(255, (3, 3, 17, 17))
    protection = Image.new("L", (20, 20), 0)
    foreground = Image.new("L", (20, 20), 255)
    result = repair_sam_mask_automatically(
        clothing, hint, protection, foreground,
    )
    try:
        assert result.added_mask.getpixel((10, 10)) == 255
        assert result.inpaint_mask.getpixel((10, 10)) == 255
        assert result.inpaint_mask.getpixel((3, 10)) == 0
        assert result.composite_mask.getpixel((3, 10)) in range(1, 255)
        assert result.added_pixel_count == 1
        assert result.hard_protected_overlap_pixel_count == 0
        assert result.soft_protected_overlap_pixel_count == 0
    finally:
        result.close()
        clothing.close()
        hint.close()
        protection.close()
        foreground.close()


def test_automatic_repair_never_ors_whole_independent_hint():
    clothing = Image.new("L", (20, 20), 0)
    clothing.paste(255, (2, 2, 8, 8))
    hint = Image.new("L", (20, 20), 0)
    hint.paste(255, (12, 12, 18, 18))
    protection = Image.new("L", (20, 20), 0)
    foreground = Image.new("L", (20, 20), 255)
    result = repair_sam_mask_automatically(
        clothing, hint, protection, foreground,
    )
    try:
        assert result.inpaint_mask.getpixel((14, 14)) == 0
        assert result.repaired_pixel_count == 36
    finally:
        result.close()
        clothing.close()
        hint.close()
        protection.close()
        foreground.close()


def test_hard_and_soft_masks_reapply_protection_and_foreground():
    clothing = Image.new("L", (20, 20), 0)
    clothing.paste(255, (2, 2, 18, 18))
    hint = Image.new("L", (20, 20), 255)
    protection = Image.new("L", (20, 20), 0)
    protection.paste(255, (9, 0, 11, 20))
    foreground = Image.new("L", (20, 20), 0)
    foreground.paste(255, (1, 1, 19, 19))
    result = repair_sam_mask_automatically(
        clothing, hint, protection, foreground,
    )
    try:
        hard = np.asarray(result.inpaint_mask) > 0
        soft = np.asarray(result.composite_mask) > 0
        protected = np.asarray(protection) > 0
        outside = np.asarray(foreground) == 0
        assert not np.any(hard & protected)
        assert not np.any(soft & protected)
        assert not np.any(hard & outside)
        assert not np.any(soft & outside)
    finally:
        result.close()
        clothing.close()
        hint.close()
        protection.close()
        foreground.close()
