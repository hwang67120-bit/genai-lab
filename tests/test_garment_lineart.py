from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from genai_lab.garment_lineart import (
    GarmentLineartError,
    GarmentLineartSettings,
    approve_garment_lineart_review,
    create_garment_lineart_review,
)
from genai_lab.garment_warp_review import GarmentWarpApprovedInput


def _approved_warp(size: tuple[int, int] = (48, 48)) -> GarmentWarpApprovedInput:
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[8:40, 10:38, :3] = (20, 80, 180)
    rgba[8:40, 10:38, 3] = 255
    rgba[22:25, 12:36, :3] = (240, 220, 40)
    alpha = rgba[:, :, 3]
    return GarmentWarpApprovedInput(
        warped_rgba=Image.fromarray(rgba),
        alpha_mask=Image.fromarray(alpha),
        component_count=1,
        component_hard_overlap_pixels=0,
        outside_soft_alpha_pixels_before_protection=0,
        outside_hard_alpha_pixels_before_protection=0,
        protected_outside_alpha_pixels=0,
        source_scale_xy=(1.0, 1.0),
        source_padding_ltrb=(0, 0, 0, 0),
    )


def test_lineart_separates_boundary_and_internal_details() -> None:
    approved = _approved_warp()
    mask = Image.new("L", (48, 48), 255)
    review = create_garment_lineart_review(approved, mask)
    try:
        assert review.visible_alpha_pixels == 32 * 28
        assert review.raw_outer_boundary_pixels > 0
        assert review.raw_internal_detail_pixels > 0
        assert review.total_edge_pixels > 0
        assert review.alpha_consistency_mismatch_pixels == 0
        assert review.protected_edge_pixels_outside_approved_mask == 0
        assert review.automatic_save_count == 0
        assert review.control_image.mode == "RGB"
    finally:
        review.close()
        mask.close()
        approved.close()


def test_transparent_rgb_is_composited_to_white() -> None:
    approved = _approved_warp()
    mask = Image.new("L", (48, 48), 255)
    review = create_garment_lineart_review(approved, mask)
    try:
        assert review.white_background_garment_preview.getpixel((0, 0)) == (
            255,
            255,
            255,
        )
        assert review.white_background_garment_preview.getpixel((15, 15)) == (
            20,
            80,
            180,
        )
    finally:
        review.close()
        mask.close()
        approved.close()


def test_approved_mask_removes_raw_outside_edges() -> None:
    approved = _approved_warp()
    mask_array = np.zeros((48, 48), dtype=np.uint8)
    mask_array[8:40, 10:38] = 255
    mask = Image.fromarray(mask_array)
    review = create_garment_lineart_review(approved, mask)
    try:
        assert review.raw_edge_pixels_outside_approved_mask > 0
        assert review.protected_edge_pixels_outside_approved_mask == 0
        assert review.total_edge_pixels < review.raw_total_edge_pixels
    finally:
        review.close()
        mask.close()
        approved.close()


def test_alpha_mismatch_is_blocked_before_review() -> None:
    approved = _approved_warp()
    approved.alpha_mask.putpixel((10, 8), 0)
    mask = Image.new("L", (48, 48), 255)
    try:
        with pytest.raises(GarmentLineartError, match="1px"):
            create_garment_lineart_review(approved, mask)
    finally:
        mask.close()
        approved.close()


def test_canvas_size_mismatch_is_blocked() -> None:
    approved = _approved_warp()
    mask = Image.new("L", (47, 48), 255)
    try:
        with pytest.raises(GarmentLineartError, match="크기가 다릅니다"):
            create_garment_lineart_review(approved, mask)
    finally:
        mask.close()
        approved.close()


def test_empty_alpha_is_blocked() -> None:
    approved = _approved_warp()
    empty_rgba = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    empty_alpha = Image.new("L", (48, 48), 0)
    empty = replace(
        approved,
        warped_rgba=empty_rgba,
        alpha_mask=empty_alpha,
    )
    mask = Image.new("L", (48, 48), 255)
    try:
        with pytest.raises(GarmentLineartError, match="0개"):
            create_garment_lineart_review(empty, mask)
    finally:
        mask.close()
        empty.close()
        approved.close()


@pytest.mark.parametrize(
    "settings",
    [
        GarmentLineartSettings(canny_lower_threshold=150, canny_upper_threshold=50),
        GarmentLineartSettings(gaussian_kernel_size=2),
        GarmentLineartSettings(outer_boundary_radius=0),
        GarmentLineartSettings(internal_erosion_radius=-1),
    ],
)
def test_invalid_numeric_settings_are_blocked(settings: GarmentLineartSettings) -> None:
    approved = _approved_warp()
    mask = Image.new("L", (48, 48), 255)
    try:
        with pytest.raises(GarmentLineartError):
            create_garment_lineart_review(approved, mask, settings)
    finally:
        mask.close()
        approved.close()


def test_explicit_approval_returns_owned_copies_and_enforces_minimum() -> None:
    approved = _approved_warp()
    mask = Image.new("L", (48, 48), 255)
    review = create_garment_lineart_review(approved, mask)
    accepted = approve_garment_lineart_review(review)
    try:
        assert accepted.control_image is not review.control_image
        assert accepted.edge_mask is not review.combined_edge_mask
        assert accepted.total_edge_pixels == review.total_edge_pixels
        blocked = replace(
            review,
            settings=replace(
                review.settings,
                minimum_edge_pixels=review.total_edge_pixels + 1,
            ),
        )
        with pytest.raises(GarmentLineartError, match="승인 최소값"):
            approve_garment_lineart_review(blocked)
    finally:
        accepted.close()
        review.close()
        mask.close()
        approved.close()
