from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from genai_lab.character_target_landmarks import (
    CharacterTargetLandmarkResult,
)
from genai_lab.clothing import ClothingCategory
from genai_lab.garment_component_matching import (
    propose_garment_component_matches,
)
from genai_lab.garment_landmarks import extract_garment_mask_landmarks
from genai_lab.garment_warp_review import (
    GarmentWarpReviewError,
    GarmentWarpReviewSettings,
    approve_garment_tps_warp_review,
    create_garment_tps_warp_review,
)


def create_source_garment(
    size: tuple[int, int],
    rectangles: tuple[tuple[int, int, int, int], ...],
) -> Image.Image:
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    colors = ((20, 90, 220), (220, 80, 40), (80, 190, 80), (180, 80, 190))
    for index, (left, top, right, bottom) in enumerate(rectangles):
        pixels[top:bottom, left:right, :3] = colors[index % len(colors)]
        pixels[top:bottom, left:right, 3] = 255
    return Image.fromarray(pixels, mode="RGBA")


def source_landmarks_from_rgba(garment: Image.Image):
    alpha = garment.getchannel("A")
    try:
        return extract_garment_mask_landmarks(alpha)
    finally:
        alpha.close()


def create_target_mask(
    size: tuple[int, int],
    box: tuple[int, int, int, int],
) -> Image.Image:
    pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
    left, top, right, bottom = box
    pixels[top:bottom, left:right] = 255
    return Image.fromarray(pixels, mode="L")


def create_target_landmarks(
    size: tuple[int, int],
    box: tuple[int, int, int, int],
) -> CharacterTargetLandmarkResult:
    left, top, right, bottom = box
    right -= 1
    bottom -= 1
    middle_y = round((top + bottom) / 2)
    mask_pixels = (right - left + 1) * (bottom - top + 1)
    return CharacterTargetLandmarkResult(
        points_xy=np.array(
            [
                [left, top],
                [right, top],
                [left, middle_y],
                [right, middle_y],
                [left, bottom],
                [right, bottom],
            ],
            dtype=np.float32,
        ),
        roles=(
            "upper_left",
            "upper_right",
            "middle_left",
            "middle_right",
            "lower_left",
            "lower_right",
        ),
        row_sources=("test_upper", "test_middle", "test_lower"),
        selected_rows_y=(top, middle_y, bottom),
        target_canvas_size=size,
        approved_mask_foreground_pixels=mask_pixels,
        approved_mask_bbox_xywh=(
            left,
            top,
            right - left + 1,
            bottom - top + 1,
        ),
        required_joint_names=("test",),
        minimum_used_joint_confidence=1.0,
        mean_used_joint_confidence=1.0,
        row_search_radius_pixels=8,
        horizontal_overlap_score=1.0,
    )


def test_single_dress_component_creates_review_and_approved_copy() -> None:
    size = (100, 100)
    garment = create_source_garment(size, ((20, 10, 80, 90),))
    source_landmarks = source_landmarks_from_rgba(garment)
    matches = propose_garment_component_matches(
        source_landmarks,
        ClothingCategory.DRESS,
    )
    target_box = (10, 5, 90, 95)
    target_mask = create_target_mask(size, target_box)
    base = Image.new("RGB", size, "white")
    review = create_garment_tps_warp_review(
        garment,
        base,
        target_mask,
        source_landmarks,
        create_target_landmarks(size, target_box),
        matches,
    )

    assert review.component_count == 1
    assert review.component_previews[0].source_points_xy.shape == (6, 2)
    assert review.component_previews[0].target_points_xy.shape == (6, 2)
    assert review.protected_outside_alpha_pixels == 0
    assert review.automatic_save_count == 0
    assert review.overlay_preview.mode == "RGB"
    approved = approve_garment_tps_warp_review(review)
    review.close()
    assert approved.warped_rgba.size == size
    assert approved.alpha_mask.getbbox() is not None
    assert approved.protected_outside_alpha_pixels == 0
    approved.close()
    garment.close()
    target_mask.close()
    base.close()


def test_source_canvas_is_fit_with_exact_scale_and_padding() -> None:
    source_size = (50, 100)
    target_size = (100, 100)
    garment = create_source_garment(source_size, ((10, 10, 40, 90),))
    source_landmarks = source_landmarks_from_rgba(garment)
    matches = propose_garment_component_matches(
        source_landmarks,
        ClothingCategory.DRESS,
    )
    target_box = (20, 5, 80, 95)
    target_mask = create_target_mask(target_size, target_box)
    base = Image.new("RGB", target_size, "white")
    review = create_garment_tps_warp_review(
        garment,
        base,
        target_mask,
        source_landmarks,
        create_target_landmarks(target_size, target_box),
        matches,
    )

    assert review.source_scale_xy == pytest.approx((1.0, 1.0))
    assert review.source_padding_ltrb == (25, 0, 25, 0)
    review.close()
    garment.close()
    target_mask.close()
    base.close()


def test_multiple_components_keep_separate_previews_and_record_shared_slot() -> None:
    size = (120, 120)
    garment = create_source_garment(
        size,
        ((10, 10, 50, 55), (70, 10, 110, 55)),
    )
    source_landmarks = source_landmarks_from_rgba(garment)
    matches = propose_garment_component_matches(
        source_landmarks,
        ClothingCategory.TOP,
    )
    target_box = (20, 10, 100, 110)
    target_mask = create_target_mask(size, target_box)
    base = Image.new("RGB", size, "white")
    review = create_garment_tps_warp_review(
        garment,
        base,
        target_mask,
        source_landmarks,
        create_target_landmarks(size, target_box),
        matches,
    )

    assert review.component_count == 2
    assert review.shared_target_slot_component_count == 2
    assert review.ambiguous_component_count == 2
    assert review.component_hard_overlap_pixels >= 0
    review.close()
    garment.close()
    target_mask.close()
    base.close()


def test_nearby_component_hard_pixels_are_excluded_and_measured() -> None:
    size = (100, 100)
    garment = create_source_garment(
        size,
        ((10, 10, 30, 60), (31, 10, 51, 60)),
    )
    source_landmarks = source_landmarks_from_rgba(garment)
    matches = propose_garment_component_matches(
        source_landmarks,
        ClothingCategory.TOP,
    )
    target_box = (10, 5, 90, 95)
    target_mask = create_target_mask(size, target_box)
    base = Image.new("RGB", size, "white")
    review = create_garment_tps_warp_review(
        garment,
        base,
        target_mask,
        source_landmarks,
        create_target_landmarks(size, target_box),
        matches,
    )

    assert review.source_foreign_hard_pixels_excluded > 0
    assert sum(
        preview.source_foreign_hard_pixels_excluded
        for preview in review.component_previews
    ) == review.source_foreign_hard_pixels_excluded
    review.close()
    garment.close()
    target_mask.close()
    base.close()


def test_protection_removes_every_outside_alpha_pixel() -> None:
    size = (100, 100)
    garment = create_source_garment(size, ((5, 5, 95, 95),))
    source_landmarks = source_landmarks_from_rgba(garment)
    matches = propose_garment_component_matches(
        source_landmarks,
        ClothingCategory.DRESS,
    )
    target_box = (30, 20, 70, 80)
    target_mask = create_target_mask(size, target_box)
    base = Image.new("RGB", size, "white")
    review = create_garment_tps_warp_review(
        garment,
        base,
        target_mask,
        source_landmarks,
        create_target_landmarks(size, target_box),
        matches,
    )

    protected_alpha = np.asarray(review.protected_alpha_mask)
    approved = np.asarray(target_mask) >= 128
    assert np.count_nonzero(protected_alpha[~approved]) == 0
    assert review.protected_outside_alpha_pixels == 0
    assert review.removed_outside_alpha_pixels == (
        review.outside_soft_alpha_pixels + review.outside_hard_alpha_pixels
    )
    review.close()
    garment.close()
    target_mask.close()
    base.close()


def test_mismatched_target_canvas_is_rejected() -> None:
    garment = create_source_garment((100, 100), ((20, 10, 80, 90),))
    source_landmarks = source_landmarks_from_rgba(garment)
    matches = propose_garment_component_matches(
        source_landmarks,
        ClothingCategory.DRESS,
    )
    target_mask = create_target_mask((100, 100), (10, 5, 90, 95))
    wrong_base = Image.new("RGB", (80, 100), "white")
    with pytest.raises(GarmentWarpReviewError, match="크기가 다릅니다"):
        create_garment_tps_warp_review(
            garment,
            wrong_base,
            target_mask,
            source_landmarks,
            create_target_landmarks((100, 100), (10, 5, 90, 95)),
            matches,
        )
    garment.close()
    target_mask.close()
    wrong_base.close()


def test_missing_one_to_one_component_proposal_is_rejected() -> None:
    garment = create_source_garment((100, 100), ((20, 10, 80, 90),))
    source_landmarks = source_landmarks_from_rgba(garment)
    matches = propose_garment_component_matches(
        source_landmarks,
        ClothingCategory.DRESS,
    )
    invalid_matches = replace(matches, proposals=(), proposal_count=0)
    target_mask = create_target_mask((100, 100), (10, 5, 90, 95))
    base = Image.new("RGB", (100, 100), "white")
    with pytest.raises(GarmentWarpReviewError, match="허용 범위"):
        create_garment_tps_warp_review(
            garment,
            base,
            target_mask,
            source_landmarks,
            create_target_landmarks((100, 100), (10, 5, 90, 95)),
            invalid_matches,
        )
    garment.close()
    target_mask.close()
    base.close()


def test_maximum_component_limit_blocks_before_tps() -> None:
    garment = create_source_garment(
        (120, 120),
        ((10, 10, 50, 55), (70, 10, 110, 55)),
    )
    source_landmarks = source_landmarks_from_rgba(garment)
    matches = propose_garment_component_matches(
        source_landmarks,
        ClothingCategory.TOP,
    )
    target_mask = create_target_mask((120, 120), (20, 10, 100, 110))
    base = Image.new("RGB", (120, 120), "white")
    with pytest.raises(GarmentWarpReviewError, match="최대=1개"):
        create_garment_tps_warp_review(
            garment,
            base,
            target_mask,
            source_landmarks,
            create_target_landmarks((120, 120), (20, 10, 100, 110)),
            matches,
            GarmentWarpReviewSettings(maximum_component_count=1),
        )
    garment.close()
    target_mask.close()
    base.close()
