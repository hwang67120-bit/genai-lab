from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from genai_lab.clothing import ClothingCategory
from genai_lab.garment_component_matching import (
    GarmentComponentMatchingError,
    GarmentComponentMatchingSettings,
    GarmentTargetSlot,
    propose_garment_component_matches,
)
from genai_lab.garment_landmarks import extract_garment_mask_landmarks


def extract_rectangles(
    rectangles: tuple[tuple[int, int, int, int], ...],
    size: tuple[int, int] = (100, 200),
):
    pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
    for left, top, right, bottom in rectangles:
        pixels[top:bottom, left:right] = 255
    mask = Image.fromarray(pixels, mode="L")
    try:
        return extract_garment_mask_landmarks(mask)
    finally:
        mask.close()


def test_top_category_maps_all_components_to_upper_body_and_marks_shared_slot() -> None:
    source = extract_rectangles(((5, 10, 35, 60), (60, 10, 90, 60)))
    result = propose_garment_component_matches(source, ClothingCategory.TOP)

    assert [proposal.target_slot for proposal in result.proposals] == [
        GarmentTargetSlot.UPPER_BODY,
        GarmentTargetSlot.UPPER_BODY,
    ]
    assert result.shared_target_slot_component_count == 2
    assert result.ambiguous_component_count == 2
    assert all(
        "shared_target_slot" in proposal.review_reasons
        for proposal in result.proposals
    )
    assert result.requires_user_approval is True
    assert result.automatic_warp_allowed is False


def test_bottom_category_maps_component_to_lower_body() -> None:
    source = extract_rectangles(((20, 70, 80, 160),))
    result = propose_garment_component_matches(source, ClothingCategory.BOTTOM)

    proposal = result.proposals[0]
    assert proposal.target_slot == GarmentTargetSlot.LOWER_BODY
    assert proposal.assignment_basis == "user_category_bottom"
    assert proposal.rule_fit_score == 1.0
    assert proposal.ambiguous is False


def test_dress_category_maps_component_to_full_body() -> None:
    source = extract_rectangles(((20, 10, 80, 180),))
    result = propose_garment_component_matches(source, ClothingCategory.DRESS)

    assert result.proposals[0].target_slot == GarmentTargetSlot.FULL_BODY
    assert result.proposals[0].assignment_basis == "user_category_dress"


def test_full_outfit_separates_upper_lower_and_left_right_footwear() -> None:
    source = extract_rectangles(
        (
            (10, 0, 90, 50),
            (20, 70, 80, 130),
            (10, 170, 30, 195),
            (70, 170, 90, 195),
        )
    )
    result = propose_garment_component_matches(
        source,
        ClothingCategory.FULL_BODY_OUTFIT,
    )

    assert [proposal.target_slot for proposal in result.proposals] == [
        GarmentTargetSlot.UPPER_BODY,
        GarmentTargetSlot.LOWER_BODY,
        GarmentTargetSlot.IMAGE_LEFT_FOOT,
        GarmentTargetSlot.IMAGE_RIGHT_FOOT,
    ]
    assert result.ambiguous_component_count == 0
    assert result.shared_target_slot_component_count == 0
    assert all(
        0.0 <= proposal.rule_fit_score <= 1.0
        for proposal in result.proposals
    )


def test_tall_full_outfit_component_maps_to_full_body() -> None:
    source = extract_rectangles(((20, 10, 80, 190),))
    result = propose_garment_component_matches(
        source,
        ClothingCategory.FULL_BODY_OUTFIT,
    )

    proposal = result.proposals[0]
    assert proposal.target_slot == GarmentTargetSlot.FULL_BODY
    assert proposal.assignment_basis == "geometry_vertical_span"
    assert proposal.normalized_height_ratio == pytest.approx(1.0)


def test_centered_footwear_component_remains_unresolved() -> None:
    source = extract_rectangles(
        (
            (10, 0, 90, 50),
            (40, 170, 61, 195),
        )
    )
    result = propose_garment_component_matches(
        source,
        ClothingCategory.FULL_BODY_OUTFIT,
    )

    footwear = result.proposals[1]
    assert footwear.target_slot == GarmentTargetSlot.FOOTWEAR_PAIR
    assert footwear.ambiguous is True
    assert "footwear_left_right_unresolved" in footwear.review_reasons


@pytest.mark.parametrize(
    "category",
    [ClothingCategory.GLOVES, ClothingCategory.SHOES],
)
def test_unsupported_accessory_category_is_rejected(
    category: ClothingCategory,
) -> None:
    source = extract_rectangles(((20, 10, 80, 180),))
    with pytest.raises(GarmentComponentMatchingError, match="지원하지 않는"):
        propose_garment_component_matches(source, category)


def test_inconsistent_retained_component_count_is_rejected() -> None:
    source = extract_rectangles(((20, 10, 80, 180),))
    invalid_source = replace(source, retained_component_count=2)
    with pytest.raises(GarmentComponentMatchingError, match="자료 수가 다릅니다"):
        propose_garment_component_matches(
            invalid_source,
            ClothingCategory.TOP,
        )


def test_component_bbox_outside_canvas_is_rejected() -> None:
    source = extract_rectangles(((20, 10, 80, 180),))
    invalid_component = replace(
        source.components[0],
        bbox_xywh=(20, 10, 100, 170),
    )
    invalid_source = replace(source, components=(invalid_component,))
    with pytest.raises(GarmentComponentMatchingError, match="캔버스 밖"):
        propose_garment_component_matches(
            invalid_source,
            ClothingCategory.TOP,
        )


def test_invalid_ratio_settings_are_rejected() -> None:
    source = extract_rectangles(((20, 10, 80, 180),))
    settings = GarmentComponentMatchingSettings(
        upper_body_end_ratio=0.80,
        footwear_start_ratio=0.70,
    )
    with pytest.raises(GarmentComponentMatchingError, match="작아야"):
        propose_garment_component_matches(
            source,
            ClothingCategory.FULL_BODY_OUTFIT,
            settings,
        )
