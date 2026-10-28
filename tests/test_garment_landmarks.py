import numpy as np
import pytest
from PIL import Image

from genai_lab.garment_landmarks import (
    LANDMARK_ROLES,
    GarmentMaskLandmarkError,
    GarmentMaskLandmarkSettings,
    extract_garment_mask_landmarks,
)


def mask_from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), mode="L")


def test_extracts_six_ordered_points_from_single_rectangle() -> None:
    pixels = np.zeros((40, 50), dtype=np.uint8)
    pixels[5:36, 10:31] = 255
    mask = mask_from_array(pixels)
    result = extract_garment_mask_landmarks(mask)

    component = result.components[0]
    assert component.roles == LANDMARK_ROLES
    assert component.bbox_xywh == (10, 5, 21, 31)
    assert component.sampled_rows_y == (8, 20, 32)
    assert np.array_equal(
        component.points_xy,
        np.array(
            [[10, 8], [30, 8], [10, 20], [30, 20], [10, 32], [30, 32]],
            dtype=np.float32,
        ),
    )
    assert component.geometry_coverage_score == pytest.approx(1.0)
    assert np.array_equal(result.single_component_tps_points(), component.points_xy)
    mask.close()


def test_soft_alpha_threshold_is_used_only_for_geometry() -> None:
    pixels = np.zeros((20, 20), dtype=np.uint8)
    pixels[2:18, 2:18] = 127
    pixels[3:17, 3:17] = 128
    mask = mask_from_array(pixels)
    result = extract_garment_mask_landmarks(mask)

    assert result.source_foreground_pixel_count == 14 * 14
    assert result.components[0].bbox_xywh == (3, 3, 14, 14)
    mask.close()


def test_small_noise_component_is_discarded_and_counted() -> None:
    pixels = np.zeros((40, 50), dtype=np.uint8)
    pixels[5:36, 10:31] = 255
    pixels[1, 1] = 255
    mask = mask_from_array(pixels)
    result = extract_garment_mask_landmarks(mask)

    assert result.source_component_count == 2
    assert result.retained_component_count == 1
    assert result.discarded_component_count == 1
    assert result.discarded_noise_pixel_count == 1
    mask.close()


def test_multiple_valid_components_remain_separate_and_block_single_tps() -> None:
    pixels = np.zeros((50, 60), dtype=np.uint8)
    pixels[3:20, 3:20] = 255
    pixels[25:45, 35:55] = 255
    mask = mask_from_array(pixels)
    result = extract_garment_mask_landmarks(mask)

    assert result.retained_component_count == 2
    assert [component.bbox_xywh for component in result.components] == [
        (3, 3, 17, 17),
        (35, 25, 20, 20),
    ]
    with pytest.raises(GarmentMaskLandmarkError, match="정확히 1개"):
        result.single_component_tps_points()
    mask.close()


def test_widest_row_is_selected_inside_search_band() -> None:
    pixels = np.zeros((30, 30), dtype=np.uint8)
    pixels[4:26, 8:22] = 255
    pixels[5, 5:25] = 255
    mask = mask_from_array(pixels)
    result = extract_garment_mask_landmarks(
        mask,
        GarmentMaskLandmarkSettings(search_band_ratio=0.10),
    )

    assert result.components[0].sampled_rows_y[0] == 5
    assert result.components[0].points_xy[0].tolist() == [5.0, 5.0]
    assert result.components[0].points_xy[1].tolist() == [24.0, 5.0]
    mask.close()


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (GarmentMaskLandmarkSettings(alpha_threshold=0), "1~255"),
        (
            GarmentMaskLandmarkSettings(vertical_ratios=(0.5, 0.1, 0.9)),
            "오름차순",
        ),
        (GarmentMaskLandmarkSettings(search_band_ratio=0.0), "0 초과"),
        (GarmentMaskLandmarkSettings(minimum_component_pixels=0), "1px 이상"),
    ],
)
def test_rejects_invalid_settings(
    settings: GarmentMaskLandmarkSettings,
    message: str,
) -> None:
    mask = Image.new("L", (20, 20), 255)
    with pytest.raises(GarmentMaskLandmarkError, match=message):
        extract_garment_mask_landmarks(mask, settings)
    mask.close()


def test_rejects_empty_mask() -> None:
    mask = Image.new("L", (20, 20), 0)
    with pytest.raises(GarmentMaskLandmarkError, match="0개"):
        extract_garment_mask_landmarks(mask)
    mask.close()


def test_rejects_component_too_narrow_for_left_right_points() -> None:
    pixels = np.zeros((30, 30), dtype=np.uint8)
    pixels[2:28, 10] = 255
    mask = mask_from_array(pixels)
    with pytest.raises(GarmentMaskLandmarkError, match="너무 작습니다"):
        extract_garment_mask_landmarks(mask)
    mask.close()
