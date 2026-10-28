import numpy as np
import pytest
from PIL import Image
from genai_lab.part_color_analysis import (
    analyze_part_colors, color_distribution_distance, save_color_evidence,
)


def test_multicolor_positions_ratios_and_background_exclusion(tmp_path):
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :2] = (0, 255, 0)
    pixels[:, 2:5] = (0, 0, 255)
    pixels[:, 5:] = (128, 0, 255)
    source = Image.fromarray(pixels)
    mask = Image.new("L", source.size)
    mask.paste(255, (2, 0, 8, 8))
    result = analyze_part_colors(source, mask, max_clusters=2)
    again = analyze_part_colors(source, mask, max_clusters=2)
    assert result.pattern == "unresolved"
    assert result.pixel_count == 48
    assert sorted(result.area_ratios) == [.5, .5]
    assert np.all(result.label_map[:, :2] == -1)
    assert result.label_map[0, 2] != result.label_map[0, 7]
    np.testing.assert_array_equal(result.label_map, again.label_map)
    assert all(rgb[1] < 5 for rgb in result.palette_rgb)
    save_color_evidence(result, tmp_path, "hair")
    assert (tmp_path / "hair_labels.npy").exists()
    assert (tmp_path / "hair_color_map.png").exists()
    source.close()
    mask.close()


def test_black_and_white_are_not_automatically_discarded():
    source = Image.new("RGB", (8, 8), "white")
    source.paste("black", (0, 0, 4, 8))
    mask = Image.new("L", source.size, 255)
    result = analyze_part_colors(source, mask)
    assert result.pixel_count == 64
    assert len(result.palette_rgb) == 2
    assert result.pattern == "unresolved"
    source.close()
    mask.close()


def test_empty_and_transparent_regions_stay_unresolved():
    source = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    mask = Image.new("L", source.size, 255)
    result = analyze_part_colors(source, mask)
    assert result.palette_rgb == [] and result.pixel_count == 0
    assert np.all(result.label_map == -1)
    source.close()
    mask.close()


@pytest.mark.parametrize("kwargs", [
    {"max_clusters": 0}, {"max_clusters": 9}, {"max_fit_pixels": 1},
    {"max_fit_pixels": 20001},
])
def test_invalid_budgets_rejected(kwargs):
    with Image.new("RGB", (8, 8)) as source, Image.new("L", (8, 8), 255) as mask:
        with pytest.raises(ValueError):
            analyze_part_colors(source, mask, **kwargs)


def test_color_analysis_is_cancellable():
    with Image.new("RGB", (8, 8)) as source, Image.new("L", (8, 8), 255) as mask:
        with pytest.raises(InterruptedError):
            analyze_part_colors(source, mask, cancelled=lambda: True)


def test_color_distribution_distance_is_symmetric_and_detects_change():
    mask = Image.new("L", (16, 16), 255)
    with Image.new("RGB", mask.size, "blue") as blue_image:
        blue = analyze_part_colors(blue_image, mask, max_clusters=1)
    with Image.new("RGB", mask.size, "red") as red_image:
        red = analyze_part_colors(red_image, mask, max_clusters=1)
    assert color_distribution_distance(blue, blue) == pytest.approx(0.0)
    assert color_distribution_distance(blue, red) == pytest.approx(
        color_distribution_distance(red, blue))
    assert color_distribution_distance(blue, red) > 18.0
    mask.close()


def test_color_distribution_distance_stays_unresolved_without_pixels():
    mask = Image.new("L", (8, 8), 0)
    with Image.new("RGB", mask.size, "blue") as image:
        empty = analyze_part_colors(image, mask)
    assert color_distribution_distance(empty, empty) is None
    assert color_distribution_distance(None, empty) is None
    mask.close()
