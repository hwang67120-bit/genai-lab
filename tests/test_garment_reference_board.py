import numpy as np
import pytest
from PIL import Image

from genai_lab.garment_reference_board import (
    GarmentReferenceBoardError,
    GarmentReferenceBoardSettings,
    create_garment_reference_board,
)


def _separated_garment() -> Image.Image:
    image = Image.new("RGBA", (120, 200), (0, 0, 0, 0))
    image.paste((10, 80, 170, 255), (10, 10, 100, 100))
    image.paste((25, 25, 25, 255), (70, 175, 105, 195))
    return image


def test_largest_component_gets_primary_board_area_and_secondary_is_visible():
    garment = _separated_garment()
    board = create_garment_reference_board(
        garment,
        GarmentReferenceBoardSettings(board_size=256, outer_padding=8),
    )
    try:
        assert board.image.mode == "RGB"
        assert board.image.size == (256, 256)
        assert board.source_component_count == 2
        assert board.retained_component_count == 2
        assert board.discarded_component_count == 0
        assert board.board_occupied_pixel_count > 0
        primary, secondary = board.components
        assert primary.foreground_pixel_count > secondary.foreground_pixel_count
        assert primary.board_bbox_xywh[1] < secondary.board_bbox_xywh[1]
        pixels = np.asarray(board.image)
        assert np.any(np.all(pixels == (10, 80, 170), axis=2))
        assert np.any(np.all(pixels == (25, 25, 25), axis=2))
    finally:
        board.close()
        garment.close()


def test_small_noise_is_discarded_without_changing_source_image():
    garment = _separated_garment()
    garment.putpixel((119, 199), (255, 0, 0, 255))
    before = garment.tobytes()
    board = create_garment_reference_board(
        garment,
        GarmentReferenceBoardSettings(
            board_size=256,
            outer_padding=8,
            minimum_component_pixels=4,
        ),
    )
    try:
        assert board.source_component_count == 3
        assert board.retained_component_count == 2
        assert board.discarded_component_count == 1
        assert garment.tobytes() == before
    finally:
        board.close()
        garment.close()


def test_rgb_without_approved_alpha_is_rejected():
    garment = Image.new("RGB", (32, 32), "blue")
    try:
        with pytest.raises(GarmentReferenceBoardError, match="알파"):
            create_garment_reference_board(garment)
    finally:
        garment.close()


@pytest.mark.parametrize(
    "settings",
    [
        GarmentReferenceBoardSettings(board_size=65),
        GarmentReferenceBoardSettings(maximum_components=0),
        GarmentReferenceBoardSettings(primary_height_ratio=0.9),
    ],
)
def test_invalid_settings_are_rejected(settings):
    garment = _separated_garment()
    try:
        with pytest.raises(GarmentReferenceBoardError):
            create_garment_reference_board(garment, settings)
    finally:
        garment.close()
