import numpy as np
import pytest
from PIL import Image, ImageDraw
from genai_lab.garment_reference_board import prepare_garment_board


def test_disconnected_garments_enlarged_without_dropping_details():
    source = Image.new('RGBA', (512, 512))
    draw = ImageDraw.Draw(source)
    draw.rectangle((220, 100, 290, 250), fill=(0, 0, 200, 255))
    draw.rectangle((230, 470, 260, 500), fill=(200, 0, 0, 255))
    draw.rectangle((295, 150, 297, 152), fill=(0, 200, 0, 255))
    before = source.tobytes()
    board, record = prepare_garment_board(source)
    pixels = np.asarray(board)
    assert len(record['regions']) == 2
    assert np.any(pixels[:, :, 1] < 100)
    assert np.count_nonzero(np.any(pixels < 245, axis=2)) > 4 * record['source_alpha_pixels']
    assert np.any((pixels[:, :, 1] > 150) & (pixels[:, :, 0] < 50) & (pixels[:, :, 2] < 50))
    assert source.tobytes() == before
    board.close()
    source.close()


def test_empty_alpha_rejected():
    with pytest.raises(ValueError, match='알파'):
        prepare_garment_board(Image.new('RGBA', (20, 20)))


def test_opaque_rgb_is_not_color_thresholded():
    board, record = prepare_garment_board(Image.new('RGB', (20, 40), 'white'))
    assert record['opaque_input'] is True
    assert record['regions'] == [(0, 0, 20, 40)]
    board.close()


def test_soft_alpha_preserved_in_compositing():
    image = Image.new('RGBA', (32, 32), (0, 0, 0, 128))
    board, record = prepare_garment_board(image)
    assert board.getpixel((256, 256)) == (127, 127, 127)
    board.close()

