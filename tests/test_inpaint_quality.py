from PIL import Image
import pytest

from genai_lab.inpaint_quality import inspect_neutral_residual


def test_only_initially_neutral_approved_pixels_are_evaluated():
    with Image.new("RGB", (4, 4), (127, 127, 127)) as initial, Image.new("RGB", (4, 4), (127, 127, 127)) as output, Image.new("L", (4, 4), 0) as mask:
        mask.paste(255, (0, 0, 2, 2))
        output.putpixel((0, 0), (20, 80, 180))
        initial.putpixel((1, 1), (255, 255, 255))
        result = inspect_neutral_residual(initial, output, mask)
        try:
            assert result.evaluated_pixel_count == 3
            assert result.suspected_pixel_count == 2
            assert result.suspected_percent == pytest.approx(200 / 3)
            assert result.mask.getpixel((3, 3)) == 0
        finally:
            result.close()


def test_no_neutral_input_is_not_evaluable():
    with Image.new("RGB", (4, 4), "white") as image, Image.new("L", (4, 4), 255) as mask:
        result = inspect_neutral_residual(image, image, mask)
        try:
            assert result.suspected_percent is None
            assert result.evaluated_pixel_count == 0
        finally:
            result.close()


def test_custom_neutral_color_and_signed_difference():
    with Image.new("RGB", (4, 4), (250, 250, 250)) as initial, Image.new("RGB", (4, 4), (0, 0, 0)) as output, Image.new("L", (4, 4), 255) as mask:
        result = inspect_neutral_residual(initial, output, mask, (250, 250, 250), 8)
        try:
            assert result.evaluated_pixel_count == 16
            assert result.suspected_pixel_count == 0
        finally:
            result.close()
