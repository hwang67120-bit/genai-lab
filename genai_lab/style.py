"""참조 그림을 읽어 생성 모델에 전달할 형태로 준비한다."""

from pathlib import Path
from typing import Any

from PIL import Image


def load_reference_image(config: dict[str, Any], project_root: Path):
    """참조 기능이 꺼져 있으면 None, 켜져 있으면 전체가 보이는 이미지를 반환한다."""
    if not config["style"]["enabled"]:
        return None

    path = Path(config["style"]["reference_image"])
    if not path.is_absolute():
        path = project_root / path
    with Image.open(path) as image:
        if "A" in image.getbands():
            rgba_image = image.convert("RGBA")
            white_background = Image.new("RGBA", rgba_image.size, "white")
            white_background.alpha_composite(rgba_image)
            reference_image = white_background.convert("RGB")
        else:
            reference_image = image.convert("RGB")

    return prepare_ip_adapter_reference_image(reference_image)


def prepare_ip_adapter_reference_image(
    reference_image: Image.Image,
) -> Image.Image:
    """참조 이미지 전체를 정사각형 여백 안에 넣어 IP-Adapter에 전달한다."""
    # IP-Adapter의 224x224 중앙 자르기 전에 원본 전체를 정사각형 안에 넣는다.
    square_size = max(reference_image.size)
    square_reference_image = Image.new(
        "RGB",
        (square_size, square_size),
        "white",
    )
    paste_x = (square_size - reference_image.width) // 2
    paste_y = (square_size - reference_image.height) // 2
    square_reference_image.paste(reference_image, (paste_x, paste_y))
    return square_reference_image


def prepare_original_image_canvas(
    reference_image: Image.Image,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """원본 전체 비율을 유지해 목표 크기의 흰색 시작 화면에 배치한다.

    반환값:
        Image-to-Image 모델이 첫 화면으로 사용할 RGB 이미지.
    """
    rgb_reference_image = reference_image.convert("RGB")
    resize_ratio = min(
        target_width / rgb_reference_image.width,
        target_height / rgb_reference_image.height,
    )
    resized_width = max(1, round(rgb_reference_image.width * resize_ratio))
    resized_height = max(1, round(rgb_reference_image.height * resize_ratio))
    resized_reference_image = rgb_reference_image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    original_image_canvas = Image.new(
        "RGB",
        (target_width, target_height),
        "white",
    )
    paste_x = (target_width - resized_width) // 2
    paste_y = (target_height - resized_height) // 2
    original_image_canvas.paste(resized_reference_image, (paste_x, paste_y))
    return original_image_canvas

