"""참조 그림을 읽어 생성 모델에 전달할 형태로 준비한다."""

from pathlib import Path
from typing import Any


def load_reference_image(config: dict[str, Any], project_root: Path):
    """참조 기능이 꺼져 있으면 None, 켜져 있으면 전체가 보이는 이미지를 반환한다."""
    if not config["style"]["enabled"]:
        return None

    from PIL import Image

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

