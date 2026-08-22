"""참조 그림을 읽어 생성 모델에 전달할 형태로 준비한다."""

from pathlib import Path
from typing import Any


def load_reference_image(config: dict[str, Any], project_root: Path):
    """참조 기능이 꺼져 있으면 None, 켜져 있으면 RGB 이미지를 반환한다."""
    if not config["style"]["enabled"]:
        return None

    from PIL import Image

    path = Path(config["style"]["reference_image"])
    if not path.is_absolute():
        path = project_root / path
    with Image.open(path) as image:
        return image.convert("RGB")

