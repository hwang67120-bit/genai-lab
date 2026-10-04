"""모델 입력 이미지가 단계 사이에서 바뀌지 않았는지 픽셀 해시로 확인한다."""

from hashlib import sha256

from PIL import Image


def calculate_image_pixel_sha256(image: Image.Image, mode: str) -> str:
    """이미지 모드·크기·픽셀 바이트를 포함한 SHA-256을 반환한다."""
    normalized_image = image.convert(mode)
    try:
        digest = sha256()
        digest.update(mode.encode("ascii"))
        digest.update(
            f"{normalized_image.width}x{normalized_image.height}".encode("ascii")
        )
        digest.update(normalized_image.tobytes())
        return digest.hexdigest()
    finally:
        normalized_image.close()
