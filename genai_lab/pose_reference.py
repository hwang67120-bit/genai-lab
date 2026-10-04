"""자세 참조 이미지의 규칙 검사와 사용자 승인 데이터를 담당한다."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class PoseReferenceSettings:
    """자세 참조 입력에 적용할 결정론적 크기 제한."""

    minimum_side_pixels: int = 64
    maximum_pixel_count: int = 40_000_000


@dataclass(frozen=True)
class PoseReferenceReviewCandidate:
    """파일 검사를 통과했지만 사용자가 아직 승인하지 않은 자세 후보."""

    source_path: Path
    image: Image.Image
    image_format: str
    width: int
    height: int
    pixel_count: int
    aspect_ratio: float
    file_size_bytes: int

    def close(self) -> None:
        """GUI 검토가 끝난 자세 후보 이미지를 메모리에서 해제한다."""
        self.image.close()


@dataclass(frozen=True)
class PoseReferenceApprovedInput:
    """사용자가 다음 관절 추출 단계에 사용하도록 승인한 자세 입력."""

    source_path: Path
    image: Image.Image
    image_format: str
    width: int
    height: int
    pixel_count: int
    aspect_ratio: float
    file_size_bytes: int

    def close(self) -> None:
        """승인 자세 이미지의 메모리 복사본을 해제한다."""
        self.image.close()


class PoseReferenceValidationError(ValueError):
    """자세 참조 파일을 안전하게 읽거나 검증하지 못한 오류."""


def load_pose_reference_candidate(
    image_path: Path,
    settings: PoseReferenceSettings = PoseReferenceSettings(),
) -> PoseReferenceReviewCandidate:
    """자세 참조를 읽고 형식·크기·비율 수치를 반환한다.

    반환값:
        파일 저장 없이 메모리에 보관하는 사용자 승인 전 RGB 이미지.

    오류:
        파일 누락, PNG·JPEG 이외 형식, 64px 미만 변 또는 4천만 픽셀
        초과 입력을 한글 오류로 거절한다.
    """
    if not image_path.is_file():
        raise PoseReferenceValidationError(
            f"자세 참조 이미지가 없습니다: {image_path}"
        )
    if settings.minimum_side_pixels < 1:
        raise PoseReferenceValidationError(
            "자세 참조 최소 변 길이는 1픽셀 이상이어야 합니다."
        )
    if settings.maximum_pixel_count < 1:
        raise PoseReferenceValidationError(
            "자세 참조 최대 픽셀 수는 1 이상이어야 합니다."
        )

    try:
        file_size_bytes = image_path.stat().st_size
        with Image.open(image_path) as opened_image:
            image_format = str(opened_image.format or "").upper()
            if image_format not in {"PNG", "JPEG"}:
                raise PoseReferenceValidationError(
                    "자세 참조는 PNG 또는 JPEG 파일만 사용할 수 있습니다: "
                    f"감지 형식={image_format or '확인 불가'}"
                )
            opened_image.load()
            oriented_image = ImageOps.exif_transpose(opened_image)
            try:
                width, height = oriented_image.size
                pixel_count = width * height
                if min(width, height) < settings.minimum_side_pixels:
                    raise PoseReferenceValidationError(
                        "자세 참조 이미지가 너무 작습니다: "
                        f"입력={width}x{height}, "
                        f"최소 변={settings.minimum_side_pixels}px"
                    )
                if pixel_count > settings.maximum_pixel_count:
                    raise PoseReferenceValidationError(
                        "자세 참조 이미지 픽셀 수가 제한을 초과했습니다: "
                        f"입력={pixel_count:,}px, "
                        f"상한={settings.maximum_pixel_count:,}px"
                    )
                rgb_image = oriented_image.convert("RGB")
            finally:
                if oriented_image is not opened_image:
                    oriented_image.close()
    except PoseReferenceValidationError:
        raise
    except (OSError, UnidentifiedImageError) as error:
        raise PoseReferenceValidationError(
            f"자세 참조 이미지를 읽을 수 없습니다: {image_path}"
        ) from error

    return PoseReferenceReviewCandidate(
        source_path=image_path,
        image=rgb_image,
        image_format=image_format,
        width=width,
        height=height,
        pixel_count=pixel_count,
        aspect_ratio=width / height,
        file_size_bytes=file_size_bytes,
    )


def approve_pose_reference_candidate(
    review_candidate: PoseReferenceReviewCandidate,
) -> PoseReferenceApprovedInput:
    """사용자가 확인한 자세 후보의 독립된 메모리 복사본을 반환한다."""
    return PoseReferenceApprovedInput(
        source_path=review_candidate.source_path,
        image=review_candidate.image.copy(),
        image_format=review_candidate.image_format,
        width=review_candidate.width,
        height=review_candidate.height,
        pixel_count=review_candidate.pixel_count,
        aspect_ratio=review_candidate.aspect_ratio,
        file_size_bytes=review_candidate.file_size_bytes,
    )
