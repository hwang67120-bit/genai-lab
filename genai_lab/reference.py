"""참조 이미지의 화질 확인, 확대 복원과 사용자 승인 데이터를 담당한다."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat


REALESRGAN_ANIME_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "RealESRGAN_x4plus_anime_6B.pth"
)


class ReferenceImageQualityStatus(str, Enum):
    """규칙으로 확인한 참조 이미지의 사용 가능 상태."""

    ORIGINAL_USABLE = "original_usable"
    ENHANCEMENT_RECOMMENDED = "enhancement_recommended"
    DETAIL_UNVERIFIABLE = "detail_unverifiable"


@dataclass(frozen=True)
class ReferenceImageQualityReport:
    """해상도와 선명도를 규칙으로 검사한 결과."""

    status: ReferenceImageQualityStatus
    width: int
    height: int
    enlargement_ratio: float
    sharpness_score: float
    reason_ko: str


@dataclass(frozen=True)
class ReferenceImageEnhancementCandidate:
    """사용자가 아직 승인하지 않은 참조 이미지 보정 후보."""

    original_image: Image.Image
    enhanced_image: Image.Image
    quality_report: ReferenceImageQualityReport
    enhancement_model_id: str


@dataclass(frozen=True)
class ApprovedReferenceImage:
    """사용자가 생성 입력으로 사용하겠다고 확정한 참조 이미지."""

    image: Image.Image
    source_name: str
    enhancement_applied: bool
    enhancement_model_id: str | None
    quality_status: ReferenceImageQualityStatus

@dataclass(frozen=True)
class ReferenceImagePreparationResult:
    """화질 검사와 필요한 확대 복원을 마친 임시 결과."""

    original_image: Image.Image
    quality_report: ReferenceImageQualityReport
    enhancement_candidate: ReferenceImageEnhancementCandidate | None



class ReferenceImagePreparationError(RuntimeError):
    """참조 이미지 화질 확인이나 확대 복원을 완료하지 못한 오류."""


def inspect_reference_image_quality(
    reference_image: Image.Image,
    target_width: int,
    target_height: int,
    minimum_short_side: int,
    minimum_sharpness_score: float,
) -> ReferenceImageQualityReport:
    """참조 이미지의 확대 필요 비율과 선명도를 규칙으로 확인한다."""
    rgb_reference_image = reference_image.convert("RGB")
    width, height = rgb_reference_image.size
    enlargement_ratio = min(target_width / width, target_height / height)
    sharpness_score = calculate_reference_image_sharpness(rgb_reference_image)

    if min(width, height) < 256 or sharpness_score < minimum_sharpness_score / 4:
        status = ReferenceImageQualityStatus.DETAIL_UNVERIFIABLE
        reason_ko = (
            "원본의 작은 눈·손·의상선을 확인하기 어렵습니다. "
            "보정본도 원본에 없는 세부 묘사를 추측할 수 있습니다."
        )
    elif (
        min(width, height) < minimum_short_side
        or enlargement_ratio > 1.05
        or sharpness_score < minimum_sharpness_score
    ):
        status = ReferenceImageQualityStatus.ENHANCEMENT_RECOMMENDED
        reason_ko = (
            "생성 크기에 비해 참조 이미지가 작거나 흐립니다. "
            "애니 그림용 확대·복원 후 비교를 권장합니다."
        )
    else:
        status = ReferenceImageQualityStatus.ORIGINAL_USABLE
        reason_ko = "현재 생성 크기에서 원본을 직접 사용할 수 있습니다."

    return ReferenceImageQualityReport(
        status=status,
        width=width,
        height=height,
        enlargement_ratio=round(enlargement_ratio, 3),
        sharpness_score=round(sharpness_score, 3),
        reason_ko=reason_ko,
    )


def prepare_reference_image_for_review(
    reference_image: Image.Image,
    target_width: int,
    target_height: int,
    quality_config: dict[str, object],
) -> ReferenceImagePreparationResult:
    """화질을 검사하고 필요할 때만 사용자 검토용 보정 후보를 만든다."""
    quality_report = inspect_reference_image_quality(
        reference_image=reference_image,
        target_width=target_width,
        target_height=target_height,
        minimum_short_side=int(quality_config["minimum_short_side"]),
        minimum_sharpness_score=float(
            quality_config["minimum_sharpness_score"]
        ),
    )
    enhancement_candidate = None
    if quality_report.status is not ReferenceImageQualityStatus.ORIGINAL_USABLE:
        enhancement_candidate = enhance_low_quality_reference_image(
            reference_image=reference_image,
            quality_report=quality_report,
            model_path=Path(str(quality_config["model_path"])),
            tile_size=int(quality_config["tile_size"]),
            tile_overlap=int(quality_config["tile_overlap"]),
            maximum_long_side=int(quality_config["maximum_long_side"]),
        )
    return ReferenceImagePreparationResult(
        original_image=reference_image.convert("RGB").copy(),
        quality_report=quality_report,
        enhancement_candidate=enhancement_candidate,
    )


def calculate_reference_image_sharpness(reference_image: Image.Image) -> float:
    """이미지 가장자리 변화량으로 비교용 선명도 점수를 계산한다."""
    grayscale_image = reference_image.convert("L")
    white_background = Image.new("L", grayscale_image.size, 255)
    content_box = ImageChops.difference(
        grayscale_image,
        white_background,
    ).getbbox()
    if content_box is None:
        return 0.0
    grayscale_image = grayscale_image.crop(content_box)

    maximum_measure_side = 1024
    if max(grayscale_image.size) > maximum_measure_side:
        resize_ratio = maximum_measure_side / max(grayscale_image.size)
        grayscale_image = grayscale_image.resize(
            (
                max(1, round(grayscale_image.width * resize_ratio)),
                max(1, round(grayscale_image.height * resize_ratio)),
            ),
            Image.Resampling.LANCZOS,
        )
    edge_image = grayscale_image.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edge_image).var[0])


def enhance_low_quality_reference_image(
    reference_image: Image.Image,
    quality_report: ReferenceImageQualityReport,
    model_path: Path,
    tile_size: int,
    tile_overlap: int,
    maximum_long_side: int,
) -> ReferenceImageEnhancementCandidate:
    """Real-ESRGAN으로 보정 후보를 만들고 원본과 함께 반환한다.

    부수 효과:
        모델 파일이 없으면 공식 Real-ESRGAN 배포 주소에서 내려받는다.
    """
    try:
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError as error:
        raise ReferenceImagePreparationError(
            "참조 이미지 확대 도구가 없습니다. "
            "가상환경에서 'python -m pip install -r requirements.txt'를 실행하세요."
        ) from error

    download_reference_enhancement_model(model_path)
    try:
        enhancement_model = ModelLoader().load_from_file(str(model_path))
    except Exception as error:
        raise ReferenceImagePreparationError(
            f"참조 이미지 확대 모델을 읽지 못했습니다: {model_path}"
        ) from error
    if not isinstance(enhancement_model, ImageModelDescriptor):
        raise ReferenceImagePreparationError(
            "선택한 참조 이미지 확대 모델이 이미지 복원 모델이 아닙니다."
        )

    execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhancement_model = enhancement_model.to(execution_device).eval()
    try:
        enhanced_image = run_tiled_reference_enhancement(
            enhancement_model,
            reference_image.convert("RGB"),
            execution_device,
            tile_size,
            tile_overlap,
        )
    except torch.cuda.OutOfMemoryError as error:
        raise ReferenceImagePreparationError(
            "참조 이미지 확대 중 GPU 메모리가 부족했습니다. "
            "다른 GPU 프로그램을 닫고 다시 시도하세요."
        ) from error
    finally:
        enhancement_model.to("cpu")
        del enhancement_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if max(enhanced_image.size) > maximum_long_side:
        resize_ratio = maximum_long_side / max(enhanced_image.size)
        enhanced_image = enhanced_image.resize(
            (
                max(1, round(enhanced_image.width * resize_ratio)),
                max(1, round(enhanced_image.height * resize_ratio)),
            ),
            Image.Resampling.LANCZOS,
        )

    return ReferenceImageEnhancementCandidate(
        original_image=reference_image.convert("RGB").copy(),
        enhanced_image=enhanced_image,
        quality_report=quality_report,
        enhancement_model_id=model_path.stem,
    )


def download_reference_enhancement_model(model_path: Path) -> None:
    """공식 Real-ESRGAN 가중치를 로컬 모델 폴더에 준비한다."""
    if model_path.is_file():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".download")
    try:
        import torch

        torch.hub.download_url_to_file(
            REALESRGAN_ANIME_MODEL_URL,
            str(temporary_path),
            progress=True,
        )
        temporary_path.replace(model_path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise ReferenceImagePreparationError(
            "참조 이미지 확대 모델을 내려받지 못했습니다. "
            "인터넷 연결과 D:\\genai-cache의 남은 공간을 확인하세요."
        ) from error


def run_tiled_reference_enhancement(
    enhancement_model: object,
    reference_image: Image.Image,
    execution_device: object,
    tile_size: int,
    tile_overlap: int,
) -> Image.Image:
    """큰 참조 이미지를 작은 타일로 나눠 GPU 메모리 초과 없이 복원한다."""
    import numpy as np
    import torch

    model_scale = int(getattr(enhancement_model, "scale", 4))
    enhanced_canvas = Image.new(
        "RGB",
        (reference_image.width * model_scale, reference_image.height * model_scale),
    )

    for top in range(0, reference_image.height, tile_size):
        for left in range(0, reference_image.width, tile_size):
            core_right = min(reference_image.width, left + tile_size)
            core_bottom = min(reference_image.height, top + tile_size)
            expanded_left = max(0, left - tile_overlap)
            expanded_top = max(0, top - tile_overlap)
            expanded_right = min(reference_image.width, core_right + tile_overlap)
            expanded_bottom = min(reference_image.height, core_bottom + tile_overlap)
            input_tile = reference_image.crop(
                (expanded_left, expanded_top, expanded_right, expanded_bottom)
            )
            tile_array = np.asarray(input_tile, dtype=np.float32) / 255.0
            tile_tensor = (
                torch.from_numpy(tile_array)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(execution_device)
            )
            with torch.inference_mode():
                enhanced_tensor = enhancement_model(tile_tensor)
            enhanced_array = (
                enhanced_tensor.squeeze(0)
                .clamp(0, 1)
                .permute(1, 2, 0)
                .mul(255)
                .byte()
                .cpu()
                .numpy()
            )
            enhanced_tile = Image.fromarray(enhanced_array)
            crop_left = (left - expanded_left) * model_scale
            crop_top = (top - expanded_top) * model_scale
            crop_right = crop_left + (core_right - left) * model_scale
            crop_bottom = crop_top + (core_bottom - top) * model_scale
            enhanced_core = enhanced_tile.crop(
                (crop_left, crop_top, crop_right, crop_bottom)
            )
            enhanced_canvas.paste(
                enhanced_core,
                (left * model_scale, top * model_scale),
            )
    return enhanced_canvas


def approve_original_reference_image(
    reference_image: Image.Image,
    source_name: str,
    quality_report: ReferenceImageQualityReport,
) -> ApprovedReferenceImage:
    """화질 검사를 통과한 원본을 생성 입력으로 확정한다."""
    return ApprovedReferenceImage(
        image=reference_image.convert("RGB").copy(),
        source_name=source_name,
        enhancement_applied=False,
        enhancement_model_id=None,
        quality_status=quality_report.status,
    )


def approve_enhanced_reference_image(
    enhancement_candidate: ReferenceImageEnhancementCandidate,
    source_name: str,
) -> ApprovedReferenceImage:
    """사용자가 선택한 보정본을 생성 입력으로 확정한다."""
    return ApprovedReferenceImage(
        image=enhancement_candidate.enhanced_image.convert("RGB").copy(),
        source_name=source_name,
        enhancement_applied=True,
        enhancement_model_id=enhancement_candidate.enhancement_model_id,
        quality_status=enhancement_candidate.quality_report.status,
    )
