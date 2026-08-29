"""의상 참조 원본을 추출 모델에 전달하기 전 정리하고 상태를 기록한다."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
import warnings
import numpy as np


from PIL import Image, ImageOps, UnidentifiedImageError

from genai_lab.clothing import ClothingCategory

if TYPE_CHECKING:
    from transformers import PretrainedConfig, Sam2Config, Sam2Model

DEFAULT_CLOTHING_DETECTION_PROMPT = (
    "clothing . outfit . dress . shirt . jacket . top . pants . skirt ."
)

class ClothingPreparationStage(str, Enum):
    """의상 준비 작업이 GUI에 전달하는 11개 상태."""

    IDLE = "idle"
    NORMALIZING = "normalizing"
    DETECTING = "detecting"
    WAITING_REGION_SELECTION = "waiting_region_selection"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    WAITING_APPROVAL = "waiting_approval"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class ClothingCategoryCandidate(str, Enum):
    """분석 모델이 제시하지만 사용자가 아직 확정하지 않은 의상 종류."""

    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    FULL_BODY_OUTFIT = "full_body_outfit"
    UNKNOWN = "unknown"


class ClothingViewSide(str, Enum):
    """의상 이미지에서 확인할 수 있는 방향."""

    FRONT = "front"
    BACK = "back"
    UNKNOWN = "unknown"


class ClothingSourceValidationCode(str, Enum):
    """의상 원본 정규화가 실패한 원인을 구분하는 코드."""

    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_NOT_FILE = "source_not_file"
    SOURCE_EMPTY = "source_empty"
    SOURCE_UNSUPPORTED_FORMAT = "source_unsupported_format"
    SOURCE_DECODE_FAILED = "source_decode_failed"
    SOURCE_UNSAFE_SIZE = "source_unsafe_size"
    SOURCE_INVALID_DIMENSIONS = "source_invalid_dimensions"


class ClothingSourceValidationError(ValueError):
    """의상 원본을 안전한 RGB 이미지로 정리할 수 없을 때 발생한다."""

    def __init__(
        self,
        code: ClothingSourceValidationCode,
        message_ko: str,
        recovery_action_ko: str,
    ) -> None:
        self.code = code
        self.message_ko = message_ko
        self.recovery_action_ko = recovery_action_ko
        super().__init__(
            f"{message_ko} 해결 방법: {recovery_action_ko}"
        )


@dataclass(frozen=True)
class ClothingSourceInput:
    """사용자가 선택한 정규화 전 의상 이미지 입력."""

    image_path: Path


@dataclass(frozen=True)
class NormalizedClothingSource:
    """방향·투명도·색상 형식을 통일한 사용자 승인 전 임시 이미지."""

    image: Image.Image
    source_name: str
    source_format: str
    source_mode: str
    source_size_bytes: int
    original_width: int
    original_height: int
    normalized_width: int
    normalized_height: int
    icc_profile: bytes | None = None


@dataclass(frozen=True)
class ClothingRegionCandidate:
    """위치 탐지 모델이 찾았지만 사용자가 아직 확인하지 않은 의상 영역."""

    box_xyxy: tuple[int, int, int, int]
    label: str
    confidence: float


@dataclass(frozen=True)
class ClothingDetectionSettings:
    """의상 위치 탐지 모델과 후보 통과 기준."""

    model_id: str = "IDEA-Research/grounding-dino-tiny"
    cache_dir: Path = Path("D:/genai-cache/huggingface")
    inference_device: str = "cpu"
    prompt: str = DEFAULT_CLOTHING_DETECTION_PROMPT
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    minimum_area_ratio: float = 0.02
    maximum_area_ratio: float = 0.95


@dataclass(frozen=True)
class ClothingRegionDetectionResult:
    """자동 탐지가 반환하는 후보와 수동 선택 필요 여부."""

    candidates: tuple[ClothingRegionCandidate, ...]
    selected_candidate: ClothingRegionCandidate | None
    elapsed_seconds: float
    requires_manual_selection: bool
    reason_ko: str


@dataclass(frozen=True)
class ClothingRegionMeasurement:
    """의상 영역 탐지 결과를 임의 종합점수 없이 수치로 기록한다."""

    detection_method: str
    confidence_percent: float | None
    area_ratio_percent: float
    width_pixels: int
    height_pixels: int


@dataclass(frozen=True)
class ClothingMaskExtractionSettings:
    """의상 마스크 후보 생성 모델과 실행 기준."""

    model_id: str = "facebook/sam2.1-hiera-tiny"
    cache_dir: Path = Path("D:/genai-cache/huggingface")
    inference_device: str = "cpu"
    maximum_candidate_count: int = 3
    maximum_region_count: int = 8
    alpha_empty_probability: float = 0.01
    alpha_solid_probability: float = 0.99


@dataclass(frozen=True)
class ClothingMaskReviewCandidate:
    """SAM2가 만들었지만 사용자가 아직 승인하지 않은 마스크 후보."""

    candidate_number: int
    mask_image: Image.Image
    model_score: float
    selected_pixel_count: int
    region_coverage_percent: float
    connected_region_count: int
    boundary_touch_pixel_count: int


@dataclass(frozen=True)
class ClothingMaskRegionCandidateGroup:
    """의상 위치 한 곳과 그 위치에서 SAM2가 만든 후보 최대 3개."""

    region_number: int
    approved_region: ClothingRegionCandidate
    candidates: tuple[ClothingMaskReviewCandidate, ...]


@dataclass(frozen=True)
class ClothingMaskExtractionResult:
    """SAM2 실행 한 번에서 여러 의상 위치별로 나온 후보와 실행 기록."""

    model_id: str
    source_model_type: str
    runtime_model_type: str
    input_width: int
    input_height: int
    region_groups: tuple[ClothingMaskRegionCandidateGroup, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class ClothingCombinedMaskCandidate:
    """사용자가 영역별로 고른 마스크를 합쳤지만 아직 최종 승인하지 않은 후보."""

    mask_image: Image.Image
    source_region_count: int
    selected_pixel_count: int
    connected_region_count: int
    boundary_touch_pixel_count: int


class ClothingMaskExtractionError(RuntimeError):
    """SAM2가 사용자 검토용 의상 마스크를 만들지 못했을 때 발생한다."""


@dataclass(frozen=True)
class ClothingPixelExtractionSettings:
    """승인 마스크로 원본 의상 픽셀을 추출하고 작은 공백을 판정하는 기준."""

    maximum_hole_area_pixels: int = 4096
    maximum_hole_area_ratio: float = 0.0025
    maximum_rgb_distance: float = 36.0
    white_clothing_luminance: float = 200.0
    maximum_white_luminance_difference: float = 48.0


@dataclass(frozen=True)
class ClothingExtractionCandidate:
    """승인 마스크로 원본 RGB를 추출했지만 사용자가 아직 확인하지 않은 후보."""

    extracted_image: Image.Image
    clothing_mask: Image.Image
    preview_crop_box: tuple[int, int, int, int]
    selected_alpha_pixel_count: int
    soft_edge_pixel_count: int
    enclosed_hole_count: int
    filled_hole_count: int
    filled_hole_pixel_count: int
    skipped_hole_count: int
    changed_rgb_pixel_count: int
    original_pixel_preservation_percent: float


@dataclass(frozen=True)
class ClothingDesignSummary:
    """추출된 의상에서 분석한 색상·장식과 확인 불가 항목."""

    dominant_rgb_colors: tuple[tuple[int, int, int], ...]
    design_tags: tuple[str, ...]
    unknown_details: tuple[str, ...]


@dataclass(frozen=True)
class ClothingDesignTagCandidate:
    """WD14가 제시했지만 사용자가 아직 승인하지 않은 일반 태그 한 개."""

    tag_name: str
    display_name: str
    score: float


@dataclass(frozen=True)
class ClothingDesignAnalysisResult:
    """WD14 실행 정보와 점수 기준을 통과한 일반 태그 후보."""

    model_id: str
    execution_provider: str
    input_width: int
    input_height: int
    model_input_size: int
    score_threshold: float
    total_label_count: int
    general_label_count: int
    excluded_rating_label_count: int
    excluded_character_label_count: int
    tag_candidates: tuple[ClothingDesignTagCandidate, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class ClothingReviewCandidate:
    """GUI에서 사용자 승인을 받기 위해 전달하는 임시 의상 결과."""

    source_image: Image.Image
    extraction: ClothingExtractionCandidate
    design: ClothingDesignSummary
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class ConfirmedClothingReference:
    """사용자가 CatVTON 적용 대상으로 승인한 확정 의상."""

    image: Image.Image
    mask: Image.Image
    category: ClothingCategory
    design: ClothingDesignSummary


@dataclass(frozen=True)
class ClothingPreparationProgress:
    """작업 스레드가 GUI에 전달하는 의상 준비 단계 정보."""

    job_id: str
    stage: ClothingPreparationStage
    stage_index: int
    total_stage_count: int
    elapsed_seconds: float
    message_ko: str
    can_cancel: bool
    can_use_manual_region: bool


@dataclass(frozen=True)
class ClothingPreparationFailure:
    """작업 스레드가 GUI와 로그에 전달하는 구조화된 실패 정보."""

    job_id: str
    failure_stage: ClothingPreparationStage
    error_code: str
    message_ko: str
    recovery_actions: tuple[str, ...]
    elapsed_seconds: float
    technical_details: str


def load_and_normalize_clothing_source(
    clothing_source_input: ClothingSourceInput,
) -> NormalizedClothingSource:
    """JPEG·PNG 원본의 방향과 투명도를 정리해 RGB 임시 이미지를 반환한다.

    반환값:
        원본 정보와 정규화된 RGB 이미지. 반환된 이미지의 해제 책임은 호출자에게 있다.

    오류:
        파일이 없거나 비어 있거나 JPEG·PNG로 안전하게 해석되지 않으면
        ClothingSourceValidationError를 발생시킨다.

    부수 효과:
        파일을 읽지만 프로젝트 출력 폴더에는 아무것도 저장하지 않는다.
    """
    image_path = clothing_source_input.image_path
    if not image_path.exists():
        raise ClothingSourceValidationError(
            ClothingSourceValidationCode.SOURCE_NOT_FOUND,
            f"의상 참조 이미지가 없습니다: {image_path}",
            "존재하는 JPEG 또는 PNG 이미지를 다시 선택하세요.",
        )
    if not image_path.is_file():
        raise ClothingSourceValidationError(
            ClothingSourceValidationCode.SOURCE_NOT_FILE,
            f"의상 참조 경로가 파일이 아닙니다: {image_path}",
            "이미지 파일 한 개를 선택하세요.",
        )

    try:
        source_size_bytes = image_path.stat().st_size
    except OSError as error:
        raise ClothingSourceValidationError(
            ClothingSourceValidationCode.SOURCE_DECODE_FAILED,
            f"의상 참조 파일 정보를 읽을 수 없습니다: {image_path}",
            "파일 권한을 확인하거나 다른 이미지를 선택하세요.",
        ) from error

    if source_size_bytes == 0:
        raise ClothingSourceValidationError(
            ClothingSourceValidationCode.SOURCE_EMPTY,
            f"의상 참조 파일의 크기가 0바이트입니다: {image_path}",
            "내용이 있는 JPEG 또는 PNG 이미지를 다시 선택하세요.",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_path) as opened_image:
                opened_image.load()
                source_format = (opened_image.format or "").upper()
                if source_format not in {"JPEG", "PNG"}:
                    raise ClothingSourceValidationError(
                        ClothingSourceValidationCode.SOURCE_UNSUPPORTED_FORMAT,
                        (
                            "현재 의상 참조 입력은 JPEG와 PNG만 지원합니다. "
                            f"실제 형식={source_format or '확인 불가'}"
                        ),
                        "JPEG 또는 PNG 이미지로 다시 선택하세요.",
                    )

                original_width, original_height = opened_image.size
                source_mode = opened_image.mode
                icc_profile = opened_image.info.get("icc_profile")
                if original_width < 1 or original_height < 1:
                    raise ClothingSourceValidationError(
                        ClothingSourceValidationCode.SOURCE_INVALID_DIMENSIONS,
                        (
                            "의상 참조 이미지 크기가 올바르지 않습니다. "
                            f"크기={original_width}x{original_height}픽셀"
                        ),
                        "가로와 세로가 각각 1픽셀 이상인 이미지를 선택하세요.",
                    )

                oriented_image = ImageOps.exif_transpose(opened_image)
                try:
                    normalized_image = composite_image_on_white_background(
                        oriented_image
                    )
                    if icc_profile is not None:
                        normalized_image.info["icc_profile"] = icc_profile
                finally:
                    if oriented_image is not opened_image:
                        oriented_image.close()

    except ClothingSourceValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ClothingSourceValidationError(
            ClothingSourceValidationCode.SOURCE_UNSAFE_SIZE,
            f"의상 참조 이미지의 픽셀 수가 안전 제한을 초과했습니다: {image_path}",
            "이미지 크기를 줄인 뒤 다시 선택하세요.",
        ) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ClothingSourceValidationError(
            ClothingSourceValidationCode.SOURCE_DECODE_FAILED,
            f"의상 참조 이미지를 읽을 수 없습니다: {image_path}",
            "손상되지 않은 JPEG 또는 PNG 이미지를 다시 선택하세요.",
        ) from error

    # NormalizedClothingSource(정규화된 의상 원본)
    # - 포함: RGB 이미지, 실제 파일 형식, 파일 크기와 정규화 전후 픽셀 크기.
    # - 생성: JPEG·PNG 방향 보정과 투명 배경 합성이 끝난 뒤 만든다.
    # - 처리: Python 픽셀 규칙만 사용하며 AI 모델은 호출하지 않는다.
    # - 저장: 저장하지 않는 임시 값이며 반환 이미지는 호출자가 닫는다.
    # - 다음 사용처: Grounding DINO 의상 위치 탐색 입력.
    return NormalizedClothingSource(
        image=normalized_image,
        source_name=image_path.name,
        source_format=source_format,
        source_mode=source_mode,
        source_size_bytes=source_size_bytes,
        original_width=original_width,
        original_height=original_height,
        normalized_width=normalized_image.width,
        normalized_height=normalized_image.height,
        icc_profile=icc_profile,
    )


def composite_image_on_white_background(source_image: Image.Image) -> Image.Image:
    """투명 영역을 흰색으로 합성하고 RGB 이미지로 반환한다."""
    has_transparency = (
        source_image.mode in {"RGBA", "LA"}
        or (
            source_image.mode == "P"
            and "transparency" in source_image.info
        )
    )
    if not has_transparency:
        return source_image.convert("RGB")

    rgba_image = source_image.convert("RGBA")
    white_background = Image.new(
        "RGBA",
        rgba_image.size,
        (255, 255, 255, 255),
    )
    try:
        return Image.alpha_composite(
            white_background,
            rgba_image,
        ).convert("RGB")
    finally:
        rgba_image.close()
        white_background.close()



def detect_clothing_regions(
    normalized_source: NormalizedClothingSource,
    settings: ClothingDetectionSettings,
) -> ClothingRegionDetectionResult:
    """Grounding DINO로 의상 위치를 찾고 유효 후보가 없으면 수동 선택을 요청한다.

    반환값:
        면적 기준을 통과한 후보, 가장 신뢰도가 높은 후보와 처리 시간.

    오류:
        모델을 불러오거나 실행하지 못하면 예외를 숨기지 않고 호출자에게 전달한다.

    부수 효과:
        모델을 처음 사용할 때 Hugging Face 캐시에 파일을 내려받을 수 있다.
        실행이 끝나면 모델 참조와 CUDA 캐시를 해제한다.
    """
    _validate_detection_settings(settings)
    started_at = perf_counter()
    processor = None
    model = None
    inputs = None
    outputs = None

    try:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        if settings.inference_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = settings.inference_device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "의상 탐지 장치가 CUDA로 설정됐지만 GPU를 사용할 수 없습니다."
            )
        processor = AutoProcessor.from_pretrained(
            settings.model_id, cache_dir=settings.cache_dir
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            settings.model_id,
            cache_dir=settings.cache_dir,
        ).to(device)
        inputs = processor(
            images=normalized_source.image,
            text=settings.prompt,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        detected = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=settings.box_threshold,
            text_threshold=settings.text_threshold,
            target_sizes=[
                (
                    normalized_source.normalized_height,
                    normalized_source.normalized_width,
                )
            ],
        )[0]

        candidates = _build_valid_clothing_region_candidates(
            boxes=detected["boxes"].detach().cpu().tolist(),
            scores=detected["scores"].detach().cpu().tolist(),
            labels=tuple(str(label) for label in detected["labels"]),
            image_size=(
                normalized_source.normalized_width,
                normalized_source.normalized_height,
            ),
            settings=settings,
        )
    finally:
        del outputs, inputs, model, processor
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    selected_candidate = candidates[0] if candidates else None
    return ClothingRegionDetectionResult(
        candidates=candidates,
        selected_candidate=selected_candidate,
        elapsed_seconds=perf_counter() - started_at,
        requires_manual_selection=selected_candidate is None,
        reason_ko=(
            "자동 탐지 후보를 찾았습니다."
            if selected_candidate is not None
            else "통과 기준을 만족한 의상 영역이 없어 수동 선택이 필요합니다."
        ),
    )


def create_manual_clothing_region(
    image_size: tuple[int, int],
    box_xyxy: tuple[int, int, int, int],
) -> ClothingRegionCandidate:
    """사용자가 그린 사각형을 이미지 경계 안의 의상 영역으로 확정한다."""
    image_width, image_height = image_size
    x1, y1, x2, y2 = box_xyxy
    if image_width < 1 or image_height < 1:
        raise ValueError("수동 선택 대상 이미지 크기는 1픽셀 이상이어야 합니다.")
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError(
            "수동 의상 영역은 이미지 안에서 가로와 세로가 각각 1픽셀 이상이어야 합니다."
        )
    return ClothingRegionCandidate(
        box_xyxy=box_xyxy,
        label="user_selected_clothing",
        confidence=0.0,
    )


def measure_clothing_region(
    candidate: ClothingRegionCandidate,
    image_size: tuple[int, int],
) -> ClothingRegionMeasurement:
    """자동·수동 의상 영역의 신뢰도와 면적 비율을 계산한다."""
    image_width, image_height = image_size
    x1, y1, x2, y2 = candidate.box_xyxy
    region_width = x2 - x1
    region_height = y2 - y1
    if image_width < 1 or image_height < 1 or region_width < 1 or region_height < 1:
        raise ValueError("의상 영역과 이미지 크기는 각각 1픽셀 이상이어야 합니다.")
    area_ratio_percent = (
        region_width * region_height / (image_width * image_height) * 100.0
    )
    is_manual = candidate.label == "user_selected_clothing"
    return ClothingRegionMeasurement(
        detection_method="manual" if is_manual else "grounding_dino",
        confidence_percent=(None if is_manual else candidate.confidence * 100.0),
        area_ratio_percent=area_ratio_percent,
        width_pixels=region_width,
        height_pixels=region_height,
    )


def _build_valid_clothing_region_candidates(
    boxes: list[list[float]],
    scores: list[float],
    labels: tuple[str, ...],
    image_size: tuple[int, int],
    settings: ClothingDetectionSettings,
) -> tuple[ClothingRegionCandidate, ...]:
    """모델 출력을 이미지 경계에 맞추고 면적 기준을 통과한 후보만 반환한다."""
    image_width, image_height = image_size
    image_area = image_width * image_height
    candidates: list[ClothingRegionCandidate] = []
    for box, score, label in zip(boxes, scores, labels):
        x1 = max(0, min(image_width, round(box[0])))
        y1 = max(0, min(image_height, round(box[1])))
        x2 = max(0, min(image_width, round(box[2])))
        y2 = max(0, min(image_height, round(box[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
        if not settings.minimum_area_ratio <= area_ratio <= settings.maximum_area_ratio:
            continue
        candidates.append(
            ClothingRegionCandidate(
                box_xyxy=(x1, y1, x2, y2),
                label=label,
                confidence=float(score),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: candidate.confidence,
            reverse=True,
        )
    )


def _validate_detection_settings(settings: ClothingDetectionSettings) -> None:
    """탐지 기준이 0.00~1.00 범위이고 최소 면적이 최대 면적보다 작은지 확인한다."""
    if settings.inference_device not in {"cpu", "cuda", "auto"}:
        raise ValueError(
            "inference_device 값은 cpu, cuda, auto 중 하나여야 합니다."
        )

    for setting_name, setting_value in (
        ("box_threshold", settings.box_threshold),
        ("text_threshold", settings.text_threshold),
        ("minimum_area_ratio", settings.minimum_area_ratio),
        ("maximum_area_ratio", settings.maximum_area_ratio),
    ):
        if not 0.0 <= setting_value <= 1.0:
            raise ValueError(f"{setting_name} 값은 0.00~1.00 범위여야 합니다.")
    if settings.minimum_area_ratio >= settings.maximum_area_ratio:
        raise ValueError("최소 의상 면적 비율은 최대 면적 비율보다 작아야 합니다.")


def extract_clothing_mask_candidates(
    normalized_source: NormalizedClothingSource,
    approved_regions: tuple[ClothingRegionCandidate, ...],
    settings: ClothingMaskExtractionSettings,
) -> ClothingMaskExtractionResult:
    """승인된 사각형 최대 8개로 영역별 SAM2 후보를 최대 3개 만든다.

    반환값:
        사용자 승인 전 마스크 후보와 SAM2 실행 기록.

    오류:
        설정, 입력 영역, 모델 로딩 또는 마스크 생성이 실패하면
        ClothingMaskExtractionError를 발생시킨다.

    부수 효과:
        첫 실행 시 모델을 캐시에 내려받을 수 있다. 결과 파일은 저장하지 않는다.
    """
    _validate_mask_extraction_settings(settings)
    if not 1 <= len(approved_regions) <= settings.maximum_region_count:
        raise ClothingMaskExtractionError(
            "SAM2 입력 의상 영역은 1개~"
            f"{settings.maximum_region_count}개여야 합니다. "
            f"실제 영역={len(approved_regions)}개"
        )
    for approved_region in approved_regions:
        _validate_region_inside_image(
            image_size=normalized_source.image.size,
            region=approved_region,
        )
    started_at = perf_counter()
    processor = None
    model = None
    model_inputs = None
    model_outputs = None

    try:
        import torch
        from transformers import Sam2Processor

        if settings.inference_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = settings.inference_device
        if device == "cuda" and not torch.cuda.is_available():
            raise ClothingMaskExtractionError(
                "SAM2 장치가 CUDA로 설정됐지만 GPU를 사용할 수 없습니다."
            )

        processor = Sam2Processor.from_pretrained(
            settings.model_id,
            cache_dir=settings.cache_dir,
        )
        model, source_model_type, runtime_model_type = (
            load_sam2_image_model(
                settings=settings,
                device=device,
            )
        )
        model_inputs = processor(
            images=normalized_source.image,
            input_boxes=[
                [list(region.box_xyxy) for region in approved_regions]
            ],
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            model_outputs = model(
                **model_inputs,
                multimask_output=True,
            )

        restored_batch_masks = processor.post_process_masks(
            model_outputs.pred_masks.detach().cpu(),
            model_inputs["original_sizes"].detach().cpu(),
            binarize=False,
        )
        restored_region_masks = restored_batch_masks[0]
        region_scores = model_outputs.iou_scores.detach().cpu()[0]
        region_groups: list[ClothingMaskRegionCandidateGroup] = []
        try:
            for region_index, approved_region in enumerate(approved_regions):
                review_candidates = build_clothing_mask_review_candidates(
                    restored_masks=restored_region_masks[region_index].numpy(),
                    model_scores=region_scores[region_index].numpy(),
                    approved_region=approved_region,
                    image_size=normalized_source.image.size,
                    maximum_candidate_count=settings.maximum_candidate_count,
                    alpha_empty_probability=settings.alpha_empty_probability,
                    alpha_solid_probability=settings.alpha_solid_probability,
                )
                if not review_candidates:
                    raise ClothingMaskExtractionError(
                        f"SAM2가 {region_index + 1}번 영역에서 선택 픽셀이 "
                        "1개 이상인 마스크를 만들지 못했습니다."
                    )
                region_groups.append(
                    ClothingMaskRegionCandidateGroup(
                        region_number=region_index + 1,
                        approved_region=approved_region,
                        candidates=review_candidates,
                    )
                )
        except Exception:
            for region_group in region_groups:
                for candidate in region_group.candidates:
                    candidate.mask_image.close()
            raise
    except ClothingMaskExtractionError:
        raise
    except Exception as error:
        raise ClothingMaskExtractionError(
            "SAM2 의상 마스크 후보 생성에 실패했습니다. "
            f"원인={type(error).__name__}: {error}"
        ) from error
    finally:
        del model_outputs, model_inputs, model, processor
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ClothingMaskExtractionResult(의상 마스크 추출 결과)
    # - 포함: SAM2 모델, 입력 크기, 승인 사각형 최대 8개와 영역별 후보 최대 3개.
    # - 생성: SAM2 실행 후 후보별 픽셀 수치를 Python 규칙으로 계산한다.
    # - 처리: 마스크 생성은 SAM2, 측정값 계산은 Python 규칙이 담당한다.
    # - 저장: 저장하지 않는 사용자 승인 전 임시 값이다.
    # - 다음 사용처: GUI 의상 마스크 후보 비교 화면.
    return ClothingMaskExtractionResult(
        model_id=settings.model_id,
        source_model_type=source_model_type,
        runtime_model_type=runtime_model_type,
        input_width=normalized_source.normalized_width,
        input_height=normalized_source.normalized_height,
        region_groups=tuple(region_groups),
        elapsed_seconds=perf_counter() - started_at,
    )


def load_sam2_image_model(
    settings: ClothingMaskExtractionSettings,
    device: str,
) -> tuple["Sam2Model", str, str]:
    """영상용 SAM2 체크포인트를 명시적인 단일 이미지 구성으로 불러온다.

    반환값:
        단일 이미지 SAM2 모델, 원본 구성 종류와 실행 구성 종류.

    오류:
        체크포인트가 sam2 또는 sam2_video 구성이 아니면 실패한다.

    부수 효과:
        첫 실행 시 Hugging Face 캐시에 모델 파일을 내려받을 수 있다.
    """
    from transformers import AutoConfig, Sam2Model

    source_config = AutoConfig.from_pretrained(
        settings.model_id,
        cache_dir=settings.cache_dir,
    )
    source_model_type = str(source_config.model_type)
    image_config = create_sam2_image_config(source_config)
    model = Sam2Model.from_pretrained(
        settings.model_id,
        config=image_config,
        cache_dir=settings.cache_dir,
    ).to(device)
    return model, source_model_type, str(model.config.model_type)


def create_sam2_image_config(
    source_config: "PretrainedConfig",
) -> "Sam2Config":
    """sam2_video 공통 가중치 설정을 경고 없는 sam2 이미지 구성으로 변환한다."""
    from transformers import Sam2Config

    source_model_type = str(source_config.model_type)
    if source_model_type not in {"sam2", "sam2_video"}:
        raise ClothingMaskExtractionError(
            "SAM2 체크포인트 구성 종류가 올바르지 않습니다. "
            f"허용=sam2 또는 sam2_video, 실제={source_model_type}"
        )

    image_config_values = source_config.to_dict()
    image_config_values["model_type"] = "sam2"
    image_config_values["architectures"] = ["Sam2Model"]
    return Sam2Config.from_dict(image_config_values)


def combine_clothing_mask_candidates(
    selected_candidates: tuple[ClothingMaskReviewCandidate, ...],
    image_size: tuple[int, int],
) -> ClothingCombinedMaskCandidate:
    """영역별 선택 마스크를 픽셀 합집합으로 합쳐 최종 확인 후보를 만든다.

    반환값:
        하나로 합친 흑백 마스크와 선택 픽셀·분리 영역·경계 접촉 수치.

    오류:
        선택 후보가 1개~8개 범위를 벗어나거나 이미지 크기가 다르면 실패한다.

    부수 효과:
        입력 마스크는 변경하거나 닫지 않으며 결과 파일도 저장하지 않는다.
    """
    if not 1 <= len(selected_candidates) <= 8:
        raise ClothingMaskExtractionError(
            "합칠 의상 마스크는 1개~8개여야 합니다. "
            f"실제 마스크={len(selected_candidates)}개"
        )

    combined_alpha_array = np.zeros(
        (image_size[1], image_size[0]),
        dtype=np.uint8,
    )
    for candidate_number, candidate in enumerate(selected_candidates, start=1):
        candidate_mask_array = np.asarray(candidate.mask_image, dtype=np.uint8)
        if candidate_mask_array.shape != combined_alpha_array.shape:
            raise ClothingMaskExtractionError(
                f"{candidate_number}번 마스크 크기가 원본 이미지와 다릅니다. "
                f"마스크={candidate.mask_image.width}x{candidate.mask_image.height}, "
                f"원본={image_size[0]}x{image_size[1]}"
            )
        combined_alpha_array = np.maximum(
            combined_alpha_array,
            candidate_mask_array,
        )

    combined_mask_image = Image.fromarray(
        combined_alpha_array,
        mode="L",
    )
    selected_binary_mask = combined_alpha_array >= 128
    return ClothingCombinedMaskCandidate(
        mask_image=combined_mask_image,
        source_region_count=len(selected_candidates),
        selected_pixel_count=int(np.count_nonzero(selected_binary_mask)),
        connected_region_count=count_connected_mask_regions(selected_binary_mask),
        boundary_touch_pixel_count=count_mask_boundary_touch_pixels(
            selected_binary_mask
        ),
    )


def build_clothing_mask_review_candidates(
    restored_masks: np.ndarray,
    model_scores: np.ndarray,
    approved_region: ClothingRegionCandidate,
    image_size: tuple[int, int],
    maximum_candidate_count: int,
    alpha_empty_probability: float = 0.01,
    alpha_solid_probability: float = 0.99,
) -> tuple[ClothingMaskReviewCandidate, ...]:
    """SAM2 배열을 흑백 마스크 후보와 규칙 기반 측정값으로 변환한다."""
    if restored_masks.ndim != 3:
        raise ClothingMaskExtractionError(
            "SAM2 마스크 배열은 후보·세로·가로의 3차원이어야 합니다. "
            f"실제 차원={restored_masks.shape}"
        )
    if model_scores.ndim != 1:
        model_scores = model_scores.reshape(-1)
    if restored_masks.shape[0] != model_scores.shape[0]:
        raise ClothingMaskExtractionError(
            "SAM2 마스크 수와 점수 수가 다릅니다. "
            f"마스크={restored_masks.shape[0]}개, 점수={model_scores.shape[0]}개"
        )

    ranked_predictions = sorted(
        zip(restored_masks, model_scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )[:maximum_candidate_count]
    review_candidates: list[ClothingMaskReviewCandidate] = []

    for candidate_number, (predicted_mask, model_score) in enumerate(
        ranked_predictions,
        start=1,
    ):
        predicted_mask_array = np.asarray(predicted_mask)
        binary_mask = predicted_mask_array.astype(bool)
        if predicted_mask_array.dtype != np.bool_:
            binary_mask = predicted_mask_array > 0.0
        if binary_mask.shape != (image_size[1], image_size[0]):
            raise ClothingMaskExtractionError(
                "복원된 SAM2 마스크 크기가 원본 이미지와 다릅니다. "
                f"마스크={binary_mask.shape[1]}x{binary_mask.shape[0]}, "
                f"원본={image_size[0]}x{image_size[1]}"
            )

        selected_pixel_count = int(np.count_nonzero(binary_mask))
        if selected_pixel_count == 0:
            continue
        mask_alpha_array = convert_sam2_mask_to_alpha(
            predicted_mask_array,
            empty_probability=alpha_empty_probability,
            solid_probability=alpha_solid_probability,
        )
        mask_image = Image.fromarray(mask_alpha_array, mode="L")
        review_candidates.append(
            ClothingMaskReviewCandidate(
                candidate_number=candidate_number,
                mask_image=mask_image,
                model_score=float(model_score),
                selected_pixel_count=selected_pixel_count,
                region_coverage_percent=calculate_mask_region_coverage_percent(
                    binary_mask,
                    approved_region.box_xyxy,
                ),
                connected_region_count=count_connected_mask_regions(
                    binary_mask
                ),
                boundary_touch_pixel_count=count_mask_boundary_touch_pixels(
                    binary_mask
                ),
            )
        )

    return tuple(review_candidates)


def convert_sam2_mask_to_alpha(
    predicted_mask: np.ndarray,
    empty_probability: float = 0.01,
    solid_probability: float = 0.99,
) -> np.ndarray:
    """SAM2 마스크를 경계 투명도를 유지하는 0~255 알파 배열로 바꾼다."""
    if not 0.0 <= empty_probability < solid_probability <= 1.0:
        raise ClothingMaskExtractionError(
            "SAM2 알파 경계 기준이 올바르지 않습니다. "
            f"빈 영역={empty_probability:.4f}, "
            f"불투명 영역={solid_probability:.4f}"
        )
    mask_array = np.asarray(predicted_mask)
    if mask_array.dtype == np.bool_:
        return np.where(mask_array, 255, 0).astype(np.uint8)

    clipped_logits = np.clip(mask_array.astype(np.float32), -20.0, 20.0)
    mask_probability = 1.0 / (1.0 + np.exp(-clipped_logits))
    mask_probability[mask_probability <= empty_probability] = 0.0
    mask_probability[mask_probability >= solid_probability] = 1.0
    return np.rint(mask_probability * 255.0).astype(np.uint8)


def calculate_mask_region_coverage_percent(
    binary_mask: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
) -> float:
    """승인 사각형 안에서 마스크가 차지하는 픽셀 비율을 계산한다."""
    x1, y1, x2, y2 = box_xyxy
    region_area = (x2 - x1) * (y2 - y1)
    if region_area < 1:
        raise ClothingMaskExtractionError(
            "의상 위치 사각형의 면적은 1픽셀 이상이어야 합니다."
        )
    selected_inside_region = int(
        np.count_nonzero(binary_mask[y1:y2, x1:x2])
    )
    return selected_inside_region / region_area * 100.0


def count_connected_mask_regions(binary_mask: np.ndarray) -> int:
    """서로 맞닿지 않은 마스크 영역 수를 8방향 연결 기준으로 계산한다."""
    try:
        import cv2
    except ImportError as error:
        raise ClothingMaskExtractionError(
            "마스크 영역 수 계산에 필요한 OpenCV를 불러올 수 없습니다."
        ) from error

    region_count, _ = cv2.connectedComponents(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )
    return max(0, int(region_count) - 1)


def count_mask_boundary_touch_pixels(binary_mask: np.ndarray) -> int:
    """마스크가 전체 이미지의 네 경계와 맞닿은 픽셀 수를 계산한다."""
    boundary_mask = np.zeros_like(binary_mask, dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    return int(np.count_nonzero(binary_mask & boundary_mask))


def extract_clothing_pixels(
    normalized_source: NormalizedClothingSource,
    approved_mask: ClothingCombinedMaskCandidate,
    settings: ClothingPixelExtractionSettings | None = None,
) -> ClothingExtractionCandidate:
    """승인 마스크를 알파로 사용해 원본 RGB 의상을 메모리에 추출한다.

    반환값:
        투명 배경 추출본, 실제 알파 마스크, 자동 공백 처리와 픽셀 보존 수치.

    오류:
        이미지·마스크 크기가 다르거나 결과 RGB가 1픽셀 이상 바뀌면 실패한다.

    부수 효과:
        모델을 호출하거나 파일을 저장하지 않으며 반환 이미지는 호출자가 닫는다.
    """
    extraction_settings = settings or ClothingPixelExtractionSettings()
    _validate_pixel_extraction_settings(extraction_settings)
    source_rgb_image = normalized_source.image.convert("RGB")
    approved_mask_image = approved_mask.mask_image.convert("L")
    try:
        if source_rgb_image.size != approved_mask_image.size:
            raise ClothingMaskExtractionError(
                "의상 원본과 승인 마스크 크기가 다릅니다. "
                f"원본={source_rgb_image.width}x{source_rgb_image.height}, "
                f"마스크={approved_mask_image.width}x{approved_mask_image.height}"
            )

        source_rgb_array = np.asarray(source_rgb_image, dtype=np.uint8).copy()
        approved_alpha_array = np.asarray(
            approved_mask_image,
            dtype=np.uint8,
        ).copy()
        (
            repaired_alpha_array,
            enclosed_hole_count,
            filled_hole_count,
            filled_hole_pixel_count,
            skipped_hole_count,
        ) = fill_enclosed_clothing_mask_holes(
            source_rgb_array,
            approved_alpha_array,
            extraction_settings,
        )
        selected_alpha_pixel_count = int(
            np.count_nonzero(repaired_alpha_array)
        )
        if selected_alpha_pixel_count == 0:
            raise ClothingMaskExtractionError(
                "승인 마스크에 추출할 픽셀이 1개도 없습니다."
            )

        extracted_rgba_array = np.empty(
            (
                source_rgb_image.height,
                source_rgb_image.width,
                4,
            ),
            dtype=np.uint8,
        )
        extracted_rgba_array[:, :, :3] = source_rgb_array
        extracted_rgba_array[:, :, 3] = repaired_alpha_array
        extracted_image = Image.fromarray(extracted_rgba_array, mode="RGBA")
        repaired_mask_image = Image.fromarray(
            repaired_alpha_array,
            mode="L",
        )
        if normalized_source.icc_profile is not None:
            extracted_image.info["icc_profile"] = normalized_source.icc_profile

        preview_crop_box = repaired_mask_image.getbbox()
        if preview_crop_box is None:
            extracted_image.close()
            repaired_mask_image.close()
            raise ClothingMaskExtractionError(
                "추출 의상의 미리보기 범위를 계산할 수 없습니다."
            )

        verification_image = extracted_image.convert("RGBA")
        try:
            verified_rgba_array = np.asarray(
                verification_image,
                dtype=np.uint8,
            ).copy()
        finally:
            verification_image.close()
        selected_for_verification = repaired_alpha_array > 0
        changed_rgb_pixel_count = int(
            np.count_nonzero(
                np.any(
                    verified_rgba_array[:, :, :3][selected_for_verification]
                    != source_rgb_array[selected_for_verification],
                    axis=1,
                )
            )
        )
        original_pixel_preservation_percent = (
            (
                selected_alpha_pixel_count - changed_rgb_pixel_count
            )
            / selected_alpha_pixel_count
            * 100.0
        )
        if changed_rgb_pixel_count > 0:
            extracted_image.close()
            repaired_mask_image.close()
            raise ClothingMaskExtractionError(
                "추출 결과가 원본 RGB를 변경했습니다. "
                f"변경 픽셀={changed_rgb_pixel_count:,}개, "
                "허용=0개"
            )

        return ClothingExtractionCandidate(
            extracted_image=extracted_image,
            clothing_mask=repaired_mask_image,
            preview_crop_box=preview_crop_box,
            selected_alpha_pixel_count=selected_alpha_pixel_count,
            soft_edge_pixel_count=int(
                np.count_nonzero(
                    (repaired_alpha_array > 0)
                    & (repaired_alpha_array < 255)
                )
            ),
            enclosed_hole_count=enclosed_hole_count,
            filled_hole_count=filled_hole_count,
            filled_hole_pixel_count=filled_hole_pixel_count,
            skipped_hole_count=skipped_hole_count,
            changed_rgb_pixel_count=changed_rgb_pixel_count,
            original_pixel_preservation_percent=(
                original_pixel_preservation_percent
            ),
        )
    finally:
        source_rgb_image.close()
        approved_mask_image.close()


def fill_enclosed_clothing_mask_holes(
    source_rgb_array: np.ndarray,
    mask_alpha_array: np.ndarray,
    settings: ClothingPixelExtractionSettings,
) -> tuple[np.ndarray, int, int, int, int]:
    """완전히 둘러싸인 작은 공백을 색상·명도 규칙으로 선별해 알파만 복원한다."""
    try:
        import cv2
    except ImportError as error:
        raise ClothingMaskExtractionError(
            "마스크 공백 검사에 필요한 OpenCV를 불러올 수 없습니다."
        ) from error

    if source_rgb_array.shape[:2] != mask_alpha_array.shape:
        raise ClothingMaskExtractionError(
            "공백 검사 입력의 이미지와 마스크 크기가 다릅니다."
        )

    repaired_alpha_array = mask_alpha_array.copy()
    selected_binary_mask = mask_alpha_array >= 128
    background_mask = (~selected_binary_mask).astype(np.uint8)
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(background_mask, connectivity=8)
    )
    image_height, image_width = mask_alpha_array.shape
    maximum_hole_area = min(
        settings.maximum_hole_area_pixels,
        max(
            1,
            int(
                round(
                    image_width
                    * image_height
                    * settings.maximum_hole_area_ratio
                )
            ),
        ),
    )
    enclosed_hole_count = 0
    filled_hole_count = 0
    filled_hole_pixel_count = 0
    skipped_hole_count = 0
    boundary_kernel = np.ones((3, 3), dtype=np.uint8)

    for component_number in range(1, component_count):
        x = int(component_stats[component_number, cv2.CC_STAT_LEFT])
        y = int(component_stats[component_number, cv2.CC_STAT_TOP])
        width = int(component_stats[component_number, cv2.CC_STAT_WIDTH])
        height = int(component_stats[component_number, cv2.CC_STAT_HEIGHT])
        area = int(component_stats[component_number, cv2.CC_STAT_AREA])
        touches_image_boundary = (
            x == 0
            or y == 0
            or x + width == image_width
            or y + height == image_height
        )
        if touches_image_boundary:
            continue

        enclosed_hole_count += 1
        if area > maximum_hole_area:
            skipped_hole_count += 1
            continue

        hole_mask = component_labels == component_number
        boundary_ring = (
            cv2.dilate(
                hole_mask.astype(np.uint8),
                boundary_kernel,
                iterations=1,
            ).astype(bool)
            & selected_binary_mask
        )
        if not np.any(boundary_ring):
            skipped_hole_count += 1
            continue

        hole_median_rgb = np.median(
            source_rgb_array[hole_mask],
            axis=0,
        )
        boundary_median_rgb = np.median(
            source_rgb_array[boundary_ring],
            axis=0,
        )
        rgb_distance = float(
            np.linalg.norm(hole_median_rgb - boundary_median_rgb)
        )
        hole_luminance = calculate_rgb_luminance(hole_median_rgb)
        boundary_luminance = calculate_rgb_luminance(
            boundary_median_rgb
        )
        passes_regular_color_rule = (
            rgb_distance <= settings.maximum_rgb_distance
        )
        passes_white_clothing_rule = (
            boundary_luminance >= settings.white_clothing_luminance
            and abs(hole_luminance - boundary_luminance)
            <= settings.maximum_white_luminance_difference
        )
        if not (
            passes_regular_color_rule
            or passes_white_clothing_rule
        ):
            skipped_hole_count += 1
            continue

        boundary_alpha = int(
            round(float(np.median(mask_alpha_array[boundary_ring])))
        )
        repaired_alpha_array[hole_mask] = max(128, boundary_alpha)
        filled_hole_count += 1
        filled_hole_pixel_count += area

    return (
        repaired_alpha_array,
        enclosed_hole_count,
        filled_hole_count,
        filled_hole_pixel_count,
        skipped_hole_count,
    )


def calculate_rgb_luminance(rgb_color: np.ndarray) -> float:
    """RGB 한 색의 명도를 sRGB 가중치로 계산한다."""
    red, green, blue = (float(channel) for channel in rgb_color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _validate_mask_extraction_settings(
    settings: ClothingMaskExtractionSettings,
) -> None:
    """SAM2 모델·장치·후보 수 설정을 검사한다."""
    if not settings.model_id.strip():
        raise ClothingMaskExtractionError(
            "SAM2 모델 ID가 비어 있습니다."
        )
    if settings.inference_device not in {"cpu", "cuda", "auto"}:
        raise ClothingMaskExtractionError(
            "SAM2 inference_device는 cpu, cuda, auto 중 하나여야 합니다."
        )
    if not 1 <= settings.maximum_candidate_count <= 3:
        raise ClothingMaskExtractionError(
            "SAM2 마스크 후보 수는 1개~3개여야 합니다."
        )
    if not 1 <= settings.maximum_region_count <= 8:
        raise ClothingMaskExtractionError(
            "SAM2 의상 영역 수는 1개~8개여야 합니다."
        )
    if not (
        0.0
        <= settings.alpha_empty_probability
        < settings.alpha_solid_probability
        <= 1.0
    ):
        raise ClothingMaskExtractionError(
            "SAM2 알파 경계 기준은 0.0~1.0 범위에서 "
            "빈 영역 기준보다 불투명 기준이 커야 합니다."
        )


def _validate_pixel_extraction_settings(
    settings: ClothingPixelExtractionSettings,
) -> None:
    """공백 자동 복원에 사용하는 5개 수치 기준을 검사한다."""
    if settings.maximum_hole_area_pixels < 1:
        raise ClothingMaskExtractionError(
            "자동 복원할 최대 공백 크기는 1픽셀 이상이어야 합니다."
        )
    if not 0.0 < settings.maximum_hole_area_ratio <= 1.0:
        raise ClothingMaskExtractionError(
            "자동 복원할 공백 면적 비율은 0.0 초과 1.0 이하여야 합니다."
        )
    if not 0.0 <= settings.maximum_rgb_distance <= 441.673:
        raise ClothingMaskExtractionError(
            "공백 RGB 거리 기준은 0.0~441.673이어야 합니다."
        )
    if not 0.0 <= settings.white_clothing_luminance <= 255.0:
        raise ClothingMaskExtractionError(
            "흰 의상 명도 기준은 0.0~255.0이어야 합니다."
        )
    if not 0.0 <= settings.maximum_white_luminance_difference <= 255.0:
        raise ClothingMaskExtractionError(
            "흰 의상 명도 차이 기준은 0.0~255.0이어야 합니다."
        )


def _validate_region_inside_image(
    image_size: tuple[int, int],
    region: ClothingRegionCandidate,
) -> None:
    """승인 사각형이 원본 이미지 경계 안에 있는지 검사한다."""
    image_width, image_height = image_size
    x1, y1, x2, y2 = region.box_xyxy
    if not (
        image_width >= 1
        and image_height >= 1
        and 0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ClothingMaskExtractionError(
            "SAM2 입력 사각형이 이미지 경계를 벗어났습니다. "
            f"이미지={image_width}x{image_height}, 영역={region.box_xyxy}"
        )
