"""사용자가 승인한 추출 의상을 WD14로 분석해 일반 태그 후보를 만든다."""

from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from genai_lab.clothing_reference import (
    ClothingDesignAnalysisResult,
    ClothingDesignTagCandidate,
    ClothingExtractionCandidate,
)


@dataclass(frozen=True)
class ClothingDesignAnalysisSettings:
    """WD14 모델 파일, 실행 장치와 태그 후보 통과 기준."""

    model_id: str = "SmilingWolf/wd-vit-tagger-v3"
    cache_dir: Path = Path("D:/genai-cache/huggingface")
    model_filename: str = "model.onnx"
    label_filename: str = "selected_tags.csv"
    execution_provider: str = "CPUExecutionProvider"
    score_threshold: float = 0.35
    maximum_tag_count: int = 30


class ClothingDesignAnalysisError(RuntimeError):
    """WD14 모델 준비·실행·출력 검증이 실패했을 때 발생한다."""


@dataclass(frozen=True)
class ClothingTagLabel:
    """WD14 CSV에서 읽은 원본 태그 이름과 분류 번호."""

    name: str
    category: int


def analyze_clothing_design(
    extraction_candidate: ClothingExtractionCandidate,
    settings: ClothingDesignAnalysisSettings,
) -> ClothingDesignAnalysisResult:
    """추출 의상 한 장을 WD14로 실행해 사용자 검토용 일반 태그를 반환한다.

    반환값:
        모델·장치·입력 크기·기준 점수와 일반 태그 후보 최대 30개.

    오류:
        의존성·모델 파일·라벨·모델 출력이 올바르지 않으면 한글 오류로 실패한다.

    부수 효과:
        최초 실행 시 모델 1개와 CSV 1개를 캐시에 내려받지만 결과 파일은 저장하지 않는다.
    """
    _validate_analysis_settings(settings)
    started_at = perf_counter()
    model_path, label_path = download_wd14_model_files(settings)
    tag_labels = load_wd14_tag_labels(label_path)

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ClothingDesignAnalysisError(
            "WD14 실행에 필요한 onnxruntime이 설치되지 않았습니다. "
            "requirements.txt를 설치한 뒤 다시 실행하세요."
        ) from error

    available_providers = ort.get_available_providers()
    if settings.execution_provider not in available_providers:
        raise ClothingDesignAnalysisError(
            "요청한 WD14 실행 장치를 사용할 수 없습니다. "
            f"요청={settings.execution_provider}, "
            f"사용 가능={available_providers}"
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=[settings.execution_provider],
    )
    model_input = session.get_inputs()[0]
    input_shape = model_input.shape
    if (
        len(input_shape) != 4
        or not isinstance(input_shape[1], int)
        or input_shape[1] < 1
        or input_shape[1] != input_shape[2]
    ):
        raise ClothingDesignAnalysisError(
            "WD14 입력 크기를 확인할 수 없습니다. "
            f"모델 입력={input_shape}"
        )
    model_input_size = int(input_shape[1])
    prepared_image = prepare_wd14_image(
        extraction_candidate.extracted_image,
        model_input_size,
    )
    model_outputs = session.run(
        None,
        {model_input.name: prepared_image},
    )
    if len(model_outputs) != 1:
        raise ClothingDesignAnalysisError(
            "WD14 출력 개수가 올바르지 않습니다. "
            f"예상=1개, 실제={len(model_outputs)}개"
        )
    model_scores = np.asarray(model_outputs[0], dtype=np.float32)
    if model_scores.ndim != 2 or model_scores.shape[0] != 1:
        raise ClothingDesignAnalysisError(
            "WD14 점수 배열 크기가 올바르지 않습니다. "
            f"실제={model_scores.shape}"
        )
    if model_scores.shape[1] != len(tag_labels):
        raise ClothingDesignAnalysisError(
            "WD14 라벨 수와 점수 수가 다릅니다. "
            f"라벨={len(tag_labels)}개, 점수={model_scores.shape[1]}개"
        )

    tag_candidates = build_general_tag_candidates(
        tag_labels=tag_labels,
        model_scores=model_scores[0],
        score_threshold=settings.score_threshold,
        maximum_tag_count=settings.maximum_tag_count,
    )
    general_label_count = sum(
        1 for label in tag_labels if label.category == 0
    )
    return ClothingDesignAnalysisResult(
        model_id=settings.model_id,
        execution_provider=settings.execution_provider,
        input_width=extraction_candidate.extracted_image.width,
        input_height=extraction_candidate.extracted_image.height,
        model_input_size=model_input_size,
        score_threshold=settings.score_threshold,
        total_label_count=len(tag_labels),
        general_label_count=general_label_count,
        excluded_rating_label_count=sum(
            1 for label in tag_labels if label.category == 9
        ),
        excluded_character_label_count=sum(
            1 for label in tag_labels if label.category == 4
        ),
        tag_candidates=tag_candidates,
        elapsed_seconds=perf_counter() - started_at,
    )


def download_wd14_model_files(
    settings: ClothingDesignAnalysisSettings,
) -> tuple[Path, Path]:
    """Hugging Face 캐시에서 WD14 ONNX 모델과 라벨 CSV 경로를 준비한다."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as error:
        raise ClothingDesignAnalysisError(
            "WD14 모델 다운로드에 필요한 huggingface-hub가 없습니다."
        ) from error

    def resolve_cached_or_downloaded_file(filename: str) -> str:
        try:
            return hf_hub_download(
                repo_id=settings.model_id,
                filename=filename,
                cache_dir=str(settings.cache_dir),
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return hf_hub_download(
                repo_id=settings.model_id,
                filename=filename,
                cache_dir=str(settings.cache_dir),
            )

    try:
        model_path = resolve_cached_or_downloaded_file(
            settings.model_filename
        )
        label_path = resolve_cached_or_downloaded_file(
            settings.label_filename
        )
    except Exception as error:
        raise ClothingDesignAnalysisError(
            "WD14 모델 파일을 준비하지 못했습니다. "
            "인터넷 연결과 D:/genai-cache 여유 공간을 확인하세요."
        ) from error
    return Path(model_path), Path(label_path)


def load_wd14_tag_labels(label_path: Path) -> tuple[ClothingTagLabel, ...]:
    """selected_tags.csv를 이름과 category 숫자가 있는 라벨로 읽는다."""
    try:
        with label_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = tuple(DictReader(stream))
    except (OSError, UnicodeError) as error:
        raise ClothingDesignAnalysisError(
            f"WD14 라벨 CSV를 읽을 수 없습니다: {label_path}"
        ) from error
    if not rows:
        raise ClothingDesignAnalysisError(
            "WD14 라벨 CSV에 데이터 행이 1개도 없습니다."
        )

    labels: list[ClothingTagLabel] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            name = str(row["name"]).strip()
            category = int(row["category"])
        except (KeyError, TypeError, ValueError) as error:
            raise ClothingDesignAnalysisError(
                "WD14 라벨 CSV 형식이 올바르지 않습니다. "
                f"행={row_number}"
            ) from error
        if not name:
            raise ClothingDesignAnalysisError(
                f"WD14 라벨 이름이 비어 있습니다. 행={row_number}"
            )
        labels.append(ClothingTagLabel(name=name, category=category))
    return tuple(labels)


def prepare_wd14_image(
    extracted_image: Image.Image,
    model_input_size: int,
) -> np.ndarray:
    """투명 의상을 흰 배경 정사각형에 배치하고 BGR float32 배열로 바꾼다."""
    if model_input_size < 1:
        raise ClothingDesignAnalysisError(
            "WD14 모델 입력 크기는 1픽셀 이상이어야 합니다."
        )
    rgba_image = extracted_image.convert("RGBA")
    crop_box = rgba_image.getchannel("A").getbbox()
    if crop_box is None:
        rgba_image.close()
        raise ClothingDesignAnalysisError(
            "WD14에 전달할 의상 알파 픽셀이 1개도 없습니다."
        )
    cropped_image = rgba_image.crop(crop_box)
    white_canvas = Image.new(
        "RGBA",
        cropped_image.size,
        (255, 255, 255, 255),
    )
    try:
        white_canvas.alpha_composite(cropped_image)
        rgb_image = white_canvas.convert("RGB")
    finally:
        rgba_image.close()
        cropped_image.close()
        white_canvas.close()

    maximum_dimension = max(rgb_image.size)
    square_image = Image.new(
        "RGB",
        (maximum_dimension, maximum_dimension),
        (255, 255, 255),
    )
    left = (maximum_dimension - rgb_image.width) // 2
    top = (maximum_dimension - rgb_image.height) // 2
    square_image.paste(rgb_image, (left, top))
    rgb_image.close()
    try:
        resized_image = square_image.resize(
            (model_input_size, model_input_size),
            Image.Resampling.BICUBIC,
        )
    finally:
        square_image.close()
    try:
        rgb_array = np.asarray(resized_image, dtype=np.float32)
        bgr_array = np.ascontiguousarray(rgb_array[:, :, ::-1])
        return np.expand_dims(bgr_array, axis=0)
    finally:
        resized_image.close()


def build_general_tag_candidates(
    tag_labels: tuple[ClothingTagLabel, ...],
    model_scores: np.ndarray,
    score_threshold: float,
    maximum_tag_count: int,
) -> tuple[ClothingDesignTagCandidate, ...]:
    """등급·캐릭터 라벨을 제외하고 기준 점수 이상의 일반 태그만 정렬한다."""
    if len(tag_labels) != len(model_scores):
        raise ClothingDesignAnalysisError(
            "태그 후보 계산의 라벨 수와 점수 수가 다릅니다."
        )
    general_candidates = [
        ClothingDesignTagCandidate(
            tag_name=label.name,
            display_name=label.name.replace("_", " "),
            score=float(score),
        )
        for label, score in zip(tag_labels, model_scores)
        if label.category == 0 and float(score) >= score_threshold
    ]
    general_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return tuple(general_candidates[:maximum_tag_count])


def _validate_analysis_settings(
    settings: ClothingDesignAnalysisSettings,
) -> None:
    """WD14 실행 전에 모델·파일·점수·후보 수 6개 설정을 검사한다."""
    if not settings.model_id.strip():
        raise ClothingDesignAnalysisError("WD14 모델 ID가 비어 있습니다.")
    if not settings.model_filename.strip() or not settings.label_filename.strip():
        raise ClothingDesignAnalysisError(
            "WD14 모델 또는 라벨 파일 이름이 비어 있습니다."
        )
    if settings.execution_provider != "CPUExecutionProvider":
        raise ClothingDesignAnalysisError(
            "현재 WD14 실행 장치는 CPUExecutionProvider만 허용합니다."
        )
    if not 0.0 <= settings.score_threshold <= 1.0:
        raise ClothingDesignAnalysisError(
            "WD14 태그 기준 점수는 0.0~1.0이어야 합니다."
        )
    if not 1 <= settings.maximum_tag_count <= 100:
        raise ClothingDesignAnalysisError(
            "WD14 최대 태그 수는 1개~100개여야 합니다."
        )
