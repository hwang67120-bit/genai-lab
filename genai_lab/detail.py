"""생성 후보의 얼굴과 손을 탐지하고 제한된 영역만 다시 그린다."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageOps

from genai_lab.style import prepare_ip_adapter_reference_image


class CharacterDetailType(str, Enum):
    """부분 보정이 지원되는 캐릭터 세부 영역."""

    FACE = "face"
    HAND = "hand"


@dataclass(frozen=True)
class CharacterDetailDetection:
    """YOLO가 생성 이미지에서 찾은 얼굴 또는 손 위치."""

    detail_type: CharacterDetailType
    bounding_box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class CharacterDetailCorrectionMask:
    """크기와 위치 검사를 마친 부분 보정 마스크."""

    detection: CharacterDetailDetection
    mask_image: Image.Image
    image_area_ratio: float
    correction_allowed: bool
    rejection_reason_ko: str | None


@dataclass(frozen=True)
class CharacterDetailCorrectionResult:
    """얼굴·손 부분 재생성과 보정 범위 검사가 끝난 결과."""

    corrected_image: Image.Image
    original_generated_image: Image.Image
    detected_face_count: int
    detected_hand_count: int
    corrected_region_count: int
    rejected_region_count: int
    pixels_changed_outside_mask: int
    verification_warning_ko: str | None


class CharacterDetailCorrectionError(RuntimeError):
    """세부 영역 탐지나 부분 재생성을 완료하지 못한 오류."""


def correct_character_candidate_details(
    generation_pipeline: Any,
    generated_image: Image.Image,
    approved_reference_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    detail_config: dict[str, Any],
    cache_dir: Path,
) -> CharacterDetailCorrectionResult:
    """YOLO로 얼굴·손을 찾고 허용된 영역만 Diffusers Inpaint로 보정한다.

    반환값:
        보정 전후 이미지와 탐지·검증 기록. 두 이미지는 파일로 저장하지 않는다.

    오류:
        탐지 모델 또는 Inpaint 실행에 실패하면 보정 전 이미지를 변경하지 않고
        CharacterDetailCorrectionError를 발생시킨다.
    """
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
    except ImportError as error:
        raise CharacterDetailCorrectionError(
            "세부 보정 도구가 없습니다. "
            "가상환경에서 'python -m pip install -r requirements.txt'를 실행하세요."
        ) from error

    original_generated_image = generated_image.convert("RGB").copy()
    detector_models = load_character_detail_detector_models(
        detail_config,
        cache_dir,
    )
    detections = detect_character_detail_regions(
        original_generated_image,
        detector_models,
        float(detail_config["minimum_confidence"]),
        int(detail_config["maximum_face_regions"]),
        int(detail_config["maximum_hand_regions"]),
    )
    correction_masks = create_character_detail_correction_masks(
        original_generated_image.size,
        detections,
        float(detail_config["mask_padding_ratio"]),
        float(detail_config["minimum_mask_area_ratio"]),
        float(detail_config["maximum_mask_area_ratio"]),
    )
    allowed_masks = [
        correction_mask
        for correction_mask in correction_masks
        if correction_mask.correction_allowed
    ]
    if not allowed_masks:
        return CharacterDetailCorrectionResult(
            corrected_image=original_generated_image.copy(),
            original_generated_image=original_generated_image,
            detected_face_count=count_detail_type(
                detections, CharacterDetailType.FACE
            ),
            detected_hand_count=count_detail_type(
                detections, CharacterDetailType.HAND
            ),
            corrected_region_count=0,
            rejected_region_count=len(correction_masks),
            pixels_changed_outside_mask=0,
            verification_warning_ko=(
                "보정 가능한 얼굴·손 영역을 찾지 못해 생성 원본을 유지했습니다."
            ),
        )

    combined_allowed_mask = Image.new("L", original_generated_image.size, 0)
    for correction_mask in allowed_masks:
        combined_allowed_mask = ImageChops.lighter(
            combined_allowed_mask,
            correction_mask.mask_image,
        )

    try:
        inpaint_pipeline = AutoPipelineForInpainting.from_pipe(
            generation_pipeline
        )
        inpaint_pipeline.enable_model_cpu_offload()
        corrected_image = original_generated_image.copy()
        ip_adapter_reference = prepare_ip_adapter_reference_image(
            approved_reference_image
        )
        for region_number, correction_mask in enumerate(allowed_masks, start=1):
            detail_prompt = create_detail_correction_prompt(
                prompt,
                correction_mask.detection.detail_type,
            )
            detail_negative_prompt = create_detail_correction_negative_prompt(
                negative_prompt,
                correction_mask.detection.detail_type,
            )
            random_start = torch.Generator(device="cpu").manual_seed(
                seed + region_number
            )
            proposed_image = inpaint_pipeline(
                prompt=detail_prompt,
                negative_prompt=detail_negative_prompt,
                image=corrected_image,
                mask_image=correction_mask.mask_image,
                ip_adapter_image=[ip_adapter_reference],
                strength=float(detail_config["inpaint_strength"]),
                num_inference_steps=int(detail_config["inpaint_steps"]),
                guidance_scale=float(detail_config["guidance_scale"]),
                padding_mask_crop=int(detail_config["padding_mask_crop"]),
                generator=random_start,
            ).images[0].convert("RGB")
            corrected_image = Image.composite(
                proposed_image,
                corrected_image,
                correction_mask.mask_image,
            )
            proposed_image.close()
    except Exception as error:
        raise CharacterDetailCorrectionError(
            "얼굴·손 부분 보정에 실패해 보정 전 후보를 유지합니다."
        ) from error
    finally:
        if "inpaint_pipeline" in locals():
            del inpaint_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pixels_changed_outside_mask = count_changed_pixels_outside_mask(
        original_generated_image,
        corrected_image,
        combined_allowed_mask,
    )
    if pixels_changed_outside_mask:
        corrected_image.close()
        raise CharacterDetailCorrectionError(
            "부분 보정 범위 밖의 픽셀이 변경되어 보정 전 후보로 복구했습니다."
        )

    detections_after_correction = detect_character_detail_regions(
        corrected_image,
        detector_models,
        float(detail_config["minimum_confidence"]),
        int(detail_config["maximum_face_regions"]),
        int(detail_config["maximum_hand_regions"]),
    )
    verification_warning_ko = create_detection_verification_warning(
        detections,
        detections_after_correction,
    )
    return CharacterDetailCorrectionResult(
        corrected_image=corrected_image,
        original_generated_image=original_generated_image,
        detected_face_count=count_detail_type(
            detections, CharacterDetailType.FACE
        ),
        detected_hand_count=count_detail_type(
            detections, CharacterDetailType.HAND
        ),
        corrected_region_count=len(allowed_masks),
        rejected_region_count=len(correction_masks) - len(allowed_masks),
        pixels_changed_outside_mask=pixels_changed_outside_mask,
        verification_warning_ko=verification_warning_ko,
    )


def load_character_detail_detector_models(
    detail_config: dict[str, Any],
    cache_dir: Path,
) -> dict[CharacterDetailType, Any]:
    """신뢰할 수 있는 저장소에서 얼굴·손 탐지 모델을 로컬 캐시에 준비한다."""
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError as error:
        raise CharacterDetailCorrectionError(
            "얼굴·손 탐지 도구가 없습니다. "
            "가상환경에서 'python -m pip install -r requirements.txt'를 실행하세요."
        ) from error

    detector_repository = str(detail_config["detector_repository"])
    model_filenames = {
        CharacterDetailType.FACE: str(detail_config["face_model"]),
        CharacterDetailType.HAND: str(detail_config["hand_model"]),
    }
    detector_models: dict[CharacterDetailType, Any] = {}
    for detail_type, model_filename in model_filenames.items():
        model_path = hf_hub_download(
            repo_id=detector_repository,
            filename=model_filename,
            cache_dir=str(cache_dir),
        )
        detector_models[detail_type] = YOLO(model_path)
    return detector_models


def detect_character_detail_regions(
    image: Image.Image,
    detector_models: dict[CharacterDetailType, Any],
    minimum_confidence: float,
    maximum_face_regions: int,
    maximum_hand_regions: int,
) -> list[CharacterDetailDetection]:
    """YOLO 실행 결과를 이름 있는 얼굴·손 위치 객체로 변환한다."""
    import numpy as np

    detections: list[CharacterDetailDetection] = []
    maximum_regions = {
        CharacterDetailType.FACE: maximum_face_regions,
        CharacterDetailType.HAND: maximum_hand_regions,
    }
    image_array = np.asarray(image.convert("RGB"))
    for detail_type, detector_model in detector_models.items():
        prediction = detector_model.predict(
            source=image_array,
            conf=minimum_confidence,
            device="cpu",
            verbose=False,
        )[0]
        detail_detections: list[CharacterDetailDetection] = []
        boxes = getattr(prediction, "boxes", None)
        if boxes is not None:
            for coordinates, confidence in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
            ):
                left, top, right, bottom = (
                    int(round(value)) for value in coordinates
                )
                detail_detections.append(
                    CharacterDetailDetection(
                        detail_type=detail_type,
                        bounding_box=(left, top, right, bottom),
                        confidence=round(float(confidence), 4),
                    )
                )
        detail_detections.sort(
            key=lambda detection: detection.confidence,
            reverse=True,
        )
        detections.extend(detail_detections[: maximum_regions[detail_type]])
    return detections


def create_character_detail_correction_masks(
    image_size: tuple[int, int],
    detections: list[CharacterDetailDetection],
    mask_padding_ratio: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
) -> list[CharacterDetailCorrectionMask]:
    """탐지 범위를 넓힌 뒤 지나치게 작거나 큰 마스크를 거절한다."""
    image_width, image_height = image_size
    image_area = image_width * image_height
    correction_masks: list[CharacterDetailCorrectionMask] = []
    for detection in detections:
        left, top, right, bottom = detection.bounding_box
        region_width = max(1, right - left)
        region_height = max(1, bottom - top)
        horizontal_padding = round(region_width * mask_padding_ratio)
        vertical_padding = round(region_height * mask_padding_ratio)
        expanded_box = (
            max(0, left - horizontal_padding),
            max(0, top - vertical_padding),
            min(image_width, right + horizontal_padding),
            min(image_height, bottom + vertical_padding),
        )
        expanded_width = max(1, expanded_box[2] - expanded_box[0])
        expanded_height = max(1, expanded_box[3] - expanded_box[1])
        area_ratio = (expanded_width * expanded_height) / image_area
        rejection_reason_ko = None
        if area_ratio < minimum_area_ratio:
            rejection_reason_ko = "탐지 영역이 너무 작아 안전하게 보정할 수 없습니다."
        elif area_ratio > maximum_area_ratio:
            rejection_reason_ko = "탐지 영역이 너무 커서 다른 디자인까지 바뀔 수 있습니다."

        mask_image = Image.new("L", image_size, 0)
        mask_draw = ImageDraw.Draw(mask_image)
        mask_draw.rectangle(
            (
                expanded_box[0],
                expanded_box[1],
                max(expanded_box[0], expanded_box[2] - 1),
                max(expanded_box[1], expanded_box[3] - 1),
            ),
            fill=255,
        )
        correction_masks.append(
            CharacterDetailCorrectionMask(
                detection=detection,
                mask_image=mask_image,
                image_area_ratio=round(area_ratio, 6),
                correction_allowed=rejection_reason_ko is None,
                rejection_reason_ko=rejection_reason_ko,
            )
        )
    return correction_masks


def create_detail_correction_prompt(
    original_prompt: str,
    detail_type: CharacterDetailType,
) -> str:
    """기존 캐릭터 조건을 유지하면서 보정 영역의 정상 형태를 요청한다."""
    detail_prompt = {
        CharacterDetailType.FACE: (
            "detailed face, symmetrical eyes, complete pupils, clean facial features"
        ),
        CharacterDetailType.HAND: (
            "detailed hand, five fingers, natural finger joints, natural hand pose"
        ),
    }[detail_type]
    return f"{original_prompt}, {detail_prompt}, same character design"


def create_detail_correction_negative_prompt(
    original_negative_prompt: str,
    detail_type: CharacterDetailType,
) -> str:
    """기존 제외 조건에 얼굴 또는 손의 실패 형태를 추가한다."""
    detail_negative_prompt = {
        CharacterDetailType.FACE: (
            "empty eyes, missing pupils, uneven eyes, deformed eyes, blurred face"
        ),
        CharacterDetailType.HAND: (
            "missing fingers, extra fingers, fused fingers, malformed hands, "
            "broken finger joints"
        ),
    }[detail_type]
    return f"{original_negative_prompt}, {detail_negative_prompt}"


def count_changed_pixels_outside_mask(
    original_image: Image.Image,
    corrected_image: Image.Image,
    allowed_mask: Image.Image,
) -> int:
    """허용 마스크 바깥에서 달라진 픽셀 수를 정확히 계산한다."""
    import numpy as np

    original_pixels = np.asarray(original_image.convert("RGB"))
    corrected_pixels = np.asarray(corrected_image.convert("RGB"))
    outside_allowed_mask = np.asarray(allowed_mask.convert("L")) == 0
    changed_pixels = np.any(original_pixels != corrected_pixels, axis=2)
    return int(np.count_nonzero(changed_pixels & outside_allowed_mask))


def count_detail_type(
    detections: list[CharacterDetailDetection],
    detail_type: CharacterDetailType,
) -> int:
    """탐지 결과에서 지정한 얼굴 또는 손의 개수를 센다."""
    return sum(
        detection.detail_type is detail_type
        for detection in detections
    )


def create_detection_verification_warning(
    detections_before_correction: list[CharacterDetailDetection],
    detections_after_correction: list[CharacterDetailDetection],
) -> str | None:
    """보정 뒤 얼굴이나 손 탐지 수가 줄었을 때 사용자 확인 경고를 만든다."""
    warnings: list[str] = []
    for detail_type, korean_name in (
        (CharacterDetailType.FACE, "얼굴"),
        (CharacterDetailType.HAND, "손"),
    ):
        count_before = count_detail_type(
            detections_before_correction,
            detail_type,
        )
        count_after = count_detail_type(
            detections_after_correction,
            detail_type,
        )
        if count_after < count_before:
            warnings.append(
                f"{korean_name} 탐지 수가 보정 전 {count_before}개에서 "
                f"보정 후 {count_after}개로 줄었습니다."
            )
    if not warnings:
        return None
    return " ".join(warnings)

