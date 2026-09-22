"""외부 편집 결과를 검증 상태가 명시된 GUI 후보로 변환한다."""

from datetime import datetime
from pathlib import Path

from PIL import Image

from genai_lab.result import CharacterGenerationCandidate


class ExternalCandidateInputError(ValueError):
    """외부 후보 이미지를 안전하게 읽을 수 없을 때 발생한다."""


def load_external_character_candidate(
    image_path: Path,
    *,
    reference_image_name: str | None,
    clothing_reference_name: str | None,
    framing_type: str,
) -> CharacterGenerationCandidate:
    """외부 편집 이미지를 자동 검증 미실행 상태의 후보로 불러온다."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise ExternalCandidateInputError(
            f"외부 편집 결과 파일을 찾을 수 없습니다: {image_path}"
        )

    try:
        with Image.open(image_path) as source_image:
            source_image.load()
            image = source_image.convert("RGB").copy()
    except (OSError, ValueError) as error:
        raise ExternalCandidateInputError(
            f"외부 편집 결과를 이미지로 읽을 수 없습니다: {error}"
        ) from error

    if image.width <= 0 or image.height <= 0:
        image.close()
        raise ExternalCandidateInputError("외부 편집 결과의 이미지 크기가 올바르지 않습니다.")

    imported_at = datetime.now().astimezone().isoformat()
    source_name = image_path.name
    unverified_warning = (
        "외부 편집 결과입니다. 의미·성별·색상·구조·유사도 게이트와 "
        "원본 픽셀 보존 검사를 실행하지 않았습니다."
    )
    return CharacterGenerationCandidate(
        image=image,
        original_generated_image=None,
        reference_image_name=reference_image_name or "not_registered",
        before_clothing_image=None,
        clothing_change_mask=None,
        clothing_reference_name=clothing_reference_name,
        clothing_category=None,
        clothing_try_on_status="external_import_pending_review",
        clothing_verification_warning_ko=unverified_warning,
        reference_enhancement_applied=False,
        reference_enhancement_model_id=None,
        reference_quality_status="external_import_unverified",
        framing_type=framing_type,
        seed=-1,
        candidate_number=0,
        prompt="",
        negative_prompt="",
        model_id="external_editor_unreported",
        reference_adapter_id="external_result_import",
        original_image_change_strength=0.0,
        reference_image_strength=0.0,
        pose_control_status="not_evaluated",
        pose_control_model_id=None,
        pose_control_conditioning_scale=None,
        pose_control_guidance_start=None,
        pose_control_guidance_end=None,
        detail_correction_status="not_evaluated",
        detected_face_count=0,
        detected_hand_count=0,
        corrected_region_count=0,
        rejected_region_count=0,
        detail_verification_warning_ko=unverified_warning,
        elapsed_seconds=0.0,
        peak_vram_bytes=0,
        generated_at=imported_at,
        design_reference_record={
            "source": "external_result_import",
            "source_file_name": source_name,
            "approval": "pending",
            "automated_semantic_gate": "not_run",
            "automated_gender_gate": "not_run",
            "automated_color_gate": "not_run",
            "automated_structure_gate": "not_run",
            "automated_similarity_gate": "not_run",
            "outside_pixel_preservation": "not_verified",
        },
    )
