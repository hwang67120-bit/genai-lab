import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog, QDialog

from genai_lab.external_candidate import (
    ExternalCandidateInputError,
    load_external_character_candidate,
)
from genai_lab.final_candidate_review import (
    APPROVED,
    create_final_review_decision,
)
from gui_main import GenAILabWindow


def test_external_candidate_is_loaded_as_unverified_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "edited-result.png"
    Image.new("RGBA", (40, 72), (20, 40, 60, 128)).save(image_path)

    candidate = load_external_character_candidate(
        image_path,
        reference_image_name="character.png",
        clothing_reference_name="outfit.jpg",
        framing_type="full_body",
    )
    try:
        assert candidate.image.mode == "RGB"
        assert candidate.image.size == (40, 72)
        assert candidate.model_id == "external_editor_unreported"
        assert candidate.seed == -1
        assert candidate.reference_quality_status == "external_import_unverified"
        assert candidate.design_reference_record == {
            "source": "external_result_import",
            "source_file_name": "edited-result.png",
            "approval": "pending",
            "automated_semantic_gate": "not_run",
            "automated_gender_gate": "not_run",
            "automated_color_gate": "not_run",
            "automated_structure_gate": "not_run",
            "automated_similarity_gate": "not_run",
            "outside_pixel_preservation": "not_verified",
        }
        assert "검사를 실행하지 않았습니다" in (
            candidate.detail_verification_warning_ko or ""
        )
    finally:
        candidate.image.close()


def test_external_candidate_rejects_non_image(tmp_path: Path) -> None:
    invalid_path = tmp_path / "not-an-image.png"
    invalid_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ExternalCandidateInputError, match="이미지로 읽을 수 없습니다"):
        load_external_character_candidate(
            invalid_path,
            reference_image_name=None,
            clothing_reference_name=None,
            framing_type="full_body",
        )
