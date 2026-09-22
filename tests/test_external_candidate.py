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


def test_gui_imports_external_result_into_existing_review_flow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    application = QApplication.instance() or QApplication([])
    image_path = tmp_path / "native-edit-candidate.png"
    Image.new("RGB", (64, 112), "navy").save(image_path)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(image_path), "이미지 파일"),
    )

    class ApprovedReviewDialog:
        def __init__(self, evidence, *args, **kwargs):
            self.selected_decision = create_final_review_decision(
                evidence,
                APPROVED,
            )

    monkeypatch.setattr(
        "gui_main.FinalCandidateReviewDialog",
        ApprovedReviewDialog,
    )
    monkeypatch.setattr(
        "gui_main.write_final_review_decision",
        lambda decision, output_directory: (
            tmp_path / "stage8-user-decision.json"
        ),
    )

    window = GenAILabWindow()
    monkeypatch.setattr(
        window,
        "execute_approval_dialog",
        lambda dialog: QDialog.DialogCode.Accepted,
    )
    window.style_path = "C:/input/character.png"
    window.selected_outfit_path = Path("C:/input/outfit.jpg")
    try:
        window.import_external_candidate()

        candidate = window.pending_character_candidate
        assert candidate is not None
        assert candidate.image.size == (64, 112)
        assert candidate.reference_image_name == "character.png"
        assert candidate.clothing_reference_name == "outfit.jpg"
        assert candidate.design_reference_record["approval"] == "pending"
        assert window.candidate_preview.pixmap() is not None
        assert window.approve_candidate_button.isEnabled()
        assert window.reject_candidate_button.isEnabled()
        assert not window.generate_button.isEnabled()
        assert not window.external_candidate_button.isEnabled()
        assert "자동 의미·성별·색상·구조·유사도 게이트" in (
            window.status_label.text()
        )

        window.approve_candidate()
        assert (
            window.pending_character_candidate.design_reference_record["approval"]
            == "user_approved"
        )
        assert window.save_candidate_button.isEnabled()
    finally:
        if window.pending_character_candidate is not None:
            window.release_pending_candidate("테스트 정리")
        window.close()
        application.processEvents()

