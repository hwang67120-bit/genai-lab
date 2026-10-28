from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from genai_lab.final_candidate_review import (
    APPROVED,
    APPROVED_WITH_REFINEMENT,
    FinalReviewEvidence,
)
from genai_lab.final_result_storage import (
    DISCARDED,
    SAVED,
    FinalResultStorageError,
    create_storage_record,
    load_storage_record,
    write_storage_record,
)
from genai_lab.result import (
    CharacterGenerationCandidate,
    CharacterSaveError,
    save_approved_character_candidate,
)
from gui_main import GenAILabWindow


def candidate(decision: str) -> CharacterGenerationCandidate:
    return CharacterGenerationCandidate(
        image=Image.new("RGB", (32, 48), "navy"),
        original_generated_image=None,
        reference_image_name="character.png",
        before_clothing_image=None,
        clothing_change_mask=None,
        clothing_reference_name="outfit.png",
        clothing_category="jacket",
        clothing_try_on_status="completed",
        clothing_verification_warning_ko=None,
        reference_enhancement_applied=False,
        reference_enhancement_model_id=None,
        reference_quality_status="reviewed",
        framing_type="full_body",
        seed=1234,
        candidate_number=2,
        prompt="1person, blue jacket",
        negative_prompt="multiple people",
        model_id="flux2_klein",
        reference_adapter_id="native_multi_reference",
        original_image_change_strength=0.35,
        reference_image_strength=0.6,
        pose_control_status="not_requested",
        pose_control_model_id=None,
        pose_control_conditioning_scale=None,
        pose_control_guidance_start=None,
        pose_control_guidance_end=None,
        detail_correction_status="completed",
        detected_face_count=1,
        detected_hand_count=2,
        corrected_region_count=0,
        rejected_region_count=0,
        detail_verification_warning_ko=None,
        elapsed_seconds=12.5,
        peak_vram_bytes=1024,
        generated_at="2026-09-21T10:00:00+09:00",
        design_reference_record={
            "final_review_decision": {
                "decision": decision,
                "eligible_for_save": decision != "rejected",
                "training_use_approved": False,
                "refinement_targets": (
                    ["hair"] if decision == APPROVED_WITH_REFINEMENT else []
                ),
            },
        },
    )


def evidence(tmp_path: Path) -> FinalReviewEvidence:
    return FinalReviewEvidence(
        candidate_id="candidate-2",
        candidate_number=2,
        result_kind="refined_candidate",
        selected_engine="flux2_klein",
        sources={
            "character_reference": "character.png",
            "garment_reference": "outfit.png",
            "approved_base": "base.png",
            "final_candidate": "selected.png",
        },
        technical_safety={"status": "PASS"},
        comparisons=(),
        guidance={"status": "not_reported"},
        refinement_targets=(),
        report_path=str(tmp_path / "run.json"),
        output_directory=str(tmp_path / "audit"),
        created_at="2026-09-21T10:00:00+09:00",
    )


def test_saves_final_result_to_user_selected_root_with_reproducibility(
    tmp_path: Path,
) -> None:
    value = candidate(APPROVED)
    character = tmp_path / "character.png"
    garment = tmp_path / "outfit.png"
    character.write_bytes(b"character-reference")
    garment.write_bytes(b"garment-reference")
    selected_root = tmp_path / "chosen"

    try:
        result = save_approved_character_candidate(
            value,
            selected_root,
            character_reference_path=character,
            clothing_reference_path=garment,
            final_review_evidence={"candidate_id": "candidate-2"},
            final_review_decision=(
                value.design_reference_record["final_review_decision"]
            ),
        )
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

        assert result.storage_class == "final_result"
        assert result.image_path.is_relative_to(selected_root / "approved")
        assert metadata["version"] == "stage9_saved_result_v1"
        assert metadata["stage"] == 9
        assert metadata["training_use_approved"] is False
        assert metadata["final_review_evidence"]["candidate_id"] == "candidate-2"
        assert metadata["source_integrity"]["character_reference"]["sha256"] == (
            hashlib.sha256(b"character-reference").hexdigest()
        )
        assert metadata["source_integrity"]["clothing_reference"]["sha256"] == (
            hashlib.sha256(b"garment-reference").hexdigest()
        )
        assert metadata["image_sha256"] == hashlib.sha256(
            result.image_path.read_bytes()
        ).hexdigest()
    finally:
        value.image.close()


def test_conditional_approval_is_saved_as_refinement_checkpoint(
    tmp_path: Path,
) -> None:
    value = candidate(APPROVED_WITH_REFINEMENT)
    try:
        result = save_approved_character_candidate(
            value,
            tmp_path,
            final_review_decision=(
                value.design_reference_record["final_review_decision"]
            ),
        )
        assert result.storage_class == "refinement_checkpoint"
        assert result.image_path.is_relative_to(tmp_path / "refinement-pending")
    finally:
        value.image.close()


def test_rejected_candidate_cannot_be_saved(tmp_path: Path) -> None:
    value = candidate("rejected")
    try:
        with pytest.raises(CharacterSaveError, match="승인 또는 조건부 승인"):
            save_approved_character_candidate(
                value,
                tmp_path,
                final_review_decision=(
                    value.design_reference_record["final_review_decision"]
                ),
            )
    finally:
        value.image.close()


@pytest.mark.parametrize("status", [SAVED, DISCARDED])
def test_stage9_storage_choice_is_separate_from_training_consent(
    tmp_path: Path,
    status: str,
) -> None:
    kwargs = {}
    if status == SAVED:
        kwargs = {
            "storage_class": "final_result",
            "selected_output_root": tmp_path / "chosen",
            "image_path": tmp_path / "chosen" / "image.png",
            "metadata_path": tmp_path / "chosen" / "image.json",
        }
    record = create_storage_record(
        candidate_id="candidate-2",
        stage8_decision=APPROVED,
        status=status,
        **kwargs,
    )
    path = write_storage_record(record, tmp_path / "audit")
    loaded = load_storage_record(path)
    assert loaded.status == status
    assert loaded.training_use_approved is False


def test_stage9_rejects_storage_for_stage8_rejection() -> None:
    with pytest.raises(FinalResultStorageError, match="승인 또는 조건부 승인"):
        create_storage_record(
            candidate_id="candidate-2",
            stage8_decision="rejected",
            status=DISCARDED,
        )


def test_gui_uses_user_selected_output_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    application = QApplication.instance() or QApplication([])
    value = candidate(APPROVED)
    review = evidence(tmp_path)
    character = tmp_path / "character.png"
    garment = tmp_path / "outfit.png"
    Image.new("RGB", (8, 8), "blue").save(character)
    Image.new("RGB", (8, 8), "green").save(garment)
    selected = tmp_path / "user-selected"

    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(selected),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: None)

    window = GenAILabWindow()
    window.pending_character_candidate = value
    window.pending_final_review_evidence = review
    window.pending_final_review_orchestrator = None
    window.candidate_is_approved = True
    window.style_path = str(character)
    window.selected_outfit_path = garment
    try:
        window.save_approved_candidate()
        images = list((selected / "approved").rglob("*.png"))
        assert len(images) == 1
        assert window.pending_character_candidate is None
        audit = load_storage_record(
            Path(review.output_directory) / "stage9-storage-decision.json"
        )
        assert audit.status == SAVED
        assert Path(audit.selected_output_root) == selected
    finally:
        if window.pending_character_candidate is not None:
            window.release_pending_candidate("테스트 정리")
        window.close()
        application.processEvents()


def test_gui_canceling_directory_picker_keeps_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    application = QApplication.instance() or QApplication([])
    value = candidate(APPROVED)
    review = evidence(tmp_path)
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "",
    )
    window = GenAILabWindow()
    window.pending_character_candidate = value
    window.pending_final_review_evidence = review
    window.candidate_is_approved = True
    try:
        window.save_approved_candidate()
        assert window.pending_character_candidate is value
        assert "현재 후보를 유지" in window.status_label.text()
    finally:
        window.release_pending_candidate("테스트 정리")
        window.close()
        application.processEvents()
