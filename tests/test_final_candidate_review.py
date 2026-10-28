import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from genai_lab.final_candidate_review import (
    APPROVED,
    APPROVED_WITH_REFINEMENT,
    REJECTED,
    FinalCandidateReviewError,
    build_final_review_evidence,
    create_final_review_decision,
    load_final_review_evidence,
    write_final_review_decision,
    write_final_review_evidence,
)
from genai_lab.final_candidate_review_gui import FinalCandidateReviewDialog


def make_evidence(tmp_path: Path):
    base = tmp_path / "approved-base.png"
    garment = tmp_path / "input_garment.png"
    final = tmp_path / "final.png"
    for path, color in ((base, "white"), (garment, "navy"), (final, "black")):
        Image.new("RGB", (32, 48), color).save(path)
    report_path = tmp_path / "run.json"
    report_path.write_text("{}", encoding="utf-8")
    result = SimpleNamespace(
        status="PASS",
        selected_engine="flux2_klein",
        selected_image_path=final,
        report_path=report_path,
        output_directory=tmp_path,
        report={
            "attempts": {
                "flux2_klein": {
                    "gate": {
                        "status": "PASS",
                        "checks": {
                            "semantic_hard_safety": True,
                            "structure_hard_safety": True,
                            "integrity": True,
                            "color_target": False,
                        },
                        "blocking_checks": [],
                        "similarity_percentages": {
                            "character": 75.99,
                            "hair": 74.04,
                            "garment": 76.03,
                            "vibe": 79.50,
                        },
                        "color_percentage": 19.96,
                        "refinement_targets": ["color"],
                    },
                    "smoke_report": {
                        "inputs": {
                            "character": {"path": str(base)},
                            "garment": {"path": str(garment)},
                        }
                    },
                }
            },
            "person_count_diagnostic": {
                "status": "PASS",
                "detected_count": 1,
                "return_blocked": False,
            },
        },
    )
    selection = SimpleNamespace(
        candidate_number=2,
        base_image=base,
        garment_reference=garment,
    )
    request = SimpleNamespace(reference_image_name="character.png")
    return build_final_review_evidence(result, selection, request=request)


def test_builds_reference_specific_comparison_and_honest_guidance(tmp_path: Path):
    evidence = make_evidence(tmp_path)

    values = {item["component"]: item for item in evidence.comparisons}
    assert evidence.candidate_id == "candidate-2"
    assert values["character"]["reference"] == "character_reference"
    assert values["garment"]["reference"] == "garment_reference"
    assert values["color"]["value"] == 19.96
    assert evidence.technical_safety["person_count"] == 1
    assert evidence.guidance["status"] == "not_reported"
    assert evidence.guidance["applied"] == []
    assert "temporal_guidance" in evidence.guidance["not_reported"]
    assert evidence.refinement_targets == ("color",)


def test_conditional_approval_preserves_result_without_training_consent(tmp_path: Path):
    evidence = make_evidence(tmp_path)
    decision = create_final_review_decision(
        evidence,
        APPROVED_WITH_REFINEMENT,
    )

    assert decision.eligible_for_save is True
    assert decision.refinement_targets == ("color",)
    assert decision.next_action == "save_or_queue_refinement"
    assert decision.training_use_approved is False
    assert decision.automatic_retry is False


@pytest.mark.parametrize(
    ("reason", "action", "preserve_seed", "scope", "revision"),
    (
        ("garment_color_mismatch", "retry_same_seed_guidance", True, "color", False),
        ("aesthetic_preference", "retry_new_seed", False, None, False),
        ("reference_analysis_error", "return_to_input_review", True, None, True),
        ("stop_without_retry", "stop", False, None, False),
    ),
)
def test_rejection_reason_maps_to_one_explicit_action(
    tmp_path: Path,
    reason: str,
    action: str,
    preserve_seed: bool,
    scope: str | None,
    revision: bool,
):
    evidence = make_evidence(tmp_path)
    decision = create_final_review_decision(
        evidence,
        REJECTED,
        reason_code=reason,
    )

    assert decision.next_action == action
    assert decision.preserve_seed is preserve_seed
    assert decision.retry_scope == scope
    assert decision.requires_input_revision is revision
    assert decision.excluded_from_final_selection is True
    assert decision.eligible_for_save is False
    assert decision.training_use_approved is False


def test_rejection_requires_reason(tmp_path: Path):
    evidence = make_evidence(tmp_path)
    with pytest.raises(FinalCandidateReviewError, match="거절 사유"):
        create_final_review_decision(evidence, REJECTED)


def test_evidence_and_decision_are_written_separately(tmp_path: Path):
    evidence = make_evidence(tmp_path)
    evidence_path = write_final_review_evidence(evidence, tmp_path)
    loaded = load_final_review_evidence(evidence_path)
    decision = create_final_review_decision(loaded, APPROVED)
    decision_path = write_final_review_decision(decision, tmp_path)

    assert loaded.candidate_id == evidence.candidate_id
    saved = json.loads(decision_path.read_text(encoding="utf-8"))
    assert saved["decision"] == APPROVED
    assert saved["training_use_approved"] is False
    assert saved["evidence_sha256"]


def test_dialog_records_rejection_reason_without_auto_retry(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    evidence = make_evidence(tmp_path)
    image = Image.new("RGB", (32, 48), "black")
    dialog = FinalCandidateReviewDialog(
        evidence,
        image,
        initial_decision=REJECTED,
    )
    try:
        dialog.reason_combo.setCurrentIndex(
            dialog.reason_combo.findData("garment_color_mismatch")
        )
        dialog.confirm()

        assert dialog.selected_decision is not None
        assert dialog.selected_decision.decision == REJECTED
        assert dialog.selected_decision.retry_scope == "color"
        assert dialog.selected_decision.automatic_retry is False
    finally:
        image.close()
        dialog.close()
        app.processEvents()
