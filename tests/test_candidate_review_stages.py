"""Intermediate candidate selection must never look like final approval."""
import os
from types import SimpleNamespace
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import QPoint
from genai_lab.visual_reference_review import VisualCandidateReview
import pytest


@pytest.mark.parametrize("stage", ["native_base", "final"])
def test_candidate_stage_and_action_match_without_approving(stage, tmp_path):
    app = QApplication.instance() or QApplication([])
    image = Image.new("RGB", (50, 100), "white")
    record = {"similarity": {"character": .8, "hair": .7, "garment": .6,
                             "vibe": .8, "warnings": []}, "approval": "pending"}
    batch = SimpleNamespace(review_stage=stage, warning="diagnostic evidence",
                            directory=tmp_path, candidates=[SimpleNamespace(
                                image=image, seed=1, design_reference_record=record)],
                            paths=[tmp_path / "candidate.png"])
    dialog = VisualCandidateReview(batch)
    dialog.show()
    app.processEvents()
    try:
        assert "FLUX" not in dialog.windowTitle() + dialog.choose_button.text()
        assert not dialog.diagnostics_label.isVisible()
        assert "저장" in dialog.next_action_hint.text()
        if stage == "native_base":
            assert "의상 적용 전" in dialog.stage_banner.text()
            assert "최종 결과가 아닙니다" in dialog.stage_banner.text()
            assert "의상 적용하기" in dialog.choose_button.text()
            assert all("의상 적용 전" in label.text() for label in dialog.candidate_stage_labels)
        else:
            assert "의상 적용 전" not in dialog.stage_banner.text()
            assert "최종 검토" in dialog.choose_button.text()
        for widget in (dialog.stage_banner, dialog.choose_button):
            origin = widget.mapTo(dialog, QPoint(0, 0))
            assert dialog.rect().contains(origin)
            assert dialog.rect().contains(origin + QPoint(widget.width()-1, widget.height()-1))
        assert dialog.result() == QDialog.DialogCode.Rejected
        dialog.diagnostics_toggle.click()
        assert dialog.diagnostics_label.isVisible()
        dialog.choose_button.click()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert record["approval"] == "pending"
        assert record["part_review"]["hair"] == "not_reviewed"
    finally:
        dialog.close()
        image.close()
