import os
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout, QLabel

from genai_lab.clothing_reference import (
    ClothingMaskExtractionResult, ClothingMaskExtractionSettings,
    ClothingMaskRegionCandidateGroup, ClothingMaskReviewCandidate, ClothingRegionCandidate,
)
from genai_lab.target_mask_review import TargetMaskReviewDialog, fit_review_dialog
from genai_lab.workflow import GenerationWorkflowContext
from gui_main import (
    ClothingRegionReviewDialog, ClothingMaskReviewDialog,
    ClothingMaskExtractionWorker, GenAILabWindow, create_pil_image_pixmap,
)

# Keep one application alive across all GUI cases; destroying it between
# worker tests invalidates Qt's queued-signal delivery.
APPLICATION = QApplication.instance() or QApplication([])


def make_dialog(source):
    return TargetMaskReviewDialog(
        source, ClothingMaskExtractionSettings(),
        ClothingRegionReviewDialog, ClothingMaskReviewDialog,
        ClothingMaskExtractionWorker, create_pil_image_pixmap,
    )


def test_explicit_protection_review_and_conflict_recovery():
    app = QApplication.instance() or QApplication([])
    with Image.new("RGB", (64, 128), "white") as source:
        dialog = make_dialog(source)
        try:
            assert not dialog.approve_button.isEnabled()
            dialog.clothing_mask.paste(255, (8, 8, 16, 32))
            dialog._refresh()
            assert not dialog.approve_button.isEnabled()
            dialog._no_special_protection()
            assert dialog.approve_button.isEnabled()
            dialog.protection_mask.putpixel((10, 10), 255)
            dialog._refresh()
            assert not dialog.approve_button.isEnabled()
            assert "충돌=1px" in dialog.status.text()
            dialog._no_special_protection()
            dialog._approve()
            assert dialog.result() == QDialog.DialogCode.Accepted
            dialog.approved_masks.validate_source(source)
            assert len(dialog.findChildren(QScrollArea)) == 1
            assert dialog.approve_button.parentWidget() is dialog
        finally:
            dialog.close_images()
            dialog.close()
            app.processEvents()


def test_existing_large_region_dialog_scrolls_with_footer_outside():
    app = QApplication.instance() or QApplication([])
    with Image.new("RGB", (64, 128), "white") as source:
        owner = make_dialog(source)
        child = ClothingRegionReviewDialog(owner.source, None, "", owner)
        fit_review_dialog(child)
        child.show()
        app.processEvents()
        try:
            available = child.screen().availableGeometry()
            assert child.height() <= available.height()
            scroll = child.findChildren(QScrollArea)[0]
            footer = child.layout().itemAt(child.layout().count() - 1).layout()
            assert footer is not None
            assert not scroll.isAncestorOf(footer.itemAt(0).widget())
        finally:
            child.close()
            owner.close_images()
            owner.close()
            app.processEvents()


class FakeRegionDialog(QDialog):
    def __init__(self, source, *args):
        super().__init__()
        self.selected_candidates = (ClothingRegionCandidate((8, 8, 32, 40), "test", 1),)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("test"))
        footer = QHBoxLayout()
        footer.addWidget(QPushButton("continue"))
        layout.addLayout(footer)
    def exec(self):
        return QDialog.DialogCode.Accepted


class FakeMaskDialog(FakeRegionDialog):
    def __init__(self, source, result, *args):
        super().__init__(source)
        self.selected_candidates = (result.region_groups[0].candidates[0],)
        self.retry_region_selection = False


class FakeWorker(QObject):
    completed = Signal(object)
    failed = Signal(str, str)
    calls = 0
    def __init__(self, source, regions, settings):
        super().__init__()
        self.source, self.regions = source, regions
    @Slot()
    def run(self):
        type(self).calls += 1
        mask = Image.new("L", self.source.image.size, 0)
        mask.paste(255, (8, 8, 32, 40))
        candidate = ClothingMaskReviewCandidate(1, mask, 1, 768, 100, 1, 0)
        group = ClothingMaskRegionCandidateGroup(1, self.regions[0], (candidate,))
        self.completed.emit(ClothingMaskExtractionResult("test", "test", "test", mask.width, mask.height, (group,), 0))


def test_sam_worker_finishes_and_selected_candidate_is_owned():
    app = QApplication.instance() or QApplication([])
    FakeWorker.calls = 0
    with Image.new("RGB", (64, 128), "white") as source:
        dialog = TargetMaskReviewDialog(source, ClothingMaskExtractionSettings(),
            FakeRegionDialog, FakeMaskDialog, FakeWorker, create_pil_image_pixmap)
        dialog._select_region("clothing")
        dialog._select_region("clothing")  # running worker must not duplicate
        deadline = monotonic() + 5
        while dialog._thread is not None and monotonic() < deadline:
            app.processEvents()
        try:
            assert dialog._thread is None
            assert FakeWorker.calls == 1
            assert dialog.clothing_mask.getpixel((10, 10)) == 255
            dialog._no_special_protection()
            assert dialog.approve_button.isEnabled()
        finally:
            if dialog._thread is not None:
                dialog._thread.quit()
                dialog._thread.wait(5000)
                app.processEvents()
            dialog.close_images()
            dialog.close()
            app.processEvents()


def test_window_review_returns_owned_copy_and_cancel_returns_none(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.config = {}
    def accept(dialog):
        dialog.clothing_mask.paste(255, (8, 8, 16, 32))
        dialog._no_special_protection()
        dialog._approve()
        return dialog.result()
    with Image.new("RGB", (64, 128), "white") as source:
        monkeypatch.setattr(window, "execute_approval_dialog", accept)
        approved = window.review_target_character_masks(source)
        try:
            approved.validate_source(source)
            assert approved.clothing_mask.getpixel((10, 10)) == 255
        finally:
            approved.close()
        monkeypatch.setattr(window, "execute_approval_dialog", lambda dialog: QDialog.DialogCode.Rejected)
        assert window.review_target_character_masks(source) is None
    window.close()
    app.processEvents()


def test_cancel_target_selection_pauses_before_body_worker(monkeypatch):
    app = APPLICATION
    window = GenAILabWindow()
    window.config = {"clothing_try_on": {
        "python_executable": "python", "repository_path": ".", "temporary_root": ".",
        "cache_dir": ".", "width": 576, "height": 1024, "timeout_seconds": 1800,
    }}
    window.style_path = "character.png"
    window.outfit_path = "outfit.png"
    window.pending_clothing_base_candidate = SimpleNamespace(image=Image.new("RGB", (64, 128)))
    window.pending_clothing_extraction = object()
    window.confirmed_clothing_design = object()
    window.workflow_context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=Path("outfit.png"),
        pose_image_path=None,
    )
    calls = []
    monkeypatch.setattr(window, "review_target_character_masks", lambda source: calls.append(source) or None)
    try:
        window.start_character_body_comparison()
        assert len(calls) == 1
        assert window.body_comparison_worker_thread is None
        assert window.generate_button.isEnabled()
        assert window.generate_button.text() == "실패 단계 다시 시도"
    finally:
        window.pending_clothing_base_candidate.image.close()
        window.pending_clothing_base_candidate = None
        window.pending_clothing_extraction = None
        window.confirmed_clothing_design = None
        window.workflow_context = None
        window.close()
        app.processEvents()


def test_open_approval_prevents_reentrant_target_review(monkeypatch):
    window = GenAILabWindow()
    window.approval_dialog_open = True
    def unexpected(source):
        raise AssertionError("duplicate target dialog")
    monkeypatch.setattr(window, "review_target_character_masks", unexpected)
    window.start_character_body_comparison()
    window.approval_dialog_open = False
    window.close()
    APPLICATION.processEvents()
