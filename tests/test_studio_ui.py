"""CPU UI checks for reachable controls and preserved approval workflow."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog

from gui_main import GenAILabWindow
from genai_lab.studio_ui import ImagePreview


def create_window():
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.show()
    app.processEvents()
    return app, window


def test_studio_small_window_keeps_generation_and_review_actions_reachable():
    app, window = create_window()
    try:
        window.resize(1024, 700)
        app.processEvents()
        for control in (window.generate_button, window.approve_candidate_button,
                        window.save_candidate_button, window.open_original_size_button):
            origin = control.mapTo(window.centralWidget(), QPoint(0, 0))
            assert window.centralWidget().rect().contains(origin)
            assert window.centralWidget().rect().contains(origin + QPoint(control.width()-1, control.height()-1))
        assert not window.generate_button.isEnabled()
        assert not window.save_candidate_button.isEnabled()
        assert window.style_label.x() >= window.style_preview.geometry().right()
        window.diagnostics_toggle.click()
        app.processEvents()
        assert window.diagnostics_panel.isVisible()
        window.input_scroll.ensureWidgetVisible(window.refinement_diagnostics_button)
        app.processEvents()
        assert window.generate_button.isVisible()
        assert not window.refinement_diagnostics_button.isEnabled()
    finally:
        window.close()


def test_reference_selection_and_clear_update_previews_without_generation(tmp_path, monkeypatch):
    app, window = create_window()
    character, outfit = tmp_path / "character.png", tmp_path / "outfit.png"
    Image.new("RGB", (40, 80), "red").save(character)
    Image.new("RGB", (60, 60), "blue").save(outfit)
    paths = iter((character, outfit))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: (str(next(paths)), ""))
    try:
        window.style_button.click()
        window.outfit_button.click()
        app.processEvents()
        assert window.style_preview.pixmap().toImage().pixelColor(0, 0) == QColor("red")
        assert window.outfit_preview.pixmap().toImage().pixelColor(0, 0) == QColor("blue")
        assert "2/2" in window.status_label.text()
        assert window.generate_button.isEnabled()
        assert window.pipeline is None
        window.clear_outfit_button.click()
        app.processEvents()
        assert window.outfit_preview.pixmap().isNull()
        assert not window.style_preview.pixmap().isNull()
        assert "1/2" in window.status_label.text()
        assert not window.generate_button.isEnabled()
        assert not window.save_candidate_button.isEnabled()
    finally:
        window.close()


def test_image_preview_resize_uses_source_and_clear_stays_empty():
    app = QApplication.instance() or QApplication([])
    preview = ImagePreview()
    original = QPixmap(400, 800)
    original.fill(QColor("blue"))
    preview.resize(200, 200)
    preview.show()
    preview.setPixmap(original)
    app.processEvents()
    assert preview.pixmap().size().height() == 200
    preview.resize(500, 600)
    app.processEvents()
    assert preview.pixmap().size().height() == 600
    assert preview.pixmap().size().width() == 300
    preview.clear()
    preview.setText("empty")
    preview.resize(400, 500)
    app.processEvents()
    assert preview.pixmap().isNull()
    assert preview.text() == "empty"
    preview.close()


def test_studio_generation_and_optional_actions_keep_existing_routes(monkeypatch):
    calls = []
    def recorder(name):
        def action(self):
            calls.append(name)
        return action
    for method in ("start_generation", "open_qwen_pose_editor", "import_external_candidate"):
        monkeypatch.setattr(GenAILabWindow, method, recorder(method))
    app, window = create_window()
    try:
        window.style_path = "character.png"
        window.selected_outfit_path = Path("outfit.png")
        window.update_input_ready_status()
        window.generate_button.click()
        window.qwen_pose_button.click()
        window.external_candidate_button.click()
        assert calls == ["start_generation", "open_qwen_pose_editor", "import_external_candidate"]
        assert not window.save_candidate_button.isEnabled()
        assert not window.pose_button.isEnabled()
    finally:
        window.close()
