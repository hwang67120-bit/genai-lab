"""A/B/C의 CPU 규칙 검사다. 원본 보존·제품 승인·참고 태그를 확인한다."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPlainTextEdit, QCheckBox, QPushButton, QFileDialog
from PySide6.QtCore import QPoint, QSettings
from genai_lab import studio_background as bg
from genai_lab import studio_controller as ui
from genai_lab import studio_generation as service
from genai_lab.studio_garment_warnings import GarmentWarningSettings, garment_warnings
from genai_lab.onepass_prompt_settings import NEGATIVE_TEMPLATE
from gui_main import GenAILabWindow
from test_studio_generation import batch_at, analysis_at, request_at, Backend, wait_for


class Foreground:
    def __init__(self, options=None):
        self.closed = False
    def alpha(self, image):
        mask = np.full((image.height, image.width), 255, dtype=np.uint8)
        mask[:, :image.width//2] = 0
        mask[:, image.width//2] = 128
        return mask
    def close(self):
        self.closed = True


def products(batch):
    for candidate in batch.candidates:
        bg.prepare_candidate(candidate, Foreground(), bg.BackgroundOptions(Path("test-model")))
    return batch


def checks():
    return dict(character=True, garment=True, exposure=True, appendages=True)


def test_background_product_raw_record_unchanged_and_export_uses_product(tmp_path):
    batch = batch_at(tmp_path)
    original = [(c.path.read_bytes(), c.record_path.read_bytes()) for c in batch.candidates]
    products(batch)
    result = service.StudioResults(batch)
    assert result.current_path.name == "product.png"
    with Image.open(result.current_path) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((8, 0)) == (128, 137, 142)  # 이진 마스크가 아닌 부드러운 알파 값이다.
    result.approve(checks())
    saved = result.export(tmp_path/"saved.png")
    assert saved.read_bytes() == result.current_path.read_bytes()
    assert [(c.path.read_bytes(), c.record_path.read_bytes()) for c in batch.candidates] == original
    metadata = json.loads(saved.with_suffix(".review.json").read_text(encoding="utf-8"))
    assert metadata["sha256"] != metadata["raw_sha256"]
    assert metadata["background"]["product_sha256"] == metadata["sha256"]
    assert metadata["automatic_gates_executed"] is False


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_wrong_model_records_visible_raw_fallback_without_loading(tmp_path, missing):
    batch = batch_at(tmp_path)
    cache = tmp_path/"cache"
    if not missing:
        model = cache/bg.MODEL_RELATIVE
        model.parent.mkdir(parents=True)
        model.write_bytes(b"wrong model")
    bg.prepare_backgrounds(batch, cache, foreground_factory=lambda _: pytest.fail("Must not load"))
    result = service.StudioResults(batch)
    assert result.current_path == result.candidate.path
    assert "배경 정리 안 됨" in result.background_notice
    assert all(bg.read_background(c)["status"] == "failed" for c in batch.candidates)
    result.approve(checks())
    assert result.export(tmp_path/"raw-fallback.png").read_bytes() == result.candidate.path.read_bytes()


@pytest.mark.parametrize("target", ["product.png", "background.json", "raw"])
def test_changed_background_product_record_or_raw_cannot_be_exported(tmp_path, target):
    result = service.StudioResults(products(batch_at(tmp_path)))
    result.approve(checks())
    file = result.candidate.path if target == "raw" else result.candidate.path.parent/target
    file.write_bytes(b'{}' if target == "background.json" else b"changed")
    with pytest.raises((ValueError, KeyError)):
        result.export(tmp_path/"bad.png")
    assert not (tmp_path/"bad.png").exists()


def test_background_failure_keeps_other_candidates_and_closes_model(tmp_path, monkeypatch):
    batch = batch_at(tmp_path)
    model = tmp_path/"cache"/bg.MODEL_RELATIVE
    model.parent.mkdir(parents=True)
    model.write_bytes(b"test")
    options_type = bg.BackgroundOptions
    monkeypatch.setattr(bg, "BackgroundOptions", lambda p: options_type(p, bg.sha(p)))
    class FailingOnce(Foreground):
        count = 0
        def alpha(self, image):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("CPU inference failed")
            return super().alpha(image)
    foreground = FailingOnce()
    bg.prepare_backgrounds(batch, tmp_path/"cache", foreground_factory=lambda _: foreground)
    assert foreground.closed
    assert [bg.read_background(c)["status"] for c in batch.candidates] == ["completed","failed","completed","completed"]
    result = service.StudioResults(batch)
    result.select(1)
    assert result.current_path == result.candidate.path
    assert "배경 정리 안 됨" in result.background_notice


def test_cancel_background_keeps_raw_and_does_not_publish_success(tmp_path):
    batch = batch_at(tmp_path)
    raw = [c.path.read_bytes() for c in batch.candidates]
    with pytest.raises(service.OnePassCancelled):
        bg.prepare_backgrounds(batch, tmp_path, cancelled=lambda: True,
            foreground_factory=lambda _: pytest.fail("cancelled"))
    assert raw == [c.path.read_bytes() for c in batch.candidates]
    assert not any((c.path.parent/"product.png").exists() for c in batch.candidates)


def test_background_rerun_refuses_to_overwrite(tmp_path):
    batch = products(batch_at(tmp_path))
    before = batch.candidates[0].path.parent.joinpath("product.png").read_bytes()
    with pytest.raises(FileExistsError):
        products(batch)
    assert batch.candidates[0].path.parent.joinpath("product.png").read_bytes() == before


@pytest.mark.parametrize("tags, rule", [
    (("black pants","miniskirt"),"bottom_types"),
    (("trousers","shorts"),"bottom_types"),
    (("pants","thighhighs","pantyhose","kneehighs"),"pants_legwear"),
    (("hands_in_pockets","arms crossed"),"garment_pose"),
    (("side-tie panties","underwear","buruma"),"negative_overlap"),
])
def test_all_configured_warnings_leave_tags_and_negative_unchanged(tags, rule):
    before = tuple(tags)
    negative = NEGATIVE_TEMPLATE
    warnings = garment_warnings(tags, negative)
    assert rule in [w["rule"] for w in warnings]
    assert tags == before and negative == NEGATIVE_TEMPLATE
    if rule == "negative_overlap":
        assert {w["term"] for w in warnings} == {"panties","underwear","buruma"}


def test_warning_terms_use_word_boundaries_and_custom_settings(tmp_path):
    assert garment_warnings(("underpants","skirt"), NEGATIVE_TEMPLATE) == []
    config = tmp_path/"rules.json"
    config.write_text(json.dumps({"negative_overlap":False,"rules":[{
        "id":"custom","terms":["hat"],"message":"custom warning"}]}))
    warnings = garment_warnings(("hat","panties","pants","skirt"), NEGATIVE_TEMPLATE,
                                GarmentWarningSettings.load(config))
    assert [w["rule"] for w in warnings] == ["custom"]


def test_approval_records_overlap_without_rewriting_negative(tmp_path):
    analysis = analysis_at(tmp_path/"inputs")
    analysis["garment_tags"] = ["panties","underwear","white shirt"]
    request = request_at(tmp_path, analysis)
    negative_before = service.prepare_onepass_gender(analysis["source"],
        ("blue_hair","medium_breasts","raccoon_ears"), NEGATIVE_TEMPLATE,
        settings=QSettings(str(tmp_path/"test-only.ini"), QSettings.Format.IniFormat)).negative_prompt
    assert request.inputs.prompt.negative == negative_before
    approval = json.loads((Path(analysis["directory"])/"approval.json").read_text(encoding="utf-8"))
    assert {w["term"] for w in approval["garment_review"]["warnings"]} == {"panties","underwear"}
    assert approval["garment_review"]["automatic_removal"] is False


def test_input_dialog_live_warnings_and_user_only_removal(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path/"inputs")
    analysis["garment_tags"] = ["pants","skirt","hands in pockets","side-tie panties"]
    monkeypatch.setattr(ui, "load_character_gender", lambda _: "male")
    def inspect(dialog):
        editor = dialog.findChild(QPlainTextEdit, "garment_tags_editor")
        warning = dialog.findChild(QLabel, "garment_warnings")
        assert editor.toPlainText() == ", ".join(analysis["garment_tags"])
        assert "하의 종류" in warning.text() and "자세" in warning.text() and "부정 문구" in warning.text()
        check = next(c for c in dialog.findChildren(QCheckBox) if c.text().startswith("얼굴에 팔"))
        check.setChecked(True)
        editor.setPlainText("pants")  # 사용자 편집만 적용한다.
        assert not check.isChecked() and warning.text() == ""
        check.setChecked(True)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", inspect)
    assert ui.confirm_inputs(None, analysis)[1] == ("pants",)


def test_gui_preview_approval_export_and_raw_button_use_separate_images(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.show()
    result_batch = products(batch_at(tmp_path))
    window.studio.analysis = analysis_at(tmp_path/"reference")
    try:
        window.studio.generated(result_batch)
        window.resize(1024,700)
        app.processEvents()
        assert window.studio.results.current_path.name == "product.png"
        assert window.studio_raw_button.isEnabled()
        assert "장식" in window.studio_background_notice.text()
        for control in (window.save_candidate_button, window.studio_raw_button):
            origin = control.mapTo(window.centralWidget(), QPoint(0,0))
            assert window.centralWidget().rect().contains(origin + QPoint(control.width()-1,control.height()-1))
        preview_bottom = window.candidate_preview.mapTo(window.centralWidget(), QPoint(0, window.candidate_preview.height()-1)).y()
        action_top = window.open_original_size_button.mapTo(window.centralWidget(), QPoint(0,0)).y()
        assert preview_bottom < action_top
        def see_raw(dialog):
            pixmap = next(label.pixmap() for label in dialog.findChildren(QLabel) if not label.pixmap().isNull())
            assert pixmap.toImage().pixelColor(0,0).red() == 1
            return 0
        monkeypatch.setattr(QDialog, "exec", see_raw)
        window.studio_raw_button.click()
        seen = []
        def review(parent, candidate, analysis):
            seen.append(candidate.path)
            return checks()
        monkeypatch.setattr(ui, "confirm_result", review)
        destination = tmp_path/"gui-saved.png"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination),""))
        expected = window.studio.results.current_path
        window.approve_candidate_button.click()
        window.save_candidate_button.click()
        assert seen == [expected] and destination.read_bytes() == expected.read_bytes()
    finally:
        window.close()


def test_tail_edit_accepts_white_product_basis_and_rejects_raw_basis(tmp_path):
    from test_qwen_tail import assets, finished_edit
    result = service.StudioResults(products(batch_at(tmp_path)))
    source, _ = assets(tmp_path/"source")
    raw_edit, _, _ = finished_edit(tmp_path/"wrong", source=source, basis=result.candidate.path)
    with pytest.raises(ValueError):
        result.adopt_tail_edit(raw_edit)
    edit, product, _ = finished_edit(tmp_path/"correct", source=source, basis=result.basis_path)
    result.adopt_tail_edit(edit)
    assert result.current_path == product
    result.approve(checks())
    assert result.export(tmp_path/"tail-export.png").read_bytes() == product.read_bytes()


def test_generate_button_releases_backend_before_cpu_background_and_records_warnings(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    analysis = analysis_at(tmp_path/"inputs")
    window.studio.output_root = tmp_path/"runs"
    window.style_path = Path(analysis["source"])
    window.selected_outfit_path = Path(analysis["directory"])/"garment.png"
    window.update_input_ready_status()
    events = []
    class TrackedBackend(Backend):
        def close(self):
            super().close()
            events.append("released")
    monkeypatch.setattr(ui, "validate_local_models", lambda _: None)
    monkeypatch.setattr(ui, "analyze_inputs", lambda *a, **kw: analysis)
    monkeypatch.setattr(ui, "confirm_inputs", lambda *a: (
        "male", ("pants", "skirt", "side-tie panties"), ui.AppearanceOverrides()))
    from test_studio_generation import Tokens
    from genai_lab.onepass_generation import generate_onepass_request
    build = service.build_request
    def build_cpu(a, tags, gender, **kw):
        return build(a, tags, gender, **kw, tokenizers=(Tokens(), Tokens()),
            seeds=(1,2,3,4), preferences=QSettings(str(tmp_path/"pref.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(ui, "build_request", build_cpu)
    monkeypatch.setattr(service.StudioRuntime, "from_environment", lambda: service.StudioRuntime(
        generation=replace(service.OnePassGenerationSettings(), width=16, height=24)))
    monkeypatch.setattr(ui, "generate_onepass_request", lambda req, directory, **kw:
        generate_onepass_request(req, directory, backend_factory=TrackedBackend, **kw))
    def prepare(batch, *a, **kw):
        assert events == ["released"]
        events.append("CPU background")
        return products(batch)
    monkeypatch.setattr(ui, "prepare_backgrounds", prepare)
    try:
        window.generate_button.click()
        wait_for(app, lambda: window.studio.task is None and window.studio.results is not None)
        assert events == ["released", "CPU background"]
        assert window.studio.results.current_path.name == "product.png"
        for candidate in window.studio.results.batch.candidates:
            report = json.loads((candidate.path.parent/"garment-review.json").read_text(encoding="utf-8"))
            assert {w["rule"] for w in report["warnings"]} == {"bottom_types","negative_overlap"}
            assert "panties" in report["negative"] and not report["automatic_removal"]
    finally:
        window.close()


def test_cancel_during_last_background_does_not_publish_completed_batch(tmp_path, monkeypatch):
    batch = batch_at(tmp_path)
    batch = replace(batch, candidates=batch.candidates[:1])
    model = tmp_path/"cache"/bg.MODEL_RELATIVE
    model.parent.mkdir(parents=True)
    model.write_bytes(b"test")
    options_type = bg.BackgroundOptions
    monkeypatch.setattr(bg, "BackgroundOptions", lambda p: options_type(p, bg.sha(p)))
    cancelled = []
    class Cancelling(Foreground):
        def alpha(self, image):
            cancelled.append(True)
            return super().alpha(image)
    foreground = Cancelling()
    with pytest.raises(service.OnePassCancelled):
        bg.prepare_backgrounds(batch, tmp_path/"cache", cancelled=lambda: bool(cancelled),
            foreground_factory=lambda _: foreground)
    assert foreground.closed and batch.candidates[0].path.is_file()
