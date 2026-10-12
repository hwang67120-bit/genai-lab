"""작업실의 원본 편집형 의상 교체 흐름(GPU 없이): 체크 → 입력 확인 → 편집 → 결과 1장 → 승인·저장."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog

from genai_lab import outfit_edit_pipeline
from genai_lab import studio_controller as ui
from genai_lab.outfit_edit import sha
from genai_lab.outfit_edit_results import OutfitEditResults
from gui_main import GenAILabWindow
from test_studio_generation import analysis_at, wait_for


def settings_file(tmp_path):
    path = tmp_path / "qwen-settings.json"
    path.write_text(json.dumps(dict(python_executable=sys.executable, model_root=str(tmp_path), gguf_file=str(tmp_path / "m.gguf"),
                                    artifact_manifest=str(tmp_path / "models.json"), artifact_manifest_sha256="a" * 64)),
                    encoding="utf-8")
    return path


def fake_outcome(directory, composited=True):
    directory = Path(directory)
    (directory / "compose").mkdir(parents=True)
    (directory / "qwen").mkdir()
    Image.new("RGB", (768, 1024), (20, 30, 40)).save(directory / "qwen" / "raw.png")
    Image.new("RGB", (768, 1024), (50, 60, 70)).save(directory / "compose" / "qwen_view.png")
    outcome = dict(directory=str(directory), composited=composited, head_accessories="keep",
                   alignment={"decision": "aligned" if composited else "unmeasured"}, raw_sha256=sha(directory / "qwen" / "raw.png"))
    if composited:
        Image.new("RGB", (768, 1024), (90, 10, 10)).save(directory / "compose" / "result.png")
        outcome.update(result=str(directory / "compose" / "result.png"), result_sha256=sha(directory / "compose" / "result.png"))
    else:
        outcome.update(reason="머리 위치를 확인할 근거(얼굴 특징점)가 부족합니다.")
    return outcome


def test_checkboxes_default_and_exclusive_with_two_pass():
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    try:
        assert not window.studio_outfit_edit_checkbox.isChecked()
        assert window.studio_head_accessory_checkbox.isChecked() and not window.studio_head_accessory_checkbox.isEnabled()
        window.studio_outfit_edit_checkbox.setChecked(True)
        assert window.studio_head_accessory_checkbox.isEnabled() and "옷만 바꾸고" in window.studio_mode_note.text()
        window.studio_proportion_checkbox.setChecked(True)
        assert not window.studio_outfit_edit_checkbox.isChecked()            # 2단계 비율 생성과 함께 쓰지 않는다
        window.studio_outfit_edit_checkbox.setChecked(True)
        assert not window.studio_proportion_checkbox.isChecked()
    finally:
        window.close()


def test_gui_outfit_edit_runs_pipeline_and_saves_approved_result(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path / "inputs")
    window = GenAILabWindow()
    window.studio.output_root = tmp_path / "runs"
    window.style_path = Path(analysis["source"])
    window.selected_outfit_path = Path(analysis["directory"]) / "garment.png"
    window.studio_outfit_edit_checkbox.setChecked(True)
    window.studio_head_accessory_checkbox.setChecked(False)
    window.update_input_ready_status()
    monkeypatch.setattr(ui, "validate_local_models", lambda _: pytest.fail("원본 편집형은 SDXL 모델을 확인하지 않는다"))
    monkeypatch.setattr(ui, "analyze_inputs", lambda *a, **k: analysis)
    monkeypatch.setattr(ui, "confirm_inputs", lambda *a: pytest.fail("SDXL 입력 확인 화면을 띄우지 않는다"))
    monkeypatch.setattr(ui, "generate_onepass_request", lambda *a, **k: pytest.fail("SDXL 생성을 하지 않는다"))
    monkeypatch.setattr(ui, "confirm_outfit_edit", lambda window, a, accessories, path: str(settings_file(tmp_path)))
    monkeypatch.setattr(ui, "remember_qwen_settings", lambda path: None)   # 실제 사용자 설정에 시험 경로를 남기지 않는다
    calls = []
    def pipeline(character, garment, **kw):
        calls.append((Path(character).name, Path(garment).name, kw["head_accessories"]))
        return fake_outcome(kw["directory"])
    monkeypatch.setattr(outfit_edit_pipeline, "run_outfit_pipeline", pipeline)
    monkeypatch.setattr(ui, "confirm_result", lambda *a: dict(character=True, garment=True, exposure=True, appendages=True))
    destination = tmp_path / "saved.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
    try:
        window.generate_button.click()
        wait_for(app, lambda: window.studio.task is None and window.studio.results is not None)
        assert calls == [("character.png", "garment.png", "remove")]       # 머리 장식 체크를 끈 선택이 전달된다
        results = window.studio.results
        assert isinstance(results, OutfitEditResults) and results.composited
        assert not window.tail_edit_button.isVisible() and not window.studio_finish_button.isVisible()
        expected = results.current_path.read_bytes()
        window.approve_candidate_button.click()
        window.save_candidate_button.click()
        assert destination.read_bytes() == expected
        record = json.loads(destination.with_suffix(".review.json").read_text(encoding="utf-8"))
        assert record["mode"] == "outfit_edit" and record["head_composited"] and record["reviewer"] == "user"
    finally:
        window.close()


def test_cancelled_confirmation_never_runs_qwen(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path / "inputs")
    window = GenAILabWindow()
    window.studio.run_directory = tmp_path / "run"
    window.studio.run_directory.mkdir()
    window.studio.outfit_edit_enabled = True
    monkeypatch.setattr(ui, "confirm_outfit_edit", lambda *a: None)
    monkeypatch.setattr(outfit_edit_pipeline, "run_outfit_pipeline", lambda *a, **k: pytest.fail("취소했는데 실행했다"))
    try:
        window.studio.inputs_ready(analysis)
        assert window.studio.task is None and window.studio.results is None
        assert "취소" in window.status_label.text()
    finally:
        window.close()


def test_result_without_head_paste_is_shown_with_reason_and_can_still_be_judged(tmp_path):
    results = OutfitEditResults(fake_outcome(tmp_path / "run", composited=False))
    assert results.current_path.name == "qwen_view.png" and "붙이지 못했습니다" in results.head_notice
    with pytest.raises(ValueError):
        results.export(tmp_path / "x.png")                                 # 승인 전 저장 불가
    results.approve(dict(character=True, garment=True, exposure=True))
    assert results.export(tmp_path / "x.png").read_bytes() == results.current_path.read_bytes()


def test_changed_result_blocks_approval(tmp_path):
    results = OutfitEditResults(fake_outcome(tmp_path / "run"))
    Image.new("RGB", (768, 1024), "white").save(results.current_path)
    with pytest.raises(ValueError):
        results.approve(dict(character=True, garment=True, exposure=True))


def test_confirmation_explains_why_start_is_disabled(tmp_path, monkeypatch):
    # 사용자 GUI 확인(2026-10-12): 설정 파일을 고르지 않아 버튼이 꺼졌는데 이유가 보이지 않았다
    from PySide6.QtWidgets import QCheckBox, QDialog, QLabel, QPushButton
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path / "inputs")
    seen = {}
    def inspect(dialog):
        hint = dialog.findChild(QLabel, "outfit_edit_hint")
        start = [b for b in dialog.findChildren(QPushButton) if b.text() == "옷 바꾸기 시작"][0]
        seen["before"] = (start.isEnabled(), hint.text())
        dialog.findChildren(QCheckBox)[0].setChecked(True)
        seen["checked_only"] = (start.isEnabled(), hint.text())
        return QDialog.DialogCode.Rejected
    monkeypatch.setattr(QDialog, "exec", inspect)
    assert ui.confirm_outfit_edit(None, analysis, "keep", "") is None
    assert seen["before"][0] is False and "실행 환경 설정 파일" in seen["before"][1] and "확인 체크" in seen["before"][1]
    assert seen["checked_only"][0] is False and "실행 환경 설정 파일" in seen["checked_only"][1]


def test_settings_path_is_remembered_only_if_file_exists(tmp_path):
    from PySide6.QtCore import QSettings
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    path = settings_file(tmp_path)
    ui.remember_qwen_settings(path, store)
    assert ui.remembered_qwen_settings(store) == str(path)
    path.unlink()
    assert ui.remembered_qwen_settings(store) == ""
