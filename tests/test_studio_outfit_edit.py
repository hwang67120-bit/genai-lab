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


def test_generation_settings_are_automatic():
    # 사용자 GUI 확인(2026-10-12): 옷 바꾸기가 기본이고(D-100), 생성 설정은 수동 체크 없이 자동이다(D-105)
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    try:
        assert window.studio_method_combo.currentData() == "edit"
        for hidden in (window.studio_method_combo, window.studio_head_accessory_checkbox,
                       window.studio_proportion_checkbox, window.studio_shoulder_checkbox):
            assert hidden.isHidden()
        assert "자동" in window.studio_mode_note.text() and "한 번 더" in window.studio_mode_note.text()
    finally:
        window.close()


def test_gui_outfit_edit_runs_pipeline_and_saves_approved_result(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path / "inputs")
    window = GenAILabWindow()
    window.studio.output_root = tmp_path / "runs"
    window.style_path = Path(analysis["source"])
    window.selected_outfit_path = Path(analysis["directory"]) / "garment.png"
    window.update_input_ready_status()
    monkeypatch.setattr(ui, "validate_local_models", lambda _: pytest.fail("원본 편집형은 SDXL 모델을 확인하지 않는다"))
    monkeypatch.setattr(ui, "analyze_inputs", lambda *a, **k: analysis)
    monkeypatch.setattr(ui, "confirm_inputs", lambda *a: pytest.fail("SDXL 입력 확인 화면을 띄우지 않는다"))
    monkeypatch.setattr(ui, "generate_onepass_request", lambda *a, **k: pytest.fail("SDXL 생성을 하지 않는다"))
    asked = []
    monkeypatch.setattr(ui, "confirm_outfit_edit", lambda window, a, accessories, saved_gender=None, saved_body_parts=None:
                        asked.append((saved_gender, saved_body_parts)) or dict(gender="male", body_parts=("tail",)))
    saved, saved_parts = {}, {}
    monkeypatch.setattr(ui, "load_character_gender", lambda source: None)
    monkeypatch.setattr(ui, "load_character_body_parts", lambda source: None)
    monkeypatch.setattr(ui, "save_character_gender", lambda source, gender: saved.update({source: gender}))
    monkeypatch.setattr(ui, "save_character_body_parts", lambda source, parts: saved_parts.update({source: parts}))
    from genai_lab import qwen_outfit_edit
    loaded = []
    monkeypatch.setattr(qwen_outfit_edit, "load_qwen_settings", lambda path: loaded.append(Path(path).name) or object())
    calls = []
    def pipeline(character, garment, **kw):
        calls.append((Path(character).name, Path(garment).name, kw["head_accessories"], kw["gender"], kw["body_parts"]))
        return fake_outcome(kw["directory"])
    monkeypatch.setattr(outfit_edit_pipeline, "run_outfit_pipeline", pipeline)
    monkeypatch.setattr(ui, "confirm_result", lambda *a: dict(character=True, garment=True, exposure=True, appendages=True,
                                                               original_outfit_gone=True))
    destination = tmp_path / "saved.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
    try:
        window.generate_button.click()
        wait_for(app, lambda: window.studio.task is None and window.studio.results is not None)
        assert calls == [("character.png", "garment.png", "keep", "male", ("tail",))]  # 머리 장식 항상 유지(D-105), 성별(D-106)·부위(D-111) 전달
        assert asked == [(None, None)] and list(saved.values()) == ["male"]   # 처음 보는 캐릭터는 고른 성별·부위를 저장한다
        assert list(saved_parts.values()) == [("tail",)]
        assert loaded == ["qwen_edit_settings.json"]                         # 사용자가 고르지 않고 작업실 기본 설정을 쓴다(D-101)
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
    monkeypatch.setattr(ui, "confirm_outfit_edit", lambda *a, **k: None)
    monkeypatch.setattr(ui, "load_character_gender", lambda source: None)
    monkeypatch.setattr(ui, "load_character_body_parts", lambda source: None)
    from genai_lab import qwen_outfit_edit
    monkeypatch.setattr(qwen_outfit_edit, "load_qwen_settings", lambda path: object())
    window.studio.runtime = ui.StudioRuntime()
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
    results.approve(dict(character=True, garment=True, exposure=True, original_outfit_gone=True))
    assert results.export(tmp_path / "x.png").read_bytes() == results.current_path.read_bytes()
    record = json.loads((results.directory / "judgement.json").read_text(encoding="utf-8"))[-1]   # D-114
    assert record["judge"] == "user" and record["source"] == "gui_approval" and record["original_garment_remaining"] == "none"


def test_changed_result_blocks_approval(tmp_path):
    results = OutfitEditResults(fake_outcome(tmp_path / "run"))
    Image.new("RGB", (768, 1024), "white").save(results.current_path)
    with pytest.raises(ValueError):
        results.approve(dict(character=True, garment=True, exposure=True, original_outfit_gone=True))


def test_confirmation_needs_only_the_check_no_settings_file(tmp_path, monkeypatch):
    # 사용자 GUI 확인(2026-10-12): 사용자가 실행 환경을 일일이 고르면 안 된다(D-101)
    from PySide6.QtWidgets import QCheckBox, QDialog, QLabel, QPushButton
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path / "inputs")
    seen = {}
    def inspect(dialog):
        start = [b for b in dialog.findChildren(QPushButton) if b.text() == "옷 바꾸기 시작"][0]
        seen["buttons"] = [b.text() for b in dialog.findChildren(QPushButton)]
        seen["before"] = start.isEnabled()
        dialog.findChild(QCheckBox, "outfit_edit_confirm").setChecked(True)
        seen["after"] = (start.isEnabled(), dialog.findChild(QLabel, "outfit_edit_hint").text())
        return QDialog.DialogCode.Rejected
    monkeypatch.setattr(QDialog, "exec", inspect)
    assert ui.confirm_outfit_edit(None, analysis, "keep") is None
    assert not any("설정" in text for text in seen["buttons"])
    assert seen["before"] is False and seen["after"][0] is True and "준비" in seen["after"][1]


def test_missing_qwen_setup_is_reported_before_confirmation(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.studio.run_directory = tmp_path / "run"
    window.studio.run_directory.mkdir()
    window.studio.runtime = ui.StudioRuntime(qwen_settings=tmp_path / "none.json")
    monkeypatch.setattr(ui, "confirm_outfit_edit", lambda *a: pytest.fail("설정이 없으면 확인 창을 띄우지 않는다"))
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "exec", lambda *a: 0)
    try:
        window.studio.begin_outfit_edit(analysis_at(tmp_path / "inputs"))
        status = json.loads((window.studio.run_directory / "gui-status.json").read_text(encoding="utf-8"))
        assert "실행 설정 파일이 없습니다" in status["error"] and window.studio.task is None
    finally:
        window.close()


def test_default_settings_resolve_manifest_next_to_config(tmp_path):
    from genai_lab.qwen_pose_settings import QwenPoseSettings
    config = tmp_path / "configs"
    config.mkdir()
    (config / "m.json").write_text("{}", encoding="utf-8")
    values = json.loads(settings_file(tmp_path).read_text(encoding="utf-8"))
    (config / "s.json").write_text(json.dumps(dict(values, artifact_manifest="m.json")), encoding="utf-8")
    assert Path(QwenPoseSettings.load(config / "s.json").artifact_manifest) == (config / "m.json").resolve()


def test_settings_path_is_remembered_only_if_file_exists(tmp_path):
    from PySide6.QtCore import QSettings
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    path = settings_file(tmp_path)
    ui.remember_qwen_settings(path, store)
    assert ui.remembered_qwen_settings(store) == str(path)
    path.unlink()
    assert ui.remembered_qwen_settings(store) == ""


def test_gender_is_prefilled_from_saved_value_or_guess(tmp_path, monkeypatch):
    # D-106: 저장된 성별은 그대로, 없으면 자동 분석 값을 채운다(자동 분석은 틀릴 수 있어 사용자가 확인)
    from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path / "inputs")
    analysis["character_scores"] = {"1girl": 0.88, "1boy": 0.05}
    chosen = {}
    def accept(dialog):
        chosen["value"] = dialog.findChild(QComboBox, "outfit_edit_gender").currentData()
        dialog.findChild(QCheckBox, "outfit_edit_confirm").setChecked(True)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", accept)
    assert ui.confirm_outfit_edit(None, analysis, "keep", saved_gender="male")["gender"] == "male"
    assert ui.confirm_outfit_edit(None, analysis, "keep")["gender"] == "female" and chosen["value"] == "female"
    analysis["character_scores"] = {}
    assert ui.confirm_outfit_edit(None, analysis, "keep")["gender"] == "unspecified"


def test_body_parts_are_prefilled_from_saved_value_or_tags(tmp_path, monkeypatch):
    # D-111: 귀·꼬리·날개는 저장값이 있으면 그대로, 없으면 태그 점수(0.5 이상)로 채우고 사용자가 확인한다
    from PySide6.QtWidgets import QCheckBox, QDialog
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path / "inputs")
    analysis["character_scores"] = {"1girl": 0.9, "cat_ears": 0.8, "tail": 0.3, "cat_tail": 0.6, "wing_collar": 0.9, "ponytail": 0.9}
    def accept(dialog):
        dialog.findChild(QCheckBox, "outfit_edit_confirm").setChecked(True)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", accept)
    assert ui.guessed_body_parts(analysis) == ("ears", "tail")                  # wing_collar·ponytail은 날개·꼬리가 아니다
    assert ui.confirm_outfit_edit(None, analysis, "keep")["body_parts"] == ("ears", "tail")
    assert ui.confirm_outfit_edit(None, analysis, "keep", saved_body_parts=())["body_parts"] == ()
    assert ui.confirm_outfit_edit(None, analysis, "keep", saved_body_parts=("wings",))["body_parts"] == ("wings",)


def test_confirmation_shows_support_warnings_and_records_them(tmp_path, monkeypatch):
    # D-113: 알려진 실패 유형(앉은 자세·걸친 겉옷·기계 몸)은 시작 전에 알리고, 출처 기록에 남긴다. 조건은 바꾸지 않는다
    from PySide6.QtWidgets import QCheckBox, QDialog, QLabel
    app = QApplication.instance() or QApplication([])
    analysis = analysis_at(tmp_path / "inputs")
    analysis["character_scores"] = {"1boy": 0.99, "fox_ears": 0.93, "tail": 0.97, "sitting": 0.69, "multiple_tails": 0.91}
    seen = {}
    def accept(dialog):
        seen["text"] = dialog.findChild(QLabel, "outfit_edit_support").text()
        dialog.findChild(QCheckBox, "outfit_edit_confirm").setChecked(True)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", accept)
    confirmed = ui.confirm_outfit_edit(None, analysis, "keep")
    assert confirmed == dict(gender="male", body_parts=("ears", "tail"))        # 경고가 조건을 바꾸지 않는다
    assert "앉거나 웅크린 자세" in seen["text"] and "꼬리가 여럿" in seen["text"]
    record = ui.confirmation_provenance(analysis, None, None, confirmed["body_parts"])
    assert [w["key"] for w in record["support"]["warnings"]] == ["pose_not_standing", "multiple_tails"]
    analysis["character_scores"] = {"1girl": 1.0, "standing": 0.7}
    ui.confirm_outfit_edit(None, analysis, "keep")
    assert "보이지 않습니다" in seen["text"]
