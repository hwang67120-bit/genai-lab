"""GUI 입력→공통 함수→사람 승인→저장 연결(CPU 모의, 실제 생성 없음)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QCheckBox, QPlainTextEdit, QPushButton
from gui_main import GenAILabWindow
from genai_lab import studio_controller as ui, studio_free_generation as free_ui, qwen_outfit_edit
from genai_lab import free_generation_pipeline as pipeline
from test_free_generation import fake_models, settings
from test_studio_generation import wait_for


def patch_preferences(monkeypatch):
    for module in (free_ui, ui):
        for name in ("load_character_gender", "load_character_body_parts", "load_character_features"):
            if hasattr(module,name):
                monkeypatch.setattr(module,name,lambda *a:None)
    for name in ("save_character_gender", "save_character_body_parts", "save_character_features"):
        monkeypatch.setattr(ui,name,lambda *a:None)


def test_free_button_needs_base_but_no_garment(tmp_path, fake_models):
    app = QApplication.instance() or QApplication([])
    base,_ = fake_models
    window = GenAILabWindow()
    try:
        assert not window.free_generation_button.isEnabled()
        window.style_path = base
        window.update_input_ready_status()
        assert window.selected_outfit_path is None and window.free_generation_button.isEnabled()
        assert not window.generate_button.isEnabled()
    finally:
        window.close()


def test_gui_uses_common_function_and_separate_four_check_approval(tmp_path, fake_models, monkeypatch):
    from test_free_generation import features
    app = QApplication.instance() or QApplication([])
    base,calls = fake_models
    window = GenAILabWindow()
    patch_preferences(monkeypatch)
    window.studio.output_root = tmp_path/"studio"
    window.style_path = base
    window.update_input_ready_status()
    monkeypatch.setattr(free_ui,"analyze_base",lambda *a:dict(source=str(base),character_scores={},groups=dict(fixed=[],appearance=[],body=[])))
    decision = dict(base_kind="original",gender="female",body_parts=(),character_features=features(),request_text="눈 내리는 거리",
                    confirmed=True,adult_confirmed=True)
    monkeypatch.setattr(free_ui,"confirm_free_inputs",lambda *a:decision)
    monkeypatch.setattr(free_ui,"confirm_free_result",lambda *a:dict(character=True,garment=True,request=True,exposure=True,finish=True))
    monkeypatch.setattr(qwen_outfit_edit,"load_qwen_settings",lambda *a:settings(tmp_path))
    monkeypatch.setattr(ui,"analyze_inputs",lambda *a,**kw:pytest.fail("옷 편집 입력 경로 사용"))
    monkeypatch.setattr(ui,"validate_local_models",lambda *a:pytest.fail("SDXL 모델 접근"))
    destination=tmp_path/"saved.png"
    monkeypatch.setattr(QFileDialog,"getSaveFileName",lambda *a:(str(destination),""))
    try:
        window.free_generation_button.click()
        wait_for(app,lambda:window.studio.task is None and window.studio.results is not None)
        assert window.studio.results.status == "awaiting_user_review"
        assert getattr(window.studio.results,"free_mode",False)
        request = [c[1] for c in calls if c[0]=="qwen"][0]
        assert request["task_kind"] == "free_generation" and "눈 내리는 거리" in request["prompt"]["positive"]
        assert not window.tail_edit_button.isVisible() and not window.studio_finish_button.isVisible()
        assert not window.save_candidate_button.isEnabled()
        window.approve_candidate_button.click()
        assert window.save_candidate_button.isEnabled()
        window.save_candidate_button.click()
        metadata=json.loads(destination.with_suffix(".review.json").read_text(encoding="utf-8"))
        assert metadata["mode"] == "free_generation" and not metadata["head_composited"]
    finally:
        window.close()


def test_cancelled_free_confirmation_does_not_call_common_generation(tmp_path, fake_models,monkeypatch):
    app=QApplication.instance() or QApplication([])
    base,calls=fake_models
    window=GenAILabWindow()
    patch_preferences(monkeypatch)
    window.studio.run_directory=tmp_path/"studio"
    window.studio.run_directory.mkdir()
    window.studio.runtime=ui.StudioRuntime()
    monkeypatch.setattr(free_ui,"confirm_free_inputs",lambda *a:None)
    monkeypatch.setattr(qwen_outfit_edit,"load_qwen_settings",lambda *a:settings(tmp_path))
    try:
        window.studio.free_inputs_ready(dict(source=str(base),character_scores={},groups=dict(fixed=[])))
        assert window.studio.task is None and not calls
    finally:
        window.close()


def test_input_confirmation_rechecks_staging_and_adult_checkbox(tmp_path, fake_models,monkeypatch):
    app=QApplication.instance() or QApplication([])
    base,_=fake_models
    patch_preferences(monkeypatch)
    seen={}
    def interact(dialog):
        start=next(b for b in dialog.findChildren(QPushButton) if b.text()=="자유 생성 시작")
        assert not start.isEnabled()
        request=dialog.findChild(QPlainTextEdit,"free_request")
        request.setPlainText("눈을 감은 표정")
        dialog.findChild(QCheckBox,"free_confirm").setChecked(True)
        seen["allowed"]=start.isEnabled()
        request.setPlainText("머리색을 바꿔")
        seen["blocked"]=not start.isEnabled()
        request.setPlainText("눈 내리는 거리")
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog,"exec",interact)
    result=free_ui.confirm_free_inputs(None,dict(source=str(base),character_scores={}))
    assert seen==dict(allowed=True,blocked=True)
    assert result["adult_confirmed"] and result["request_text"]=="눈 내리는 거리"
    assert "experiment" not in result                                          # 기본은 40단계(제품 설정)


def test_quick_preview_checkbox_requests_the_8_step_accelerator(tmp_path, fake_models, monkeypatch):
    app=QApplication.instance() or QApplication([])
    base,_=fake_models
    patch_preferences(monkeypatch)
    def interact(dialog):
        dialog.findChild(QCheckBox,"free_quick_preview").setChecked(True)
        dialog.findChild(QCheckBox,"free_confirm").setChecked(True)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog,"exec",interact)
    result=free_ui.confirm_free_inputs(None,dict(source=str(base),character_scores={}))
    assert result["experiment"]==dict(lightning8=True,single_attempt=True)


def test_missing_qwen_settings_are_reported_instead_of_stopping(tmp_path, fake_models, monkeypatch):
    # 검토 6번: 설정 파일 문제는 편집 경로와 같이 작업실 오류 안내로 보인다
    app = QApplication.instance() or QApplication([])
    base,_ = fake_models
    window = GenAILabWindow()
    patch_preferences(monkeypatch)
    window.studio.output_root = tmp_path/"studio"
    window.studio.runtime = SimpleNamespace(qwen_settings=tmp_path/"missing.json")
    failures = []
    monkeypatch.setattr(window.studio, "fail", lambda error, detail: failures.append(str(error)))
    monkeypatch.setattr(free_ui, "confirm_free_inputs", lambda *a: pytest.fail("설정이 없으면 확인 창을 띄우지 않는다"))
    try:
        window.studio.free_inputs_ready(dict(source=str(base), character_scores={}, groups=dict(fixed=[], appearance=[], body=[])))
        assert failures and "설정 파일" in failures[0]
    finally:
        window.close()
