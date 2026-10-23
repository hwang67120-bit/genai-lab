"""최초 모델 준비(D-173): 공통 깊은 검사·저장·재검사, 화면을 막지 않는 검사, 중복 실행·폴더 바뀜·취소 처리, 파일 검증 ≠ GPU 생성 검증."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from genai_lab import model_check, model_setup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = """
import json, sys, time
home, out, run_id, mode, sleep, fingerprint = sys.argv[1:7]
print("progress 가짜 검사", flush=True)
time.sleep(float(sleep))
if mode == "crash":
    sys.exit(3)
result = dict(schema="model_setup/1", run_id=run_id, home=home if mode != "other_home" else home + "-other", deep=True,
              started="s", finished="f", seconds=0.1, fingerprint=json.loads(fingerprint),
              ok=mode == "ok", problems=[] if mode == "ok" else [dict(component="wd_tagger")], problem_count=0 if mode == "ok" else 1,
              message="" if mode == "ok" else "필요한 모델 파일이 없거나 달라서 시작하지 않았습니다.")
open(out, "w", encoding="utf-8").write(json.dumps(result))
"""


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def fake_command(tmp_path, mode, sleep=0.2):
    script = tmp_path / "fake_check.py"
    script.write_text(FAKE, encoding="utf-8")
    fingerprint = json.dumps(model_setup.manifest_fingerprint())
    return lambda home, out, run_id: [sys.executable, str(script), str(home), str(out), run_id, mode, str(sleep), fingerprint]


def wait(app, condition, seconds=20):
    end = time.time() + seconds
    while not condition() and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    assert condition()


def window_for(tmp_path, mode, sleep=0.2, opened=None):
    from genai_lab.model_setup_ui import ModelSetupWindow
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    record = tmp_path / "model-setup.json"
    w = ModelSetupWindow(on_ready=(opened.append if opened is not None else None), command=fake_command(tmp_path, mode, sleep),
                         record_file=record)
    w.set_folder(home)
    return w, home, record


def test_success_is_saved_as_files_verified_not_gpu_verified_and_opens_the_studio(app, tmp_path, monkeypatch):
    monkeypatch.setenv(model_setup.ENV, "placeholder")      # 시험이 끝나면 원래대로(없음) 되돌리도록 먼저 기록
    monkeypatch.delenv(model_setup.ENV)
    opened = []
    w, home, record = window_for(tmp_path, "ok", opened=opened)
    assert not w.open_button.isEnabled()
    assert w.start_check() and w.running()
    assert not w.choose_button.isEnabled() and not w.check_button.isEnabled() and w.cancel_button.isEnabled()
    wait(app, lambda: not w.running())
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["status"] == model_setup.FILES_VERIFIED and saved["gpu_generation_verified"] is False
    assert "모델 파일 검증: 완료" in w.files_status.text() and "GPU 생성 검증: 아직 안 됨" in w.gpu_status.text()
    assert w.check_button.text() == "재검사" and w.open_button.isEnabled()
    assert w.open_studio() and opened == [home] and os.environ[model_setup.ENV] == str(home)
    w.close()


def test_duplicate_start_and_folder_change_are_refused_while_checking(app, tmp_path):
    w, home, record = window_for(tmp_path, "ok", sleep=1.5)
    assert w.start_check()
    assert w.start_check() is False                                  # 중복 실행 안 됨
    assert w.set_folder(tmp_path / "elsewhere") is False and w.folder == home   # 검사 중 폴더 바꾸기 안 됨
    wait(app, lambda: not w.running())
    assert json.loads(record.read_text(encoding="utf-8"))["model_home"] == str(home.resolve())
    w.close()


def test_cancelled_and_crashed_checks_are_not_saved(app, tmp_path):
    w, home, record = window_for(tmp_path, "ok", sleep=30)
    assert w.start_check()
    wait(app, lambda: "가짜 검사" in w.status.text(), seconds=10)
    assert w.cancel()
    wait(app, lambda: not w.running())
    assert "취소" in w.status.text() and not record.exists() and not w.open_button.isEnabled()
    w.close()
    w2, _, record2 = window_for(tmp_path, "crash")
    assert w2.start_check()
    wait(app, lambda: not w2.running())
    assert "중간에 끝났습니다" in w2.status.text() and not record2.exists()
    w2.close()


def test_result_for_another_folder_is_discarded(app, tmp_path):
    w, home, record = window_for(tmp_path, "other_home")
    assert w.start_check()
    wait(app, lambda: not w.running())
    assert "버렸습니다" in w.status.text() and not record.exists()
    w.close()


def test_problems_are_saved_as_failed_with_guidance_and_studio_stays_closed(app, tmp_path):
    w, home, record = window_for(tmp_path, "problems")
    assert w.start_check()
    wait(app, lambda: not w.running())
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["status"] == model_setup.FILES_CHECK_FAILED and "필요한 모델 파일" in w.details.toPlainText()
    assert not w.open_button.isEnabled() and w.open_studio() is False
    w.close()


def test_run_check_uses_the_shared_deep_check_on_the_chosen_folder(tmp_path, monkeypatch):
    calls = []
    def fake(runtime, settings, flow, *, deep=False, progress=None, **kw):
        calls.append((runtime, settings, flow, deep))
        return dict(ok=True, problems=[], locations={})
    monkeypatch.setattr(model_check, "check_models", fake)
    home = tmp_path / "home"
    home.mkdir()
    result = model_setup.run_check(home, "r1", progress=lambda text: None)
    runtime, settings, flow, deep = calls[0]
    assert flow == "outfit" and deep is True and result["ok"] is True and result["run_id"] == "r1"
    assert Path(runtime.model_cache) == home.resolve() / "hf" and Path(settings.model_root) == home.resolve() / "qwen/Qwen-Image-Edit-2511"
    assert settings.python_executable == sys.executable


def test_real_check_of_an_empty_folder_reports_problems_and_a_missing_folder_is_not_success(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = model_setup.run_check(empty, "r2", progress=lambda text: None)
    assert result["ok"] is False and result["problem_count"] > 0 and "할 일" in result["message"]
    gone = model_setup.run_check(tmp_path / "missing", "r3", progress=lambda text: None)
    assert gone["ok"] is False and "모델 폴더가 없습니다" in gone["message"]


def test_unfinished_results_are_refused_and_records_need_matching_lists(tmp_path):
    with pytest.raises(ValueError):
        model_setup.save_result(dict(schema=model_setup.SCHEMA, run_id="x", home=str(tmp_path), ok=True), path=tmp_path / "r.json")
    good = dict(schema=model_setup.SCHEMA, run_id="x", home=str(tmp_path), ok=True, finished="f", seconds=1,
                fingerprint=model_setup.manifest_fingerprint())
    record = model_setup.save_result(good, path=tmp_path / "r.json")
    assert model_setup.verified_home(record) == tmp_path
    assert model_setup.verified_home(dict(record, fingerprint={"model_home_files": "old"})) is None   # 목록이 바뀌면 다시 검사


def test_startup_decision(tmp_path):
    verified = dict(schema=model_setup.SCHEMA, model_home=str(tmp_path), status=model_setup.FILES_VERIFIED,
                    fingerprint=model_setup.manifest_fingerprint())
    assert model_setup.startup_decision({model_setup.ENV: "X"}, record={}) == ("studio", Path("X"))
    assert model_setup.startup_decision({}, record=verified) == ("studio", tmp_path)
    assert model_setup.startup_decision({}, record={}, quick_check=lambda: True) == ("studio", None)       # 개발 PC 기존 경로
    assert model_setup.startup_decision({}, record={}, quick_check=lambda: False)[0] == "setup"
    failed = dict(verified, status=model_setup.FILES_CHECK_FAILED)
    assert model_setup.startup_decision({}, record=failed, quick_check=lambda: False)[0] == "setup"


def test_studio_start_keeps_the_quick_check_and_routes_through_the_decision():
    source = (ROOT / "gui_main.py").read_text(encoding="utf-8")
    assert "startup_decision(quick_check=None if packaged else current_paths_pass, packaged=packaged)" in source and "--model-setup" in source
    product = (ROOT / "genai_lab/studio_product.py").read_text(encoding="utf-8")
    assert "model_check.require_product_models(" in product                     # '이미지 생성' 전 빠른 검사 유지
