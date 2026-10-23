"""배포판 모드(D-174): 설치 폴더 런타임을 쓰고, 개발 기본 경로로 준비 화면을 우회하지 않으며, 옮긴·없는 모델 폴더를 안내한다."""
from pathlib import Path

import pytest

from genai_lab import model_home, model_setup
from genai_lab.app_home import ENV, app_home, runtime_python
from genai_lab.child_env import child_environment
from genai_lab.studio_generation import StudioRuntime

ROOT = Path(__file__).resolve().parents[1]


def test_development_run_is_unchanged(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert app_home() is None and runtime_python("pose") is None
    assert StudioRuntime().pose_python == Path("D:/genai-cache/catvton-venv/Scripts/python.exe")


def test_packaged_run_uses_its_runtimes_and_never_falls_back_to_development_paths(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    monkeypatch.delenv(model_home.ENV, raising=False)
    runtime = StudioRuntime()
    assert runtime.pose_python == tmp_path / "runtime-pose" / "python.exe"
    for value in (runtime.model_cache, runtime.pose_models, runtime.head_cache, runtime.head_semantic_repo, runtime.head_semantic_checkpoint):
        assert Path(value).is_relative_to(tmp_path) and "genai-cache" not in str(value) and "192.168" not in str(value)
    assert runtime.qwen_settings.name == "qwen_edit_settings.model_home.json"


def test_bad_install_folder_is_reported(monkeypatch):
    monkeypatch.setenv(ENV, "relative/install")
    with pytest.raises(ValueError, match=ENV):
        app_home()


def test_packaged_first_run_without_a_record_goes_to_setup_even_if_development_paths_pass():
    decision, reason = model_setup.startup_decision({}, record={}, quick_check=lambda: True, packaged=True)
    assert decision == "setup" and "아직 준비되지 않았습니다" in reason
    assert model_setup.startup_decision({}, record={}, quick_check=lambda: True, packaged=False) == ("studio", None)


def test_moved_or_missing_model_folder_and_changed_lists_are_explained(tmp_path):
    verified = dict(schema=model_setup.SCHEMA, model_home=str(tmp_path / "moved away"), status=model_setup.FILES_VERIFIED,
                    fingerprint=model_setup.manifest_fingerprint())
    decision, reason = model_setup.startup_decision({}, record=verified, packaged=True)
    assert decision == "setup" and "찾을 수 없습니다" in reason and str(tmp_path / "moved away") in reason and "새 위치" in reason
    here = dict(verified, model_home=str(tmp_path), fingerprint={"model_home_files": "old"})
    assert "목록이 바뀌어" in model_setup.startup_decision({}, record=here, packaged=True)[1]
    failed = dict(verified, model_home=str(tmp_path), status=model_setup.FILES_CHECK_FAILED)
    assert "문제가 있었습니다" in model_setup.startup_decision({}, record=failed, packaged=True)[1]


def test_children_learn_the_install_folder(tmp_path):
    assert child_environment({ENV: str(tmp_path)})[ENV] == str(tmp_path)


def test_studio_start_does_not_bypass_setup_when_packaged():
    source = (ROOT / "gui_main.py").read_text(encoding="utf-8")
    assert "quick_check=None if packaged else current_paths_pass" in source and "reason=value" in source
