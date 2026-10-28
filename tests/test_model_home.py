"""모델 폴더 한 곳(GENAI_MODEL_HOME): 정하면 그 아래 배치에서, 정하지 않으면 기존 경로 그대로. 실행 환경은 다루지 않는다."""
import json
import sys
from pathlib import Path

import pytest

from genai_lab import model_home
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.studio_generation import StudioRuntime

ROOT = Path(__file__).resolve().parents[1]


def test_without_model_home_the_existing_paths_are_unchanged():
    runtime = StudioRuntime()
    assert runtime.pose_models == Path("D:/genai-cache/huggingface/easy-dwpose/checkpoints")
    assert runtime.model_cache == Path("D:/genai-cache/huggingface")
    assert runtime.head_cache == ROOT / "outputs/pose-identity-20260927/identity-v2/model-cache/hub"
    assert runtime.head_semantic_repo == Path("D:/genai-cache/tools/see-through")
    assert runtime.head_semantic_checkpoint == Path("D:/genai-cache/huggingface") / model_home.L2D_SAM_SNAPSHOT
    assert runtime.qwen_settings == ROOT / "configs/qwen_edit_settings.json"
    assert runtime.pose_python == Path("D:/genai-cache/catvton-venv/Scripts/python.exe")      # 실행 환경은 이번 범위 밖


def test_model_home_puts_every_product_model_under_one_folder(tmp_path, monkeypatch):
    monkeypatch.setenv(model_home.ENV, str(tmp_path))
    runtime = StudioRuntime()
    for key, attr in (("pose_models", "pose_models"), ("model_cache", "model_cache"), ("head_cache", "head_cache"),
                      ("head_semantic_repo", "head_semantic_repo"), ("head_semantic_checkpoint", "head_semantic_checkpoint")):
        assert getattr(runtime, attr) == tmp_path / model_home.LAYOUT[key]
        assert getattr(runtime, attr).is_relative_to(tmp_path)
    assert "192.168" not in str(runtime.head_cache) and "outputs" not in str(runtime.head_cache)
    assert runtime.qwen_settings == ROOT / "configs/qwen_edit_settings.model_home.json"
    settings = QwenPoseSettings.load(runtime.qwen_settings)
    assert Path(settings.model_root) == tmp_path / "qwen/Qwen-Image-Edit-2511"
    assert Path(settings.gguf_file) == tmp_path / model_home.LAYOUT["qwen_gguf"]
    assert settings.python_executable == sys.executable
    assert Path(settings.artifact_manifest) == ROOT / "configs/qwen_model_files.json"


def test_explicit_runtime_file_still_wins_over_model_home(tmp_path, monkeypatch):
    monkeypatch.setenv(model_home.ENV, str(tmp_path))
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps(dict(model_cache=str(tmp_path / "elsewhere"))), encoding="utf-8")
    monkeypatch.setenv("GENAI_STUDIO_RUNTIME", str(config))
    runtime = StudioRuntime.from_environment()
    assert runtime.model_cache == tmp_path / "elsewhere" and runtime.pose_models == tmp_path / model_home.LAYOUT["pose_models"]


@pytest.mark.parametrize("value", ["relative/models", "Z:/no/such/folder/for/genai"])
def test_bad_model_home_is_reported_not_guessed(monkeypatch, value):
    monkeypatch.setenv(model_home.ENV, value)
    with pytest.raises(ValueError, match="GENAI_MODEL_HOME"):
        StudioRuntime()


def test_relative_model_paths_need_a_model_home():
    with pytest.raises(ValueError, match="GENAI_MODEL_HOME"):
        QwenPoseSettings.load(ROOT / "configs/qwen_edit_settings.model_home.json")


def test_installed_settings_differ_from_the_verified_settings_only_in_paths():
    base = json.loads((ROOT / "configs/qwen_edit_settings.json").read_text(encoding="utf-8-sig"))
    home = json.loads((ROOT / "configs/qwen_edit_settings.model_home.json").read_text(encoding="utf-8-sig"))
    changed = {k for k in base if base[k] != home.get(k)}
    assert set(home) == set(base) and changed == {"python_executable", "model_root", "gguf_file"}
