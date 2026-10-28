"""제품 하위 프로세스 환경 변수(D-170): 필요한 이름만 넘기고, 개발 환경 변수는 막고, 오프라인은 고정."""
import subprocess
import sys
from pathlib import Path

import pytest

from genai_lab.child_env import child_environment

PARENT = {"SystemRoot": "C:\\Windows", "PATH": "C:\\Windows\\System32", "TEMP": "C:\\Temp", "APPDATA": "C:\\Users\\u\\AppData\\Roaming",
          "COMSPEC": "C:\\Windows\\system32\\cmd.exe", "NUMBER_OF_PROCESSORS": "16",
          "CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "4",
          "GENAI_MODEL_HOME": "D:\\models", "GENAI_CHARACTER_MEMORY": "D:\\mem", "HF_HOME": "D:\\hf",
          "PYTHONPATH": "C:\\other\\site", "PYTHONHOME": "C:\\Python39", "PYTHONSTARTUP": "x.py", "VIRTUAL_ENV": "C:\\venv",
          "CONDA_PREFIX": "C:\\conda", "SOME_TOOL_TOKEN": "secret", "HF_HUB_OFFLINE": "0"}


def test_only_needed_names_pass_and_development_python_settings_do_not():
    env = child_environment(PARENT)
    for name in ("SystemRoot", "PATH", "TEMP", "APPDATA", "COMSPEC", "NUMBER_OF_PROCESSORS", "CUDA_VISIBLE_DEVICES",
                 "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "GENAI_MODEL_HOME", "GENAI_CHARACTER_MEMORY", "HF_HOME"):
        assert env[name] == PARENT[name]
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX", "SOME_TOOL_TOKEN"):
        assert name not in env
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == env["PYTHONDONTWRITEBYTECODE"] == "1"   # 사용자가 0으로 둬도 오프라인


def test_overrides_win_and_extra_names_are_explicit():
    env = child_environment(dict(PARENT, GENAI_CHILD_ENV_EXTRA="PYTHONPATH, GENAI_OPEN_TRACE", GENAI_OPEN_TRACE="t.jsonl"),
                            CUDA_VISIBLE_DEVICES="", ONNX_MODE="cpu")
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["ONNX_MODE"] == "cpu"
    assert env["PYTHONPATH"] == PARENT["PYTHONPATH"] and env["GENAI_OPEN_TRACE"] == "t.jsonl"
    assert "PYTHONHOME" not in env


def test_a_real_child_python_starts_with_the_explicit_environment(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text("import os, sys, json; print(json.dumps(dict(ok=True, offline=os.environ.get('HF_HUB_OFFLINE'), "
                      "pythonpath=os.environ.get('PYTHONPATH'))))", encoding="utf-8")
    import os
    env = child_environment(dict(os.environ, PYTHONPATH=str(tmp_path / "elsewhere")))
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert '"ok": true' in out.stdout and '"offline": "1"' in out.stdout and '"pythonpath": null' in out.stdout


@pytest.mark.parametrize("module, needle, count", [
    ("studio_generation.py", 'env = child_environment(CUDA_VISIBLE_DEVICES="", ONNX_MODE="cpu")', 1),   # 분석(관절·특징)
    ("outfit_edit_pipeline.py", "env = child_environment()", 2),                                     # 머리 분할 · 관절 검사
    ("qwen_pose_edit.py", "env = child_environment()", 1),                                           # Qwen 작업 프로세스
])
def test_product_launchers_use_the_explicit_environment(module, needle, count):
    source = (Path(__file__).resolve().parents[1] / "genai_lab" / module).read_text(encoding="utf-8")
    assert source.count(needle) == count and "env = dict(os.environ" not in source
