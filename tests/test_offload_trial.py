"""요청 39(D-185): 측정 전용 오프로드 이름은 받되 제품 설정은 group_block1 그대로, 실행기는 settings.offload만 바꾼다."""
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from genai_lab.qwen_pose_settings import OFFLOAD_ARGS, TRIAL_OFFLOADS, QwenPoseSettings

ROOT = Path(__file__).resolve().parents[1]


def settings(tmp):
    return QwenPoseSettings(sys.executable, str(tmp), str(tmp / "m.gguf"), str(tmp / "m.json"), "a" * 64)


def test_product_settings_stay_on_the_current_offload_with_the_same_arguments():
    for name in ("qwen_edit_settings.json", "qwen_edit_settings.model_home.json"):
        assert json.loads((ROOT / "configs" / name).read_text(encoding="utf-8-sig"))["offload"] == "group_block1"
    assert OFFLOAD_ARGS["group_block1"] == dict(num_blocks_per_group=1)        # 지금 제품과 같은 인자(겹치기 없음)
    assert set(TRIAL_OFFLOADS) == set(OFFLOAD_ARGS) - {"group_block1"}
    source = (ROOT / "genai_lab" / "qwen_pose_worker.py").read_text(encoding="utf-8")
    assert source.count("enable_group_offload(") == 1 and "**OFFLOAD_ARGS[settings.offload]" in source


def test_trial_names_are_accepted_and_unknown_names_are_not(tmp_path):
    for name in TRIAL_OFFLOADS:
        assert replace(settings(tmp_path), offload=name).offload == name
    with pytest.raises(ValueError, match="오프로드"):
        replace(settings(tmp_path), offload="sequential")


def test_runner_changes_only_the_offload_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("GENAI_QWEN_SESSION_REUSE", "0")
    spec = importlib.util.spec_from_file_location("offload_trial", ROOT / "scripts/offload_trial.py")
    trial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial)
    base = dict(task_kind="free_generation", seed=42, prompt=dict(positive="p"), settings=settings(tmp_path).record())
    (tmp_path / "request.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "raw.png").write_bytes(b"raw")
    out = tmp_path / "out"
    assert trial.main([str(tmp_path / "request.json"), str(tmp_path / "raw.png"), "python", str(out), "--dry"]) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert [r["offload"] for r in report["runs"]] == list(trial.ORDER) and trial.ORDER[0] == trial.ORDER[-1] == "group_block1"
    assert [r["request_changes"] for r in report["runs"]] == [[], ["settings.offload"], ["settings.offload"], []]
    for run in report["runs"]:
        request = json.loads((out / f"{run['index']:02d}-{run['offload']}" / "request.json").read_text(encoding="utf-8"))
        assert request["settings"]["offload"] == run["offload"] and request["seed"] == 42 and request["prompt"] == base["prompt"]
    assert trial.main([str(tmp_path / "request.json"), str(tmp_path / "raw.png"), "python", str(tmp_path / "out2"), "--dry", "nope"]) == 1
