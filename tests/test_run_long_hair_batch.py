"""긴 머리 GPU 확인 스크립트: 쓰기 사전 확인이 실패하면 모델 시작 전에 원인을 남긴다."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_long_hair_batch", ROOT / "scripts/run_long_hair_batch.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_write_probe_leaves_no_folder(tmp_path):
    result = script.probe_write(tmp_path)
    assert result["status"] == "ok" and result["attempts"] == 1
    assert list((tmp_path / "auto-head-pastes").iterdir()) == []


def test_write_probe_records_network_denial_and_retries(tmp_path, monkeypatch):
    def deny(self, *args, **kwargs):
        error = PermissionError(13, "네트워크 액세스가 거부되었습니다")
        error.winerror = 65
        raise error
    monkeypatch.setattr(Path, "mkdir", deny)
    result = script.probe_write(tmp_path, attempts=2, wait=0)
    assert result["status"] == "failed" and result["attempts"] == 2
    assert [row["winerror"] for row in result["errors"]] == [65, 65]


def test_failure_outside_worker_is_written_to_summary(tmp_path, monkeypatch):
    import json, sys
    def boom(args, summary):
        summary["rule"] = "G3+H2+L1"
        raise PermissionError("거부")
    monkeypatch.setattr(script, "run", boom)
    monkeypatch.setattr(sys, "argv", ["x", "--run", "r", "--preview", "p", "--output", str(tmp_path / "out")])
    with pytest.raises(PermissionError):
        script.main()
    summary = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and summary["error_type"] == "PermissionError"
    assert "Traceback" in summary["traceback"] and summary["rule"] == "G3+H2+L1"
