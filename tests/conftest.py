import pytest


@pytest.fixture(autouse=True)
def isolate_removal_diagnostic_files(tmp_path, monkeypatch):
    """테스트 기록과 실제 사용자 실행 기록을 섞지 않는다."""
    monkeypatch.setattr("genai_lab.removal_diagnostics.DIAGNOSTIC_ROOT",
                        tmp_path / "removal_logs")
