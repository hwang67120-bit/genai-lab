import pytest


@pytest.fixture(autouse=True)
def isolate_removal_diagnostic_files(tmp_path, monkeypatch):
    """테스트 기록과 실제 사용자 실행 기록을 섞지 않는다."""
    monkeypatch.setattr("genai_lab.removal_diagnostics.DIAGNOSTIC_ROOT",
                        tmp_path / "removal_logs")
    monkeypatch.setenv("GENAI_CHARACTER_MEMORY", str(tmp_path / "character-memory"))   # 실제 사용자 캐릭터 기억과 섞지 않는다
    monkeypatch.delenv("GENAI_MODEL_HOME", raising=False)    # 모델 폴더 설정이 켜진 PC에서도 기존 경로 기준으로 시험한다
    monkeypatch.setenv("GENAI_STUDIO_DEVELOPER", "1")        # 기존 시험은 개발·평가 화면 기준. 사용자 화면 시험은 이 값을 지운다
