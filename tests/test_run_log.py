from pathlib import Path
import logging

import pytest

from genai_lab.run_log import create_generation_run_log


def test_generation_logs_are_unique_and_close_is_explicit(tmp_path: Path) -> None:
    first = create_generation_run_log(tmp_path)
    second = create_generation_run_log(tmp_path)
    try:
        assert first.run_id != second.run_id
        assert first.file_path != second.file_path
        assert first.logger.name not in logging.Logger.manager.loggerDict
        assert second.logger.name not in logging.Logger.manager.loggerDict

        first.write_stage("후속 단계", "Native 정밀화 기록")
        first.close()
        first.close()

        assert "Native 정밀화 기록" in first.file_path.read_text(
            encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="이미 종료된 생성 로그"):
            first.write_stage("늦은 기록", "누락되면 안 됨")
    finally:
        first.close()
        second.close()
