"""이미지 생성 단계와 오류를 로컬 텍스트 파일에 기록한다.

공식 참고 문서:
- Python logging: https://docs.python.org/3.10/library/logging.html
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import sys


@dataclass(frozen=True)
class GenerationRunLog:
    """한 번의 이미지 생성 실행을 추적하는 로그."""

    run_id: str
    file_path: Path
    logger: logging.Logger

    def write_stage(self, stage: str, message: str) -> None:
        """현재 실행 단계와 확인할 값을 기록한다."""
        self.logger.info("[%s] %s", stage, message)

    def write_failure(
        self,
        stage: str,
        error: Exception,
        recovery_action: str,
    ) -> None:
        """오류, 전체 추적 정보와 사용자가 취할 행동을 기록한다."""
        self.logger.error(
            "[%s] 실패: %s: %s",
            stage,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
        self.logger.error("[권장 행동] %s", recovery_action)

    def close(self) -> None:
        """파일 기록을 끝내고 운영체제의 파일 자원을 해제한다."""
        for handler in tuple(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)


def create_generation_run_log(project_root: Path) -> GenerationRunLog:
    """새 실행 ID와 UTF-8 로그 파일을 만들고 추적 객체를 반환한다."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_directory = project_root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file_path = log_directory / f"{run_id}.log"

    logger = logging.getLogger(f"genai_lab.generation.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        log_file_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    generation_run_log = GenerationRunLog(
        run_id=run_id,
        file_path=log_file_path,
        logger=logger,
    )
    generation_run_log.write_stage(
        "실행 시작",
        f"실행 ID={run_id}, 로그 파일={log_file_path}",
    )
    return generation_run_log


def find_recovery_action(error: Exception) -> str:
    """오류 문구를 규칙으로 확인해 사용자가 먼저 할 행동을 반환한다."""
    error_message = str(error).lower()
    if "out of memory" in error_message or "메모리" in error_message:
        return "GPU를 사용하는 다른 프로그램을 종료한 뒤 다시 실행하세요."
    if "파일" in error_message or "file" in error_message:
        return "참조 이미지와 설정 파일 경로를 확인하세요."
    if "cuda" in error_message or "gpu" in error_message:
        return "NVIDIA 드라이버와 CUDA용 PyTorch 설치 상태를 확인하세요."
    return "로그의 오류 발생 위치와 전체 추적 정보를 확인하세요."
