"""제품 최종 정밀화 단계의 결과 기록이다."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class RefinementExecutionResult:
    status: str
    selected_engine: str | None
    selected_image_path: Path
    report_path: Path
    output_directory: Path
    report: dict[str, Any]
