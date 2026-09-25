"""Result record for the product final-refinement stage."""
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
