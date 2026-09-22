"""Stage 9 storage choice audit records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

STORAGE_RECORD_VERSION = "stage9_storage_decision_v1"
SAVED = "saved"
DISCARDED = "discarded"


class FinalResultStorageError(ValueError):
    """Stage 9 storage choice is inconsistent with the Stage 8 decision."""


@dataclass(frozen=True)
class FinalResultStorageRecord:
    candidate_id: str
    stage8_decision: str
    status: str
    storage_class: str | None
    selected_output_root: str | None
    image_path: str | None
    metadata_path: str | None
    training_use_approved: bool
    recorded_at: str

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value["version"] = STORAGE_RECORD_VERSION
        return value


def create_storage_record(
    *,
    candidate_id: str,
    stage8_decision: str,
    status: str,
    storage_class: str | None = None,
    selected_output_root: Path | str | None = None,
    image_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> FinalResultStorageRecord:
    if stage8_decision not in {"approved", "approved_with_refinement"}:
        raise FinalResultStorageError(
            "8단계 승인 또는 조건부 승인 결과만 9단계에서 처리할 수 있습니다."
        )
    if status not in {SAVED, DISCARDED}:
        raise FinalResultStorageError(f"지원하지 않는 저장 결정입니다: {status}")
    if status == SAVED:
        if storage_class not in {"final_result", "refinement_checkpoint"}:
            raise FinalResultStorageError("저장 결과 등급이 올바르지 않습니다.")
        if not selected_output_root or not image_path or not metadata_path:
            raise FinalResultStorageError("저장된 결과 경로가 빠졌습니다.")
    else:
        storage_class = None
        selected_output_root = None
        image_path = None
        metadata_path = None

    return FinalResultStorageRecord(
        candidate_id=str(candidate_id),
        stage8_decision=stage8_decision,
        status=status,
        storage_class=storage_class,
        selected_output_root=(
            str(selected_output_root) if selected_output_root else None
        ),
        image_path=str(image_path) if image_path else None,
        metadata_path=str(metadata_path) if metadata_path else None,
        training_use_approved=False,
        recorded_at=datetime.now().astimezone().isoformat(),
    )


def write_storage_record(
    record: FinalResultStorageRecord,
    audit_directory: Path | str,
) -> Path:
    directory = Path(audit_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stage9-storage-decision.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record.record(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_storage_record(path: Path | str) -> FinalResultStorageRecord:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FinalResultStorageError("9단계 저장 기록 형식이 올바르지 않습니다.")
    return FinalResultStorageRecord(
        candidate_id=str(value["candidate_id"]),
        stage8_decision=str(value["stage8_decision"]),
        status=str(value["status"]),
        storage_class=value.get("storage_class"),
        selected_output_root=value.get("selected_output_root"),
        image_path=value.get("image_path"),
        metadata_path=value.get("metadata_path"),
        training_use_approved=bool(value.get("training_use_approved", False)),
        recorded_at=str(value["recorded_at"]),
    )
