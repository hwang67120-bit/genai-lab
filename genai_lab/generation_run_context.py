"""Per-request artifact isolation for product generation runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


class GenerationRunContextError(ValueError):
    """Run directories or artifact contracts are unsafe or inconsistent."""


def _project_path(value: object, project_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_path_isolation(
    run_root: Path,
    training_roots: tuple[Path, ...],
) -> None:
    resolved_run = run_root.resolve()
    for training_root in training_roots:
        resolved_training = training_root.resolve()
        if (
            resolved_run == resolved_training
            or _is_relative_to(resolved_run, resolved_training)
            or _is_relative_to(resolved_training, resolved_run)
        ):
            raise GenerationRunContextError(
                "생성 결과 경로와 학습 자료 경로가 겹칩니다: "
                f"run_root={resolved_run}, training_root={resolved_training}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class GenerationRunContext:
    """Owns one request's paths and append-only diagnostic metadata."""

    run_id: str
    refinement_mode: str | None
    run_directory: Path
    inputs_directory: Path
    base_directory: Path
    masks_directory: Path
    refinement_directory: Path
    diagnostics_directory: Path
    manifest_path: Path
    record: dict[str, Any] = field(default_factory=dict)

    VERSION = "generation_run_context_v1"

    @classmethod
    def create(
        cls,
        config: Mapping[str, Any],
        project_root: Path,
        *,
        refinement_mode: str | None,
    ) -> "GenerationRunContext":
        section = config.get("generation_run_context", {})
        if section is None:
            section = {}
        if not isinstance(section, Mapping):
            raise GenerationRunContextError(
                "generation_run_context는 항목 묶음이어야 합니다."
            )
        output_root = _project_path(
            section.get("root", "outputs/generation-runs"), project_root
        )
        raw_training = section.get(
            "training_roots", ("training-data", "datasets")
        )
        if not isinstance(raw_training, (tuple, list)) or not all(
            isinstance(value, str) and value.strip() for value in raw_training
        ):
            raise GenerationRunContextError(
                "generation_run_context.training_roots는 경로 문자열 목록이어야 합니다."
            )
        training_roots = tuple(
            _project_path(value, project_root) for value in raw_training
        )
        _validate_path_isolation(output_root, training_roots)
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{uuid4().hex[:8]}"
        run_directory = output_root / run_id
        directories = {
            "inputs": run_directory / "inputs",
            "base": run_directory / "base",
            "masks": run_directory / "masks",
            "refinement": run_directory / "refinement",
            "diagnostics": run_directory / "diagnostics",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=False)
        context = cls(
            run_id=run_id,
            refinement_mode=refinement_mode,
            run_directory=run_directory,
            inputs_directory=directories["inputs"],
            base_directory=directories["base"],
            masks_directory=directories["masks"],
            refinement_directory=directories["refinement"],
            diagnostics_directory=directories["diagnostics"],
            manifest_path=run_directory / "run-context.json",
            record={
                "version": cls.VERSION,
                "run_id": run_id,
                "status": "created",
                "refinement_mode": refinement_mode,
                "created_at": datetime.now().astimezone().isoformat(),
                "run_directory": str(run_directory),
                "training_roots": [str(path) for path in training_roots],
                "artifacts": {},
                "events": [],
            },
        )
        context.write()
        return context

    def directory_for_mode(self, mode: str) -> Path:
        if mode != self.refinement_mode:
            raise GenerationRunContextError(
                "요청에 봉인된 정밀화 모드와 실행 모드가 다릅니다: "
                f"sealed={self.refinement_mode}, requested={mode}"
            )
        directory = self.refinement_directory / mode
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def record_artifact(
        self,
        name: str,
        path: Path,
        *,
        digest: bool = False,
    ) -> None:
        artifact = {"path": str(Path(path))}
        if digest and Path(path).is_file():
            artifact["sha256"] = sha256_file(Path(path))
        self.record.setdefault("artifacts", {})[name] = artifact
        self.write()

    def event(self, stage: str, status: str, **details: Any) -> None:
        entry = {"stage": stage, "status": status, **details}
        self.record.setdefault("events", []).append(entry)
        self.write()

    def finish(self, status: str, **details: Any) -> None:
        self.record["status"] = status
        self.record["finished_at"] = datetime.now().astimezone().isoformat()
        self.record.update(details)
        self.write()

    def write(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)
