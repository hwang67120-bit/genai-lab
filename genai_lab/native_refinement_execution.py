"""GUI에서 로컬 FLUX.2 Klein 전체 이미지 정밀화를 관리한다."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Callable


@dataclass(frozen=True)
class NativeRefinementExecutionSettings:
    python_executable: Path
    runner_path: Path
    model_cache_dir: Path
    width: int
    height: int
    flux_steps: int
    minimum_similarity: float
    minimum_color_similarity: float
    timeout_seconds: float
    local_files_only: bool = True


@dataclass(frozen=True)
class NativeRefinementExecutionResult:
    status: str
    selected_engine: str | None
    selected_image_path: Path
    report_path: Path
    output_directory: Path
    report: dict[str, Any]


class NativeRefinementExecutionError(RuntimeError):
    """로컬 정밀화 실행 또는 결과 계약이 올바르지 않을 때 발생한다."""


def _resolved_project_path(value: object, project_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def resolve_native_refinement_settings(
    config: dict[str, Any],
    project_root: Path,
) -> NativeRefinementExecutionSettings:
    raw = config.get("native_refinement", {})
    if not raw.get("enabled", False):
        raise NativeRefinementExecutionError(
            "로컬 FLUX.2 Klein 정밀화가 설정에서 비활성화되어 있습니다."
        )

    settings = NativeRefinementExecutionSettings(
        python_executable=_resolved_project_path(
            raw.get(
                "python_executable",
                "D:/genai-cache/venv/Scripts/python.exe",
            ),
            project_root,
        ),
        runner_path=_resolved_project_path(
            raw.get(
                "runner_path",
                "scripts/run_native_refinement.py",
            ),
            project_root,
        ),
        model_cache_dir=Path(
            str(
                raw.get(
                    "model_cache_dir",
                    Path.home() / ".cache" / "huggingface",
                )
            )
        ),
        width=int(raw.get("width", 320)),
        height=int(raw.get("height", 512)),
        flux_steps=int(raw.get("flux_steps", 4)),
        minimum_similarity=float(raw.get("minimum_similarity", 0.65)),
        minimum_color_similarity=float(
            raw.get("minimum_color_similarity", 0.55)
        ),
        timeout_seconds=float(raw.get("timeout_seconds", 1800)),
        local_files_only=not bool(raw.get("allow_download", False)),
    )
    for label, path in (
        ("Python 실행 파일", settings.python_executable),
        ("정밀화 실행기", settings.runner_path),
        ("모델 캐시", settings.model_cache_dir),
    ):
        if not path.exists():
            raise NativeRefinementExecutionError(f"{label}을 찾을 수 없습니다: {path}")
    if settings.width <= 0 or settings.height <= 0:
        raise NativeRefinementExecutionError("정밀화 출력 크기는 양수여야 합니다.")
    if settings.flux_steps <= 0:
        raise NativeRefinementExecutionError("정밀화 단계 수는 양수여야 합니다.")
    if settings.timeout_seconds <= 0:
        raise NativeRefinementExecutionError("정밀화 제한 시간은 양수여야 합니다.")
    return settings


def build_native_refinement_command(
    settings: NativeRefinementExecutionSettings,
    *,
    base_image: Path,
    candidate_record: Path,
    approved_run: Path,
    garment_reference: Path,
    output_directory: Path,
    input_mode: str = "approved_base",
) -> list[str]:
    command = [
        str(settings.python_executable),
        "-u",
        str(settings.runner_path),
        "--base-image",
        str(base_image),
        "--input-mode",
        input_mode,
        "--candidate-record",
        str(candidate_record),
        "--approved-run",
        str(approved_run),
        "--garment-reference",
        str(garment_reference),
        "--output-dir",
        str(output_directory),
        "--python",
        str(settings.python_executable),
        "--model-cache-dir",
        str(settings.model_cache_dir),
        "--width",
        str(settings.width),
        "--height",
        str(settings.height),
        "--flux-steps",
        str(settings.flux_steps),
        "--minimum-similarity",
        str(settings.minimum_similarity),
        "--minimum-color-similarity",
        str(settings.minimum_color_similarity),
    ]
    if not settings.local_files_only:
        command.append("--allow-download")
    return command


def create_native_output_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    stem = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = output_root / stem
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stem}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _read_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_log_tail(path: Path, limit: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
    except OSError:
        return ""


def _progress_message(report: dict[str, Any] | None) -> str:
    if not report:
        return "FLUX.2 Klein 로컬 모델 준비 중..."
    attempts = report.get("attempts", {})
    if "flux2_klein" not in attempts:
        return "FLUX.2 Klein 단판 후보 생성 및 게이트 검사 중..."
    if report.get("status") == "PASS":
        engine = report.get("decision", {}).get("selected_engine")
        return f"최종 게이트 통과 - {engine} 결과 준비 중..."
    if report.get("status") == "BASE_LOCKED":
        return "FLUX 정밀화 후보 hard safety 실패 - 승인 Animagine Base 복원 중..."
    if report.get("status") == "SOURCE_LOCKED":
        return "FLUX 직접 편집 후보 hard safety 실패 - 명시적 Animagine fallback 필요"
    return "로컬 정밀화 최종 보고서 작성 중..."


def execute_native_refinement(
    settings: NativeRefinementExecutionSettings,
    *,
    project_root: Path,
    base_image: Path,
    candidate_record: Path,
    approved_run: Path,
    garment_reference: Path,
    output_directory: Path,
    input_mode: str = "approved_base",
    status_callback: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> NativeRefinementExecutionResult:
    if input_mode not in {"approved_base", "source_character_direct"}:
        raise NativeRefinementExecutionError(
            f"지원하지 않는 Native 입력 모드입니다: {input_mode}"
        )
    required = (
        (
            "승인 원본 캐릭터"
            if input_mode == "source_character_direct"
            else "승인 Base",
            base_image,
        ),
        ("Base 후보 기록", candidate_record),
        ("승인 실행 기록", approved_run),
        ("FLUX 의상 참조", garment_reference),
    )
    for label, path in required:
        if not Path(path).is_file():
            raise NativeRefinementExecutionError(f"{label} 파일이 없습니다: {path}")
    from genai_lab.native_pipeline_contract import (
        require_isolated_garment_board,
    )
    try:
        garment_reference = require_isolated_garment_board(
            garment_reference
        )
    except ValueError as error:
        raise NativeRefinementExecutionError(str(error)) from error

    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "run.json"
    command = build_native_refinement_command(
        settings,
        base_image=base_image,
        candidate_record=candidate_record,
        approved_run=approved_run,
        garment_reference=garment_reference,
        output_directory=output_directory,
        input_mode=input_mode,
    )
    environment = os.environ.copy()
    if settings.local_files_only:
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DIFFUSERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )

    stdout_path = output_directory / "launcher-stdout.log"
    stderr_path = output_directory / "launcher-stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_log, \
            stderr_path.open("w", encoding="utf-8") as stderr_log:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            text=True,
            stdout=stdout_log,
            stderr=stderr_log,
        )
        started = monotonic()
        last_message = ""
        last_report_read_at = float("-inf")
        report = None
        while process.poll() is None:
            now = monotonic()
            if cancelled():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise NativeRefinementExecutionError(
                    "사용자가 로컬 정밀화를 취소했습니다."
                )
            if now - started > settings.timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise NativeRefinementExecutionError(
                    f"로컬 정밀화 제한 시간 {settings.timeout_seconds:g}초를 초과했습니다."
                )
            # SMB 출력 경로에서는 run.json 전체 읽기를 제한하되 취소 확인은
            # 짧은 주기로 유지한다.
            if now - last_report_read_at >= 2.0:
                report = _read_report(report_path)
                last_report_read_at = now
                message = _progress_message(report)
                if status_callback is not None and message != last_message:
                    status_callback(message)
                    last_message = message
            sleep(0.25)
        process.wait()
    report = _read_report(report_path)
    if report is None:
        raise NativeRefinementExecutionError(
            "로컬 정밀화 실행기가 run.json을 만들지 않았습니다."
        )

    status = str(report.get("status", "ERROR"))
    if status == "SOURCE_LOCKED":
        reason = report.get("decision", {}).get(
            "reason", "all_native_candidates_failed_hard_safety"
        )
        raise NativeRefinementExecutionError(
            "원본 직접 편집 후보가 모두 hard safety에 실패했습니다. "
            "Animagine fallback은 자동 실행하지 않습니다. "
            f"진단={reason}, 보고서={report_path}"
        )
    if status not in {"PASS", "BASE_LOCKED"}:
        error = (
            report.get("error")
            or _read_log_tail(stderr_path)
            or _read_log_tail(stdout_path)
            or f"exit code {process.returncode}"
        )
        raise NativeRefinementExecutionError(
            f"로컬 정밀화 실행 실패: {error or '원인 기록 없음'}"
        )
    if status == "PASS":
        selected = report.get("selected_output")
        if not selected or not Path(selected).is_file():
            raise NativeRefinementExecutionError(
                "최종 게이트 통과 결과 파일이 없습니다."
            )
        selected_path = Path(selected)
        selected_engine = report.get("decision", {}).get("selected_engine")
    else:
        selected_path = Path(base_image)
        selected_engine = None

    return NativeRefinementExecutionResult(
        status=status,
        selected_engine=(
            str(selected_engine) if selected_engine is not None else None
        ),
        selected_image_path=selected_path,
        report_path=report_path,
        output_directory=output_directory,
        report=report,
    )


