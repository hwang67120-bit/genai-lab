"""결과 폴더와 result.json 기록만 담당한다."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def create_new_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = output_root / name
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{name}-{suffix}"
        suffix += 1
    (candidate / "images").mkdir(parents=True)
    return candidate


def resolve_resume_directory(candidate: Path, output_root: Path) -> Path:
    candidate = candidate.resolve()
    root = output_root.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"이어서 실행할 폴더는 outputs 안에 있어야 합니다: {candidate}")
    if not (candidate / "result.json").is_file():
        raise ValueError(f"이어서 실행할 result.json이 없습니다: {candidate}")
    if not (candidate / "images").is_dir():
        raise ValueError(f"이어서 실행할 images 폴더가 없습니다: {candidate}")
    return candidate


def load_existing_result(run_directory: Path, fingerprint: str) -> dict[str, Any]:
    try:
        result = json.loads(
            (run_directory / "result.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"기존 result.json을 읽을 수 없습니다: {error}") from error
    if result.get("config_fingerprint") != fingerprint:
        raise ValueError(
            "기존 실행과 현재 설정 또는 prompts.csv가 다릅니다. "
            "같은 설정으로 다시 실행하거나 새 실행을 시작하세요."
        )
    return result


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def initial_result(
    config: dict[str, Any],
    prompts: list[Any],
    environment: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": None,
        "config_fingerprint": fingerprint,
        "environment": environment,
        "configuration": config,
        "requests": [
            {**asdict(item), "filename": item.filename, "status": "pending"}
            for item in prompts
        ],
        "summary": {"total": len(prompts), "completed": 0, "failed": 0},
    }


def update_request_result(
    result: dict[str, Any], request_id: str, **updates: Any
) -> None:
    for request in result["requests"]:
        if request["request_id"] == request_id:
            request.update(updates)
            return
    raise ValueError(f"실행 기록에서 요청 번호를 찾을 수 없습니다: {request_id}")


def refresh_summary(result: dict[str, Any]) -> None:
    statuses = [item["status"] for item in result["requests"]]
    result["summary"] = {
        "total": len(statuses),
        "completed": statuses.count("completed") + statuses.count("skipped"),
        "failed": statuses.count("failed"),
    }

