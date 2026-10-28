"""메모리 후보와 사용자 승인 후 결과 저장을 담당한다."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from genai_lab.try_on_metrics import TryOnEffectMetricsResult


@dataclass(frozen=True)
class CharacterGenerationCandidate:
    """AI가 생성했지만 아직 승인되거나 저장되지 않은 이미지 후보."""

    image: Image.Image
    original_generated_image: Image.Image | None
    reference_image_name: str
    before_clothing_image: Image.Image | None
    clothing_change_mask: Image.Image | None
    clothing_reference_name: str | None
    clothing_category: str | None
    clothing_try_on_status: str
    clothing_verification_warning_ko: str | None
    reference_enhancement_applied: bool
    reference_enhancement_model_id: str | None
    reference_quality_status: str
    framing_type: str
    seed: int
    candidate_number: int
    prompt: str
    negative_prompt: str
    model_id: str
    reference_adapter_id: str
    original_image_change_strength: float
    reference_image_strength: float
    pose_control_status: str
    pose_control_model_id: str | None
    pose_control_conditioning_scale: float | None
    pose_control_guidance_start: float | None
    pose_control_guidance_end: float | None
    detail_correction_status: str
    detected_face_count: int
    detected_hand_count: int
    corrected_region_count: int
    rejected_region_count: int
    detail_verification_warning_ko: str | None
    elapsed_seconds: float
    peak_vram_bytes: int
    generated_at: str
    raw_clothing_try_on_image: Image.Image | None = None
    clothing_difference_image: Image.Image | None = None
    clothing_effect_metrics: TryOnEffectMetricsResult | None = None
    design_reference_record: dict[str, Any] | None = None


@dataclass(frozen=True)
class CharacterSaveResult:
    """사용자 승인 후 디스크에 기록된 이미지와 설정 파일 경로."""

    image_path: Path
    metadata_path: Path
    storage_class: str
    output_root: Path


class CharacterSaveError(OSError):
    """승인한 후보의 PNG 또는 설정 파일을 저장하지 못한 오류."""


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


def _file_integrity(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_provided", "path": None, "sha256": None}
    source = Path(path)
    if not source.is_file():
        return {
            "status": "not_available",
            "path": str(source),
            "sha256": None,
        }
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "status": "verified",
        "path": str(source.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": source.stat().st_size,
    }


def _mapping_record(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    record = getattr(value, "record", None)
    if callable(record):
        result = record()
        if isinstance(result, dict):
            return dict(result)
    raise CharacterSaveError("8단계 증거 또는 결정 형식이 올바르지 않습니다.")


def save_approved_character_candidate(
    character_candidate: CharacterGenerationCandidate,
    output_root: Path,
    *,
    character_reference_path: Path | str | None = None,
    clothing_reference_path: Path | str | None = None,
    final_review_evidence: Any | None = None,
    final_review_decision: Any | None = None,
) -> CharacterSaveResult:
    """사용자가 저장을 승인한 후보만 PNG와 JSON으로 기록한다.

    반환값:
        저장이 끝난 이미지와 생성 설정 파일 경로.

    오류:
        파일 기록에 실패하면 생성 중인 임시 파일을 제거하고 오류를 발생시킨다.
    """
    output_root = Path(output_root)
    evidence_record = _mapping_record(final_review_evidence)
    decision_record = _mapping_record(final_review_decision)
    if not decision_record:
        candidate_record = character_candidate.design_reference_record or {}
        decision_record = _mapping_record(
            candidate_record.get("final_review_decision")
        )
    decision = decision_record.get("decision")
    if decision not in {"approved", "approved_with_refinement"}:
        raise CharacterSaveError(
            "8단계에서 승인 또는 조건부 승인된 결과만 저장할 수 있습니다."
        )
    if decision_record.get("eligible_for_save") is False:
        raise CharacterSaveError("8단계 결정에서 저장이 허용되지 않은 후보입니다.")

    storage_class = (
        "refinement_checkpoint"
        if decision == "approved_with_refinement"
        else "final_result"
    )
    storage_directory_name = (
        "refinement-pending"
        if storage_class == "refinement_checkpoint"
        else "approved"
    )
    saved_at = datetime.now().astimezone()
    approved_directory = (
        output_root / storage_directory_name / saved_at.strftime("%Y-%m-%d")
    )
    approved_directory.mkdir(parents=True, exist_ok=True)

    framing_filename_codes = {
        "full_body": "full",
        "upper_body": "upper",
        "face": "face",
    }
    framing_filename_code = framing_filename_codes.get(
        character_candidate.framing_type,
        "unknown",
    )
    filename_base = (
        f"{saved_at.strftime('%Y%m%d_%H%M%S')}_"
        f"{framing_filename_code}_"
        f"{character_candidate.candidate_number:02d}_"
        f"seed{character_candidate.seed}"
    )
    image_path, metadata_path = find_available_approved_paths(
        approved_directory,
        filename_base,
    )
    temporary_image_path = image_path.with_suffix(".png.tmp")
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")

    saved_metadata = {
        "version": "stage9_saved_result_v1",
        "stage": 9,
        "status": "saved",
        "storage_class": storage_class,
        "selected_output_root": str(output_root.resolve()),
        "saved_at": saved_at.isoformat(),
        "training_use_approved": False,
        "final_review_evidence": evidence_record or None,
        "final_review_decision": decision_record,
        "source_integrity": {
            "character_reference": _file_integrity(
                character_reference_path
            ),
            "clothing_reference": _file_integrity(
                clothing_reference_path
            ),
        },
        "generated_at": character_candidate.generated_at,
        "reference_image_name": character_candidate.reference_image_name,
        "reference_enhancement_applied": (
            character_candidate.reference_enhancement_applied
        ),
        "reference_enhancement_model_id": (
            character_candidate.reference_enhancement_model_id
        ),
        "reference_quality_status": character_candidate.reference_quality_status,
        "framing_type": character_candidate.framing_type,
        "detail_correction_status": character_candidate.detail_correction_status,
        "clothing_reference_name": character_candidate.clothing_reference_name,
        "clothing_category": character_candidate.clothing_category,
        "clothing_try_on_status": character_candidate.clothing_try_on_status,
        "design_reference_record": character_candidate.design_reference_record,
        "clothing_verification_warning_ko": (
            character_candidate.clothing_verification_warning_ko
        ),
        "clothing_effect_metrics": (
            character_candidate.clothing_effect_metrics.to_dict()
            if character_candidate.clothing_effect_metrics is not None
            else None
        ),
        "detected_face_count": character_candidate.detected_face_count,
        "detected_hand_count": character_candidate.detected_hand_count,
        "corrected_region_count": character_candidate.corrected_region_count,
        "rejected_region_count": character_candidate.rejected_region_count,
        "detail_verification_warning_ko": (
            character_candidate.detail_verification_warning_ko
        ),
        "seed": character_candidate.seed,
        "candidate_number": character_candidate.candidate_number,
        "width": character_candidate.image.width,
        "height": character_candidate.image.height,
        "prompt": character_candidate.prompt,
        "negative_prompt": character_candidate.negative_prompt,
        "model_id": character_candidate.model_id,
        "reference_adapter_id": character_candidate.reference_adapter_id,
        "original_image_change_strength": (
            character_candidate.original_image_change_strength
        ),
        "reference_image_strength": (
            character_candidate.reference_image_strength
        ),
        "pose_control_status": character_candidate.pose_control_status,
        "pose_control_model_id": character_candidate.pose_control_model_id,
        "pose_control_conditioning_scale": (
            character_candidate.pose_control_conditioning_scale
        ),
        "pose_control_guidance_start": (
            character_candidate.pose_control_guidance_start
        ),
        "pose_control_guidance_end": (
            character_candidate.pose_control_guidance_end
        ),
        "elapsed_seconds": character_candidate.elapsed_seconds,
        "peak_vram_bytes": character_candidate.peak_vram_bytes,
        "image_file": image_path.name,
    }

    try:
        character_candidate.image.save(temporary_image_path, format="PNG")
        saved_metadata["image_sha256"] = _file_integrity(
            temporary_image_path
        )["sha256"]
        temporary_metadata_path.write_text(
            json.dumps(saved_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_image_path.replace(image_path)
        temporary_metadata_path.replace(metadata_path)
    except OSError as error:
        for unfinished_path in (
            temporary_image_path,
            temporary_metadata_path,
            image_path,
            metadata_path,
        ):
            unfinished_path.unlink(missing_ok=True)
        raise CharacterSaveError(
            f"승인한 이미지와 설정을 저장하지 못했습니다: {error}"
        ) from error

    return CharacterSaveResult(
        image_path=image_path,
        metadata_path=metadata_path,
        storage_class=storage_class,
        output_root=output_root,
    )


def find_available_approved_paths(
    approved_directory: Path,
    filename_base: str,
) -> tuple[Path, Path]:
    """기존 승인 결과를 덮어쓰지 않는 PNG와 JSON 경로를 반환한다."""
    suffix_number = 0
    while True:
        suffix = "" if suffix_number == 0 else f"_{suffix_number}"
        image_path = approved_directory / f"{filename_base}{suffix}.png"
        metadata_path = approved_directory / f"{filename_base}{suffix}.json"
        if not image_path.exists() and not metadata_path.exists():
            return image_path, metadata_path
        suffix_number += 1


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

