"""Load a GPU-approved Animagine Base for an exact visual fingerprint."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from PIL import Image

from genai_lab.result import CharacterGenerationCandidate


VERSION = "approved_base_replay_v1"


class ApprovedBaseReplayError(RuntimeError):
    """The configured approved Base cannot be trusted or reconstructed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_file(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ApprovedBaseReplayError(f"{label} path is required.")
    root = Path(project_root).resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ApprovedBaseReplayError(
            f"{label} must stay inside the project: {candidate}"
        ) from error
    if not candidate.is_file():
        raise ApprovedBaseReplayError(f"{label} file is missing: {candidate}")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ApprovedBaseReplayError(
            f"{label} JSON cannot be read: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ApprovedBaseReplayError(f"{label} must be a JSON object.")
    return value


def _require_hash(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ApprovedBaseReplayError(f"{label} SHA-256 is invalid.")
    actual = _sha256(path)
    if actual != expected.lower():
        raise ApprovedBaseReplayError(
            f"{label} SHA-256 mismatch: expected={expected[:12]}, "
            f"actual={actual[:12]}"
        )
    return actual


def _require_passed_record(record: Mapping[str, Any]) -> None:
    checks = {
        "integrity": (
            (record.get("generated_image_integrity") or {}).get("status"),
            "passed",
        ),
        "semantic": (
            (record.get("candidate_semantic_gate") or {}).get("action"),
            "pass",
        ),
        "structure": (
            (record.get("candidate_structure_gate") or {}).get("action"),
            "pass",
        ),
        "similarity": (
            (record.get("candidate_decision") or {}).get("action"),
            "keep",
        ),
    }
    failures = [
        f"{name}={actual!r}"
        for name, (actual, expected) in checks.items()
        if str(actual).lower() != expected
    ]
    if failures:
        raise ApprovedBaseReplayError(
            "Approved Base gate record is not fully passed: "
            + ", ".join(failures)
        )


def _request_value(request: Any, name: str, default: Any = None) -> Any:
    return getattr(request, name, default)


def load_approved_base_batch(
    config: Mapping[str, Any],
    request: Any,
    project_root: Path,
    inputs: Any,
    approval: Any,
):
    """Return one exact approved Base batch, or None for a nonmatching input."""
    section = (config or {}).get("approved_base_replay", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ApprovedBaseReplayError(
            "approved_base_replay must be a mapping."
        )
    enabled = section.get("enabled", False)
    if enabled is False:
        return None
    if enabled is not True:
        raise ApprovedBaseReplayError(
            "approved_base_replay.enabled must be true or false."
        )

    manifest_path = _project_file(
        Path(project_root), section.get("manifest"), "Approved Base manifest"
    )
    manifest = _read_json(manifest_path, "Approved Base manifest")
    if manifest.get("version") != VERSION:
        raise ApprovedBaseReplayError(
            f"Unsupported Approved Base version: {manifest.get('version')!r}"
        )
    if manifest.get("status") != "approved_gpu_baseline":
        raise ApprovedBaseReplayError(
            "Approved Base manifest is not marked approved_gpu_baseline."
        )

    if not hasattr(approval, "record"):
        raise ApprovedBaseReplayError(
            "Current approved input record is unavailable."
        )
    current_approval = approval.record()
    current_visual = current_approval.get("visual_fingerprint")
    expected_visual = manifest.get("visual_fingerprint")
    if current_visual != expected_visual:
        return None

    expected_model = manifest.get("model_id")
    if _request_value(request, "model_id") != expected_model:
        raise ApprovedBaseReplayError(
            "Current model does not match the GPU-approved Base model."
        )
    for name in ("width", "height"):
        if _request_value(request, name) != manifest.get(name):
            raise ApprovedBaseReplayError(
                f"Current {name} does not match the GPU-approved Base."
            )
    current_gender = current_approval.get("gender")
    if current_gender != manifest.get("gender"):
        raise ApprovedBaseReplayError(
            "Current gender contract does not match the GPU-approved Base."
        )

    base_path = _project_file(
        manifest_path.parent,
        manifest.get("base_image"),
        "Approved Base image",
    )
    record_path = _project_file(
        manifest_path.parent,
        manifest.get("candidate_record"),
        "Approved Base candidate record",
    )
    base_hash = _require_hash(
        base_path, manifest.get("base_sha256"), "Approved Base image"
    )
    _require_hash(
        record_path,
        manifest.get("candidate_record_sha256"),
        "Approved Base candidate record",
    )
    source_record = _read_json(
        record_path, "Approved Base candidate record"
    )
    _require_passed_record(source_record)

    candidate_number = manifest.get("candidate_number")
    candidate_seed = manifest.get("candidate_seed")
    if (
        type(candidate_number) is not int
        or source_record.get("candidate_number") != candidate_number
        or type(candidate_seed) is not int
        or source_record.get("seed") != candidate_seed
    ):
        raise ApprovedBaseReplayError(
            "Approved Base seed or candidate number does not match its record."
        )

    with Image.open(base_path) as opened:
        image = opened.convert("RGB").copy()
    if image.size != (manifest.get("width"), manifest.get("height")):
        image.close()
        raise ApprovedBaseReplayError(
            "Approved Base dimensions do not match its manifest."
        )

    directory = Path(tempfile.mkdtemp(prefix="genai-approved-base-"))
    candidate_path = directory / f"candidate_{candidate_number}.png"
    shutil.copyfile(base_path, candidate_path)

    replay_record = dict(source_record)
    replay_record["approved_generation_fingerprint"] = approval.fingerprint
    replay_record["approval"] = "pending"
    replay_record["approved_base_replay"] = {
        "version": VERSION,
        "status": "replayed_exact_gpu_baseline",
        "visual_fingerprint": expected_visual,
        "base_sha256": base_hash,
        "source_approval_fingerprint": manifest.get(
            "source_approval_fingerprint"
        ),
        "current_approval_fingerprint": approval.fingerprint,
        "source_log": manifest.get("source_log"),
        "image_merge_used": False,
    }
    candidate_path.with_suffix(".json").write_text(
        json.dumps(replay_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    current_approval = dict(current_approval)
    current_approval["approval_fingerprint"] = approval.fingerprint
    (directory / "approved_generation.json").write_text(
        json.dumps(current_approval, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    inputs.source.save(directory / "input_source.png")
    inputs.garment.save(directory / "input_garment.png")

    framing = _request_value(request, "framing_type")
    framing_value = getattr(framing, "value", framing)
    candidate = CharacterGenerationCandidate(
        image=image,
        original_generated_image=None,
        reference_image_name=_request_value(
            request, "reference_image_name", "approved-reference"
        ),
        before_clothing_image=None,
        clothing_change_mask=None,
        clothing_reference_name="input_garment.png",
        clothing_category=None,
        clothing_try_on_status="approved_base_replay",
        clothing_verification_warning_ko=None,
        reference_enhancement_applied=bool(
            _request_value(request, "reference_enhancement_applied", False)
        ),
        reference_enhancement_model_id=_request_value(
            request, "reference_enhancement_model_id"
        ),
        reference_quality_status=_request_value(
            request, "reference_quality_status", "approved"
        ),
        framing_type=str(framing_value),
        seed=candidate_seed,
        candidate_number=candidate_number,
        prompt=str(source_record.get("prompt") or ""),
        negative_prompt=str(source_record.get("negative") or ""),
        model_id=str(expected_model),
        reference_adapter_id=str(
            _request_value(request, "reference_adapter_id", "")
        ),
        original_image_change_strength=float(
            _request_value(request, "original_image_change_strength", 0.0)
        ),
        reference_image_strength=float(
            _request_value(request, "reference_image_strength", 0.0)
        ),
        pose_control_status="not_used",
        pose_control_model_id=None,
        pose_control_conditioning_scale=None,
        pose_control_guidance_start=None,
        pose_control_guidance_end=None,
        detail_correction_status="approved_base_replay",
        detected_face_count=0,
        detected_hand_count=0,
        corrected_region_count=0,
        rejected_region_count=0,
        detail_verification_warning_ko=None,
        elapsed_seconds=0.0,
        peak_vram_bytes=0,
        generated_at=datetime.now().astimezone().isoformat(),
        design_reference_record=replay_record,
    )

    from genai_lab.visual_reference import CandidateBatch

    return CandidateBatch(
        candidates=[candidate],
        paths=[candidate_path],
        directory=directory,
        warning=(
            "\uac80\uc99d\ub41c GPU Base\ub97c \ub3d9\uc77c\ud55c "
            "\uc2dc\uac01 \uc785\ub825 \uc9c0\ubb38\uc5d0\uc11c "
            "\uc7ac\uc0ac\uc6a9\ud588\uc2b5\ub2c8\ub2e4."
        ),
        quarantined=[],
        review_stage="native_base",
    )
