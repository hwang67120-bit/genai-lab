"""Fail-closed policy for FLUX.2 Klein full-image refinement.

The FLUX result is a whole image. This module never merges, pastes, or
composites candidate pixels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


CONTRACT_VERSION = "native_refinement_policy_v1"
ENGINE_NAME = "flux2_klein"


class NativeRefinementContractError(ValueError):
    """Raised when the approved Animagine Base contract is incomplete."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy_settings(return_policy: str):
    from types import SimpleNamespace
    if return_policy not in {"strict_all", "hard_safety_only"}:
        raise NativeRefinementContractError(
            f"unsupported native return policy: {return_policy}")
    return SimpleNamespace(return_policy=return_policy)


def _quality_score(
    similarity: Mapping[str, float | None],
    color_similarity: float | None,
) -> float:
    values = []
    for name in ("character", "hair", "garment", "vibe"):
        value = similarity.get(name)
        values.append(
            max(0.0, min(1.0, float(value)))
            if isinstance(value, (int, float)) else 0.0
        )
    values.append(
        max(0.0, min(1.0, float(color_similarity)))
        if isinstance(color_similarity, (int, float)) else 0.0
    )
    return sum(values) / len(values)


def validate_approved_base(
    base_image: str | Path,
    candidate_record: Mapping[str, Any],
    approved_run: Mapping[str, Any],
    *,
    minimum_similarity: float = 0.65,
    return_policy: str = "hard_safety_only",
) -> dict[str, Any]:
    path = Path(base_image)
    if not path.is_file():
        raise NativeRefinementContractError(f"approved Base image not found: {path}")

    candidate_fingerprint = candidate_record.get("approved_generation_fingerprint")
    approval_fingerprint = approved_run.get("approval_fingerprint")
    if (
        not isinstance(candidate_fingerprint, str)
        or len(candidate_fingerprint) != 64
        or candidate_fingerprint != approval_fingerprint
    ):
        raise NativeRefinementContractError(
            "candidate and approved run fingerprints do not match"
        )

    seed = candidate_record.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise NativeRefinementContractError("approved Base seed is invalid")

    from genai_lab.candidate_pipeline import (
        semantic_blocking_reasons,
        similarity_percentages,
        structure_blocking_reasons,
    )
    policy = _policy_settings(return_policy)
    semantic_report = candidate_record.get("candidate_semantic_gate", {})
    structure_report = candidate_record.get("candidate_structure_gate", {})
    semantic_blocking = semantic_blocking_reasons(semantic_report, policy)
    structure_blocking = structure_blocking_reasons(structure_report, policy)
    similarity = candidate_record.get("similarity", {})
    percentages = similarity_percentages(
        similarity, ("character", "hair"))
    refinement_targets = [
        f"similarity_unresolved:{name}"
        if percentages[name] is None
        else f"similarity_below_threshold:{name}"
        for name in ("character", "hair")
        if percentages[name] is None
        or percentages[name] < minimum_similarity * 100.0
    ]
    required = {
        "integrity": (
            candidate_record.get("generated_image_integrity", {}).get("status")
            == "passed"
        ),
        "semantic_hard_safety": not semantic_blocking,
        "structure_hard_safety": not structure_blocking,
    }
    if return_policy == "strict_all" and refinement_targets:
        raise NativeRefinementContractError(
            "approved Base failed strict similarity gates: "
            + ", ".join(refinement_targets)
        )
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise NativeRefinementContractError(
            "approved Base failed required hard-safety gates: "
            + ", ".join(failed)
        )

    return {
        "version": CONTRACT_VERSION,
        "base_image": str(path.resolve()),
        "base_sha256": sha256_file(path),
        "seed": seed,
        "candidate_number": int(candidate_record.get("candidate_number", 0)),
        "approval_fingerprint": approval_fingerprint,
        "prompt": candidate_record.get("positive") or approved_run.get("prompt") or "",
        "gender": approved_run.get("gender"),
        "base_gate": {name: "PASS" for name in required},
        "return_policy": return_policy,
        "quality_diagnostic": {
            "similarity_percentages": percentages,
            "target_percentage": round(minimum_similarity * 100.0, 2),
            "status": (
                "refinement_required"
                if refinement_targets else "meets_target"
            ),
            "refinement_required": bool(refinement_targets),
            "refinement_targets": refinement_targets,
            "percentage_is_probability": False,
        },
        "output_policy": "whole_image_candidate_no_composite",
    }


def validate_approved_source(
    source_image: str | Path,
    source_record: Mapping[str, Any],
    approved_run: Mapping[str, Any],
    *,
    return_policy: str = "hard_safety_only",
) -> dict[str, Any]:
    """승인된 원본 캐릭터를 Animagine Base 검사 없이 Native 입력으로 봉인한다."""
    path = Path(source_image)
    if not path.is_file():
        raise NativeRefinementContractError(
            f"approved source character not found: {path}"
        )
    if source_record.get("input_mode") != "source_character_direct":
        raise NativeRefinementContractError(
            "direct source record has an invalid input mode"
        )
    source_fingerprint = source_record.get("approved_generation_fingerprint")
    approval_fingerprint = approved_run.get("approval_fingerprint")
    if (
        not isinstance(source_fingerprint, str)
        or len(source_fingerprint) != 64
        or source_fingerprint != approval_fingerprint
    ):
        raise NativeRefinementContractError(
            "source character and approved run fingerprints do not match"
        )
    seed = source_record.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise NativeRefinementContractError("direct source seed is invalid")
    _policy_settings(return_policy)
    return {
        "version": "native_direct_source_contract_v1",
        "input_mode": "source_character_direct",
        "base_image": str(path.resolve()),
        "base_sha256": sha256_file(path),
        "seed": seed,
        "candidate_number": int(source_record.get("candidate_number", 1)),
        "approval_fingerprint": approval_fingerprint,
        "prompt": approved_run.get("prompt") or "",
        "gender": approved_run.get("gender"),
        "base_gate": "not_applicable_original_source",
        "return_policy": return_policy,
        "quality_diagnostic": {
            "status": "pending_native_candidate_evaluation",
            "percentage_is_probability": False,
        },
        "output_policy": "whole_image_candidate_no_composite",
        "animagine_base_used": False,
    }


def gate_native_candidate(
    *,
    semantic_report: Mapping[str, Any],
    structure_report: Mapping[str, Any],
    integrity_report: Mapping[str, Any],
    similarity: Mapping[str, float | None],
    minimum_similarity: float,
    color_similarity: float | None,
    minimum_color_similarity: float,
    garment_baseline: float | None = None,
    color_baseline: float | None = None,
    return_policy: str = "hard_safety_only",
) -> dict[str, Any]:
    from genai_lab.candidate_pipeline import (
        semantic_blocking_reasons,
        semantic_refinement_reasons,
        similarity_percentages,
        structure_blocking_reasons,
        structure_refinement_reasons,
    )
    policy = _policy_settings(return_policy)
    similarity_failures: dict[str, Any] = {}
    for name in ("character", "hair", "garment", "vibe"):
        value = similarity.get(name)
        if value is None or float(value) < minimum_similarity:
            similarity_failures[name] = value

    garment = similarity.get("garment")
    if (
        garment_baseline is not None
        and garment is not None
        and float(garment) < garment_baseline
    ):
        similarity_failures["garment_regression"] = garment

    color_passed = (
        color_similarity is not None
        and float(color_similarity) >= minimum_color_similarity
        and (
            color_baseline is None
            or float(color_similarity) >= float(color_baseline)
        )
    )
    semantic_hard = semantic_blocking_reasons(semantic_report, policy)
    structure_hard = structure_blocking_reasons(structure_report, policy)
    semantic_soft = semantic_refinement_reasons(semantic_report, policy)
    structure_soft = structure_refinement_reasons(structure_report, policy)
    hard_checks = {
        "semantic_hard_safety": not semantic_hard,
        "structure_hard_safety": not structure_hard,
        "integrity": integrity_report.get("status") == "passed",
    }
    refinement_targets = [
        *(f"similarity:{name}" for name in similarity_failures),
        *(f"semantic:{name}" for name in semantic_soft),
        *(f"structure:{name}" for name in structure_soft),
    ]
    if not color_passed:
        refinement_targets.append("color")
    strict_quality = return_policy == "strict_all"
    quality_passed = not refinement_targets
    status = (
        "PASS"
        if all(hard_checks.values()) and (quality_passed or not strict_quality)
        else "FAIL"
    )
    percentages = similarity_percentages(
        similarity, ("character", "hair", "garment", "vibe"))
    quality_score = _quality_score(similarity, color_similarity)
    return {
        "status": status,
        "checks": {
            **hard_checks,
            "similarity_target": not similarity_failures,
            "color_target": color_passed,
        },
        "return_policy": return_policy,
        "blocking_checks": [
            name for name, passed in hard_checks.items() if not passed
        ],
        "semantic_blocking_reasons": list(semantic_hard),
        "structure_blocking_reasons": list(structure_hard),
        "similarity_failures": similarity_failures,
        "similarity_percentages": percentages,
        "color_similarity": color_similarity,
        "color_percentage": (
            round(max(0.0, min(1.0, float(color_similarity))) * 100.0, 2)
            if isinstance(color_similarity, (int, float)) else None
        ),
        "quality_score": quality_score,
        "overall_similarity_percentage": round(quality_score * 100.0, 2),
        "quality_status": (
            "refinement_required" if refinement_targets else "meets_target"
        ),
        "refinement_required": bool(refinement_targets),
        "refinement_targets": refinement_targets,
        "minimum_similarity": minimum_similarity,
        "target_percentage": round(minimum_similarity * 100.0, 2),
        "garment_baseline": garment_baseline,
        "minimum_color_similarity": minimum_color_similarity,
        "color_baseline": color_baseline,
        "percentage_is_probability": False,
        "pixel_composite_used": False,
    }


def decide_refinement(attempts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Select the one FLUX candidate or keep the approved input locked."""
    attempt = attempts.get(ENGINE_NAME)
    if attempt is None:
        return {
            "status": "RUN_NEXT",
            "action": "run_flux2_klein",
            "reason": "flux2_klein_not_attempted",
        }
    gate = attempt.get("gate", {})
    if gate.get("status") == "PASS":
        refinement_required = (
            gate.get("quality_status", "meets_target") != "meets_target"
        )
        return {
            "status": "PASS_WITH_REFINEMENT" if refinement_required else "PASS",
            "action": "select_existing_candidate",
            "selected_engine": ENGINE_NAME,
            "reason": (
                "flux2_klein_hard_safe_candidate_selected_for_refinement"
                if refinement_required else "flux2_klein_meets_quality_target"
            ),
            "refinement_required": refinement_required,
            "overall_similarity_percentage": gate.get(
                "overall_similarity_percentage"
            ),
            "refinement_targets": list(gate.get("refinement_targets", ())),
        }
    return {
        "status": "BASE_LOCKED",
        "action": "keep_approved_base_and_seed",
        "selected_engine": None,
        "reason": "flux2_klein_failed_hard_safety",
    }
