from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.run_native_refinement import (
    common_instruction,
    decision_for_input_mode,
    engine_prompt,
    engine_garment_reference,
)

from genai_lab.native_refinement_policy import (
    NativeRefinementContractError,
    decide_refinement,
    gate_native_candidate,
    validate_approved_base,
    validate_approved_source,
)


def approved_contract(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (32, 48), "white").save(base)
    fingerprint = "a" * 64
    candidate = {
        "approved_generation_fingerprint": fingerprint,
        "seed": 7,
        "candidate_number": 3,
        "positive": "1boy, blue hair",
        "generated_image_integrity": {"status": "passed"},
        "candidate_semantic_gate": {"action": "pass"},
        "candidate_structure_gate": {"action": "pass"},
        "similarity": {"character": 0.72, "hair": 0.74},
    }
    approved = {"approval_fingerprint": fingerprint, "gender": "male"}
    return base, candidate, approved


def passing_gate():
    return {
        "gate": {"status": "PASS"},
        "output": "candidate.png",
    }


def failing_gate():
    return {
        "gate": {"status": "FAIL"},
        "output": "candidate.png",
    }


def test_approved_base_contract_records_no_composite_policy(tmp_path):
    base, candidate, approved = approved_contract(tmp_path)
    contract = validate_approved_base(base, candidate, approved)
    assert contract["seed"] == 7
    assert contract["output_policy"] == "whole_image_candidate_no_composite"
    assert len(contract["base_sha256"]) == 64


def test_approved_base_contract_rejects_fingerprint_mismatch(tmp_path):
    base, candidate, approved = approved_contract(tmp_path)
    approved["approval_fingerprint"] = "b" * 64
    with pytest.raises(NativeRefinementContractError, match="fingerprints"):
        validate_approved_base(base, candidate, approved)


def test_approved_base_contract_records_weak_hair_for_refinement(tmp_path):
    base, candidate, approved = approved_contract(tmp_path)
    candidate["similarity"]["hair"] = 0.61
    contract = validate_approved_base(base, candidate, approved)
    quality = contract["quality_diagnostic"]
    assert quality["refinement_required"] is True
    assert quality["similarity_percentages"]["hair"] == 61.0
    assert quality["refinement_targets"] == [
        "similarity_below_threshold:hair"
    ]


def test_strict_base_contract_still_rejects_weak_hair(tmp_path):
    base, candidate, approved = approved_contract(tmp_path)
    candidate["similarity"]["hair"] = 0.61
    with pytest.raises(NativeRefinementContractError, match="strict similarity"):
        validate_approved_base(
            base, candidate, approved, return_policy="strict_all")


def test_flux_pass_is_selected():
    decision = decide_refinement({"flux2_klein": passing_gate()})
    assert decision["selected_engine"] == "flux2_klein"
    assert decision["action"] == "select_existing_candidate"


def test_low_quality_safe_flux_is_returned_for_refinement():
    decision = decide_refinement({
        "flux2_klein": {
            "gate": {
                "status": "PASS",
                "quality_status": "refinement_required",
                "overall_similarity_percentage": 52.0,
                "refinement_targets": ["similarity:hair"],
            },
        },
    })
    assert decision["selected_engine"] == "flux2_klein"
    assert decision["status"] == "PASS_WITH_REFINEMENT"
    assert decision["overall_similarity_percentage"] == 52.0


def test_flux_failure_locks_approved_base_and_seed():
    decision = decide_refinement({"flux2_klein": failing_gate()})
    assert decision == {
        "status": "BASE_LOCKED",
        "action": "keep_approved_base_and_seed",
        "selected_engine": None,
        "reason": "flux2_klein_failed_hard_safety",
    }


def test_low_similarity_native_candidate_passes_for_refinement():
    report = gate_native_candidate(
        semantic_report={"action": "retry", "violations": [
            "ears_species_conflict:fox ears"]},
        structure_report={"action": "reject", "violations": [
            "character_occupancy_too_small"]},
        integrity_report={"status": "passed"},
        similarity={
            "character": 0.48,
            "hair": 0.57,
            "garment": 0.59,
            "vibe": 0.61,
        },
        minimum_similarity=0.65,
        color_similarity=0.52,
        minimum_color_similarity=0.55,
        return_policy="hard_safety_only",
    )
    assert report["status"] == "PASS"
    assert report["quality_status"] == "refinement_required"
    assert report["similarity_percentages"] == {
        "character": 48.0,
        "hair": 57.0,
        "garment": 59.0,
        "vibe": 61.0,
    }
    assert report["blocking_checks"] == []
    assert "color" in report["refinement_targets"]


def test_gender_conflict_is_refinement_diagnostic_in_relaxed_policy():
    report = gate_native_candidate(
        semantic_report={"action": "retry", "violations": [
            "gender_conflict:1girl"]},
        structure_report={"action": "pass", "violations": []},
        integrity_report={"status": "passed"},
        similarity={
            "character": 0.9, "hair": 0.9,
            "garment": 0.9, "vibe": 0.9,
        },
        minimum_similarity=0.65,
        color_similarity=0.9,
        minimum_color_similarity=0.55,
        return_policy="hard_safety_only",
    )
    assert report["status"] == "PASS"
    assert report["semantic_blocking_reasons"] == []
    assert report["refinement_required"] is True
    assert "semantic:gender_conflict:1girl" in report["refinement_targets"]


def test_candidate_gate_has_no_pixel_composite_check():
    report = gate_native_candidate(
        semantic_report={"action": "pass"},
        structure_report={"action": "pass"},
        integrity_report={"status": "passed"},
        similarity={
            "character": 0.8,
            "hair": 0.8,
            "garment": 0.8,
            "vibe": 0.8,
        },
        minimum_similarity=0.65,
        color_similarity=0.7,
        minimum_color_similarity=0.55,
        garment_baseline=0.6,
        color_baseline=0.5,
    )
    assert report["status"] == "PASS"
    assert report["pixel_composite_used"] is False
    assert "pixel" not in report["checks"]

def test_native_instruction_uses_separated_approved_contract():
    instruction = common_instruction(
        {
            "positive": {
                "effective": "1girl, red hair, pants",
                "required_prompt": "unused",
            }
        },
        {
            "approved_character_tags": ["blue hair", "animal ears", "tail"],
            "approved_tags": ["skirt", "blue jacket"],
            "approved_detail_tags": ["white shirt"],
            "gender": "male",
        },
        None,
    )
    assert "blue hair" in instruction
    assert "skirt" in instruction
    assert "male character" in instruction
    assert "red hair" not in instruction
    assert "1girl" not in instruction
    assert "do not create pants" in instruction


def test_direct_source_contract_does_not_require_animagine_gates(tmp_path):
    source = tmp_path / "input_source.png"
    Image.new("RGB", (32, 48), "white").save(source)
    fingerprint = "d" * 64
    source_record = {
        "input_mode": "source_character_direct",
        "approved_generation_fingerprint": fingerprint,
        "seed": 17,
        "candidate_number": 1,
    }
    approved = {
        "approval_fingerprint": fingerprint,
        "gender": "male",
    }

    contract = validate_approved_source(source, source_record, approved)

    assert contract["input_mode"] == "source_character_direct"
    assert contract["animagine_base_used"] is False
    assert contract["base_gate"] == "not_applicable_original_source"


def test_direct_engine_prompt_names_original_character_authority():
    prompt = engine_prompt(
        "flux2_klein",
        "Keep identity and apply only the garment.",
        "source_character_direct",
    )
    assert "approved original character reference" in prompt
    assert "approved Animagine Base" not in prompt


def test_direct_failure_requires_explicit_animagine_fallback():
    decision = decision_for_input_mode(
        {
            "status": "BASE_LOCKED",
            "action": "keep_approved_base_and_seed",
            "selected_engine": None,
            "reason": "all_native_candidates_failed_hard_safety",
        },
        "source_character_direct",
    )
    assert decision["status"] == "SOURCE_LOCKED"
    assert decision["action"] == "require_explicit_animagine_fallback"


def test_engine_garment_reference_accepts_only_flux(tmp_path):
    from argparse import Namespace
    flux_board = tmp_path / "input_garment.png"
    args = Namespace(garment_reference=flux_board)
    assert engine_garment_reference("flux2_klein", args) == flux_board
    with pytest.raises(ValueError, match="unsupported native engine"):
        engine_garment_reference("other", args)


def test_relaxed_policy_records_diagnostic_gender_conflict_for_refinement():
    report = gate_native_candidate(
        semantic_report={
            "action": "pass",
            "violations": [],
            "unresolved": [],
            "diagnostics": ["gender_conflict:1boy"],
        },
        structure_report={"action": "pass", "violations": []},
        integrity_report={"status": "passed"},
        similarity={
            "character": 0.9, "hair": 0.9,
            "garment": 0.9, "vibe": 0.9,
        },
        minimum_similarity=0.65,
        color_similarity=0.9,
        minimum_color_similarity=0.55,
        return_policy="hard_safety_only",
    )
    assert report["status"] == "PASS"
    assert report["refinement_required"] is True
    assert "semantic:gender_conflict:1boy" in report["refinement_targets"]
