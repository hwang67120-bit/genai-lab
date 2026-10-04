from genai_lab.candidate_pipeline import (
    CandidateAction,
    candidate_rank_key,
    final_candidate_decision,
    preliminary_candidate_decision,
    resolve_candidate_pipeline,
    select_retry_phase,
    summarize_candidate_failures,
)


def test_adaptive_candidate_settings_use_approved_candidate_count_as_attempt_cap():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 2,
            "repair_best_candidate_only": True,
        },
    }, 4)
    assert settings.target_valid_candidates == 2
    assert settings.maximum_attempts == 4
    assert settings.gender_retry_attempts == 0
    assert settings.record()["gender_retry_policy"] == "disabled"
    assert settings.quality_retry_attempts == 0


def test_gender_retry_attempts_are_bounded_and_recorded():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "gender_retry_attempts": 2,
            "quality_retry_attempts": 1,
        },
    }, 2)
    assert settings.gender_retry_attempts == 2
    assert settings.quality_retry_attempts == 1
    assert settings.record()["gender_retry_policy"] == (
        "only_after_initial_pool_all_gender_conflicts")
    assert settings.record()["quality_retry_policy"] == (
        "after_any_non_gender_base_failure")


def test_preliminary_decision_quarantines_mechanical_corruption():
    decision = preliminary_candidate_decision({
        "final_latent_integrity": {"status": "passed"},
        "generated_image_integrity": {"status": "corrupted"},
    })
    assert decision.action == CandidateAction.QUARANTINE
    assert decision.reasons == ("generated_image_integrity",)


def test_ranking_prioritizes_balanced_minimum_then_average():
    stronger_identity = {
        "similarity": {"character": .9, "garment": .1, "vibe": .1},
    }
    stronger_garment = {
        "similarity": {"character": .8, "garment": 1.0, "vibe": 1.0},
    }
    assert candidate_rank_key(stronger_identity) < candidate_rank_key(
        stronger_garment)


def test_preliminary_decision_retries_when_one_component_is_below_gate():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "minimum_component_similarity": .65,
            "require_all_similarity_scores": True,
        },
    }, 2)
    decision = preliminary_candidate_decision({
        "similarity": {"character": .80, "garment": .51, "vibe": .72},
    }, settings)
    assert decision.action == CandidateAction.RETRY
    assert decision.reasons == ("similarity_below_threshold:garment",)


def test_failed_post_audit_is_quarantined_and_unresolved_requires_review():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "reject_failed_post_audit": True,
        },
    }, 2)
    assert final_candidate_decision(
        {"status": "FAIL"}, settings).action == CandidateAction.QUARANTINE
    assert final_candidate_decision(
        {"status": "UNRESOLVED"}, settings).action == CandidateAction.REVIEW


def test_post_repair_mechanical_corruption_is_always_quarantined():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "reject_failed_post_audit": False,
        },
    }, 2)
    decision = final_candidate_decision(
        {"status": "UNRESOLVED"},
        settings,
        {"status": "corrupted"},
    )
    assert decision.action == CandidateAction.QUARANTINE
    assert decision.reasons == ("post_repair_image_integrity",)


def test_approved_color_unresolved_is_quarantined_even_when_generic_fail_rejection_is_off():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "reject_failed_post_audit": False,
        },
    }, 2)
    decision = final_candidate_decision({
        "status": "UNRESOLVED",
        "blocking_color_checks": {
            "animal_ears_color": "output_part_audit_missing",
        },
    }, settings)
    assert decision.action == CandidateAction.QUARANTINE
    assert decision.reasons == (
        "approved_part_color_audit_failed_or_unresolved",
    )



def test_retry_phase_uses_gender_only_when_every_initial_candidate_conflicts():
    settings = resolve_candidate_pipeline({"candidate_pipeline": {
        "enabled": True, "target_valid_candidates": 1,
        "maximum_attempts": 3, "gender_retry_attempts": 2,
        "quality_retry_attempts": 2,
    }}, 3)
    assert select_retry_phase(
        [True, True, True], [False, False, False], settings) == ("gender", 2)


def test_retry_phase_uses_quality_for_semantic_structure_pass_similarity_failure():
    settings = resolve_candidate_pipeline({"candidate_pipeline": {
        "enabled": True, "target_valid_candidates": 1,
        "maximum_attempts": 3, "gender_retry_attempts": 2,
        "quality_retry_attempts": 2,
    }}, 3)
    assert select_retry_phase(
        [False, True, False], [False, False, True], settings) == ("quality", 2)


def test_failure_summary_aggregates_machine_readable_categories():
    settings = resolve_candidate_pipeline({"candidate_pipeline": {
        "enabled": True, "target_valid_candidates": 1,
        "maximum_attempts": 2, "quality_retry_attempts": 2,
    }}, 2)
    summary = summarize_candidate_failures([
        {"failure_stage": "candidate_semantic_gate", "report": {
            "violations": ["gender_conflict:1girl"]}},
        {"failure_stage": "candidate_similarity_gate", "report": {
            "reasons": ["similarity_below_threshold:character"]}},
        {"failure_stage": "candidate_structure_gate", "report": {
            "violations": ["large_detached_object_above_character"]}},
    ], settings, retry_phase="quality")
    assert summary["failure_category_counts"] == {
        "gender_conflict": 1, "similarity_weak": 1,
        "structure_invalid": 1,
    }
    assert summary["next_action"] == "quality_retry_exhausted"


def test_staged_base_similarity_uses_character_and_hair_only():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "minimum_component_similarity": .65,
            "require_all_similarity_scores": True,
            "output_coordinate_similarity": True,
            "base_similarity_components": ["character", "hair"],
            "final_similarity_components": ["character", "hair", "garment", "vibe"],
        },
    }, 2)
    base = preliminary_candidate_decision({
        "similarity": {
            "character": .66, "hair": .69, "garment": .20, "vibe": .10,
        },
    }, settings, stage="base")
    assert base.action == CandidateAction.KEEP
    assert base.report["gating_components"] == ["character", "hair"]
    final = preliminary_candidate_decision({
        "similarity": {
            "character": .66, "hair": .69, "garment": .20, "vibe": .70,
        },
    }, settings, stage="final")
    assert final.action == CandidateAction.RETRY
    assert final.reasons == ("similarity_below_threshold:garment",)


def test_output_coordinate_hair_gate_fails_closed_when_score_missing():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "output_coordinate_similarity": True,
            "base_similarity_components": ["character", "hair"],
        },
    }, 2)
    decision = preliminary_candidate_decision({
        "similarity": {"character": .80, "hair": None},
    }, settings, stage="base")
    assert settings.require_all_similarity_scores is True
    assert decision.action == CandidateAction.RETRY
    assert decision.reasons == ("similarity_unresolved:hair",)


def test_ranking_uses_the_components_that_gated_the_candidate():
    record = {
        "similarity": {
            "character": .72, "hair": .68, "garment": .05, "vibe": .10,
        },
        "candidate_decision": {
            "gating_components": ["character", "hair"],
        },
    }
    assert candidate_rank_key(record) == (.68, .70, .72, .68)

def test_relaxed_policy_keeps_low_similarity_for_refinement():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "minimum_component_similarity": .65,
            "require_all_similarity_scores": True,
            "return_policy": "hard_safety_only",
        }
    }, 2)
    decision = preliminary_candidate_decision({
        "similarity": {
            "character": .48,
            "hair": .57,
            "garment": .59,
            "vibe": .61,
        }
    }, settings, stage="final")
    assert decision.action is CandidateAction.KEEP
    assert decision.report["quality_status"] == "refinement_required"
    assert decision.report["similarity_percentages"]["character"] == 48.0
    assert decision.report["quality_scores_block_return"] is False


def test_relaxed_final_decision_does_not_block_gender_diagnostic():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "return_policy": "hard_safety_only",
        }
    }, 2)
    soft = final_candidate_decision({
        "status": "FAIL",
        "checks": {
            "hair": {"status": "FAIL", "reason": "different"},
            "garment": {"status": "FAIL", "reason": "different"},
        },
        "blocking_color_checks": {"tail_color": "different"},
    }, settings, {"status": "passed"})
    assert soft.action is CandidateAction.REVIEW
    assert soft.report["refinement_required"] is True

    gender = final_candidate_decision({
        "status": "FAIL",
        "checks": {
            "gender": {"status": "FAIL", "reason": "gender_conflict:1girl"},
        },
    }, settings, {"status": "passed"})
    assert gender.action is CandidateAction.REVIEW
    assert gender.report["refinement_required"] is True



def test_needs_refinement_status_is_returned_for_review():
    settings = resolve_candidate_pipeline({
        "candidate_pipeline": {
            "enabled": True,
            "target_valid_candidates": 1,
            "maximum_attempts": 2,
            "return_policy": "hard_safety_only",
        }
    }, 2)
    decision = final_candidate_decision({
        "status": "PASS",
        "checks": {
            "similarity": {
                "status": "NEEDS_REFINEMENT",
                "reason": "below_target",
            },
        },
    }, settings, {"status": "passed"})
    assert decision.action is CandidateAction.REVIEW
    assert decision.report["refinement_targets"] == ["similarity"]



def test_failure_summary_separates_reference_measurement_failure():
    from genai_lab.candidate_pipeline import (
        resolve_candidate_pipeline,
        summarize_candidate_failures,
    )
    settings = resolve_candidate_pipeline(
        {"candidate_pipeline": {"enabled": True}},
        configured_candidate_count=3,
    )
    report = summarize_candidate_failures([{
        "failure_stage": "reference_base_mask_validation",
        "report": {
            "reasons": ["hair:base_mask_below_measurement_minimum"],
        },
    }], settings)
    assert report["failure_category_counts"] == {
        "reference_not_measurable": 1
    }
