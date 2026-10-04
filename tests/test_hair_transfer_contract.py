from genai_lab.hair_transfer_contract import (
    build_hair_observed_evidence,
    build_hair_transfer_contract,
    build_target_hair_plan,
)


def analysis_report():
    return {
        "status": "analyzed",
        "evidence": [
            {
                "tag": "blunt_bangs",
                "group": "front",
                "max_score": .82,
                "status": "model_candidate",
                "evidence_views": ("front",),
                "eligible_for_prompt": True,
            },
            {
                "tag": "long_hair",
                "group": "length",
                "max_score": .71,
                "status": "geometry_conflict",
                "evidence_views": ("whole",),
                "eligible_for_prompt": False,
            },
        ],
        "rear_silhouette_observation": {
            "status": "observed_candidate",
            "semantic_identity_confirmed": False,
        },
    }


def test_observations_do_not_become_intent_without_user_approval():
    contract = build_hair_transfer_contract(("blue_hair",), analysis_report())

    assert (
        contract["observed_evidence"]["components"]["front"]["state"]
        == "observed_candidate"
    )
    assert (
        contract["observed_evidence"]["components"]["length"]["state"]
        == "conflict"
    )
    assert contract["intent"]["approved_tags"] == ["blue hair"]
    assert "blunt bangs" not in contract["stage_plan"]["pass1_structure_tags"]
    assert contract["uncertainty_policy"]["unresolved_is_not_absent"] is True


def test_pass1_and_pass2_responsibilities_are_separate():
    contract = build_hair_transfer_contract((
        "long_hair", "blunt_bangs", "high_ponytail", "blue_hair",
    ))

    assert contract["stage_plan"]["pass1_structure_tags"] == [
        "long hair", "blunt bangs", "high ponytail",
    ]
    assert contract["stage_plan"]["pass1_appearance_tags"] == ["blue hair"]
    assert contract["stage_plan"]["pass2_local_eligible_tags"] == [
        "blunt bangs", "blue hair",
    ]
    assert contract["stage_plan"]["base_retry_on_mismatch_tags"] == [
        "long hair", "high ponytail",
    ]


def test_missing_detector_candidate_is_unresolved_not_absent():
    evidence = build_hair_observed_evidence({"status": "analyzed"})

    assert evidence["components"]["side"]["state"] == "unresolved"
    assert evidence["components"]["side"]["absence_confirmed"] is False
    assert evidence["missing_candidate_is_not_absence"] is True


def test_target_plan_uses_only_output_coordinates():
    contract = build_hair_transfer_contract(("blunt_bangs",))
    plan = build_target_hair_plan(
        contract,
        {"status": "SELECTED", "selected_regions": ["front"]},
        {"status": "PARTITIONED", "semantic_identity_confirmed": False},
    )

    assert plan["action"] == "local_refine"
    assert plan["coordinate_space"] == "generated_output"
    assert plan["source_coordinates_reused"] is False
    assert plan["selected_regions"] == ["front"]


def test_global_structure_mismatch_routes_to_base_regeneration():
    contract = build_hair_transfer_contract(("long_hair",))
    plan = build_target_hair_plan(
        contract,
        {
            "status": "UNRESOLVED",
            "reason": "pass1_structure_requires_base_regeneration",
            "deferred_to_base_retry": ["length"],
        },
        {"status": "PARTITIONED"},
    )

    assert plan["action"] == "regenerate_base"
    assert plan["automatic_action_authorized"] is False
    assert plan["deferred_to_base_retry"] == ["length"]