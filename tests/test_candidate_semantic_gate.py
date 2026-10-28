from types import SimpleNamespace

from genai_lab.candidate_semantic_gate import (
    CandidateSemanticGateSettings,
    compare_lower_body_transition,
    evaluate_candidate_semantics,
    reset_candidate_runtime,
)


def approved():
    return {
        "gender": "male",
        "approved_character_tags": ["raccoon ears", "raccoon tail"],
        "approved_tags": ["skirt", "jacket", "uniform"],
    }


def test_gender_is_diagnostic_while_species_and_pants_still_retry():
    report = evaluate_candidate_semantics(approved(), {
        "1girl": .82,
        "1boy": .20,
        "cat_ears": .75,
        "raccoon_ears": .10,
        "pants": .80,
    }, CandidateSemanticGateSettings())
    assert report["action"] == "retry"
    assert "gender_conflict:1girl" in report["diagnostics"]
    assert "gender_conflict:1girl" not in report["violations"]
    assert "ears_species_conflict:cat ears" in report["violations"]
    assert "unapproved_lower_body_garment:pants" in report["violations"]


def test_base_stage_gender_conflict_is_diagnostic_only():
    contract = approved()
    contract["approved_character_tags"] = []
    report = evaluate_candidate_semantics(
        contract,
        {"1girl": .98, "1boy": .03},
        CandidateSemanticGateSettings(),
        stage="base",
    )
    assert report["action"] == "pass"
    assert report["violations"] == []
    assert report["diagnostics"] == ["gender_conflict:1girl"]
    assert report["checks"]["gender"]["status"] == "DIAGNOSTIC"
    assert report["checks"]["gender"]["enforced"] is False


def test_gender_gate_can_only_be_reenabled_explicitly():
    report = evaluate_candidate_semantics(
        approved(), {"1girl": .98, "1boy": .03},
        CandidateSemanticGateSettings(enforce_gender=True), stage="base")
    assert report["action"] == "retry"
    assert report["violations"] == ["gender_conflict:1girl"]


def test_missing_detector_evidence_is_not_a_false_failure():
    report = evaluate_candidate_semantics(
        approved(), {}, CandidateSemanticGateSettings())
    assert report["action"] == "pass"
    assert report["absence_is_not_failure"] is True


def test_tuple_scores_from_wd14_are_supported():
    report = evaluate_candidate_semantics(
        approved(), (("1boy", .91), ("raccoon_ears", .73)),
        CandidateSemanticGateSettings())
    assert report["action"] == "pass"


def test_two_piece_topology_mismatch_is_diagnostic_only():
    contract = approved()
    from genai_lab.garment_topology import resolve_garment_topology
    contract["garment_topology"] = resolve_garment_topology(("bikini",))
    report = evaluate_candidate_semantics(
        contract,
        {"bodysuit": .82, "bikini": .12},
        CandidateSemanticGateSettings(),
    )

    assert report["action"] == "pass"
    assert report["checks"]["garment_topology"]["status"] == "DIAGNOSTIC"
    assert "garment_topology:possible_topology_mismatch" in report["diagnostics"]
    assert report["violations"] == []


def test_approved_stockings_are_not_treated_as_novel_garment():
    contract = approved()
    contract["approved_tags"].append("stockings")
    report = evaluate_candidate_semantics(
        contract, {"stockings": .9}, CandidateSemanticGateSettings())
    assert report["action"] == "pass"


def test_descriptive_trousers_approve_equivalent_pants_detector_label():
    contract = approved()
    contract["approved_tags"].append("black wide-leg trousers")
    report = evaluate_candidate_semantics(
        contract, {"pants": .9}, CandidateSemanticGateSettings())

    assert report["action"] == "pass"
    check = report["checks"]["unapproved_lower_body_garment"]
    assert check["status"] == "PASS"
    assert check["approved"] == ["pants", "trousers"]


def test_runtime_reset_recreates_scheduler_and_restores_scales(monkeypatch):
    emptied = []
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: emptied.append(True))

    class Scheduler:
        config = {"name": "test"}

        @classmethod
        def from_config(cls, config):
            assert config == {"name": "test"}
            return cls()

    scales = []
    pipeline = SimpleNamespace(
        scheduler=Scheduler(), _interrupt=True,
        maybe_free_model_hooks=lambda: None,
        set_ip_adapter_scale=scales.append,
    )
    visual_inputs = SimpleNamespace(
        identity=object(), identity_mask=object(), garment=object(),
        garment_mask=object(), hair_reference=None, extra_references=(),
        scene_condition=None,
    )
    report = reset_candidate_runtime(pipeline, visual_inputs, .7, .45)
    assert report["scheduler_recreated"] is True
    assert pipeline._interrupt is False
    assert scales == [[[.7, .45]]]
    assert emptied == [True]


def test_hair_structure_mismatch_is_diagnostic_only():
    from genai_lab.hair_structure import resolve_hair_structure

    contract = approved()
    contract["hair_structure"] = resolve_hair_structure(
        ("short_hair", "blunt_bangs")
    )
    report = evaluate_candidate_semantics(
        contract,
        {"long_hair": .82, "short_hair": .10, "bangs": .73},
        CandidateSemanticGateSettings(),
    )

    assert report["action"] == "pass"
    assert report["checks"]["hair_structure"]["status"] == "DIAGNOSTIC"
    assert "hair_structure:possible_hair_part_mismatch" in report["diagnostics"]
    assert report["violations"] == []


def test_lower_body_transition_distinguishes_inherited_and_introduced():
    inherited = compare_lower_body_transition(
        {
            "checks": {
                "unapproved_lower_body_garment": {
                    "detected_conflicts": {"thighhighs": .64},
                },
            },
        },
        {
            "checks": {
                "unapproved_lower_body_garment": {
                    "detected_conflicts": {"thighhighs": .95},
                },
            },
        },
    )
    assert inherited["status"] == "INHERITED_FROM_BASE"
    assert inherited["inherited_conflicts"] == {"thighhighs": .95}
    assert inherited["introduced_conflicts"] == {}
    assert inherited["base_is_not_neutral_when_inherited"] is True

    introduced = compare_lower_body_transition(
        {"checks": {"unapproved_lower_body_garment": {
            "detected_conflicts": {},
        }}},
        {"checks": {"unapproved_lower_body_garment": {
            "detected_conflicts": {"pants": .81},
        }}},
    )
    assert introduced["status"] == "INTRODUCED_BY_REFINEMENT"
    assert introduced["introduced_conflicts"] == {"pants": .81}


def test_lower_body_transition_records_resolution():
    report = compare_lower_body_transition(
        {"checks": {"unapproved_lower_body_garment": {
            "detected_conflicts": {"pants": .70},
        }}},
        {"checks": {"unapproved_lower_body_garment": {
            "detected_conflicts": {},
        }}},
    )
    assert report["status"] == "PASS"
    assert report["resolved_conflicts"] == {"pants": .70}
    assert report["final_violation_remains"] is False
