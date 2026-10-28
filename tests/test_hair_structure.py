from genai_lab.hair_structure import (
    evaluate_hair_structure_diagnostic,
    hair_structure_guidance,
    resolve_hair_structure,
)


def test_hair_parts_are_separate_and_missing_parts_stay_unknown():
    contract = resolve_hair_structure(
        ("short_hair", "blunt_bangs", "sidelocks", "high_ponytail", "blue_hair"),
        {"views": [{"name": "rear_silhouette_candidate"}]},
    )

    assert contract["status"] == "PARTIAL"
    assert contract["components"]["length"]["approved_tags"] == ["short hair"]
    assert contract["components"]["front"]["approved_tags"] == ["blunt bangs"]
    assert contract["components"]["side"]["approved_tags"] == ["sidelocks"]
    assert contract["components"]["arrangement"]["approved_tags"] == ["high ponytail"]
    assert contract["components"]["texture"]["state"] == "unknown"
    assert contract["components"]["rear_silhouette"]["state"] == "observed_candidate"
    assert contract["components"]["rear_silhouette"]["semantic_identity_confirmed"] is False
    assert contract["absence_is_not_failure"] is True


def test_guidance_contains_only_confirmed_parts():
    contract = resolve_hair_structure(("long_hair", "wavy_hair"))

    guidance = hair_structure_guidance(contract)

    assert "overall length (long hair)" in guidance
    assert "hair texture (wavy hair)" in guidance
    assert "front hair" not in guidance


def test_part_diagnostic_is_advisory_and_separate():
    contract = resolve_hair_structure(("short_hair", "blunt_bangs", "blue_hair"))
    diagnostic = evaluate_hair_structure_diagnostic(
        contract,
        {"long_hair": .82, "short_hair": .10, "bangs": .74, "blue_hair": .71},
    )

    assert diagnostic["status"] == "DIAGNOSTIC"
    assert diagnostic["reason"] == "possible_hair_part_mismatch"
    assert diagnostic["component_checks"]["length"]["status"] == "DIAGNOSTIC"
    assert diagnostic["component_checks"]["front"]["status"] == "PASS"
    assert diagnostic["component_checks"]["color"]["status"] == "PASS"
    assert diagnostic["enforced"] is False


def test_missing_output_evidence_is_unresolved_not_failure():
    contract = resolve_hair_structure(("short_hair", "sidelocks"))
    diagnostic = evaluate_hair_structure_diagnostic(contract, {})

    assert diagnostic["status"] == "UNRESOLVED"
    assert diagnostic["reason"] == "hair_part_evidence_incomplete"
    assert diagnostic["component_checks"]["side"]["status"] == "UNRESOLVED"
    assert diagnostic["absence_is_not_failure"] is True
