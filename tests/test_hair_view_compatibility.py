from genai_lab.hair_view_compatibility import (
    HairViewGateSettings,
    classify_hair_view,
    compare_hair_views,
)


def test_front_reference_accepts_front_three_quarter_candidate():
    settings = HairViewGateSettings()
    reference = classify_hair_view({"looking_at_viewer": .81}, settings)
    candidate = classify_hair_view({
        "looking_at_viewer": .72,
        "from_side": .44,
    }, settings)
    report = compare_hair_views(reference, candidate, settings)
    assert reference["view"] == "front"
    assert candidate["view"] == "front_three_quarter"
    assert report["status"] == "compatible"
    assert report["inpaint_allowed"] is True
    assert report["allowed_candidate_views"] == [
        "front", "front_three_quarter"]


def test_front_reference_rejects_side_candidate():
    settings = HairViewGateSettings()
    reference = classify_hair_view({"facing_viewer": .76}, settings)
    candidate = classify_hair_view({"profile": .74}, settings)
    report = compare_hair_views(reference, candidate, settings)
    assert candidate["view"] == "side"
    assert report["status"] == "incompatible"
    assert report["reason"] == "unobserved_hair_geometry_required"


def test_ambiguous_candidate_does_not_authorize_inpaint():
    settings = HairViewGateSettings()
    reference = classify_hair_view({"looking_at_viewer": .8}, settings)
    candidate = classify_hair_view({"profile": .37, "facing_viewer": .33}, settings)
    report = compare_hair_views(reference, candidate, settings)
    assert candidate["view"] == "unresolved"
    assert report["status"] == "unresolved"
    assert report["inpaint_allowed"] is False
