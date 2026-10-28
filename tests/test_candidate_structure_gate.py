from PIL import Image, ImageDraw

from genai_lab.candidate_structure_gate import (
    CandidateStructureGateSettings,
    evaluate_candidate_structure,
)


def settings(**overrides):
    values = {"enabled": True, **overrides}
    return CandidateStructureGateSettings(**values)


def test_normal_full_body_silhouette_passes():
    image = Image.new("RGB", (200, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((78, 30, 122, 74), fill="navy")
    draw.rectangle((68, 70, 132, 220), fill="navy")
    draw.rectangle((74, 215, 94, 292), fill="navy")
    draw.rectangle((106, 215, 126, 292), fill="navy")

    report = evaluate_candidate_structure(image, settings())

    assert report["action"] == "pass"
    assert report["checks"]["character_occupancy"]["status"] == "PASS"
    assert report["checks"]["detached_overhead_object"]["status"] == "PASS"


def test_large_detached_object_above_character_is_rejected():
    image = Image.new("RGB", (200, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((75, 95, 125, 278), fill="navy")
    draw.rectangle((18, 14, 182, 62), fill="royalblue")

    report = evaluate_candidate_structure(image, settings())

    assert report["action"] == "reject"
    assert "large_detached_object_above_character" in report["violations"]


def test_nearby_overhead_component_is_reviewed_as_possible_raised_limb():
    image = Image.new("RGB", (200, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((65, 95, 135, 278), fill="navy")
    draw.rectangle((55, 45, 145, 88), fill="royalblue")

    report = evaluate_candidate_structure(image, settings())

    assert report["action"] == "review"
    assert "large_detached_object_above_character" not in report["violations"]
    assert "overhead_component_near_primary_pose_ambiguous" in report["unresolved"]
    assert report["checks"]["detached_overhead_object"]["status"] == "UNRESOLVED"

def test_tiny_character_is_rejected_without_pose_assumption():
    image = Image.new("RGB", (200, 320), "white")
    ImageDraw.Draw(image).rectangle((86, 150, 114, 245), fill="navy")

    report = evaluate_candidate_structure(image, settings())

    assert report["action"] == "reject"
    assert "character_occupancy_too_small" in report["violations"]
    assert report["pose_policy"] == (
        "pose_invariant_no_front_or_standing_requirement")



def test_tall_convex_enclosure_cannot_masquerade_as_character():
    image = Image.new("RGB", (200, 320), "white")
    ImageDraw.Draw(image).ellipse((62, 24, 138, 296), fill="navy")

    report = evaluate_candidate_structure(image, settings())

    assert report["action"] == "reject"
    assert "tall_convex_enclosure_not_character" in report["violations"]
    assert report["checks"]["primary_component_shape"]["status"] == "FAIL"

def test_canvas_spanning_foreground_cannot_masquerade_as_character():
    image = Image.new('RGB', (200, 320), 'white')
    ImageDraw.Draw(image).rectangle((0, 15, 199, 290), fill='lightsteelblue')
    report = evaluate_candidate_structure(image, settings())
    assert report['action'] != 'pass'
    assert 'foreground_bridges_horizontal_borders' in report['unresolved']



