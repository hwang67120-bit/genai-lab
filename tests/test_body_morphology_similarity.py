from PIL import Image, ImageDraw

from genai_lab.body_morphology import BodyMorphologyVector, freeze_body_morphology
from genai_lab.body_morphology_similarity import (
    BodyMorphologySimilaritySettings,
    evaluate_body_morphology_similarity,
)


def silhouette():
    image = Image.new("RGB", (200, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((78, 30, 122, 74), fill="navy")
    draw.rectangle((68, 70, 132, 220), fill="navy")
    draw.rectangle((74, 215, 94, 292), fill="navy")
    draw.rectangle((106, 215, 126, 292), fill="navy")
    return image


def settings():
    return BodyMorphologySimilaritySettings(enabled=True)


def test_records_only_three_observable_silhouette_components():
    image = silhouette()
    try:
        report = evaluate_body_morphology_similarity(
            image, freeze_body_morphology(1234), settings())
    finally:
        image.close()
    assert report["status"] == "RECORDED"
    assert report["observable_components"] == [
        "shoulder_width", "pelvis_width", "waist_hip_ratio"]
    assert report["observable_component_count"] == 3
    assert report["coverage_percentage"] == 33.33
    assert report["component_similarity"]["chest_volume"] is None
    assert report["blocks_return"] is False
    assert report["used_for_retry"] is False
    assert report["percentage_is_probability"] is False


def test_matching_observed_proxy_values_score_one_hundred_percent():
    image = silhouette()
    first = evaluate_body_morphology_similarity(
        image, freeze_body_morphology(7), settings())
    observed = first["observed"]
    target = BodyMorphologyVector(
        source_seed=7,
        shoulder_width=observed["shoulder_width"],
        pelvis_width=observed["pelvis_width"],
        waist_hip_ratio=observed["waist_hip_ratio"],
        chest_volume=.5,
        muscle_mass=.5,
        body_fat=.5,
        jaw_angularity=.5,
        height_ratio=.5,
        limb_thickness=.5,
    )
    try:
        matched = evaluate_body_morphology_similarity(
            image, target, settings())
    finally:
        image.close()
    assert matched["overall_similarity"] == 1.0
    assert matched["overall_similarity_percentage"] == 100.0


def test_non_full_body_result_is_unresolved_without_false_score():
    image = silhouette()
    try:
        report = evaluate_body_morphology_similarity(
            image,
            freeze_body_morphology(5),
            settings(),
            framing_type="upper_body",
        )
    finally:
        image.close()
    assert report["status"] == "UNRESOLVED"
    assert report["reason"] == "full_body_frame_required"
    assert report["overall_similarity_percentage"] is None


def test_structure_rejection_prevents_misleading_body_score():
    image = silhouette()
    try:
        report = evaluate_body_morphology_similarity(
            image,
            freeze_body_morphology(5),
            settings(),
            structure_report={"action": "reject"},
        )
    finally:
        image.close()
    assert report["status"] == "UNRESOLVED"
    assert report["reason"] == "structure_gate_not_passed"
    assert report["blocks_return"] is False


def test_disabled_similarity_has_no_side_effects():
    image = silhouette()
    try:
        report = evaluate_body_morphology_similarity(
            image,
            freeze_body_morphology(5),
            BodyMorphologySimilaritySettings(),
        )
    finally:
        image.close()
    assert report["status"] == "DISABLED"
    assert report["blocks_return"] is False
