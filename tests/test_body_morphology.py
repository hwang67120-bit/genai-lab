from dataclasses import replace

from PIL import Image

from genai_lab.body_morphology import (
    BodyMorphologyVector,
    COMPONENT_NAMES,
    freeze_body_morphology,
)
from genai_lab.request import (
    CharacterFramingType,
    CharacterGenerationRequest,
)


def request(seed=1234):
    return CharacterGenerationRequest(
        reference_image=Image.new("RGB", (8, 8), "white"),
        reference_image_name="character.png",
        reference_enhancement_applied=False,
        reference_enhancement_model_id=None,
        reference_quality_status="good",
        framing_type=CharacterFramingType.FULL_BODY,
        width=8,
        height=8,
        prompt="prompt",
        negative_prompt="negative",
        seed=seed,
        candidate_number=1,
        inference_steps=20,
        guidance_scale=5.5,
        original_image_change_strength=.25,
        reference_image_strength=.7,
        model_id="model",
        reference_adapter_id="adapter",
    )


def test_same_seed_freezes_the_same_body_morphology():
    first = freeze_body_morphology(1234)
    second = freeze_body_morphology(1234)
    assert first == second
    assert first.source_seed == 1234
    assert set(first.values()) == set(COMPONENT_NAMES)
    assert all(.05 <= value <= .95 for value in first.values().values())


def test_different_request_seed_changes_body_morphology():
    assert freeze_body_morphology(1234) != freeze_body_morphology(1235)


def test_candidate_seed_changes_do_not_resample_request_vector():
    base = request(1234)
    candidate = replace(base, seed=1235, candidate_number=2)
    try:
        assert candidate.body_morphology is base.body_morphology
        assert candidate.body_morphology.source_seed == 1234
    finally:
        base.reference_image.close()


def test_body_morphology_record_round_trip():
    vector = freeze_body_morphology(88)
    restored = BodyMorphologyVector.from_record(vector.record())
    assert restored == vector
    assert vector.record()["gender_gate_input"] is False
    assert vector.record()["model_conditioning_applied"] is False
