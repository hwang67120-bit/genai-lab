"""전달 순서 실험: 잘라낸 부위별 이미지와 강도의 대응을 검증한다."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

from genai_lab.reference_order import (
    adapter_references,
    adapter_references_for_stage,
    adapter_reference_names,
)
from genai_lab.scene_reference import PartReference
from genai_lab.visual_reference import VisualInputs, configure_visual_condition
from genai_lab.reference_experiment_approval import approve_reference_experiment, require_reference_experiment_approval


def mask(x):
    image = Image.new("L", (64, 64))
    image.paste(255, (x, 0, x+8, 48))
    return image


@pytest.fixture
def inputs():
    value = VisualInputs(
        Image.new("RGB", (64, 64), "white"),
        Image.new("RGB", (16, 16), "red"),
        Image.new("RGB", (16, 16), "blue"), mask(0), mask(48),
        extra_references=(
            PartReference("tail", Image.new("RGB", (16, 16), "cyan"), mask(24), .31),
            PartReference("ears", Image.new("RGB", (16, 16), "purple"), mask(12), .19)),
        analysis_record={"part_region_experiment": {"requires_input_review": True}})
    yield value
    value.close()


def test_rgb_mask_scale_move_together_without_mutating_inputs(inputs):
    original_extras = inputs.extra_references
    original_masks = [inputs.identity_mask.tobytes(), inputs.garment_mask.tobytes(),
                      *[r.region.tobytes() for r in inputs.extra_references]]
    seen = {}
    pipeline = SimpleNamespace(_execution_device="cpu",
        set_ip_adapter_scale=lambda value: seen.update(scales=value),
        prepare_ip_adapter_image_embeds=lambda **kw: seen.update(images=kw["ip_adapter_image"]) or ["encoded"])
    approve_reference_experiment(inputs)
    result = configure_visual_condition(pipeline, inputs, .71, .43)
    tail, ears = inputs.extra_references
    assert seen["images"] == [[inputs.identity, ears.rgb, tail.rgb, inputs.garment]]
    assert seen["scales"] == pytest.approx(.41)
    assert result == {"ip_adapter_image_embeds": ["encoded"]}
    assert "cross_attention_kwargs" not in result
    assert adapter_reference_names(inputs) == ["identity", "ears", "tail", "garment"]
    assert inputs.extra_references is original_extras
    assert original_masks == [inputs.identity_mask.tobytes(), inputs.garment_mask.tobytes(),
                             *[r.region.tobytes() for r in inputs.extra_references]]


def test_missing_parts_and_extra_parts_have_stable_order(inputs):
    borrowed = SimpleNamespace(identity=inputs.identity, garment=inputs.garment,
                               identity_mask=inputs.identity_mask, garment_mask=inputs.garment_mask)
    assert adapter_reference_names(borrowed) == ["identity", "garment"]
    borrowed.extra_references = (inputs.extra_references[0],)
    assert adapter_reference_names(borrowed) == ["identity", "tail", "garment"]
    wings = PartReference("wings", Image.new("RGB", (16, 16)), mask(36), .2)
    try:
        borrowed.extra_references = (wings, *inputs.extra_references)
        assert adapter_reference_names(borrowed) == ["identity", "ears", "tail", "wings", "garment"]
    finally:
        wings.close()


def test_separate_hair_reference_is_between_identity_and_animal_parts(inputs):
    hair_reference = Image.new("RGB", (20, 20), (128, 128, 128))
    hair_mask = mask(8)
    old_hair_mask = inputs.hair_mask
    inputs.hair_reference = hair_reference
    inputs.hair_mask = hair_mask
    inputs.hair_reference_scale = .61
    try:
        assert adapter_reference_names(inputs) == [
            "identity", "hair", "ears", "tail", "garment"]
        assert [entry.scale for entry in adapter_references(inputs, .7, .45)] == [
            .7, .61, .19, .31, .45]
    finally:
        inputs.hair_reference = None
        inputs.hair_mask = old_hair_mask
        hair_reference.close()
        hair_mask.close()


def test_base_stage_contains_only_full_character_identity(inputs):
    hair_reference = Image.new('RGB', (20, 20), (128, 128, 128))
    hair_mask = mask(8)
    full_character_reference = Image.new(
        "RGB", (32, 32), (64, 96, 128)
    )
    old_hair_mask = inputs.hair_mask
    inputs.vibe_reference = full_character_reference
    inputs.hair_reference = hair_reference
    inputs.hair_mask = hair_mask
    inputs.hair_reference_scale = .61
    try:
        entries = adapter_references_for_stage(
            inputs, 'base', .7, .45
        )
        assert [entry.name for entry in entries] == ['identity']
        assert [entry.scale for entry in entries] == [.7]
        assert entries[0].image is full_character_reference
        assert entries[0].image is not inputs.identity
        assert entries[0].image is not hair_reference
    finally:
        inputs.hair_reference = None
        inputs.hair_mask = old_hair_mask
        hair_reference.close()
        hair_mask.close()


def test_ear_stage_keeps_human_and_animal_references_separate(inputs):
    human = PartReference(
        "human_ears", Image.new("RGB", (16, 16), "pink"), mask(8), .18)
    animal = PartReference(
        "animal_ears", Image.new("RGB", (16, 16), "purple"), mask(16), .22)
    original = inputs.extra_references
    inputs.extra_references = (human, animal, original[0])
    try:
        entries = adapter_references_for_stage(inputs, "ears", .7, .45)
        assert [entry.name for entry in entries] == [
            "human_ears", "animal_ears"]
        assert [entry.scale for entry in entries] == [.18, .22]
    finally:
        inputs.extra_references = original
        human.close()
        animal.close()


def test_hair_reference_change_after_approval_requires_review(inputs):
    hair_reference = Image.new("RGB", (20, 20), (128, 128, 128))
    hair_mask = mask(8)
    old_hair_mask = inputs.hair_mask
    inputs.hair_reference = hair_reference
    inputs.hair_mask = hair_mask
    try:
        approve_reference_experiment(inputs)
        hair_reference.putpixel((0, 0), (1, 2, 3))
        with pytest.raises(ValueError, match="승인 이후"):
            require_reference_experiment_approval(inputs)
    finally:
        inputs.hair_reference = None
        inputs.hair_mask = old_hair_mask
        hair_reference.close()
        hair_mask.close()


def test_changed_transmission_order_requires_new_approval(inputs, monkeypatch):
    approve_reference_experiment(inputs)
    original = adapter_references(inputs)
    monkeypatch.setattr("genai_lab.reference_order.adapter_references",
                        lambda *args: tuple(reversed(original)))
    with pytest.raises(ValueError, match="승인 이후"):
        require_reference_experiment_approval(inputs)


def test_scene_specific_order_is_not_changed_by_this_experiment(inputs, monkeypatch):
    scene = SimpleNamespace(references=[
        PartReference("identity", inputs.identity, inputs.identity_mask, .7),
        PartReference("garment", inputs.garment, inputs.garment_mask, .45),
        inputs.extra_references[0]])
    borrowed = SimpleNamespace(scene_condition=scene)
    monkeypatch.setattr("genai_lab.scene_generation.reference_scales", lambda *a: [.7, .45, .31])
    assert adapter_reference_names(borrowed) == ["identity", "garment", "tail"]
