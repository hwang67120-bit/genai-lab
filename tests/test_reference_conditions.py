"""독립 조건/조립 연결 검사. 실제 생성 품질 시험이 아니다."""
from dataclasses import replace, FrozenInstanceError
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
import pytest

from genai_lab.reference_conditions import (
    ReferenceCondition, ReferenceConditions, plan_source_regions, source_part_conditions, SOURCE_LAYOUT, DISJOINT_LAYOUT,
)
from genai_lab.regional_reference import DetectedPart
from genai_lab.reference_experiment_approval import approve_reference_experiment, require_reference_experiment_approval
from genai_lab.reference_contract import validate_visual_inputs
from genai_lab.visual_reference import VisualInputs, prepare_visual_inputs, configure_visual_condition


def mask(box, size=(32, 48)):
    image = Image.new("L", size)
    image.paste(255, box)
    return image


def condition(name, box, color, scale=None):
    with Image.new("RGB", (16, 16), color) as rgb, mask(box) as region:
        return ReferenceCondition.capture(name, rgb, region, scale)


@pytest.fixture
def bundle():
    return ReferenceConditions((
        condition("identity", (8, 0, 24, 12), "red"),
        condition("garment", (8, 12, 24, 44), "blue"),
        condition("animal_ears", (8, 0, 12, 4), "cyan", .35),
        condition("tail", (1, 24, 12, 40), "purple", .35),
    ), SOURCE_LAYOUT)


def inputs_for(bundle):
    refs, layout = bundle.materialize()
    return VisualInputs(Image.new("RGB", (32, 48), "white"), refs[0].rgb, refs[1].rgb,
        refs[0].region, refs[1].region, extra_references=refs[2:],
        analysis_record={"condition_separation": bundle.record(), "part_region_experiment": layout},
        reference_conditions=bundle)


def test_snapshots_do_not_alias_original_or_materialized_images(bundle):
    old = bundle.fingerprint()
    refs, report = bundle.materialize()
    try:
        refs[0].rgb.putpixel((0, 0), (1, 2, 3))
        refs[1].region.putpixel((8, 12), 0)
        refs[-1].scale = .9
        assert bundle.fingerprint() == old
        assert report["identity_overlap_removed"] == 16
        assert report["garment_overlap_removed"] == 64
    finally:
        for ref in refs:
            ref.close()
    with pytest.raises(FrozenInstanceError):
        bundle.conditions[0].name = "garment"


def test_external_images_can_change_or_close_without_changing_condition():
    rgb, region = Image.new("RGB", (16, 16), "red"), mask((8, 0, 24, 12))
    frozen = ReferenceCondition.capture("identity", rgb, region)
    rgb.paste("blue", (0, 0, 16, 16))
    region.paste(0, (0, 0, 32, 48))
    rgb.close()
    region.close()
    with frozen.image() as original, frozen.region() as original_mask:
        assert original.getpixel((0, 0)) == (255, 0, 0)
        assert original_mask.getpixel((8, 0)) == 255


@pytest.mark.parametrize("name", ["garment", "animal_ears", "tail"])
def test_changing_one_rgb_condition_does_not_change_other_conditions(bundle, name):
    changed = tuple(replace(c, rgb_bytes=bytes(len(c.rgb_bytes))) if c.name == name else c
                    for c in bundle.conditions)
    other = replace(bundle, conditions=changed)
    assert other.fingerprint() != bundle.fingerprint()
    for before, after in zip(bundle.conditions, other.conditions):
        if before.name != name:
            assert before == after
    first, _ = bundle.materialize()
    second, _ = other.materialize()
    try:
        for a, b in zip(first, second):
            assert a.region.tobytes() == b.region.tobytes()
            if a.name != name:
                assert a.rgb.tobytes() == b.rgb.tobytes()
    finally:
        for ref in (*first, *second):
            ref.close()


def test_layout_retains_old_output_without_changing_source_masks(bundle):
    refs, _ = bundle.materialize()
    try:
        with bundle.conditions[0].region() as original_head, bundle.conditions[1].region() as original_garment:
            assert original_head.getpixel((8, 0)) == 255
            assert original_garment.getpixel((8, 24)) == 255
            assert refs[0].region.getpixel((8, 0)) == 0
            assert refs[1].region.getpixel((8, 24)) == 0
        occupied = np.zeros((48, 32), dtype=bool)
        for ref in refs:
            selected = np.asarray(ref.region) > 0
            assert not np.any(occupied & selected)
            occupied |= selected
    finally:
        for ref in refs:
            ref.close()


def test_part_preparation_owns_only_its_source_conditions():
    part = DetectedPart("tail", mask((1, 24, 7, 40)), None, .35)
    before = part.source_mask.tobytes()
    with Image.new("RGB", (32, 48), "purple") as source:
        conditions = source_part_conditions(source, [part])
    assert part.target_region is None
    assert part.source_mask.tobytes() == before
    part.close()
    with conditions[0].region() as captured:
        assert captured.tobytes() == before


def test_encoder_uses_reviewed_assembled_conditions(bundle):
    inputs = inputs_for(bundle)
    seen = {}
    pipe = SimpleNamespace(_execution_device="cpu",
        set_ip_adapter_scale=lambda scales: seen.update(scales=scales),
        prepare_ip_adapter_image_embeds=lambda **kw: seen.update(images=kw["ip_adapter_image"]) or ["encoded"])
    try:
        with pytest.raises(ValueError, match="먼저 승인"):
            configure_visual_condition(pipe, inputs)
        assert not seen
        approve_reference_experiment(inputs)
        output = configure_visual_condition(pipe, inputs, .7, .45)
        assert seen["scales"] == pytest.approx(.4625)
        expected = [bundle.conditions[i].rgb_bytes for i in (0, 2, 3, 1)]
        assert [image.tobytes() for image in seen["images"][0]] == expected
        assert output == {"ip_adapter_image_embeds": ["encoded"]}
    finally:
        inputs.close()


@pytest.mark.parametrize("change", ["condition", "effective", "remove"])
def test_approval_cannot_hide_condition_changes(bundle, change):
    inputs = inputs_for(bundle)
    try:
        approve_reference_experiment(inputs)
        if change == "condition":
            inputs.reference_conditions = replace(bundle, conditions=(
                *bundle.conditions[:-1], replace(bundle.conditions[-1], scale=.5)))
            with pytest.raises(ValueError, match="승인 이후"):
                require_reference_experiment_approval(inputs)
        elif change == "effective":
            inputs.garment.putpixel((0, 0), (1, 2, 3))
        else:
            inputs.reference_conditions = None
        with pytest.raises(ValueError):
            validate_visual_inputs(inputs)
    finally:
        inputs.close()


def test_diagnostics_preserve_source_and_assembled_distinction(bundle, tmp_path):
    bundle.save(tmp_path)
    saved = json.loads((tmp_path / "conditions.json").read_text(encoding="utf-8"))
    assert saved["fingerprint"] == bundle.fingerprint()
    assert saved["layout_changed"] is False
    assert saved["generation_isolated"] is False
    with Image.open(tmp_path / "garment_source_region.png") as raw:
        assert raw.getpixel((8, 24)) == 255


def test_disjoint_layout_never_silently_changes_overlapping_conditions(bundle):
    with pytest.raises(ValueError, match="겹칩니다"):
        replace(bundle, layout_policy=DISJOINT_LAYOUT).materialize()


def test_real_preparation_path_separates_parts_before_layout(monkeypatch, tmp_path):
    from genai_lab.reference_regions import ReferenceRegions
    regions = ReferenceRegions({
        "identity": mask((8, 0, 24, 12)), "garment": mask((8, 12, 24, 44)),
        "hair": mask((8, 0, 24, 12)), "face": Image.new("L", (32, 48)),
        "foreground": mask((0, 0, 32, 48)),
    }, {})
    original_garment = regions.masks["garment"].tobytes()
    monkeypatch.setattr("genai_lab.reference_regions.analyze_reference_regions", lambda *a, **k: regions)
    monkeypatch.setattr("genai_lab.visual_reference.prepare_garment_board", lambda image: (image.copy(), {}))
    class Analyzer:
        preview_only = True
        report = {"parts": {"tail": {"status": "detected", "target_status": "unresolved"}}}
        closed = False
        def analyze(self, source, size, **kwargs):
            self.parts = [DetectedPart("tail", mask((1, 24, 12, 40)), None, .35)]
            return self.parts
        def close(self):
            self.closed = True
    analyzer = Analyzer()
    config = {"reference_analysis": {"color_analysis_enabled": False, "part_analyzer": analyzer,
              "part_detection": {"conditioning_mode": "source_regions_experiment"}}}
    with Image.new("RGB", (32, 48), "red") as source, Image.new("RGB", (16, 16), "blue") as garment:
        inputs = prepare_visual_inputs(source, garment, config, tmp_path)
    try:
        bundle = inputs.reference_conditions
        assert bundle.conditions[1].region_bytes == original_garment
        assert inputs.garment_mask.tobytes() != original_garment
        assert inputs.analysis_record["condition_separation"]["source_conditions_mutated"] is False
        assert analyzer.parts[0].target_region is None
        assert analyzer.closed
        validate_visual_inputs(inputs)
        assert inputs.extra_references[0].name == "tail"
    finally:
        inputs.close()


def test_returned_hair_mask_recalculates_ear_generation_region(
        monkeypatch, tmp_path):
    from genai_lab.reference_regions import ReferenceRegions
    regions = ReferenceRegions({
        "identity": mask((8, 0, 24, 20)),
        "garment": mask((8, 20, 24, 44)),
        "hair": mask((8, 0, 24, 12)),
        "face": mask((10, 10, 22, 20)),
        "foreground": mask((0, 0, 32, 48)),
    }, {})
    monkeypatch.setattr(
        "genai_lab.reference_regions.analyze_reference_regions",
        lambda *a, **k: regions)
    monkeypatch.setattr(
        "genai_lab.visual_reference.prepare_garment_board",
        lambda image: (image.copy(), {}))

    class Analyzer:
        preview_only = True
        report = {"parts": {"animal_ears": {
            "status": "detected",
            "accepted_masks": 1,
            "accepted_boxes": [[10, 0, 14, 4]],
            "detection_scores": [0.9],
            "target_status": "unresolved"}}}

        def analyze(self, source, size, **kwargs):
            return [DetectedPart("animal_ears", mask((10, 0, 14, 4)),
                                 None, .35)]

        def close(self):
            pass

    calls = []

    def fake_strengthen(name, source_mask, settings, blocked_mask=None):
        calls.append(None if blocked_mask is None else blocked_mask.tobytes())
        if blocked_mask is None:
            expanded = source_mask.copy()
            expanded.paste(255, (8, 0, 16, 6))
            return expanded, {"status": "strengthened",
                              "blocked_pixels_removed": 0}
        return source_mask.copy(), {
            "status": "blocked_fallback_source",
            "blocked_pixels_removed": 32,
        }

    monkeypatch.setattr(
        "genai_lab.generation_region_strengthening."
        "strengthen_generation_region", fake_strengthen)
    config = {"reference_analysis": {
        "color_analysis_enabled": False,
        "part_analyzer": Analyzer(),
        "part_detection": {
            "conditioning_mode": "source_regions_experiment",
            "generation_region_strengthening": {},
        },
        "hair_mask_refinement": {
            "enabled": True,
            "minimum_hair_pixels": 1,
            "minimum_retained_ratio": .5,
            "maximum_missing_ratio": .2,
            "protection_padding_pixels": 0,
            "hole_inspection_radius": 1,
            "retry_close_radii": [0],
        },
        "hair_visual_reference": {
            "enabled": True,
            "reference_scale": .6,
            "padding_ratio": .15,
            "neutral_background": [128, 128, 128],
        },
    }}
    with Image.new("RGB", (32, 48), "blue") as source:
        with Image.new("RGB", (16, 16), "white") as garment:
            inputs = prepare_visual_inputs(
                source, garment, config, tmp_path)
    try:
        assert len(calls) == 2
        assert calls[1] == inputs.hair_mask.tobytes()
        ear_region = np.asarray(inputs.extra_references[0].region) == 255
        hair_region = np.asarray(inputs.hair_mask) == 255
        assert not np.any(ear_region & hair_region)
        audit = inputs.analysis_record[
            "part_region_experiment"
        ]["generation_region_strengthening"]["animal_ears"]
        assert audit["status"] == "blocked_fallback_source"
        assert audit["blocked_pixels_removed"] == 32
        validate_visual_inputs(inputs)
    finally:
        inputs.close()


@pytest.mark.parametrize("name", ["../tail", "con", "identity"])
def test_invalid_extra_names_rejected(name):
    with Image.new("RGB", (32, 48)) as rgb, mask((1, 1, 3, 3)) as region:
        if name == "identity":
            with pytest.raises(ValueError):
                ReferenceCondition.capture(name, rgb, region, .35)
        else:
            with pytest.raises(ValueError):
                ReferenceCondition.capture(name, rgb, region, .35)


def test_source_region_plan_accepts_no_optional_parts_as_no_op():
    with mask((8, 0, 24, 12)) as identity, mask((8, 12, 24, 44)) as garment:
        planned_identity, planned_garment, report = plan_source_regions(
            identity,
            garment,
            [],
        )
        try:
            assert planned_identity.tobytes() == identity.tobytes()
            assert planned_garment.tobytes() == garment.tobytes()
            assert report["status"] == "no_optional_parts"
            assert report["conditioned_parts"] == []
            assert report["requires_input_review"] is False
            assert report["semantic_accuracy_verified"] == "not_applicable"
        finally:
            planned_identity.close()
            planned_garment.close()
