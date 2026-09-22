from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from genai_lab.part_error_correction import (
    PartCorrectionContractError,
    correct_detected_parts,
    resolve_part_error_correction,
)
from genai_lab.part_color_descriptions import (
    PartColorDescription,
    description_records,
)
from genai_lab.part_color_analysis import analyze_part_colors
from genai_lab.regional_reference import DetectedPart
from genai_lab.scene_reference import PartReference


REQUEST = SimpleNamespace(
    prompt="1boy, animal ears, tail",
    negative_prompt="1girl",
    seed=7,
    guidance_scale=5.5,
)


def config(**changes):
    values = {
        "enabled": True,
        "part_names": ["animal_ears", "tail"],
        "minimum_similarity": .6,
        "minimum_improvement": .02,
        "inpaint_strength": .3,
        "inference_steps": 10,
        "guidance_scale": 5.5,
        "mask_padding_pixels": 0,
        "mask_feather_pixels": 0,
        "timeout_seconds": 300,
    }
    values.update(changes)
    return {"reference_analysis": {
        "part_error_correction": values,
        "part_detection": {"enabled": True},
    }}


class Analyzer:
    def __init__(self, parts):
        self.parts = parts

    def analyze(self, image, output_size, *, cancelled, deadline):
        assert image.size == output_size
        assert not cancelled()
        assert deadline > 0
        return self.parts

    def close(self):
        raise AssertionError("주입받은 분석기는 호출자가 소유합니다.")


class InpaintPipeline:
    def __init__(self, proposed_color, output_size=None):
        self.proposed_color = proposed_color
        self.output_size = output_size
        self.calls = []

    def set_ip_adapter_scale(self, scale):
        assert scale == .35

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["width"] == kwargs["image"].width
        assert kwargs["height"] == kwargs["image"].height
        values = {"latents": object()}
        assert kwargs["callback_on_step_end"](self, 0, 0, values) is values
        return SimpleNamespace(images=[
            Image.new("RGB", self.output_size or
                      (kwargs["width"], kwargs["height"]), self.proposed_color)])


def feature(_pipeline, image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    center = rgb[rgb.shape[0] // 2, rgb.shape[1] // 2]
    return np.array([1., 0.]) if center[0] > center[2] else np.array([0., 1.])


def inputs(mask, descriptions=(), color_evidence=None):
    return SimpleNamespace(extra_references=(
        PartReference("tail", Image.new("RGB", (16, 16), "red"), mask.copy(), .35),
    ), part_color_descriptions=descriptions,
       color_evidence={} if color_evidence is None else color_evidence)


def color_description(name="tail", colors=("blue",)):
    return PartColorDescription(name, colors, "proposed", (), b"{}", "test")


def approved_record(cfg, request, descriptions=()):
    approved = tuple(
        description for description in descriptions
        if description.status == "proposed"
    )
    return {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "part_error_correction": resolve_part_error_correction(
            cfg, request).record(),
        "approved_part_color_descriptions": list(
            description_records(approved)),
    }


def run_correction(pipeline, image, visual, cfg, request, *,
                   approved_descriptions=None, **kwargs):
    descriptions = (
        tuple(getattr(visual, "part_color_descriptions", ()))
        if approved_descriptions is None else tuple(approved_descriptions)
    )
    return correct_detected_parts(
        pipeline, image, visual, cfg, request,
        approved_run_record=approved_record(cfg, request, descriptions),
        **kwargs,
    )


def solid_evidence(color, size=(32, 32)):
    with Image.new("RGB", size, color) as image, Image.new("L", size, 255) as mask:
        return analyze_part_colors(image, mask, max_clusters=1)


def test_only_detected_mask_is_changed_and_improved_result_is_accepted(monkeypatch):
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", feature)
    inpaint = InpaintPipeline("red")
    monkeypatch.setattr("genai_lab.part_error_correction._inpaint_pipeline_from",
                        lambda pipeline: inpaint)
    mask = Image.new("L", (32, 32), 0)
    mask.paste(255, (8, 8, 24, 24))
    source = Image.new("RGB", (32, 32), "blue")
    source.putpixel((0, 0), (1, 2, 3))
    visual = inputs(mask)
    result = run_correction(
        SimpleNamespace(), source, visual, config(), REQUEST,
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        assert result.report["corrected_count"] == 1
        assert result.report["parts"]["tail"]["status"] == "corrected_and_accepted"
        assert result.image.getpixel((16, 16)) == (255, 0, 0)
        assert result.image.getpixel((0, 0)) == (1, 2, 3)
        assert len(inpaint.calls) == 1
    finally:
        result.image.close()
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_color_prompt_is_applied_only_to_matching_part_inpaint(monkeypatch):
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", feature)
    inpaint = InpaintPipeline("red")
    monkeypatch.setattr("genai_lab.part_error_correction._inpaint_pipeline_from",
                        lambda pipeline: inpaint)
    mask = Image.new("L", (32, 32), 0)
    mask.paste(255, (8, 8, 24, 24))
    source = Image.new("RGB", (32, 32), "blue")
    visual = inputs(mask, (color_description(),))
    request = SimpleNamespace(**dict(
        vars(REQUEST), prompt="1boy, animal ears, blue tail"))
    result = run_correction(
        SimpleNamespace(), source, visual, config(), request,
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        assert inpaint.calls[0]["prompt"] == request.prompt
        assert inpaint.calls[0]["negative_prompt"] == request.negative_prompt
        assert result.report["parts"]["tail"]["local_color_prompt"] == "blue tail"
        assert request.prompt == "1boy, animal ears, blue tail"
    finally:
        result.image.close()
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_unapproved_color_observation_is_diagnostic_only(monkeypatch):
    monkeypatch.setattr(
        "genai_lab.visual_reference.image_feature",
        lambda pipeline, image: np.array([1., 0.]),
    )
    monkeypatch.setattr(
        "genai_lab.part_error_correction._inpaint_pipeline_from",
        lambda pipeline: pytest.fail("미승인 색상으로 보정하면 안 됩니다."),
    )
    mask = Image.new("L", (32, 32), 255)
    source = Image.new("RGB", mask.size, "blue")
    proposed = color_description(colors=("red",))
    visual = inputs(
        mask, (proposed,), {"tail": solid_evidence("red", mask.size)})
    result = run_correction(
        SimpleNamespace(), source, visual, config(), REQUEST,
        approved_descriptions=(),
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        part = result.report["parts"]["tail"]
        assert part["status"] == "accepted_without_correction"
        assert part["before_color_distance"] is None
        assert part["correction_triggers"] == ()
        assert part["trace"]["source_reference"]["color_evidence"] is None
        assert part["trace"]["source_reference"][
            "color_condition_approved"] is False
        assert result.report["approval_contract"][
            "unapproved_color_policy"] == "diagnostic_only"
    finally:
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_approved_color_missing_from_shared_prompt_is_added_only_locally(
        monkeypatch):
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", feature)
    inpaint = InpaintPipeline("blue")
    monkeypatch.setattr(
        "genai_lab.part_error_correction._inpaint_pipeline_from",
        lambda pipeline: inpaint)
    mask = Image.new("L", (32, 32), 255)
    source = Image.new("RGB", mask.size, "blue")
    visual = inputs(mask, (color_description(),))
    result = run_correction(
        SimpleNamespace(), source, visual, config(), REQUEST,
        analyzer=Analyzer([
            DetectedPart("tail", mask.copy(), None, .35)
        ]))
    try:
        assert inpaint.calls[0]["prompt"] == REQUEST.prompt + ", blue tail"
        assert result.report["parts"]["tail"]["local_color_prompt"] == "blue tail"
    finally:
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_color_mismatch_triggers_correction_when_clip_similarity_passes(monkeypatch):
    monkeypatch.setattr(
        "genai_lab.visual_reference.image_feature",
        lambda pipeline, image: np.array([1., 0.]),
    )
    inpaint = InpaintPipeline("red")
    monkeypatch.setattr("genai_lab.part_error_correction._inpaint_pipeline_from",
                        lambda pipeline: inpaint)
    mask = Image.new("L", (48, 80), 255)
    source = Image.new("RGB", mask.size, "blue")
    visual = inputs(
        mask,
        (color_description(colors=("red",)),),
        {"tail": solid_evidence("red", mask.size)},
    )
    request = SimpleNamespace(**dict(
        vars(REQUEST), prompt="1boy, animal ears, red tail"))
    result = run_correction(
        SimpleNamespace(), source, visual,
        config(maximum_color_distance=5.0, minimum_color_improvement=1.0),
        request,
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        part = result.report["parts"]["tail"]
        assert part["before_similarity"] == pytest.approx(1.0)
        assert part["correction_triggers"] == ("color_distance",)
        assert part["status"] == "corrected_and_accepted"
        assert part["after_color_distance"] < part["before_color_distance"]
        assert inpaint.calls[0]["prompt"] == request.prompt
        assert inpaint.calls[0]["width"] == 48
        assert inpaint.calls[0]["height"] == 80
    finally:
        result.image.close()
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_color_correction_is_rejected_when_absolute_target_is_not_met(monkeypatch):
    monkeypatch.setattr(
        "genai_lab.visual_reference.image_feature",
        lambda pipeline, image: np.array([1., 0.]),
    )
    distances = iter((44.5367, 42.1305))
    monkeypatch.setattr(
        "genai_lab.part_color_analysis.color_distribution_distance",
        lambda source, target: next(distances),
    )
    inpaint = InpaintPipeline("brown")
    monkeypatch.setattr("genai_lab.part_error_correction._inpaint_pipeline_from",
                        lambda pipeline: inpaint)
    mask = Image.new("L", (48, 80), 255)
    source = Image.new("RGB", mask.size, "blue")
    visual = inputs(
        mask,
        (color_description(colors=("blue",)),),
        {"tail": solid_evidence("blue", mask.size)},
    )
    request = SimpleNamespace(**dict(
        vars(REQUEST), prompt="1boy, animal ears, blue tail"))
    result = run_correction(
        SimpleNamespace(), source, visual,
        config(maximum_color_distance=18.0, minimum_color_improvement=2.0),
        request,
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        part = result.report["parts"]["tail"]
        assert part["color_improvement"] == pytest.approx(2.4062)
        assert part["after_color_distance"] == pytest.approx(42.1305)
        assert part["status"] == "rejected_keep_original"
        assert part["reason"] == "absolute_color_target_not_met"
        assert result.image is source
    finally:
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_excessive_output_mask_growth_skips_unsafe_correction(monkeypatch):
    monkeypatch.setattr(
        "genai_lab.part_error_correction._inpaint_pipeline_from",
        lambda pipeline: pytest.fail("과대 검출 마스크로 보정하면 안 됩니다."))
    source_mask = Image.new("L", (32, 32), 0)
    source_mask.paste(255, (8, 8, 24, 24))
    output_mask = Image.new("L", (32, 32), 255)
    source = Image.new("RGB", (32, 32), "blue")
    visual = inputs(source_mask)
    result = run_correction(
        SimpleNamespace(), source, visual,
        config(maximum_region_area_growth=2.0), REQUEST,
        analyzer=Analyzer([DetectedPart("tail", output_mask.copy(), None, .35)]))
    try:
        part = result.report["parts"]["tail"]
        assert result.image is source
        assert part["status"] == "unresolved_keep_original"
        assert part["reason"] == "output_mask_area_growth_exceeded"
        assert part["output_to_source_area_ratio"] == pytest.approx(4.0)
        assert result.report["quality_warnings"][0]["part_name"] == "tail"
    finally:
        source.close()
        visual.extra_references[0].close()
        source_mask.close()
        output_mask.close()


def test_latent_empty_output_mask_is_unresolved_without_aborting_batch(monkeypatch):
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", feature)
    inpaint = InpaintPipeline("red")
    monkeypatch.setattr(
        "genai_lab.part_error_correction._inpaint_pipeline_from",
        lambda pipeline: inpaint)

    def reject_empty_latent_mask(*args, **kwargs):
        raise ValueError("output correction mask is empty at latent resolution")

    monkeypatch.setattr(
        "genai_lab.part_error_correction.run_isolated_inpaint",
        reject_empty_latent_mask)
    mask = Image.new("L", (32, 32), 0)
    mask.paste(255, (8, 8, 24, 24))
    source = Image.new("RGB", mask.size, "blue")
    visual = inputs(mask)
    result = run_correction(
        SimpleNamespace(), source, visual,
        config(mask_feather_pixels=10), REQUEST,
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        part = result.report["parts"]["tail"]
        assert result.image is source
        assert part["status"] == "unresolved_keep_original"
        assert part["reason"] == "output_mask_empty_at_latent_resolution"
        assert part["trace"]["decision"] == "keep_original"
    finally:
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_wrong_inpaint_output_size_is_recorded_and_original_is_kept(monkeypatch):
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", feature)
    inpaint = InpaintPipeline("red", output_size=(1024, 1024))
    monkeypatch.setattr("genai_lab.part_error_correction._inpaint_pipeline_from",
                        lambda pipeline: inpaint)
    mask = Image.new("L", (48, 80), 255)
    source = Image.new("RGB", mask.size, "blue")
    visual = inputs(mask)
    result = run_correction(
        SimpleNamespace(), source, visual, config(), REQUEST,
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        part = result.report["parts"]["tail"]
        assert result.image is source
        assert part["status"] == "failed_keep_original"
        assert part["reason"] == "inpaint_output_size_mismatch"
        assert part["trace"]["inpaint_output"]["size"] == [1024, 1024]
    finally:
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_unresolved_color_does_not_change_local_prompt(monkeypatch):
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", feature)
    inpaint = InpaintPipeline("red")
    monkeypatch.setattr("genai_lab.part_error_correction._inpaint_pipeline_from",
                        lambda pipeline: inpaint)
    unresolved = PartColorDescription(
        "tail", (), "unresolved", ("ambiguous",), b"{}", "test")
    mask = Image.new("L", (32, 32), 255)
    source = Image.new("RGB", (32, 32), "blue")
    visual = inputs(mask, (unresolved,))
    result = run_correction(
        SimpleNamespace(), source, visual, config(), REQUEST,
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        assert inpaint.calls[0]["prompt"] == REQUEST.prompt
        assert result.report["parts"]["tail"]["local_color_prompt"] is None
    finally:
        result.image.close()
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_non_improving_correction_is_rejected(monkeypatch):
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", feature)
    inpaint = InpaintPipeline("blue")
    monkeypatch.setattr("genai_lab.part_error_correction._inpaint_pipeline_from",
                        lambda pipeline: inpaint)
    mask = Image.new("L", (32, 32), 0)
    mask.paste(255, (8, 8, 24, 24))
    source = Image.new("RGB", (32, 32), "blue")
    visual = inputs(mask)
    result = run_correction(
        SimpleNamespace(), source, visual, config(), REQUEST,
        analyzer=Analyzer([DetectedPart("tail", mask.copy(), None, .35)]))
    try:
        assert result.image is source
        assert result.report["corrected_count"] == 0
        assert result.report["parts"]["tail"]["status"] == "rejected_keep_original"
        assert result.report["parts"]["tail"]["reason"] == "similarity_requirement_not_met"
    finally:
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_undetected_part_is_unresolved_and_never_inpainted(monkeypatch):
    monkeypatch.setattr(
        "genai_lab.part_error_correction._inpaint_pipeline_from",
        lambda pipeline: pytest.fail("미검출 부위를 임의 보정하면 안 됩니다."))
    mask = Image.new("L", (32, 32), 255)
    source = Image.new("RGB", (32, 32), "blue")
    visual = inputs(mask)
    result = run_correction(
        SimpleNamespace(), source, visual, config(), REQUEST,
        analyzer=Analyzer([]))
    try:
        assert result.image is source
        assert result.report["parts"]["tail"]["status"] == "unresolved"
    finally:
        source.close()
        visual.extra_references[0].close()
        mask.close()


def test_human_ears_cannot_be_enabled_as_independent_correction_target():
    with pytest.raises(ValueError, match="human_ears"):
        resolve_part_error_correction(
            config(part_names=["human_ears", "animal_ears", "tail"]), REQUEST)


def test_animal_ears_and_tail_run_as_independent_sequential_passes(monkeypatch):
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", feature)
    inpaint = InpaintPipeline("red")
    monkeypatch.setattr(
        "genai_lab.part_error_correction._inpaint_pipeline_from",
        lambda pipeline: inpaint,
    )
    ears = Image.new("L", (32, 32), 0)
    ears.paste(255, (4, 8, 12, 24))
    tail = Image.new("L", (32, 32), 0)
    tail.paste(255, (20, 8, 28, 24))
    source = Image.new("RGB", (32, 32), "blue")
    visual = SimpleNamespace(
        extra_references=(
            PartReference(
                "animal_ears", Image.new("RGB", (16, 16), "red"),
                ears.copy(), .35),
            PartReference(
                "tail", Image.new("RGB", (16, 16), "red"),
                tail.copy(), .35),
        ),
        part_color_descriptions=(),
        color_evidence={},
    )
    result = run_correction(
        SimpleNamespace(), source, visual, config(), REQUEST,
        analyzer=Analyzer([
            DetectedPart("animal_ears", ears.copy(), None, .35),
            DetectedPart("tail", tail.copy(), None, .35),
        ]),
    )
    try:
        assert result.report["execution_order"] == ["animal_ears", "tail"]
        assert len(inpaint.calls) == 2
        assert result.report["corrected_count"] == 2
        assert result.image.getpixel((8, 16)) == (255, 0, 0)
        assert result.image.getpixel((24, 16)) == (255, 0, 0)
        assert result.image.getpixel((16, 16)) == (0, 0, 255)
        for name in ("animal_ears", "tail"):
            part = result.report["parts"][name]
            assert part["status"] == "corrected_and_accepted"
            assert part["outside_changed_pixels"] == 0
            assert part["trace"]["dual_masks"]["soft_outside_hard_pixels"] == 0
    finally:
        result.image.close()
        source.close()
        for reference in visual.extra_references:
            reference.close()
        ears.close()
        tail.close()


@pytest.mark.parametrize("change", [
    {"minimum_similarity": 1.1},
    {"minimum_improvement": -1},
    {"maximum_color_distance": 0},
    {"minimum_color_improvement": -1},
    {"maximum_similarity_regression": 1.1},
    {"maximum_region_area_growth": .9},
    {"inpaint_strength": .6},
    {"inference_steps": 0},
    {"mask_padding_pixels": 33},
    {"mask_feather_pixels": 33},
    {"timeout_seconds": 0},
])
def test_invalid_settings_are_rejected(change):
    with pytest.raises(ValueError):
        resolve_part_error_correction(config(**change), REQUEST)


def test_inpaint_wrapper_is_cached_and_cpu_offloaded(monkeypatch):
    import sys
    from genai_lab.part_error_correction import _inpaint_pipeline_from
    calls = []
    converted = SimpleNamespace(
        enable_model_cpu_offload=lambda: calls.append("offload"))
    factory = SimpleNamespace(
        from_pipe=lambda pipeline: calls.append(pipeline) or converted)
    monkeypatch.setitem(
        sys.modules, "diffusers",
        SimpleNamespace(AutoPipelineForInpainting=factory))
    pipeline = SimpleNamespace()
    assert _inpaint_pipeline_from(pipeline) is converted
    assert _inpaint_pipeline_from(pipeline) is converted
    assert calls == [pipeline, "offload"]


def test_generation_adapter_scales_are_restored_in_approved_order():
    from genai_lab.part_error_correction import _restore_generation_scales
    calls = []
    pipeline = SimpleNamespace(set_ip_adapter_scale=lambda value: calls.append(value))
    identity = Image.new("RGB", (8, 8))
    identity_mask = Image.new("L", (8, 8), 255)
    garment = Image.new("RGB", (8, 8))
    garment_mask = Image.new("L", (8, 8), 255)
    tail_mask = Image.new("L", (8, 8), 255)
    visual = SimpleNamespace(
        identity=identity,
        identity_mask=identity_mask,
        garment=garment,
        garment_mask=garment_mask,
        extra_references=(PartReference(
            "tail", Image.new("RGB", (8, 8)), tail_mask.copy(), .35),),
        scene_condition=None,
    )
    settings = {"clothing_reference_generation": {
        "identity_reference_scale": .7,
        "garment_reference_scale": .45,
    }}
    try:
        result = _restore_generation_scales(pipeline, visual, settings)
        assert calls == [[[.7, .35, .45]]]
        assert result["reference_order"] == ["identity", "tail", "garment"]
        assert result["status"] == "restored"
    finally:
        identity.close()
        identity_mask.close()
        garment.close()
        garment_mask.close()
        visual.extra_references[0].close()
        tail_mask.close()


@pytest.fixture(autouse=True)
def isolate_model_dependencies(monkeypatch):
    # These tests exercise promotion decisions using synthetic proposals. The
    # real control contract is exercised separately in test_output_coordinate_control.
    monkeypatch.setattr('genai_lab.part_error_correction.run_isolated_inpaint',
                        lambda pipe, control_report, **kwargs: pipe(**kwargs))
    monkeypatch.setattr('genai_lab.part_error_correction.isolate_output_mask',
                        lambda image, mask, config, **kwargs: mask.copy())
