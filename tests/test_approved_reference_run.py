"""승인 해시/실제 텐서 드리프트 테스트. GPU 생성 품질 시험은 아니다."""
from dataclasses import dataclass, replace, FrozenInstanceError
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from genai_lab.visual_reference import VisualInputs, configure_visual_condition, generate_visual_batch
from genai_lab.reference_experiment_approval import approve_reference_experiment
from genai_lab.approved_reference_run import (
    approve_reference_run, require_reference_run,
    seal_encoded_condition, verify_pipeline_boundary, verify_refinement_boundary,
)


@dataclass(frozen=True)
class Request:
    prompt: str = "1boy, blue hair, wearing skirt"
    negative_prompt: str = "1girl"
    width: int = 64
    height: int = 64
    inference_steps: int = 20
    guidance_scale: float = 7.
    model_id: str = "test-model"
    reference_adapter_id: str = "test-adapter"
    seed: int = 2**32 - 1
    candidate_number: int = 1
    reference_image_strength: float = .8
    original_image_change_strength: float = .25


@pytest.fixture
def data():
    head, garment = Image.new("L", (64, 64)), Image.new("L", (64, 64))
    head.paste(255, (0, 0, 64, 16))
    garment.paste(255, (0, 16, 64, 64))
    inputs = VisualInputs(Image.new("RGB", (64, 64), "white"),
                          Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue"),
                          head, garment, analysis_record={"part_region_experiment": {}})
    request = Request()
    config = {"generation": {"mode": "image_to_image", "original_image_change_strength": .25},
              "clothing_reference_generation": {"enabled": True, "candidate_count": 2,
              "character_gender": "male", "approved_tags": ("skirt",),
              "approved_prompt_pair": (request.prompt, request.negative_prompt),
              "identity_reference_scale": .7, "garment_reference_scale": .45}}
    approve_reference_experiment(inputs)
    approve_reference_run(inputs, config, request)
    yield inputs, config, request
    inputs.close()


def test_unapproved_and_changed_snapshot_are_rejected(data):
    inputs, config, request = data
    approval = inputs.approved_generation
    with pytest.raises(FrozenInstanceError):
        approval.payload = b"changed"
    record = approval.record()
    record["prompt"] = "changed"
    assert approval.record()["prompt"] == request.prompt
    inputs.approved_generation = None
    with pytest.raises(ValueError, match="승인되지"):
        require_reference_run(inputs, config, request)


@pytest.mark.parametrize("field,value", [
    ("identity_reference_scale", .6), ("garment_reference_scale", .3),
    ("character_gender", "female"), ("candidate_count", 3),
    ("approved_tags", ("skirt", "trousers")), ("approved_prompt_pair", ("changed", "1girl")),
    ("approved_prompt_pair", ("1boy, blue hair, wearing skirt", "changed")),
])
def test_changed_config_is_rejected(data, field, value):
    inputs, config, request = data
    config["clothing_reference_generation"][field] = value
    with pytest.raises(ValueError, match="승인 이후"):
        require_reference_run(inputs, config, request)


@pytest.mark.parametrize("field,value", [
    ("width", 128), ("height", 128), ("inference_steps", 30),
    ("guidance_scale", 8.), ("seed", 7), ("model_id", "other"),
])
def test_changed_request_is_rejected(data, field, value):
    inputs, config, request = data
    with pytest.raises(ValueError):
        require_reference_run(inputs, config, replace(request, **{field: value}))


@pytest.mark.parametrize("target", ["identity", "garment", "garment_mask"])
def test_changed_pixels_are_rejected(data, target):
    inputs, config, request = data
    image = getattr(inputs, target)
    image.putpixel((0, 0), (4, 5, 6) if image.mode == "RGB" else 255)
    with pytest.raises(ValueError):
        require_reference_run(inputs, config, request)


def test_actual_encoding_and_both_candidate_seeds(data):
    inputs, config, request = data
    pipeline = SimpleNamespace(_execution_device="cpu", set_ip_adapter_scale=lambda value: None,
                               prepare_ip_adapter_image_embeds=lambda **kw: [torch.ones(2, 2, 4)])
    condition = configure_visual_condition(pipeline, inputs)
    config["clothing_reference_generation"]["encoded_reference_receipt"] = seal_encoded_condition(inputs, condition)
    for number, seed in ((1, 2**32-1), (2, 0)):
        current = replace(request, candidate_number=number, seed=seed)
        arguments = dict(condition, prompt=current.prompt, negative_prompt=current.negative_prompt,
                         width=64, height=64, num_inference_steps=20, guidance_scale=7.,
                         generator=torch.Generator().manual_seed(seed),
                         image=inputs.source, strength=current.original_image_change_strength)
        assert verify_pipeline_boundary(inputs, config, current, arguments) == inputs.approved_generation.fingerprint
    with pytest.raises(ValueError, match="시드"):
        require_reference_run(inputs, config, replace(request, candidate_number=2), prepared=True)


def test_approved_gender_retry_extends_sealed_seed_budget(data):
    inputs, config, request = data
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 1,
        "maximum_attempts": 2,
        "gender_retry_attempts": 2,
    }
    approve_reference_run(inputs, config, request)
    record = require_reference_run(inputs, config, request).record()
    assert record["candidate_count"] == 2
    assert record["maximum_candidate_attempts"] == 4
    retry = replace(
        request,
        candidate_number=3,
        seed=(request.seed + 2) % (2**32),
    )
    assert require_reference_run(
        inputs, config, retry, prepared=True).fingerprint


def test_approved_quality_retry_extends_sealed_seed_budget(data):
    inputs, config, request = data
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 1,
        "maximum_attempts": 2,
        "gender_retry_attempts": 1,
        "quality_retry_attempts": 3,
    }
    approve_reference_run(inputs, config, request)
    record = require_reference_run(inputs, config, request).record()
    assert record["maximum_candidate_attempts"] == 5
    retry = replace(
        request,
        candidate_number=5,
        seed=(request.seed + 4) % (2**32),
    )
    assert require_reference_run(
        inputs, config, retry, prepared=True).fingerprint


@pytest.mark.parametrize("target", ["embed", "spatial_mask", "prompt", "negative", "seed", "initial"])
def test_changed_final_pipeline_inputs_are_rejected(data, target):
    inputs, config, request = data
    condition = {"ip_adapter_image_embeds": [torch.ones(2, 1, 4)]}
    config["clothing_reference_generation"]["encoded_reference_receipt"] = seal_encoded_condition(inputs, condition)
    arguments = dict(condition, prompt=request.prompt, negative_prompt=request.negative_prompt,
                     width=64, height=64, num_inference_steps=20, guidance_scale=7.,
                     generator=torch.Generator().manual_seed(request.seed),
                     image=inputs.source, strength=request.original_image_change_strength)
    if target == "embed":
        arguments["ip_adapter_image_embeds"][0][0, 0, 0] = 2
    elif target == "spatial_mask":
        arguments["cross_attention_kwargs"] = {
            "ip_adapter_masks": [torch.zeros(1, 2, 64, 64)]}
    elif target == "prompt":
        arguments["prompt"] = "1girl"
    elif target == "negative":
        arguments["negative_prompt"] = "changed"
    elif target == "seed":
        arguments["generator"] = torch.Generator().manual_seed(9)
    else:
        changed_initial = inputs.source.copy()
        changed_initial.putpixel((0, 0), (0, 0, 0))
        arguments["image"] = changed_initial
    with pytest.raises(ValueError):
        verify_pipeline_boundary(inputs, config, request, arguments)


def test_reference_batch_cannot_encode_without_approval(data, tmp_path):
    inputs, config, request = data
    inputs.approved_generation = None
    with pytest.raises(ValueError, match="승인되지"):
        generate_visual_batch(SimpleNamespace(), config, request, tmp_path,
                              SimpleNamespace(), inputs, lambda: False, lambda message: None)


@pytest.mark.parametrize("gender", ["unspecified", "male", "female"])
def test_gender_is_not_inferred_from_clothes(data, gender):
    inputs, config, request = data
    config["clothing_reference_generation"]["character_gender"] = gender
    approve_reference_run(inputs, config, request)
    record = require_reference_run(inputs, config, request).record()
    assert record["gender"] == gender
    assert record["approved_tags"] == ["skirt"]
    assert "pants" not in record["prompt"]


def test_invalid_gender_and_zero_garment_strength_are_not_defaulted(data):
    inputs, config, request = data
    section = config["clothing_reference_generation"]
    section["character_gender"] = "typo"
    with pytest.raises(ValueError):
        approve_reference_run(inputs, config, request)
    section["character_gender"] = "male"
    section["garment_reference_scale"] = 0
    with pytest.raises(ValueError):
        approve_reference_run(inputs, config, request)


def test_batch_uses_one_encoding_and_checks_both_candidates(data, tmp_path, monkeypatch):
    inputs, config, request = data
    encodings, seeds = [], []
    pipe = SimpleNamespace(_execution_device="cpu", set_ip_adapter_scale=lambda value: None)
    def encode(**kw):
        encodings.append(kw)
        return [torch.ones(2, 2, 4)]
    pipe.prepare_ip_adapter_image_embeds = encode
    monkeypatch.setattr("genai_lab.visual_reference.tempfile.mkdtemp", lambda **kw: str(tmp_path))
    monkeypatch.setattr("genai_lab.visual_reference.image_feature", lambda *args: np.array([1., 0.]))

    @dataclass
    class Candidate:
        image: Image.Image
        prompt: str
        design_reference_record: dict | None = None

    def generate(pipeline, local, current, root, log):
        section = local["clothing_reference_generation"]
        arguments = dict(section["visual_condition"], prompt=current.prompt,
                         negative_prompt=current.negative_prompt, width=64, height=64,
                         num_inference_steps=20, guidance_scale=7.,
                         generator=torch.Generator().manual_seed(current.seed),
                         image=inputs.source, strength=current.original_image_change_strength)
        verify_pipeline_boundary(inputs, local, current, arguments)
        seeds.append(current.seed)
        return Candidate(inputs.source.copy(), current.prompt)
    monkeypatch.setattr("genai_lab.generator.generate_character_candidate", generate)
    batch = generate_visual_batch(pipe, config, request, tmp_path,
        SimpleNamespace(write_stage=lambda *args: None), inputs, lambda: False, lambda message: None)
    try:
        assert len(encodings) == 1
        assert seeds == [2**32-1, 0]
        assert len(batch.candidates) == 2 and not batch.warning
        assert (tmp_path / "approved_generation.json").exists()
    finally:
        batch.close()


def test_gui_only_seals_after_acceptance(data, monkeypatch):
    from threading import Event
    from PySide6.QtWidgets import QApplication, QDialog
    from gui_main import GenAILabWindow
    inputs, config, request = data
    inputs.approved_generation = None
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    worker = SimpleNamespace(config=config, generation_request=request, cancel_requested=Event(),
                             review_done=Event(), inputs_approved=False)
    worker.orchestrator = SimpleNamespace(
        approve_visual_inputs=lambda value: approve_reference_run(
            value, config, request))
    window.worker = worker
    def accept(dialog):
        dialog.accept()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(window, "execute_approval_dialog", accept)
    try:
        window.review_visual_generation_inputs(inputs)
        assert worker.inputs_approved and worker.review_done.is_set()
        assert require_reference_run(inputs, config, request)
    finally:
        window.worker = None
        window.close()


def test_reject_revokes_previous_approval(data):
    from PySide6.QtWidgets import QApplication
    from genai_lab.visual_reference_review import VisualInputReview
    inputs, config, request = data
    app = QApplication.instance() or QApplication([])
    dialog = VisualInputReview(inputs)
    dialog.reject()
    try:
        assert inputs.approved_generation is None
        with pytest.raises(ValueError):
            require_reference_run(inputs, config, request)
    finally:
        dialog.close()


def test_color_description_changes_require_new_approval(data):
    from genai_lab.part_color_descriptions import PartColorDescription
    inputs, config, request = data
    desc = (PartColorDescription("tail", ("blue",), "proposed", (), b"{}", "test"),)
    inputs.part_color_descriptions = desc
    config["clothing_reference_generation"]["part_color_descriptions"] = desc
    approve_reference_run(inputs, config, request)
    changed = (replace(desc[0], colors=("red",)),)
    config["clothing_reference_generation"]["part_color_descriptions"] = changed
    with pytest.raises(ValueError, match="색상 설명"):
        require_reference_run(inputs, config, request)
    inputs.part_color_descriptions = changed
    with pytest.raises(ValueError, match="승인 이후"):
        require_reference_run(inputs, config, request)


def test_step_schedule_change_requires_new_approval(data):
    inputs, config, request = data
    config["reference_analysis"] = {"reference_step_schedule": {
        "enabled": True,
        "phases": [
            {"end_ratio": .5, "scale_factors": {
                "identity": 1., "garment": 1.}},
            {"end_ratio": 1., "scale_factors": {
                "identity": .8, "garment": .7}},
        ],
    }}
    approve_reference_run(inputs, config, request)
    config["reference_analysis"]["reference_step_schedule"]["phases"][1][
        "scale_factors"]["garment"] = .6
    with pytest.raises(ValueError, match="승인 이후"):
        require_reference_run(inputs, config, request)


def test_enabled_step_schedule_is_sealed_at_pipeline_boundary(data):
    inputs, config, request = data
    config["reference_analysis"] = {"reference_step_schedule": {
        "enabled": True,
        "phases": [
            {"end_ratio": .5, "scale_factors": {
                "identity": 1., "garment": 1.}},
            {"end_ratio": 1., "scale_factors": {
                "identity": .8, "garment": .7}},
        ],
    }}
    approve_reference_run(inputs, config, request)
    pipeline = SimpleNamespace(
        _execution_device="cpu",
        set_ip_adapter_scale=lambda value: None,
        prepare_ip_adapter_image_embeds=lambda **kw: [torch.ones(2, 2, 4)],
    )
    condition = configure_visual_condition(pipeline, inputs)
    config["clothing_reference_generation"][
        "encoded_reference_receipt"] = seal_encoded_condition(inputs, condition)
    from genai_lab.reference_order import adapter_references
    from genai_lab.reference_step_schedule import (
        build_reference_scale_schedule,
        effective_denoising_steps,
    )
    denoising_steps = effective_denoising_steps(
        request.inference_steps,
        "image_to_image",
        request.original_image_change_strength,
    )
    schedule, _ = build_reference_scale_schedule(
        config, adapter_references(inputs, .7, .45), denoising_steps)
    arguments = dict(
        condition,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        generator=torch.Generator().manual_seed(request.seed),
        image=inputs.source,
        strength=request.original_image_change_strength,
        callback_on_step_end=schedule,
    )
    assert verify_pipeline_boundary(
        inputs, config, request, arguments) == inputs.approved_generation.fingerprint
    schedule.reference_step_schedule_record["phases"][0][
        "resolved_scales"]["identity"] = .1
    with pytest.raises(ValueError, match="단계별 참조 강도"):
        verify_pipeline_boundary(inputs, config, request, arguments)


def test_latent_refinement_is_approved_at_both_pipeline_boundaries(data):
    inputs, config, request = data
    config["reference_analysis"] = {"latent_refinement": {
        "enabled": True,
        "inference_steps": 10,
        "strength": .25,
        "guidance_scale": 5.5,
        "latent_scale_factor": 1.,
    }}
    approve_reference_run(inputs, config, request)
    condition = {
        "ip_adapter_image_embeds": [torch.ones(2, 1, 4)],
    }
    config["clothing_reference_generation"][
        "encoded_reference_receipt"] = seal_encoded_condition(inputs, condition)

    base_arguments = dict(
        condition,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        generator=torch.Generator().manual_seed(request.seed),
        image=inputs.source,
        strength=request.original_image_change_strength,
        output_type="latent",
    )
    assert verify_pipeline_boundary(
        inputs, config, request, base_arguments) == inputs.approved_generation.fingerprint
    without_latent = dict(base_arguments)
    without_latent.pop("output_type")
    with pytest.raises(ValueError, match="latent 출력"):
        verify_pipeline_boundary(inputs, config, request, without_latent)

    latents = torch.zeros(1, 4, 8, 8)
    def callback(pipe, step, timestep, values):
        return values
    callback.latent_refinement_record = inputs.approved_generation.record()[
        "latent_refinement"]
    refinement_arguments = dict(
        condition,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        image=latents,
        strength=.25,
        num_inference_steps=10,
        guidance_scale=5.5,
        generator=torch.Generator().manual_seed(request.seed),
        callback_on_step_end=callback,
    )
    assert verify_refinement_boundary(
        inputs, config, request, refinement_arguments, latents
    ) == inputs.approved_generation.fingerprint
    with pytest.raises(ValueError, match="검증된 1차 latent"):
        verify_refinement_boundary(
            inputs, config, request, refinement_arguments, latents.clone())


def test_latent_refinement_change_requires_new_approval(data):
    inputs, config, request = data
    config["reference_analysis"] = {"latent_refinement": {
        "enabled": True, "inference_steps": 10, "strength": .25,
        "guidance_scale": 5.5, "latent_scale_factor": 1.,
    }}
    approve_reference_run(inputs, config, request)
    config["reference_analysis"]["latent_refinement"]["strength"] = .35
    with pytest.raises(ValueError, match="승인 이후"):
        require_reference_run(inputs, config, request)


def test_part_error_correction_change_requires_new_approval(data):
    inputs, config, request = data
    config["reference_analysis"] = {"part_error_correction": {
        "enabled": True,
        "part_names": ["ears", "tail"],
        "minimum_similarity": .6,
        "minimum_improvement": .02,
        "maximum_color_distance": 18.0,
        "minimum_color_improvement": 2.0,
        "maximum_similarity_regression": .03,
        "inpaint_strength": .3,
        "inference_steps": 10,
        "guidance_scale": 5.5,
        "mask_padding_pixels": 4,
        "timeout_seconds": 300,
    }}
    approve_reference_run(inputs, config, request)
    config["reference_analysis"]["part_error_correction"][
        "maximum_color_distance"] = 17.0
    with pytest.raises(ValueError, match="승인 이후"):
        require_reference_run(inputs, config, request)


def test_generated_image_integrity_change_requires_new_approval(data):
    inputs, config, request = data
    config["generated_image_integrity"] = {"enabled": True, "retry_count": 1}
    approve_reference_run(inputs, config, request)
    config["generated_image_integrity"]["retry_count"] = 2
    with pytest.raises(ValueError, match="승인 이후"):
        require_reference_run(inputs, config, request)


def test_generated_image_integrity_callback_is_sealed_at_pipeline_boundary(data):
    inputs, config, request = data
    config["generated_image_integrity"] = {"enabled": True, "retry_count": 1}
    approve_reference_run(inputs, config, request)
    condition = {
        "ip_adapter_image_embeds": [torch.ones(2, 1, 4)],
    }
    config["clothing_reference_generation"][
        "encoded_reference_receipt"] = seal_encoded_condition(inputs, condition)

    def callback(pipe, step, timestep, values):
        return values

    callback.generated_image_integrity_record = inputs.approved_generation.record()[
        "generated_image_integrity"]
    arguments = dict(
        condition,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        generator=torch.Generator().manual_seed(request.seed),
        image=inputs.source,
        strength=request.original_image_change_strength,
        callback_on_step_end=callback,
    )
    assert verify_pipeline_boundary(
        inputs, config, request, arguments) == inputs.approved_generation.fingerprint
    callback.generated_image_integrity_record = dict(
        callback.generated_image_integrity_record, retry_count=2)
    with pytest.raises(ValueError, match="무결성 콜백"):
        verify_pipeline_boundary(inputs, config, request, arguments)


def test_active_body_proportion_control_is_accepted_and_sealed(data):
    from genai_lab.body_proportion_presets import (
        PROJECT_ROOT,
        prepare_body_proportion_control,
    )

    inputs, config, base_request = data
    config["model"] = {"family": "sdxl"}
    config["body_proportion_presets"] = {
        "enabled": True,
        "status": "animagine_base_connected",
        "model_id": "xinsir/controlnet-canny-sdxl-1.0",
        "local_files_only": True,
        "conditioning_scale": .42,
        "guidance_start": 0.0,
        "guidance_end": .72,
        "user_selection_required": True,
    }
    request = SimpleNamespace(
        **base_request.__dict__,
        body_proportion_preset_id="standard_7_5h_hip",
        framing_type=SimpleNamespace(value="full_body"),
    )
    approve_reference_run(inputs, config, request)
    pipeline = SimpleNamespace(
        _execution_device="cpu",
        set_ip_adapter_scale=lambda value: None,
        prepare_ip_adapter_image_embeds=lambda **kw: [torch.ones(2, 2, 4)],
    )
    condition = configure_visual_condition(pipeline, inputs)
    config["clothing_reference_generation"]["encoded_reference_receipt"] = (
        seal_encoded_condition(inputs, condition)
    )
    control, record = prepare_body_proportion_control(
        config, request, PROJECT_ROOT
    )
    try:
        arguments = dict(
            condition,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.inference_steps,
            guidance_scale=request.guidance_scale,
            generator=torch.Generator().manual_seed(request.seed),
            image=inputs.source,
            strength=request.original_image_change_strength,
            control_image=control,
            controlnet_conditioning_scale=.42,
            control_guidance_start=0.0,
            control_guidance_end=.72,
        )
        assert record["status"] == "active"
        assert verify_pipeline_boundary(
            inputs, config, request, arguments
        ) == inputs.approved_generation.fingerprint

        changed = control.copy()
        changed.putpixel((0, 0), (255, 255, 255))
        changed_arguments = {**arguments, "control_image": changed}
        with pytest.raises(ValueError, match="체형 프리셋 ControlNet"):
            verify_pipeline_boundary(
                inputs, config, request, changed_arguments
            )
        changed.close()
    finally:
        control.close()