from types import SimpleNamespace
import sys

import pytest
import torch

from genai_lab.latent_refinement import (
    prepare_refinement_latents,
    refinement_pipeline_from,
    resolve_latent_refinement,
)


REQUEST = SimpleNamespace(width=64, height=112, guidance_scale=5.5)


def config(**values):
    return {"reference_analysis": {"latent_refinement": values}}


def test_disabled_by_default_and_enabled_record_is_explicit():
    assert resolve_latent_refinement({}, REQUEST).record()["enabled"] is False
    record = resolve_latent_refinement(
        config(enabled=True, inference_steps=10, strength=.25,
               guidance_scale=5.5, latent_scale_factor=1.), REQUEST
    ).record()
    assert record == {
        "version": "sdxl_latent_img2img_refinement_v1",
        "enabled": True,
        "inference_steps": 10,
        "strength": .25,
        "guidance_scale": 5.5,
        "latent_scale_factor": 1.,
        "validation": "tensor_shape_channels_and_finite_values",
        "semantic_output_verification": False,
        "pipeline_reuse": "AutoPipelineForImage2Image.from_pipe",
    }


@pytest.mark.parametrize("values", [
    {"inference_steps": 0},
    {"strength": 0},
    {"strength": .51},
    {"guidance_scale": 0},
    {"latent_scale_factor": .99},
    {"latent_scale_factor": 1.51},
])
def test_invalid_settings_are_rejected(values):
    with pytest.raises(ValueError):
        resolve_latent_refinement(config(enabled=True, **values), REQUEST)


def test_latent_validation_and_optional_bicubic_resize():
    settings = resolve_latent_refinement(
        config(enabled=True, latent_scale_factor=1.5), REQUEST)
    source = torch.zeros(1, 4, 14, 8)
    resized = prepare_refinement_latents(source, REQUEST, settings)
    assert resized.shape == (1, 4, 21, 12)
    with pytest.raises(ValueError, match="크기가 요청과"):
        prepare_refinement_latents(torch.zeros(1, 4, 8, 8), REQUEST, settings)
    source[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="유한하지"):
        prepare_refinement_latents(source, REQUEST, settings)


def test_from_pipe_wrapper_is_cached_without_reloading(monkeypatch):
    made = []
    converted = SimpleNamespace()
    factory = SimpleNamespace(from_pipe=lambda pipeline: made.append(pipeline) or converted)
    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(AutoPipelineForImage2Image=factory))
    pipeline = SimpleNamespace()
    assert refinement_pipeline_from(pipeline) is converted
    assert refinement_pipeline_from(pipeline) is converted
    assert made == [pipeline]
