import torch
from types import SimpleNamespace

from genai_lab.common_candidate_latents import (
    correlated_candidate_latents,
    create_common_base_latents,
    resolve_common_candidate_latents,
    validate_candidate_latent,
)


def pipeline():
    return SimpleNamespace(
        vae_scale_factor=8,
        unet=SimpleNamespace(config=SimpleNamespace(in_channels=4), dtype=torch.float32),
    )


def request(seed=42):
    return SimpleNamespace(seed=seed, width=768, height=1344, candidate_number=1)


def test_candidates_share_base_noise_but_keep_deterministic_variation():
    settings = resolve_common_candidate_latents({
        "common_candidate_latents": {
            "enabled": True, "correlation": .96, "variation_seed_offset": 10000,
        }
    })
    base = create_common_base_latents(pipeline(), request(), settings)
    first, first_record = correlated_candidate_latents(base, 42, 0, settings, "cpu")
    repeated, repeated_record = correlated_candidate_latents(base, 42, 0, settings, "cpu")
    second, second_record = correlated_candidate_latents(base, 42, 1, settings, "cpu")
    assert torch.equal(first, repeated)
    assert first_record == repeated_record
    assert not torch.equal(first, second)
    assert first_record["variation_seed"] == 10042
    assert second_record["variation_seed"] == 10043
    pair_correlation = torch.corrcoef(torch.stack((first.flatten(), second.flatten())))[0, 1]
    assert .90 < float(pair_correlation) < .95


def test_disabled_policy_keeps_legacy_independent_noise_path():
    settings = resolve_common_candidate_latents({})
    assert settings.enabled is False
    assert settings.record()["correlation"] == .96


def test_boundary_validation_supports_wrapped_candidate_seed():
    settings = resolve_common_candidate_latents({
        "common_candidate_latents": {
            "enabled": True, "correlation": .96, "variation_seed_offset": 10000,
        }
    })
    base_request = request(seed=2**32 - 1)
    base = create_common_base_latents(pipeline(), base_request, settings)
    candidate, record = correlated_candidate_latents(
        base, base_request.seed, 1, settings, "cpu")
    wrapped = request(seed=0)
    wrapped.candidate_number = 2
    validate_candidate_latent(
        {"latents": candidate}, wrapped, settings.record(), record)


def test_retry_candidates_use_lower_approved_correlation():
    settings = resolve_common_candidate_latents({
        "common_candidate_latents": {
            "enabled": True,
            "correlation": .88,
            "retry_correlation": .35,
            "variation_seed_offset": 10000,
        },
        "candidate_pipeline": {"maximum_attempts": 2},
    })
    base_request = request()
    base = create_common_base_latents(pipeline(), base_request, settings)
    initial, initial_record = correlated_candidate_latents(
        base, base_request.seed, 1, settings, "cpu")
    retry, retry_record = correlated_candidate_latents(
        base, base_request.seed, 2, settings, "cpu")

    assert initial_record["latent_phase"] == "initial"
    assert initial_record["correlation"] == .88
    assert retry_record["latent_phase"] == "retry"
    assert retry_record["correlation"] == .35
    retry_request = request(seed=44)
    retry_request.candidate_number = 3
    validate_candidate_latent(
        {"latents": retry}, retry_request, settings.record(), retry_record)
    observed = torch.corrcoef(torch.stack((base.flatten(), retry.cpu().flatten())))[0, 1]
    assert .30 < float(observed) < .40


def test_img2img_disables_configured_common_latents_to_preserve_source_image():
    settings = resolve_common_candidate_latents({
        "generation": {"mode": "image_to_image"},
        "common_candidate_latents": {
            "enabled": True, "correlation": .88, "variation_seed_offset": 10000,
        },
    })
    assert settings.enabled is False
    assert settings.configured_enabled is True
    assert settings.disabled_reason == "image_to_image_must_encode_the_approved_source_image"
    assert settings.record()["enabled"] is False
