import numpy as np
import pytest
import torch
from PIL import Image

from genai_lab.generated_image_integrity import (
    GeneratedImageIntegritySettings,
    GeneratedOutputIntegrityError,
    analyze_generated_image_integrity,
    resolve_generated_image_integrity,
    wrap_final_latent_audit,
)


def test_smooth_rgb_image_passes_mechanical_integrity_check():
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    report = analyze_generated_image_integrity(
        image, GeneratedImageIntegritySettings(enabled=True))
    assert report["status"] == "passed"
    assert report["failed_check_count"] == 0
    assert report["semantic_quality_judgement"] is False


def test_repeated_high_contrast_stripes_are_rejected_as_corruption():
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 1::2] = 255
    image = Image.fromarray(pixels, "RGB")
    report = analyze_generated_image_integrity(
        image, GeneratedImageIntegritySettings(enabled=True))
    assert report["status"] == "corrupted"
    assert report["failed_check_count"] >= 3
    assert report["failure_stage"] == "decoded_rgb"
    assert report["retryable"] is False


def test_adjacent_and_extreme_pair_rejects_even_below_minimum_failed_count():
    ramp = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    image = Image.fromarray(np.repeat(ramp[:, :, None], 3, axis=2), "RGB")
    settings = GeneratedImageIntegritySettings(
        enabled=True,
        maximum_adjacent_difference=.1,
        maximum_high_jump_ratio=1.0,
        maximum_laplacian_variance=1_000_000,
        maximum_extreme_pixel_ratio=0.0,
        minimum_failed_checks=3,
        reject_adjacent_and_extreme_pair=True,
    )
    report = analyze_generated_image_integrity(image, settings)
    assert report["failed_checks"] == [
        "adjacent_difference", "extreme_pixel_ratio"]
    assert report["critical_pair_triggered"] is True
    assert report["status"] == "corrupted"


def test_final_latent_audit_preserves_existing_schedule_contract():
    def existing(pipe, step, timestep, callback_kwargs):
        return callback_kwargs

    existing.reference_step_schedule_record = {"enabled": True}
    audit = {}
    policy = GeneratedImageIntegritySettings(enabled=True).record()
    callback = wrap_final_latent_audit(existing, 2, audit, policy)
    callback(None, 0, None, {"latents": torch.ones(1, 4, 8, 8)})
    assert audit == {}
    callback(None, 1, None, {"latents": torch.ones(1, 4, 8, 8)})
    assert audit["status"] == "passed"
    assert callback.reference_step_schedule_record == {"enabled": True}
    assert callback.generated_image_integrity_record == policy


def test_non_finite_final_latent_aborts_generation():
    callback = wrap_final_latent_audit(
        None, 1, {}, GeneratedImageIntegritySettings(enabled=True).record())
    latent = torch.zeros(1, 4, 8, 8)
    latent[0, 0, 0, 0] = float("nan")
    with pytest.raises(GeneratedOutputIntegrityError) as caught:
        callback(None, 0, None, {"latents": latent})
    assert caught.value.report["reason"] == "final_latents_non_finite"
    assert caught.value.report["failure_stage"] == "final_latent"
    assert caught.value.report["retryable"] is True


def test_integrity_policy_is_resolved_without_changing_defaults():
    settings = resolve_generated_image_integrity({
        "generated_image_integrity": {"enabled": True, "retry_count": 1}
    })
    assert settings.enabled is True
    assert settings.retry_count == 1
    assert settings.retry_final_latent_failure is True
    assert settings.retry_decoded_rgb_failure is False
    assert settings.continue_after_corruption is True
    assert settings.preserve_corrupted_image is True
    assert settings.minimum_failed_checks == 3


def test_decoded_rgb_corruption_can_be_configured_for_one_bounded_retry():
    settings = resolve_generated_image_integrity({
        "generated_image_integrity": {
            "enabled": True,
            "retry_count": 1,
            "retry_decoded_rgb_failure": True,
        }
    })
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 1::2] = 255
    report = analyze_generated_image_integrity(
        Image.fromarray(pixels, "RGB"), settings)
    assert report["status"] == "corrupted"
    assert report["retryable"] is True
