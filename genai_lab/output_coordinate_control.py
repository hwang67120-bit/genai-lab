"""Output-coordinate isolation for the supported SDXL four-channel inpaint path.

Diffusers 0.38 StableDiffusionXLInpaintPipeline restores the complement of its
mask after EVERY scheduler step with image_latents + the original noise at the
next timestep, and clean image_latents on the final step. Do not replace this
with a clean latent at intermediate timesteps or apply it to a nine-channel UNet.
"""
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np
from PIL import Image, ImageFilter


def create_dual_masks(detected_mask, feather_radius=10):
    """Create a binary authorization mask and an inward-only soft mask."""
    if type(feather_radius) is not int or not 0 <= feather_radius <= 32:
        raise ValueError("mask feather radius must be an integer from 0 to 32")
    if isinstance(detected_mask, Image.Image):
        with detected_mask.convert("L") as gray:
            values = np.asarray(gray)
    else:
        values = np.asarray(detected_mask)
    if values.ndim != 2:
        raise ValueError("detected mask must be a two-dimensional grayscale mask")
    hard = (values > 128).astype(np.uint8) * 255
    if not hard.any():
        raise ValueError("hard authorized mask is empty")
    effective_radius = feather_radius
    if feather_radius == 0:
        soft = hard.copy()
    else:
        # Thin output detections (especially individual animal ears) can be
        # narrower than the configured feather radius. Keep the requested
        # radius as an upper bound and reduce it until erosion leaves a safe
        # interior. This never expands beyond the binary authorization mask.
        eroded = None
        for radius in range(feather_radius, 0, -1):
            size = radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            candidate = cv2.erode(hard, kernel, iterations=1)
            if candidate.any():
                effective_radius = radius
                eroded = candidate
                break
        if eroded is None:
            raise ValueError("soft inpaint mask is empty after inward erosion")
        size = effective_radius * 2 + 1
        blurred = cv2.GaussianBlur(
            eroded.astype(np.float32), (size, size), 0)
        soft = np.clip(
            blurred * (hard.astype(np.float32) / 255.0), 0, 255
        ).astype(np.uint8)
        if not soft.any():
            raise ValueError("soft inpaint mask is empty after feathering")
    hard_image = Image.fromarray(hard, mode="L")
    soft_image = Image.fromarray(soft, mode="L")
    soft_image.info["requested_feather_radius"] = feather_radius
    soft_image.info["effective_feather_radius"] = effective_radius
    return hard_image, soft_image


@dataclass
class OutputProtectionAnalysis:
    hard_protection: Image.Image
    soft_boundary_protection: Image.Image
    conditional_protection: Image.Image
    diagnostic_masks: dict[str, Image.Image]
    record: dict

    def close(self):
        self.hard_protection.close()
        self.soft_boundary_protection.close()
        self.conditional_protection.close()
        for mask in self.diagnostic_masks.values():
            mask.close()


def analyze_output_protection_masks(image, config, *, target, check):
    """Re-detect protected parts in generated-candidate coordinates."""
    from genai_lab.hair_error_correction import create_output_hair_analyzer

    analyzer = create_output_hair_analyzer(config)
    parts = []
    try:
        parts = analyzer.analyze(
            image, image.size,
            cancelled=lambda: (check() or False),
            deadline=perf_counter() + 300)
        masks = {part.name: part.source_mask for part in parts}
        if "output_face" not in masks:
            raise ValueError("output face protection could not be resolved")

        hard = np.zeros((image.height, image.width), dtype=bool)
        conditional = np.zeros_like(hard)
        names = {
            "output_face", "output_animal_ears",
            "output_ears", "output_hair_accessory",
        }
        if target == "garment":
            names.add("output_hair")
        if target in {"ears", "animal_ears"}:
            names -= {
                "output_human_ears", "output_animal_ears", "output_ears",
            }
        if target == "human_ears":
            names.discard("output_human_ears")

        for name in names:
            if name not in masks:
                continue
            if masks[name].size != image.size:
                raise ValueError(
                    "protected mask coordinates differ from candidate")
            with masks[name].convert("L") as gray:
                with gray.filter(ImageFilter.MaxFilter(5)) as expanded:
                    hard |= np.asarray(expanded) >= 128

        # A tail is identity-bearing but may be occluded by some garments.
        # Keep it conditional and preserve it by default; a later reviewed
        # occlusion policy may provide an explicit release mask.
        if target == "garment" and "output_tail" in masks:
            with masks["output_tail"].convert("L") as gray:
                with gray.filter(ImageFilter.MaxFilter(5)) as expanded:
                    conditional |= np.asarray(expanded) >= 128

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expanded_hard = cv2.dilate(
            hard.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = expanded_hard & ~hard
        diagnostics = {}
        for name in (
            "output_face", "output_human_ears", "output_animal_ears",
            "output_ears", "output_hair", "output_hair_accessory",
            "output_tail",
        ):
            if name in masks:
                diagnostics[name] = masks[name].convert("L").copy()
        return OutputProtectionAnalysis(
            hard_protection=Image.fromarray(
                hard.astype(np.uint8) * 255, mode="L"),
            soft_boundary_protection=Image.fromarray(
                boundary.astype(np.uint8) * 255, mode="L"),
            conditional_protection=Image.fromarray(
                conditional.astype(np.uint8) * 255, mode="L"),
            diagnostic_masks=diagnostics,
            record={
                "version": "output_protection_analysis_v1",
                "coordinate_source": "generated_candidate",
                "hard_parts": sorted(
                    name for name in names if name in masks),
                "conditional_parts": (
                    ["output_tail"] if conditional.any() else []
                ),
                "diagnostic_only_parts": (
                    ["output_human_ears"]
                    if "output_human_ears" in masks else []
                ),
                "hard_pixels": int(np.count_nonzero(hard)),
                "conditional_pixels": int(np.count_nonzero(conditional)),
                "soft_boundary_pixels": int(np.count_nonzero(boundary)),
            },
        )
    finally:
        for part in parts:
            part.close()
        analyzer.close()


def isolate_output_mask(image, mask, config, *, target, check):
    if mask.size != image.size:
        raise ValueError("output mask coordinates differ from candidate")
    analysis = analyze_output_protection_masks(
        image, config, target=target, check=check)
    try:
        protected = (
            (np.asarray(analysis.hard_protection) >= 128)
            | (np.asarray(analysis.conditional_protection) >= 128)
        )
        with mask.convert("L") as gray:
            clean = (np.asarray(gray) >= 128) & ~protected
        if not clean.any():
            raise ValueError("empty safe output-coordinate correction mask")
        return Image.fromarray(clean.astype("uint8") * 255, mode="L")
    finally:
        analysis.close()


def run_isolated_inpaint(
        pipeline, *, control_report, hard_authorized_mask=None, **kwargs):
    from diffusers import StableDiffusionXLInpaintPipeline
    from diffusers.image_processor import (
        IPAdapterMaskProcessor, VaeImageProcessor,
    )
    if not isinstance(pipeline, StableDiffusionXLInpaintPipeline):
        raise ValueError(
            "unverified inpaint pipeline cannot guarantee latent isolation")
    if pipeline.unet.config.in_channels != 4:
        raise ValueError(
            "latent preservation requires the SDXL four-channel inpaint path")
    mask = kwargs["mask_image"]
    image = kwargs["image"]
    values = np.asarray(mask)
    if mask.mode != "L" or mask.size != image.size or not values.any():
        raise ValueError(
            "a nonempty grayscale output-coordinate mask is required")
    hard_mask = (
        hard_authorized_mask
        if hard_authorized_mask is not None else mask
    )
    hard_values = np.asarray(hard_mask)
    if (hard_mask.mode != "L" or hard_mask.size != image.size
            or not np.isin(hard_values, (0, 255)).all()
            or not hard_values.any()):
        raise ValueError(
            "a nonempty binary hard authorization mask is required")
    if np.any(values[hard_values == 0]):
        raise ValueError(
            "soft inpaint mask exceeds hard authorization boundary")
    masks = IPAdapterMaskProcessor(do_binarize=False).preprocess(
        mask, height=image.height, width=image.width)
    import torch
    latent_mask = torch.nn.functional.interpolate(
        masks,
        size=(
            image.height // pipeline.vae_scale_factor,
            image.width // pipeline.vae_scale_factor,
        ),
        mode="nearest",
    )
    if not latent_mask.any().item():
        control_report["output_coordinate_control"] = {
            "status": "rejected",
            "reason": "empty_output_mask_at_latent_resolution",
            "mask_coordinates": "generated_candidate",
            "latent_restore_steps": 0,
        }
        raise ValueError("output correction mask is empty at latent resolution")
    kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": [masks]}
    previous = kwargs.get("callback_on_step_end")
    report = {
        "mask_coordinates": "generated_candidate",
        "latent_policy":
            "diffusers_sdxl_4ch_next_timestep_original_latents",
        "attention_policy": "same_output_mask_single_reference",
        "mask_policy": "hard_boundary_with_inward_soft_feather",
        "hard_authorized_pixels": int(np.count_nonzero(hard_values)),
        "soft_nonzero_pixels": int(np.count_nonzero(values)),
        "latent_restore_steps": 0,
    }
    control_report["output_coordinate_control"] = report

    def after_step(pipe, step, timestep, tensors):
        # Native blending has already run before this callback.
        report["latent_restore_steps"] += 1
        if previous is not None:
            tensors = previous(pipe, step, timestep, tensors)
        return tensors

    kwargs["callback_on_step_end"] = after_step
    original_mask_processor = getattr(pipeline, "mask_processor", None)
    if original_mask_processor is None:
        raise ValueError(
            "inpaint pipeline mask processor is unavailable for soft masking")
    pipeline.mask_processor = VaeImageProcessor(
        vae_scale_factor=pipeline.vae_scale_factor,
        do_normalize=False,
        do_binarize=False,
        do_convert_grayscale=True,
    )
    report["inpaint_mask_processor"] = (
        "temporary_non_binary_processor_restored_after_call")
    try:
        return pipeline(**kwargs)
    finally:
        pipeline.mask_processor = original_mask_processor


def enforce_hard_paste(
        original, proposed, hard_authorized_mask, report=None):
    """Restore every RGB pixel outside the hard output-coordinate boundary."""
    if (proposed.size != original.size
            or hard_authorized_mask.size != original.size):
        raise ValueError(
            "pixel restoration requires identical output coordinates")
    with original.convert("RGB") as rgb:
        base = np.array(rgb)
    with proposed.convert("RGB") as rgb:
        result = np.array(rgb)
    allowed = np.asarray(hard_authorized_mask) == 255
    result[~allowed] = base[~allowed]
    changed = int(np.count_nonzero(
        np.any(result[~allowed] != base[~allowed], axis=-1)))
    if report is not None:
        report["outside_changed_pixels"] = changed
        report["outside_changed_ratio"] = 0.0
    if changed:
        raise ValueError("outside RGB pixel preservation failed")
    return Image.fromarray(result)


def measure_outside_rgb_change(original, proposed, hard_authorized_mask):
    """Measure decoder spill outside the Hard domain without compositing."""
    if (proposed.size != original.size
            or hard_authorized_mask.size != original.size):
        raise ValueError(
            "outside-change measurement requires identical coordinates")
    with original.convert("RGB") as rgb:
        base = np.asarray(rgb, dtype=np.uint8).copy()
    with proposed.convert("RGB") as rgb:
        result = np.asarray(rgb, dtype=np.uint8).copy()
    allowed = np.asarray(hard_authorized_mask) >= 128
    outside = ~allowed
    changed = np.any(result != base, axis=-1) & outside
    outside_pixels = int(np.count_nonzero(outside))
    changed_pixels = int(np.count_nonzero(changed))
    absolute = np.abs(result.astype(np.int16) - base.astype(np.int16))
    return {
        "outside_pixel_count": outside_pixels,
        "outside_changed_pixels": changed_pixels,
        "outside_changed_ratio": (
            changed_pixels / outside_pixels if outside_pixels else 0.0
        ),
        "outside_mean_absolute_difference": (
            float(absolute[outside].mean()) if outside_pixels else 0.0
        ),
        "outside_pixel_restoration_applied": False,
    }

def exact_pixel_composite(original, proposed, mask, report):
    """Backward-compatible name for the hard pixel restoration gate."""
    return enforce_hard_paste(original, proposed, mask, report)

