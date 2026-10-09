"""지원되는 SDXL 4채널 인페인트의 결과 좌표를 분리한다. Diffusers 0.38은 매 스케줄러 단계 후 마스크 밖을 다음 시점의 원본 노이즈와 이미지
잠재값으로 복원하고 마지막에는 깨끗한 이미지 잠재값을 쓴다. 중간 시점에 깨끗한 값만 쓰거나 9채널 UNet에 적용하면 안 된다.
"""
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np
from PIL import Image, ImageFilter


def create_dual_masks(detected_mask, feather_radius=10):
    """이진 승인 마스크와 안쪽으로만 흐려지는 마스크를 만든다."""
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
        # 가는 검출 영역, 특히 동물 귀는 설정된
        # 흐림 반경보다 좁을 수 있다. 요청 반경을
        # 상한으로 두고 안전한 내부가 남을 때까지 줄인다.
        # 이진 승인 영역 밖으로 넓히지 않는다.
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
    """생성 후보 좌표에서 보호 부위를 다시 검출한다."""
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

        # 꼬리는 동일성 특징이지만 일부 의상에 가려질 수 있다.
        # 기본으로 보호하며, 나중에 검토한 가림 정책에서
        # 명시적인 보호 해제 마스크를 제공할 수 있다.
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
        pipeline, *, control_report, hard_authorized_mask=None,
        ip_adapter_mask=None, **kwargs):
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
    # 위 잠재값·편집 마스크는 항상 W다. 이미지 어텐션만 다를 수 있다.
    # 공식 마스크 설명: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter#masking
    attention_masks = masks
    if ip_adapter_mask is not None:
        if (not isinstance(ip_adapter_mask, Image.Image)
                or ip_adapter_mask.mode != "L"
                or ip_adapter_mask.size != image.size):
            raise ValueError("IP-Adapter mask must use grayscale output coordinates")
        attention_masks = IPAdapterMaskProcessor(do_binarize=False).preprocess(
            ip_adapter_mask, height=image.height, width=image.width)
    kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": [attention_masks]}
    previous = kwargs.get("callback_on_step_end")
    report = {
        "mask_coordinates": "generated_candidate",
        "latent_policy":
            "diffusers_sdxl_4ch_next_timestep_original_latents",
        "attention_policy": (
            "same_output_mask_single_reference" if ip_adapter_mask is None
            else "explicit_condition_mask_single_reference"),
        "mask_policy": "hard_boundary_with_inward_soft_feather",
        "hard_authorized_pixels": int(np.count_nonzero(hard_values)),
        "soft_nonzero_pixels": int(np.count_nonzero(values)),
        "latent_restore_steps": 0,
    }
    control_report["output_coordinate_control"] = report

    def after_step(pipe, step, timestep, tensors):
        # 이 콜백 전에 기본 합성이 이미 실행됐다.
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
    """결과 좌표의 필수 경계 밖 모든 RGB 픽셀을 복원한다."""
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
    """합성하지 않고 필수 영역 밖으로 번진 디코더 결과를 측정한다."""
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
    """필수 픽셀 복원 검사에 대한 구형 호환 이름이다."""
    return enforce_hard_paste(original, proposed, mask, report)

