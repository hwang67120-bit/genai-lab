'''Selected-candidate garment-only inpaint stage.'''

from genai_lab.output_coordinate_control import (
    run_isolated_inpaint,
)
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from genai_lab.garment_topology import topology_guidance


class GarmentCorrectionContractError(ValueError):
    pass


@dataclass(frozen=True)
class SelectedGarmentCorrectionSettings:
    enabled: bool = False
    reference_scale: float = .45
    inpaint_strength: float = .40
    inference_steps: int = 10
    guidance_scale: float = 5.5
    minimum_similarity_improvement: float = 0.0
    mask_padding_pixels: int = 2
    timeout_seconds: float = 300.0
    feather_radius: int = 10
    target_growth_pixels: int = 12
    soft_boundary_strength: float = 0.5
    reject_similarity_regression: bool = False

    def record(self):
        return {
            'version': 'selected_garment_correction_v2',
            **self.__dict__,
            'reference_policy': 'garment_only',
            'mask_policy': 'generated_output_hard_soft_coordinates',
            'outside_mask_policy': 'measure_without_pixel_composite',
        }


def resolve_selected_garment_correction(config):
    staged = config.get('staged_reference_generation', {})
    raw = staged.get('garment', {}) if staged.get('enabled', False) else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('staged garment settings must be an object')
    allowed = {field.name for field in fields(SelectedGarmentCorrectionSettings)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f'unknown staged garment settings: {sorted(unknown)}')
    settings = SelectedGarmentCorrectionSettings(**raw)
    for name in (
        'reference_scale', 'inpaint_strength',
        'minimum_similarity_improvement',
    ):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be numeric')
    if not 0 <= settings.reference_scale <= 1:
        raise ValueError('reference_scale must be between 0 and 1')
    if not 0 < settings.inpaint_strength <= 1:
        raise ValueError('inpaint_strength must be between 0 and 1')
    if type(settings.inference_steps) is not int or settings.inference_steps < 1:
        raise ValueError('inference_steps must be a positive integer')
    if settings.guidance_scale <= 0 or settings.timeout_seconds <= 0:
        raise ValueError('guidance_scale and timeout_seconds must be positive')
    if (type(settings.mask_padding_pixels) is not int
            or settings.mask_padding_pixels < 0):
        raise ValueError('mask_padding_pixels must be a non-negative integer')
    for name in ('feather_radius', 'target_growth_pixels'):
        value = getattr(settings, name)
        if type(value) is not int or not 0 <= value <= 64:
            raise ValueError(f'{name} must be an integer from 0 to 64')
    if (isinstance(settings.soft_boundary_strength, bool)
            or not isinstance(settings.soft_boundary_strength, (int, float))
            or not 0 <= settings.soft_boundary_strength <= 1):
        raise ValueError('soft_boundary_strength must be between 0 and 1')
    if type(settings.reject_similarity_regression) is not bool:
        raise ValueError('reject_similarity_regression must be boolean')
    return settings


def _similarity(pipeline, image, mask, reference_feature):
    from genai_lab.regional_reference import crop_reference
    from genai_lab.visual_reference import image_feature

    with crop_reference(image, mask) as crop:
        candidate_feature = image_feature(pipeline, crop)
    return float(np.clip(np.dot(candidate_feature, reference_feature), -1, 1))


def build_garment_refinement_prompt(prompt, garment_topology):
    guidance = topology_guidance(garment_topology)
    if not guidance:
        return prompt, False
    separator = ", " if prompt.strip() else ""
    return f"{prompt.strip()}{separator}{guidance}", True




def _diagnostic_prompt(assembled_prompt, override):
    """Explicit diagnostic text only; never alter the request or default assembly."""
    import hashlib
    if not isinstance(override, str) or not override.strip():
        raise ValueError("refinement_prompt_override must be a nonempty string")
    digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    return override, {
        "applied": True,
        "scope": "refinement_positive_prompt_only",
        "assembled_prompt": assembled_prompt,
        "actual_prompt": override,
        "assembled_sha256": digest(assembled_prompt),
        "actual_sha256": digest(override),
    }


def _load_or_redetect_regions(
    generated_image,
    config,
    root,
    output_regions_directory,
    *,
    check,
    run_log,
    report,
):
    """Reuse valid Base-coordinate masks; reject stale masks and redetect safely."""
    from genai_lab.reference_regions import (
        analyze_reference_regions,
        load_reference_regions,
    )

    if output_regions_directory is not None:
        try:
            regions = load_reference_regions(
                Path(output_regions_directory),
                generated_image.size,
                required_regions=("identity", "garment"),
                analysis_scope="garment_edit_input",
            )
        except ValueError as error:
            report["stored_output_regions"] = {
                "status": "rejected_redetected",
                "reason": str(error),
                "error_type": type(error).__name__,
                "directory": str(output_regions_directory),
            }
        else:
            report["output_region_source"] = (
                "reused_redetected_base_candidate_coordinates"
            )
            report["output_regions_directory"] = str(output_regions_directory)
            from genai_lab.provenance import recorder
            observation = recorder(config)
            if observation is not None:
                observation.branches.append({"name": "garment_region_source",
                    "choice": "stored_output_regions", "reason": "valid stored masks"})
            return regions

    from genai_lab.provenance import recorder
    observation = recorder(config)
    if observation is not None:
        observation.branches.append({"name": "garment_region_source",
            "choice": "redetected_output_regions"})
    regions = analyze_reference_regions(
        generated_image,
        config,
        root,
        cancelled=lambda: (check() or False),
        run_log=run_log,
        analysis_scope="garment_edit_input",
    )
    report["output_region_source"] = "redetected_for_selected_candidate"
    return regions


def _diagnostic_inputs(base, soft_mask, condition_mask, *, exclusion_mask=None,
                       prefill_mask=None, prefill_rgb=None):
    """Explicit diagnostic inputs only; never mutate Base or edit authorization.

    Attention masking follows run_isolated_inpaint's independent mask channel:
    https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter#masking
    """
    import hashlib

    def binary(mask, label):
        if mask.mode != "L" or mask.size != base.size:
            raise ValueError(f"{label} must be an L mask in Base coordinates")
        values = np.asarray(mask)
        if not np.isin(values, (0, 255)).all() or not values.any():
            raise ValueError(f"{label} must be nonempty and binary")
        return values == 255

    exclusion = binary(exclusion_mask, "IP exclusion") if exclusion_mask is not None else None
    prefill = binary(prefill_mask, "initial prefill") if prefill_mask is not None else None
    if (prefill is None) != (prefill_rgb is None):
        raise ValueError("initial prefill requires both mask and RGB color")
    if prefill is not None:
        if (base.mode != "RGB" or len(prefill_rgb) != 3
                or any(type(v) is not int or not 0 <= v <= 255 for v in prefill_rgb)):
            raise ValueError("initial prefill requires three integer RGB channels")
    attention = None
    initial = None
    evidence = {"scope": "diagnostic_only", "ip_exclusion_pixels": 0,
                "initial_prefill_pixels": 0, "initial_prefill_rgb": None}
    if exclusion is not None:
        values = np.array(condition_mask if condition_mask is not None else soft_mask)
        values[exclusion] = 0
        attention = Image.fromarray(values, mode="L")
        evidence.update(ip_exclusion_pixels=int(exclusion.sum()),
                        exclusion_mask_pixel_sha256=hashlib.sha256(exclusion_mask.tobytes()).hexdigest())
    if prefill is not None:
        pixels = np.array(base)
        pixels[prefill] = prefill_rgb
        initial = Image.fromarray(pixels, mode="RGB")
        evidence.update(initial_prefill_pixels=int(prefill.sum()),
                        initial_prefill_rgb=list(prefill_rgb),
                        prefill_mask_pixel_sha256=hashlib.sha256(prefill_mask.tobytes()).hexdigest(),
                        initial_pixel_sha256=hashlib.sha256(initial.tobytes()).hexdigest())
    return attention, initial, evidence


def correct_selected_garment(
    generation_pipeline,
    generated_image,
    visual_inputs,
    config,
    request,
    root,
    *,
    approved_run_record,
    check_running=None,
    run_log=None,
    output_regions_directory=None,
    diagnostic_directory=None,
    restore_pipeline_state=True,
    ip_adapter_mask_mode="edit",
    ip_adapter_exclusion_mask=None,
    initial_prefill_mask=None,
    initial_prefill_rgb=None,
    refinement_prompt_override=None,
):
    import torch
    from genai_lab.part_error_correction import (
        _expanded_mask,
        _inpaint_pipeline_from,
        _restore_generation_scales,
    )
    from genai_lab.reference_regions import (
        analyze_reference_regions, load_reference_regions,
    )
    from genai_lab.visual_reference import image_feature
    from genai_lab.garment_edit_plan import (
        build_garment_edit_plan,
        project_target_garment_coverage,
    )
    from genai_lab.output_coordinate_control import (
        analyze_output_protection_masks,
        measure_outside_rgb_change,
    )

    if ip_adapter_mask_mode not in ("edit", "target_pre_union"):
        raise ValueError("unknown garment IP-Adapter mask mode")
    settings = resolve_selected_garment_correction(config)
    report = settings.record()
    report.update(status='disabled' if not settings.enabled else 'running')
    if not settings.enabled:
        return generated_image, report
    if (approved_run_record.get('prompt') != request.prompt
            or approved_run_record.get('negative_prompt')
            != request.negative_prompt
            or approved_run_record.get('staged_reference_generation')
            != config.get('staged_reference_generation', {})):
        raise GarmentCorrectionContractError(
            'garment correction differs from the sealed generation contract'
        )
    garment_reference = getattr(visual_inputs, 'garment', None)
    if garment_reference is None:
        report.update(status='skipped', reason='garment_reference_absent')
        return generated_image, report

    topology_contract = approved_run_record.get("garment_topology")
    refinement_prompt, topology_guidance_applied = (
        build_garment_refinement_prompt(request.prompt, topology_contract)
    )
    report["garment_topology"] = topology_contract
    report["garment_topology_guidance_applied"] = topology_guidance_applied
    if refinement_prompt_override is not None:
        refinement_prompt, prompt_evidence = _diagnostic_prompt(
            refinement_prompt, refinement_prompt_override)
        report["refinement_prompt_override"] = prompt_evidence

    started = perf_counter()

    def check():
        if check_running is not None:
            check_running()
        if perf_counter() - started >= settings.timeout_seconds:
            raise TimeoutError('selected garment correction timed out')

    regions = None
    raw_proposed = None
    proposed = None
    correction_mask = None
    condition_mask = None
    initial_image = None
    hard_mask = None
    target_coverage = None
    protection = None
    plan = None
    try:
        check()
        regions = _load_or_redetect_regions(
            generated_image,
            config,
            root,
            output_regions_directory,
            check=check,
            run_log=run_log,
            report=report,
        )
        source_mask = _expanded_mask(
            regions.masks["garment"],
            generated_image.size,
            settings.mask_padding_pixels,
        )
        try:
            approved_tags = tuple(approved_run_record.get("approved_tags", ()))
            approved_tags += tuple(
                approved_run_record.get("approved_detail_tags", ()))
            target_coverage = project_target_garment_coverage(
                source_mask,
                regions.masks["foreground"],
                approved_tags,
                growth_pixels=settings.target_growth_pixels,
            )
            if target_coverage.record.get("status") == "UNRESOLVED":
                raise GarmentCorrectionContractError(
                    f"garment_category_unresolved: approved_tags={approved_tags}"
                )
            protection = analyze_output_protection_masks(
                generated_image, config, target="garment", check=check)
            plan = build_garment_edit_plan(
                generated_image,
                source_mask,
                target_coverage.mask,
                protection.hard_protection,
                regions.masks["foreground"],
                garment_growth_envelope=target_coverage.growth_envelope,
                soft_boundary_protection=protection.soft_boundary_protection,
                conditional_protection=protection.conditional_protection,
                feather_radius=settings.feather_radius,
                soft_boundary_strength=settings.soft_boundary_strength,
            )
            metrics = plan.record.get("metrics", {})
            if (metrics.get("hard_protection_pixels", 0)
                    >= metrics.get("requested_pixels", 0)):
                raise GarmentCorrectionContractError(
                    "protection_exceeds_requested_region: "
                    f"protection={metrics.get('hard_protection_pixels')} "
                    f"requested={metrics.get('requested_pixels')}"
                )
        finally:
            source_mask.close()

        report["target_garment_coverage"] = target_coverage.record
        report["output_protection_analysis"] = protection.record
        report["garment_edit_plan"] = plan.record
        diagnostic_root = (
            Path(diagnostic_directory)
            if diagnostic_directory is not None
            else (
                Path(output_regions_directory) / "local_refinement_masks"
                if output_regions_directory is not None
                else None
            )
        )
        if diagnostic_root is not None:
            if target_coverage.pre_union_mask is not None:
                diagnostic_root.mkdir(parents=True, exist_ok=True)
                target_coverage.pre_union_mask.save(
                    diagnostic_root / "target_pre_union.png")
            report["mask_diagnostic_manifest"] = str(
                plan.save(diagnostic_root))
            report["mask_diagnostic_directory"] = str(diagnostic_root)
            report["diagnostic_images"] = {
                name: str(diagnostic_root / f"{name}.png")
                for name in plan.images
            }
            for name, image in protection.diagnostic_masks.items():
                path = diagnostic_root / f"{name}.png"
                image.save(path)
                report["diagnostic_images"][name] = str(path)

        correction_mask = plan.soft_guidance.copy()
        report["ip_adapter_mask_mode"] = ip_adapter_mask_mode
        if ip_adapter_mask_mode == "target_pre_union":
            if target_coverage.pre_union_mask is None:
                raise GarmentCorrectionContractError("target_pre_union_unavailable")
            # Binary restriction preserves the existing soft weights exactly.
            condition_mask = Image.fromarray(
                np.where(np.asarray(target_coverage.pre_union_mask) == 255,
                         np.asarray(correction_mask), 0).astype(np.uint8), mode="L")
            if diagnostic_root is not None:
                condition_mask.save(diagnostic_root / "ip_adapter_condition_mask.png")
        if (ip_adapter_exclusion_mask is not None or initial_prefill_mask is not None
                or initial_prefill_rgb is not None):
            diagnostic_attention, initial_image, evidence = _diagnostic_inputs(
                generated_image, correction_mask, condition_mask,
                exclusion_mask=ip_adapter_exclusion_mask,
                prefill_mask=initial_prefill_mask, prefill_rgb=initial_prefill_rgb)
            if diagnostic_attention is not None:
                if condition_mask is not None:
                    condition_mask.close()
                condition_mask = diagnostic_attention
            report["diagnostic_overrides"] = evidence
            if diagnostic_root is not None:
                if condition_mask is not None:
                    condition_mask.save(diagnostic_root / "ip_adapter_condition_mask.png")
                if initial_image is not None:
                    initial_image.save(diagnostic_root / "diagnostic_inpaint_initial.png")
        hard_mask = plan.hard_edit_domain.copy()
        selected_pixels = int(np.count_nonzero(
            np.asarray(hard_mask) >= 128
        ))
        if selected_pixels == 0:
            report.update(status="unresolved", reason="output_garment_mask_empty")
            return generated_image, report

        reference_feature = image_feature(
            generation_pipeline, garment_reference
        )
        before = _similarity(
            generation_pipeline,
            generated_image,
            hard_mask,
            reference_feature,
        )
        if hasattr(generation_pipeline, 'maybe_free_model_hooks'):
            generation_pipeline.maybe_free_model_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inpaint_pipeline = _inpaint_pipeline_from(generation_pipeline)
        inpaint_pipeline.set_ip_adapter_scale(settings.reference_scale)
        from genai_lab.provenance import observe_pipeline
        observe_pipeline(config, inpaint_pipeline, "garment_inpaint_scale_set")

        def on_step_end(pipe, step, timestep, values):
            del pipe, step, timestep
            check()
            return values

        from genai_lab.provenance import recorder
        observation = recorder(config)
        if observation is not None:
            observation.loaded["call_arguments"] = {
                "prompt": refinement_prompt, "negative_prompt": request.negative_prompt,
                "strength": settings.inpaint_strength,
                "num_inference_steps": settings.inference_steps,
                "guidance_scale": settings.guidance_scale,
                "width": generated_image.width, "height": generated_image.height,
                "padding_mask_crop": None,
                "image": ("diagnostic_prefilled_selected_base_rgb" if initial_image is not None
                          else "selected_base_rgb"), "ip_adapter_image": "isolated_garment",
            }
            if "refinement_prompt_override" in report:
                observation.loaded["refinement_prompt_override"] = report["refinement_prompt_override"]
            if "diagnostic_overrides" in report:
                observation.loaded["diagnostic_overrides"] = report["diagnostic_overrides"]
            observation.loaded["input_image_size"] = list(generated_image.size)
        try:
            raw_proposed = run_isolated_inpaint(
                inpaint_pipeline, control_report=report,
                hard_authorized_mask=hard_mask,
                **({"ip_adapter_mask": condition_mask} if condition_mask is not None else {}),
                prompt=refinement_prompt,
                negative_prompt=request.negative_prompt,
                image=initial_image if initial_image is not None else generated_image,
                mask_image=correction_mask,
                ip_adapter_image=[garment_reference],
                width=generated_image.width,
                height=generated_image.height,
                strength=settings.inpaint_strength,
                num_inference_steps=settings.inference_steps,
                guidance_scale=settings.guidance_scale,
                generator=torch.Generator(device='cpu').manual_seed(
                    (request.seed + 5000) % (2**32)
                ),
                callback_on_step_end=on_step_end,
            ).images[0]
        finally:
            if restore_pipeline_state:
                report['pipeline_state_restore'] = _restore_generation_scales(
                    generation_pipeline, visual_inputs, config
                )
            else:
                report['pipeline_state_restore'] = {
                    'status': 'deferred_to_request_boundary_reset',
                }

        proposed = raw_proposed.convert('RGB')
        if proposed is not raw_proposed:
            raw_proposed.close()
            raw_proposed = None
        if proposed.size != generated_image.size:
            report.update(status='rejected_keep_original',
                          reason='inpaint_output_size_mismatch')
            return generated_image, report

        report["outside_change"] = measure_outside_rgb_change(
            generated_image, proposed, hard_mask
        )
        corrected = proposed.copy()
        after = _similarity(
            generation_pipeline,
            corrected,
            hard_mask,
            reference_feature,
        )
        improvement = after - before
        review_reasons = list(
            target_coverage.record.get("review_reasons", ()))
        review_reasons.extend(plan.record.get("review_reasons", ()))
        if improvement < settings.minimum_similarity_improvement:
            review_reasons.append("garment_similarity_regressed")
        if report["outside_change"]["outside_changed_pixels"]:
            review_reasons.append("outside_rgb_changed_without_composite")
        review_reasons = list(dict.fromkeys(review_reasons))
        report.update(
            before_similarity=before,
            after_similarity=after,
            similarity_improvement=improvement,
            selected_pixels=selected_pixels,
            elapsed_seconds=round(perf_counter() - started, 3),
            ip_adapter_inputs=["garment"],
            blocked_ip_adapter_inputs=[
                "identity", "hair", "human_ears",
                "animal_ears", "ears", "tail",
            ],
            image_merge_used=False,
            hard_paste_used=False,
            mask_composite_used=False,
            review_reasons=review_reasons,
        )
        if (
            improvement < settings.minimum_similarity_improvement
            and settings.reject_similarity_regression
        ):
            corrected.close()
            report.update(
                status="rejected_keep_original",
                reason="garment_similarity_regressed",
            )
            return generated_image, report
        if review_reasons:
            report.update(
                status="corrected_review_required",
                reason="diagnostics_recorded",
            )
        else:
            report.update(status="corrected_and_accepted", reason="verified")
        return corrected, report
    finally:
        if proposed is not None:
            proposed.close()
        if raw_proposed is not None:
            raw_proposed.close()
        if initial_image is not None:
            initial_image.close()
        if condition_mask is not None:
            condition_mask.close()
        if correction_mask is not None:
            correction_mask.close()
        if hard_mask is not None:
            hard_mask.close()
        if plan is not None:
            plan.close()
        if protection is not None:
            protection.close()
        if target_coverage is not None:
            target_coverage.close()
        if regions is not None:
            regions.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
