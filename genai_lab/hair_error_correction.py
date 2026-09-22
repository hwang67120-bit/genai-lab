"""Selected-candidate hair repair with non-hair IP-Adapter isolation."""

from genai_lab.output_coordinate_control import (
    isolate_output_mask, run_isolated_inpaint, exact_pixel_composite,
)
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageFilter

from genai_lab.hair_structure import (
    hair_structure_guidance,
    resolve_hair_structure,
)
from genai_lab.hair_part_partition import (
    partition_hair_regions,
    select_hair_correction_scope,
)
from genai_lab.hair_transfer_contract import (
    build_hair_transfer_contract,
    build_target_hair_plan,
)
from genai_lab.hair_inpaint_guard import (
    evaluate_hair_promotion,
    guarded_hair_composite,
    resolve_hair_boundary_guard,
    resolve_hair_promotion_gate,
)
from genai_lab.reference_regions import read_binary_mask
from genai_lab.regional_reference import DetectedPart, crop_reference


class HairCorrectionContractError(ValueError):
    pass


def _canonical_contract(value):
    """Compare contracts with the same JSON representation used by approval."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def build_hair_refinement_prompt(prompt, contract):
    """Append only user-approved, part-scoped hair guidance."""
    guidance = hair_structure_guidance(contract).strip()
    if not guidance:
        return str(prompt)
    return f"{str(prompt).rstrip(' ,')}, {guidance}"


@dataclass(frozen=True)
class HairErrorCorrectionSettings:
    enabled: bool
    minimum_similarity: float
    minimum_improvement: float
    hair_reference_scale: float
    inpaint_strength: float
    inference_steps: int
    guidance_scale: float
    mask_padding_pixels: int
    protection_padding_pixels: int
    timeout_seconds: float
    view_gate: dict
    boundary_guard: object
    promotion_gate: object

    def record(self):
        return {
            "version": "isolated_hair_inpaint_v3",
            **{
                name: value for name, value in self.__dict__.items()
                if name not in {"boundary_guard", "promotion_gate"}
            },
            "boundary_guard": self.boundary_guard.record(),
            "promotion_gate": self.promotion_gate.record(),
            "adapter_policy": {
                "hair_reference": self.hair_reference_scale,
                "identity": 0.0,
                "human_ears": 0.0,
                "animal_ears": 0.0,
                "ears": 0.0,
                "tail": 0.0,
                "garment": 0.0,
            },
            "outside_mask_policy": "pixel_preserving_composite",
            "required_output_masks": ["output_hair", "output_face"],
        }


def resolve_hair_error_correction(config):
    from genai_lab.hair_view_compatibility import resolve_hair_view_gate
    raw = config.get("reference_analysis", {}).get(
        "hair_error_correction", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("헤어 오류 보정 설정은 객체여야 합니다.")
    settings = HairErrorCorrectionSettings(
        enabled=bool(raw.get("enabled", False)),
        minimum_similarity=float(raw.get("minimum_similarity", .68)),
        minimum_improvement=float(raw.get("minimum_improvement", .01)),
        hair_reference_scale=float(raw.get("hair_reference_scale", .55)),
        inpaint_strength=float(raw.get("inpaint_strength", .45)),
        inference_steps=int(raw.get("inference_steps", 10)),
        guidance_scale=float(raw.get("guidance_scale", 5.5)),
        mask_padding_pixels=int(raw.get("mask_padding_pixels", 3)),
        protection_padding_pixels=int(
            raw.get("protection_padding_pixels", 3)),
        timeout_seconds=float(raw.get("timeout_seconds", 300)),
        view_gate=resolve_hair_view_gate(config).record(),
        boundary_guard=resolve_hair_boundary_guard(
            raw.get("boundary_guard", {"enabled": False})),
        promotion_gate=resolve_hair_promotion_gate(
            raw.get("promotion_gate", {"enabled": False})),
    )
    ratios = (
        settings.minimum_similarity,
        settings.hair_reference_scale,
        settings.inpaint_strength,
    )
    if (any(not np.isfinite(value) or not 0 <= value <= 1
            for value in ratios)
            or not np.isfinite(settings.minimum_improvement)
            or settings.minimum_improvement < 0
            or settings.inference_steps < 1
            or not np.isfinite(settings.guidance_scale)
            or settings.guidance_scale <= 0
            or settings.mask_padding_pixels < 0
            or settings.protection_padding_pixels < 0
            or not np.isfinite(settings.timeout_seconds)
            or settings.timeout_seconds <= 0):
        raise ValueError("헤어 오류 보정 설정값이 올바르지 않습니다.")
    return settings


def create_output_hair_analyzer(config):
    from genai_lab.extra_parts_analysis import (
        AdditionalPartsAnalyzer,
        TransformersPartsBackend,
    )
    detection = config["clothing_preparation"]
    segmentation = config["clothing_mask_extraction"]
    return AdditionalPartsAnalyzer(
        TransformersPartsBackend(
            detection["detector_model_id"],
            segmentation["model_id"],
            detection["cache_dir"],
        ),
        box_threshold=.30,
        mask_threshold=.75,
        scale=.55,
        part_queries=(
            {"name": "output_hair", "query": "hair.", "hair_context": False},
            {"name": "output_face", "query": "face.", "hair_context": False},
            {"name": "output_human_ears",
             "query": "human ears. person ears.",
             "hair_context": False},
            {"name": "output_animal_ears", "query": "animal ears.",
             "hair_context": False},
            {"name": "output_hair_accessory",
             "query": "hair ornament. hair accessory. hair clip.",
             "hair_context": False},
        ),
    )


def _expanded(mask, radius):
    size = max(1, radius * 2 + 1)
    return mask.convert("L").filter(ImageFilter.MaxFilter(size))


def _isolated_hair_masks(
        detected, size, settings, require_ears,
        *, require_human_ears=False, diagnostics=None):
    by_name = {part.name: part.source_mask for part in detected}
    if "output_hair" not in by_name or "output_face" not in by_name:
        return None, None, "hair_or_face_not_detected"
    if require_ears and not (
            "output_animal_ears" in by_name or "output_ears" in by_name):
        return None, None, "expected_animal_ears_not_detected"
    if require_human_ears and "output_human_ears" not in by_name:
        return None, None, "expected_human_ears_not_detected"
    hair = read_binary_mask(by_name["output_hair"], size)
    with Image.fromarray(hair.astype(np.uint8) * 255) as raw_hair:
        with _expanded(raw_hair, settings.mask_padding_pixels) as expanded:
            expanded_hair = read_binary_mask(expanded, size)
    protected = np.zeros((size[1], size[0]), dtype=bool)
    # The face mask already protects human-shaped ears. Keep the independent
    # human-ear detection diagnostic-only: broad false positives otherwise
    # remove the entire hair edit region. Animal ears remain separately
    # protected. A hair-accessory candidate that covers most of verified hair
    # is also diagnostic-only because it is not a localized decoration mask.
    if diagnostics is not None and "output_human_ears" in by_name:
        diagnostics["human_ears"] = {"status": "diagnostic_only"}
    for name in (
            "output_face", "output_animal_ears",
            "output_ears", "output_hair_accessory"):
        mask = by_name.get(name)
        if mask is None:
            continue
        mask_values = read_binary_mask(mask, size)
        overlap_ratio = float(
            np.count_nonzero(mask_values & expanded_hair)
            / max(1, np.count_nonzero(expanded_hair))
        )
        if name == "output_hair_accessory" and overlap_ratio > .5:
            if diagnostics is not None:
                diagnostics["hair_accessory"] = {
                    "status": "diagnostic_only_implausibly_broad",
                    "hair_overlap_ratio": overlap_ratio,
                    "maximum_protection_overlap_ratio": .5,
                }
            continue
        if diagnostics is not None:
            diagnostics[name.removeprefix("output_")] = {
                "status": "protected",
                "hair_overlap_ratio": overlap_ratio,
            }
        with _expanded(mask, settings.protection_padding_pixels) as expanded:
            protected |= read_binary_mask(expanded, size)
    isolated = expanded_hair & ~protected
    if not isolated.any():
        return None, None, "isolated_hair_mask_empty"
    return (
        Image.fromarray(isolated.astype(np.uint8) * 255),
        Image.fromarray(protected.astype(np.uint8) * 255),
        None,
    )


def _isolated_hair_mask(
        detected, size, settings, require_ears,
        *, require_human_ears=False):
    """Compatibility wrapper retained for mask-only callers and tests."""
    repair, protected, reason = _isolated_hair_masks(
        detected, size, settings, require_ears,
        require_human_ears=require_human_ears)
    if protected is not None:
        protected.close()
    return repair, reason


def correct_selected_hair(
        generation_pipeline, generated_image, visual_inputs, config, request,
        *, approved_run_record, check_running=None, analyzer=None,
        inpaint_pipeline=None, view_analyzer=None,
        output_regions_directory=None):
    """Repair hair only when its isolated CLIP similarity is below contract."""
    import torch
    from genai_lab.part_error_correction import (
        _inpaint_pipeline_from,
        _restore_generation_scales,
        _similarity,
    )
    from genai_lab.visual_reference import image_feature

    settings = resolve_hair_error_correction(config)
    report = settings.record()
    hair_structure = approved_run_record.get("hair_structure")
    if not isinstance(hair_structure, dict):
        hair_structure = resolve_hair_structure(
            approved_run_record.get("approved_character_tags", ()),
            approved_run_record.get("hair_detail_analysis"),
        )
    refinement_prompt = build_hair_refinement_prompt(
        request.prompt, hair_structure
    )
    hair_transfer_contract = approved_run_record.get("hair_transfer_contract")
    if not isinstance(hair_transfer_contract, dict):
        hair_transfer_contract = build_hair_transfer_contract(
            approved_run_record.get("approved_character_tags", ()),
            approved_run_record.get("hair_detail_analysis"),
        )
    report.update(
        status="disabled" if not settings.enabled else "running",
        hair_structure=hair_structure,
        hair_structure_guidance_applied=(refinement_prompt != request.prompt),
        hair_transfer_contract=hair_transfer_contract,
    )
    if not settings.enabled:
        return generated_image, report
    contract_differences = []
    if approved_run_record.get("prompt") != request.prompt:
        contract_differences.append("prompt")
    if approved_run_record.get("negative_prompt") != request.negative_prompt:
        contract_differences.append("negative_prompt")
    if _canonical_contract(approved_run_record.get("hair_error_correction")) \
            != _canonical_contract(settings.record()):
        contract_differences.append("hair_error_correction")
    if contract_differences:
        raise HairCorrectionContractError(
            "헤어 보정 조건이 봉인된 생성 계약과 다릅니다. "
            f"불일치={contract_differences}")
    if visual_inputs.hair_mask is None:
        report.update(status="unresolved", reason="approved_hair_mask_missing")
        return generated_image, report

    started = perf_counter()

    def check():
        if check_running is not None:
            check_running()
        if perf_counter() - started >= settings.timeout_seconds:
            raise TimeoutError("헤어 국소 보정 시간 제한을 초과했습니다.")

    owns_analyzer = analyzer is None
    owns_view_analyzer = view_analyzer is None
    if analyzer is None:
        analyzer = create_output_hair_analyzer(config)
    detected = []
    source_hair = None
    source_scope_mask = None
    repair_mask = None
    protected_mask = None
    post_repair_mask = None
    post_protected_mask = None
    post_detected = []
    proposed = None
    corrected = None
    try:
        check()
        detected = analyzer.analyze(
            generated_image,
            generated_image.size,
            cancelled=lambda: (check() or False),
            deadline=started + settings.timeout_seconds,
        )
        if output_regions_directory is not None:
            from genai_lab.reference_regions import load_reference_regions
            regions = load_reference_regions(
                Path(output_regions_directory), generated_image.size,
                required_regions=("identity",),
                analysis_scope="base_output")
            try:
                retained = []
                for part in detected:
                    if part.name in {"output_hair", "output_face"}:
                        part.close()
                    else:
                        retained.append(part)
                detected = retained + [
                    DetectedPart(
                        "output_hair", regions.masks["hair"].copy(), None, 1.0),
                    DetectedPart(
                        "output_face", regions.masks["face"].copy(), None, 1.0),
                ]
                report["output_region_source"] = (
                    "reused_redetected_base_candidate_coordinates")
                report["output_regions_directory"] = str(
                    output_regions_directory)
            finally:
                regions.close()
        require_ears = any(
            part.name in {"animal_ears", "ears"}
            for part in getattr(visual_inputs, "extra_references", ()))
        require_human_ears = any(
            part.name == "human_ears"
            for part in getattr(visual_inputs, "extra_references", ()))
        protection_diagnostics = {}
        repair_mask, protected_mask, reason = _isolated_hair_masks(
            detected, generated_image.size, settings, require_ears,
            require_human_ears=require_human_ears,
            diagnostics=protection_diagnostics)
        report["protection_diagnostics"] = protection_diagnostics
        if repair_mask is None:
            report.update(status="unresolved", reason=reason)
            return generated_image, report

        output_by_name = {part.name: part.source_mask for part in detected}
        output_regions, output_partition = partition_hair_regions(
            repair_mask, output_by_name["output_face"]
        )
        legacy_scope = (
            "hair_structure" not in approved_run_record
            and "approved_character_tags" not in approved_run_record
        )
        scoped_repair, output_scope = select_hair_correction_scope(
            hair_structure, output_regions,
            legacy_whole_if_missing=legacy_scope,
        )
        report["output_hair_partition"] = output_partition
        report["output_hair_scope"] = output_scope
        report["target_hair_plan"] = build_target_hair_plan(
            hair_transfer_contract, output_scope, output_partition)
        if scoped_repair is None:
            report.update(
                status="unresolved",
                reason=output_scope.get("reason"),
            )
            return generated_image, report
        repair_mask.close()
        repair_mask = scoped_repair

        source_base_mask = (
            getattr(visual_inputs, "hair_source_mask", None)
            or visual_inputs.hair_mask
        )
        source_face_mask = getattr(visual_inputs, "face_mask", None)
        if source_face_mask is None:
            if output_scope.get("mode") == "confirmed_local_parts":
                report.update(
                    status="unresolved",
                    reason="source_face_anchor_missing_for_local_hair_scope",
                )
                return generated_image, report
            source_scope_mask = source_base_mask.copy()
            source_partition = {
                "status": "UNRESOLVED",
                "reason": "source_face_anchor_missing",
            }
            source_scope = {
                "status": "SELECTED", "mode": "whole_fallback",
                "selected_regions": ["whole"],
            }
        else:
            source_regions, source_partition = partition_hair_regions(
                source_base_mask, source_face_mask
            )
            source_scope_mask, source_scope = select_hair_correction_scope(
                hair_structure, source_regions,
                legacy_whole_if_missing=legacy_scope,
            )
            if source_scope_mask is None:
                report["source_hair_partition"] = source_partition
                report["source_hair_scope"] = source_scope
                report.update(
                    status="unresolved",
                    reason=source_scope.get("reason"),
                )
                return generated_image, report
        report["source_hair_partition"] = source_partition
        report["source_hair_scope"] = source_scope
        source_hair = crop_reference(
            visual_inputs.source, source_scope_mask)
        reference_feature = image_feature(generation_pipeline, source_hair)
        before = _similarity(
            generation_pipeline, generated_image, repair_mask,
            reference_feature)
        report["before_similarity"] = before
        report["repair_mask_pixels"] = int(np.count_nonzero(
            np.asarray(repair_mask) >= 128))
        before_by_name = {part.name: part.source_mask for part in detected}
        before_promotion = evaluate_hair_promotion(
            visual_inputs.source, visual_inputs.hair_mask,
            generated_image, before_by_name['output_hair'],
            before_by_name['output_face'], generated_image,
            before_by_name['output_hair'], before_by_name['output_face'],
            settings.promotion_gate)
        report['before_promotion_gate'] = before_promotion
        if before >= settings.minimum_similarity and before_promotion['passed']:
            report.update(
                status="accepted_without_correction",
                elapsed_seconds=round(perf_counter() - started, 3),
            )
            return generated_image, report

        from genai_lab.hair_view_compatibility import (
            WdHairViewAnalyzer,
            resolve_hair_view_gate,
        )
        view_settings = resolve_hair_view_gate(config)
        if view_settings.enabled:
            if view_analyzer is None:
                view_analyzer = WdHairViewAnalyzer(config)
            view_report = view_analyzer.analyze_pair(
                visual_inputs.source,
                generated_image,
                view_settings,
                check_running=check,
            )
            report["view_compatibility"] = view_report
            if not view_report["inpaint_allowed"]:
                report.update(
                    status=("skipped_view_incompatible"
                            if view_report["status"] == "incompatible"
                            else "skipped_view_unresolved"),
                    reason=view_report["reason"],
                    elapsed_seconds=round(perf_counter() - started, 3),
                )
                return generated_image, report
        else:
            report["view_compatibility"] = {
                **view_settings.record(),
                "status": "disabled",
                "inpaint_allowed": True,
            }

        check()
        if inpaint_pipeline is None:
            if hasattr(generation_pipeline, "maybe_free_model_hooks"):
                generation_pipeline.maybe_free_model_hooks()
            torch.cuda.empty_cache()
            inpaint_pipeline = _inpaint_pipeline_from(generation_pipeline)
        # 단일 헤어 크롭만 전달한다. 다른 참조 이미지가 들어갈 슬롯이 없다.
        inpaint_pipeline.set_ip_adapter_scale(settings.hair_reference_scale)

        def on_step_end(pipe, step, timestep, values):
            del pipe, step, timestep
            check()
            return values

        try:
            raw_proposed = run_isolated_inpaint(
                inpaint_pipeline, control_report=report,
                prompt=refinement_prompt,
                negative_prompt=request.negative_prompt,
                image=generated_image,
                mask_image=repair_mask,
                ip_adapter_image=[source_hair],
                width=generated_image.width,
                height=generated_image.height,
                strength=settings.inpaint_strength,
                num_inference_steps=settings.inference_steps,
                guidance_scale=settings.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(
                    (request.seed + 2001) % (2**32)),
                callback_on_step_end=on_step_end,
            ).images[0]
            proposed = raw_proposed.convert("RGB")
            if proposed is not raw_proposed:
                raw_proposed.close()
        finally:
            report["pipeline_state_restore"] = _restore_generation_scales(
                generation_pipeline, visual_inputs, config)

        if settings.boundary_guard.enabled:
            corrected, boundary_report = guarded_hair_composite(
                generated_image, proposed, repair_mask, protected_mask,
                settings.boundary_guard)
            report["boundary_guard_result"] = boundary_report
            if corrected is None:
                report.update(
                    status="rejected_keep_original",
                    reason=boundary_report["reason"],
                    elapsed_seconds=round(perf_counter() - started, 3),
                )
                return generated_image, report
        else:
            corrected = exact_pixel_composite(
                generated_image, proposed, repair_mask, report)
            report["boundary_guard_result"] = {
                **settings.boundary_guard.record(),
                "status": "disabled", "reason": None,
            }

        if settings.promotion_gate.enabled:
            check()
            post_detected = analyzer.analyze(
                corrected, corrected.size,
                cancelled=lambda: (check() or False),
                deadline=started + settings.timeout_seconds,
            )
            post_repair_mask, post_protected_mask, post_reason = (
                _isolated_hair_masks(
                    post_detected, corrected.size, settings, require_ears,
                    require_human_ears=require_human_ears))
            if post_repair_mask is None:
                corrected.close()
                corrected = None
                report["promotion_gate_result"] = {
                    **settings.promotion_gate.record(),
                    "status": "unresolved", "passed": False,
                    "reason": post_reason,
                }
                report.update(
                    status="rejected_keep_original", reason=post_reason,
                    elapsed_seconds=round(perf_counter() - started, 3),
                )
                return generated_image, report
            before_by_name = {
                part.name: part.source_mask for part in detected}
            after_by_name = {
                part.name: part.source_mask for part in post_detected}
            promotion_report = evaluate_hair_promotion(
                visual_inputs.source, visual_inputs.hair_mask,
                generated_image, before_by_name["output_hair"],
                before_by_name["output_face"], corrected,
                after_by_name["output_hair"],
                after_by_name["output_face"], settings.promotion_gate)
            report["promotion_gate_result"] = promotion_report
            if not promotion_report["passed"]:
                corrected.close()
                corrected = None
                report.update(
                    status="rejected_keep_original",
                    reason=promotion_report["reason"],
                    elapsed_seconds=round(perf_counter() - started, 3),
                )
                return generated_image, report
        else:
            report["promotion_gate_result"] = {
                **settings.promotion_gate.record(),
                "status": "disabled", "passed": True,
                "reason": None,
            }
        after = _similarity(
            generation_pipeline, corrected, repair_mask, reference_feature)
        improvement = after - before
        report.update(
            after_similarity=after,
            similarity_improvement=improvement,
            elapsed_seconds=round(perf_counter() - started, 3),
        )
        required_improvement = (
            0.0 if before >= settings.minimum_similarity
            else settings.minimum_improvement)
        if improvement < required_improvement:
            corrected.close()
            corrected = None
            report.update(
                status="rejected_keep_original",
                reason="hair_similarity_not_improved",
            )
            return generated_image, report
        report["status"] = "corrected_and_accepted"
        return corrected, report
    finally:
        if proposed is not None:
            proposed.close()
        if repair_mask is not None:
            repair_mask.close()
        if protected_mask is not None:
            protected_mask.close()
        if post_repair_mask is not None:
            post_repair_mask.close()
        if post_protected_mask is not None:
            post_protected_mask.close()
        if source_hair is not None:
            source_hair.close()
        if source_scope_mask is not None:
            source_scope_mask.close()
        for part in detected:
            part.close()
        for part in post_detected:
            part.close()
        if owns_analyzer:
            analyzer.close()
        if owns_view_analyzer and view_analyzer is not None:
            view_analyzer.close()
