"""생성 후보에서 확인된 추가 부위만 제한 영역 Inpaint로 보정한다."""

from genai_lab.output_coordinate_control import (
    create_dual_masks, enforce_hard_paste, isolate_output_mask,
    run_isolated_inpaint,
)
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from genai_lab.regional_reference import crop_reference


class PartCorrectionContractError(ValueError):
    """승인 후 변경되거나 미승인인 부위 보정 조건."""


@dataclass(frozen=True)
class PartErrorCorrectionSettings:
    enabled: bool
    part_names: tuple[str, ...]
    minimum_similarity: float
    minimum_improvement: float
    maximum_color_distance: float
    minimum_color_improvement: float
    maximum_similarity_regression: float
    maximum_region_area_growth: float
    inpaint_strength: float
    inference_steps: int
    guidance_scale: float
    mask_padding_pixels: int
    mask_feather_pixels: int
    timeout_seconds: float

    def record(self):
        return {
            "version": "detected_part_masked_inpaint_v4",
            "enabled": self.enabled,
            "part_names": list(self.part_names),
            "minimum_similarity": self.minimum_similarity,
            "minimum_improvement": self.minimum_improvement,
            "maximum_color_distance": self.maximum_color_distance,
            "minimum_color_improvement": self.minimum_color_improvement,
            "maximum_similarity_regression": self.maximum_similarity_regression,
            "maximum_region_area_growth": self.maximum_region_area_growth,
            "inpaint_strength": self.inpaint_strength,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "mask_padding_pixels": self.mask_padding_pixels,
            "mask_feather_pixels": self.mask_feather_pixels,
            "timeout_seconds": self.timeout_seconds,
            "verification": "independent_clip_lab_and_region_structure_checks",
            "undetected_policy": "unresolved_keep_original",
            "outside_mask_policy": "hard_authorized_pixel_restore",
            "mask_policy": "inward_eroded_soft_mask_inside_hard_boundary",
        }


@dataclass
class PartErrorCorrectionResult:
    image: Image.Image
    report: dict[str, Any]


def resolve_part_error_correction(config, request):
    raw = config.get("reference_analysis", {}).get("part_error_correction", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("부위 오류 보정 설정은 객체여야 합니다.")
    enabled = bool(raw.get("enabled", False))
    names = tuple(raw.get(
        "part_names", ("animal_ears", "tail")))
    if (not names or len(names) != len(set(names))
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("부위 오류 보정 대상 이름이 잘못됐습니다.")
    similarity = float(raw.get("minimum_similarity", .60))
    improvement = float(raw.get("minimum_improvement", .02))
    color_distance = float(raw.get("maximum_color_distance", 18.0))
    color_improvement = float(raw.get("minimum_color_improvement", 2.0))
    similarity_regression = float(raw.get("maximum_similarity_regression", .03))
    area_growth = float(raw.get("maximum_region_area_growth", 2.0))
    strength = float(raw.get("inpaint_strength", .30))
    steps = raw.get("inference_steps", 10)
    guidance = float(raw.get("guidance_scale", request.guidance_scale))
    padding = raw.get("mask_padding_pixels", 4)
    feather = raw.get("mask_feather_pixels", 10)
    timeout = float(raw.get("timeout_seconds", 300))
    if not all(math.isfinite(value) and 0 <= value <= 1
               for value in (similarity, improvement)):
        raise ValueError("부위 유사도 기준은 0~1이어야 합니다.")
    if (not math.isfinite(color_distance) or color_distance <= 0
            or not math.isfinite(color_improvement) or color_improvement < 0):
        raise ValueError("부위 색상 거리 기준이 잘못됐습니다.")
    if (not math.isfinite(similarity_regression)
            or not 0 <= similarity_regression <= 1):
        raise ValueError("부위 유사도 허용 하락값은 0~1이어야 합니다.")
    if not math.isfinite(area_growth) or area_growth < 1.0:
        raise ValueError("부위 마스크 면적 증가 한도는 1.0 이상이어야 합니다.")
    if not math.isfinite(strength) or not 0 < strength <= .5:
        raise ValueError("부위 보정 strength는 0 초과 0.5 이하여야 합니다.")
    if type(steps) is not int or not 1 <= steps <= 30:
        raise ValueError("부위 보정 반복 횟수는 1~30이어야 합니다.")
    if not math.isfinite(guidance) or guidance <= 0:
        raise ValueError("부위 보정 guidance_scale이 잘못됐습니다.")
    if type(padding) is not int or not 0 <= padding <= 32:
        raise ValueError("부위 보정 마스크 여백은 0~32px이어야 합니다.")
    if type(feather) is not int or not 0 <= feather <= 32:
        raise ValueError("부위 보정 마스크 내부 페더는 0~32px이어야 합니다.")
    if "human_ears" in names:
        raise ValueError(
            "human_ears: 사람형태 귀는 얼굴 보호 영역이며 독립 부위 보정 대상이 아닙니다.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("부위 보정 시간 제한이 잘못됐습니다.")
    return PartErrorCorrectionSettings(
        enabled, names, similarity, improvement, color_distance,
        color_improvement, similarity_regression, area_growth, strength, steps,
        guidance, padding, feather, timeout)


def _expanded_mask(mask, size, padding):
    hard = mask.convert("L")
    if hard.size != size:
        hard.close()
        raise ValueError("검출된 부위 마스크 크기가 후보 이미지와 다릅니다.")
    if padding:
        expanded = hard.filter(ImageFilter.MaxFilter(padding * 2 + 1))
        hard.close()
        hard = expanded
    return hard.point(lambda value: 255 if value >= 128 else 0)


def _similarity(pipeline, image, mask, reference_feature):
    from genai_lab.visual_reference import image_feature
    with crop_reference(image, mask) as crop:
        candidate_feature = image_feature(pipeline, crop)
    return float(np.clip(np.dot(candidate_feature, reference_feature), -1, 1))


def _inpaint_pipeline_from(pipeline):
    cached = getattr(pipeline, "_genai_lab_part_inpaint_pipeline", None)
    if cached is not None:
        return cached
    from diffusers import AutoPipelineForInpainting
    cached = AutoPipelineForInpainting.from_pipe(pipeline)
    cached.enable_model_cpu_offload()
    pipeline._genai_lab_part_inpaint_pipeline = cached
    return cached


def _approved_color_records(approved_run_record):
    """봉인된 실행 계약에서만 색상 조건을 읽는다."""
    if not isinstance(approved_run_record, dict):
        raise PartCorrectionContractError(
            "부위 보정에는 봉인된 승인 조건이 필요합니다.")
    records = approved_run_record.get("approved_part_color_descriptions", ())
    if not isinstance(records, (list, tuple)):
        raise PartCorrectionContractError(
            "승인된 부위 색상 조건 형식이 잘못됐습니다.")
    names = []
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("part_name"), str)
                or record.get("status") != "proposed"
                or not isinstance(record.get("prompt_text"), str)
                or not record["prompt_text"].strip()):
            raise PartCorrectionContractError(
                "승인된 부위 색상 조건 내용이 잘못됐습니다.")
        names.append(record["part_name"])
    if len(names) != len(set(names)):
        raise PartCorrectionContractError(
            "승인된 부위 색상 조건 이름이 중복됩니다.")
    return tuple(records)


def _part_local_prompt(request_prompt, approved_color_records, part_name):
    """지역 Inpaint에서도 승인된 최종 프롬프트를 그대로 사용한다."""
    matches = tuple(
        record for record in approved_color_records
        if record["part_name"] == part_name
    )
    if len(matches) > 1:
        raise PartCorrectionContractError(
            "승인된 부위 색상 조건 이름이 중복됩니다.")
    color_prompt = matches[0]["prompt_text"].strip() if matches else ""
    if not color_prompt:
        return request_prompt, ""
    # 공유 T2I 프롬프트는 그대로 두고, 봉인된 승인 색상만 해당 부위
    # Inpaint 호출에 국소적으로 추가한다. 이미 포함된 경우 중복하지 않는다.
    prompt_terms = {
        term.strip().lower() for term in request_prompt.split(",")
    }
    local_prompt = (
        request_prompt if color_prompt.lower() in prompt_terms
        else f"{request_prompt}, {color_prompt}"
    )
    return local_prompt, color_prompt


def _image_trace(image):
    return {"size": list(image.size), "mode": image.mode}


def _mask_trace(mask):
    with mask.convert("L") as gray:
        pixels = int(np.count_nonzero(np.asarray(gray) >= 128))
    return {"size": list(mask.size), "mode": mask.mode, "selected_pixels": pixels}


def _color_trace(evidence):
    return None if evidence is None else evidence.record()


def _restore_generation_scales(pipeline, visual_inputs, config):
    """from_pipe가 공유한 IP-Adapter scale을 승인 기본값으로 되돌린다."""
    if not hasattr(pipeline, "set_ip_adapter_scale"):
        return {"status": "not_supported"}
    from genai_lab.reference_order import (
        adapter_references,
        adapter_references_for_stage,
        unmasked_adapter_scale,
    )
    section = config.get("clothing_reference_generation", {})
    identity_scale = float(section.get("identity_reference_scale", .7))
    garment_scale = float(section.get("garment_reference_scale", .45))
    entries = (
        adapter_references_for_stage(
            visual_inputs, 'base', identity_scale, garment_scale
        )
        if config.get('staged_reference_generation', {}).get('enabled', False)
        else adapter_references(
            visual_inputs, identity_scale, garment_scale
        )
    )
    scales = [float(entry.scale) for entry in entries]
    staged = config.get('staged_reference_generation', {}).get(
        'enabled', False
    )
    pipeline.set_ip_adapter_scale(
        unmasked_adapter_scale(scales) if staged else [scales]
    )
    return {
        "status": "restored",
        "policy": "approved_base_scales",
        "reference_order": [entry.name for entry in entries],
        "scales": scales,
    }


def correct_detected_parts(
    generation_pipeline,
    generated_image,
    visual_inputs,
    config,
    request,
    *,
    approved_run_record,
    check_running=None,
    analyzer=None,
):
    """후보에서 다시 찾은 부위만 보정하고 유사도가 개선된 결과만 채택한다."""
    import torch
    from genai_lab.visual_reference import image_feature

    settings = resolve_part_error_correction(config, request)
    approved_colors = _approved_color_records(approved_run_record)
    if (approved_run_record.get("prompt") != request.prompt
            or approved_run_record.get("negative_prompt") != request.negative_prompt
            or approved_run_record.get("part_error_correction") != settings.record()):
        raise PartCorrectionContractError(
            "부위 보정 조건이 봉인된 생성 계약과 다릅니다.")
    approved_color_parts = {
        record["part_name"] for record in approved_colors
    }
    report = settings.record()
    report["approval_contract"] = {
        "status": "verified",
        "prompt_policy": "exact_approved_prompt_only",
        "approved_color_parts": sorted(approved_color_parts),
        "unapproved_color_policy": "diagnostic_only",
    }
    report.update(status="disabled" if not settings.enabled else "running", parts={})
    if not settings.enabled:
        return PartErrorCorrectionResult(generated_image, report)

    references = {
        reference.name: reference
        for reference in getattr(visual_inputs, "extra_references", ())
        if reference.name in settings.part_names
    }
    if not references:
        report.update(status="unresolved", reason="approved_part_reference_missing")
        return PartErrorCorrectionResult(generated_image, report)
    report["execution_order"] = [
        name for name in settings.part_names if name in references
    ]

    owns_analyzer = analyzer is None
    if analyzer is None:
        from genai_lab.extra_parts_analysis import create_extra_parts_analyzer
        analyzer = create_extra_parts_analyzer(config)
    if analyzer is None:
        report.update(status="unresolved", reason="output_part_analyzer_unavailable")
        return PartErrorCorrectionResult(generated_image, report)

    started = perf_counter()
    def check():
        if check_running is not None:
            check_running()
        if perf_counter() - started >= settings.timeout_seconds:
            raise TimeoutError("후보 부위 오류 보정 시간 제한을 초과했습니다.")

    detected = []
    try:
        check()
        detected = analyzer.analyze(
            generated_image,
            generated_image.size,
            cancelled=lambda: (check() or False),
            deadline=started + settings.timeout_seconds,
        )
        report["output_part_detection"] = getattr(analyzer, "report", {})
        from genai_lab.ear_contract import evaluate_output_ear_contract
        source_ear_contract = approved_run_record.get("ear_contract")
        report["ear_output_contract"] = evaluate_output_ear_contract(
            source_ear_contract,
            report["output_part_detection"],
            generated_image.size,
        )
        detected_by_name = {part.name: part for part in detected}
        current = generated_image
        inpaint_pipeline = None
        corrected_count = 0
        for index, name in enumerate(report["execution_order"], start=1):
            reference = references[name]
            check()
            # 원본 RGB 측정값은 해당 부위 색상이 승인된 경우에만
            # 생성 및 채택 판단에 참여한다. 미승인 값은 분석 기록에만 남는다.
            source_color = None
            if name in approved_color_parts:
                source_color = (
                    getattr(visual_inputs, "color_evidence", None) or {}
                ).get(name)
            source_mask_trace = _mask_trace(reference.region)
            part_report = {
                "status": "unresolved",
                "trace": {
                    "candidate_number": getattr(request, "candidate_number", None),
                    "part_name": name,
                    "generated_input": _image_trace(current),
                    "source_reference": {
                        "image": _image_trace(reference.rgb),
                        "region": source_mask_trace,
                        "scale": float(reference.scale),
                        "color_evidence": _color_trace(source_color),
                        "color_condition_approved": name in approved_color_parts,
                    },
                },
            }
            report["parts"][name] = part_report
            target = detected_by_name.get(name)
            if target is None:
                part_report["reason"] = "output_part_not_detected_keep_original"
                part_report["trace"]["output_detection"] = {"detected": False}
                continue

            output_mask_trace = _mask_trace(target.source_mask)
            source_pixels = source_mask_trace["selected_pixels"]
            output_pixels = output_mask_trace["selected_pixels"]
            area_growth_ratio = (
                output_pixels / source_pixels if source_pixels > 0 else None
            )
            part_report["trace"]["output_detection"] = {
                "detected": True,
                "mask": output_mask_trace,
                "source_selected_pixels": source_pixels,
                "output_selected_pixels": output_pixels,
                "area_growth_ratio": area_growth_ratio,
                "maximum_region_area_growth": settings.maximum_region_area_growth,
            }
            region_structure_ok = (
                area_growth_ratio is not None
                and area_growth_ratio <= settings.maximum_region_area_growth
            )
            if not region_structure_ok:
                part_report.update(
                    status="unresolved_keep_original",
                    reason="output_mask_area_growth_exceeded",
                    suspected_duplicate_or_merged_detection=True,
                    output_to_source_area_ratio=area_growth_ratio,
                )
                part_report["trace"]["decision"] = "skip_unsafe_inpaint_and_keep_original"
                report.setdefault("quality_warnings", []).append({
                    "part_name": name,
                    "reason": "output_mask_area_growth_exceeded",
                    "area_growth_ratio": area_growth_ratio,
                })
                continue

            reference_feature = image_feature(generation_pipeline, reference.rgb)
            before = _similarity(
                generation_pipeline, current, target.source_mask, reference_feature)
            from genai_lab.part_color_analysis import (
                analyze_part_colors, color_distribution_distance,
            )
            before_color = analyze_part_colors(
                current, target.source_mask, max_clusters=3, cancelled=lambda: (check() or False))
            before_color_distance = color_distribution_distance(source_color, before_color)
            shape_mismatch = before < settings.minimum_similarity
            color_mismatch = (
                before_color_distance is not None
                and before_color_distance > settings.maximum_color_distance
            )
            triggers = tuple(name for name, active in (
                ("clip_similarity", shape_mismatch),
                ("color_distance", color_mismatch),
            ) if active)
            part_report.update(
                before_similarity=before,
                before_color_distance=before_color_distance,
                correction_triggers=triggers,
            )
            part_report["trace"]["before_verification"] = {
                "clip_similarity": before,
                "minimum_similarity": settings.minimum_similarity,
                "color_evidence": _color_trace(before_color),
                "color_distance": before_color_distance,
                "maximum_color_distance": settings.maximum_color_distance,
                "triggers": list(triggers),
            }
            if not triggers:
                part_report.update(status="accepted_without_correction",
                                   reason="all_available_checks_met")
                part_report["trace"]["decision"] = "keep_original"
                continue

            # 승인 프롬프트 검증은 모델 생성과 IP-Adapter 상태 변경보다
            # 반드시 먼저 수행한다.
            local_prompt, color_prompt = _part_local_prompt(
                request.prompt, approved_colors, name)
            if inpaint_pipeline is None:
                if hasattr(generation_pipeline, "maybe_free_model_hooks"):
                    generation_pipeline.maybe_free_model_hooks()
                torch.cuda.empty_cache()
                inpaint_pipeline = _inpaint_pipeline_from(generation_pipeline)
            inpaint_pipeline.set_ip_adapter_scale(float(reference.scale))
            part_report["local_color_prompt"] = color_prompt or None
            with _expanded_mask(
                    target.source_mask, current.size,
                    settings.mask_padding_pixels) as expanded_mask, isolate_output_mask(
                        current, expanded_mask, config, target=name,
                        check=check) as authorized_mask:
                try:
                    hard_mask, soft_mask = create_dual_masks(
                        authorized_mask, settings.mask_feather_pixels)
                except ValueError as error:
                    part_report.update(
                        status="unresolved_keep_original",
                        reason="safe_dual_mask_unavailable",
                    )
                    part_report["trace"]["mask_failure"] = str(error)
                    continue
                with hard_mask, soft_mask:
                    part_report["trace"]["dual_masks"] = {
                        "hard": _mask_trace(hard_mask),
                        "soft": _mask_trace(soft_mask),
                        "requested_feather_pixels": settings.mask_feather_pixels,
                        "effective_feather_pixels": soft_mask.info.get(
                            "effective_feather_radius",
                            settings.mask_feather_pixels),
                        "soft_outside_hard_pixels": int(np.count_nonzero(
                            (np.asarray(soft_mask) > 0)
                            & (np.asarray(hard_mask) == 0))),
                    }
                    part_report["trace"]["inpaint_input"] = {
                        "image": _image_trace(current),
                        "mask": _mask_trace(soft_mask),
                        "hard_authorized_mask": _mask_trace(hard_mask),
                        "reference_image": _image_trace(reference.rgb),
                        "width": current.width,
                        "height": current.height,
                        "ip_adapter_scale": float(reference.scale),
                        "local_color_prompt": color_prompt or None,
                    }

                    def on_step_end(pipe, step, timestep, values):
                        del pipe, step, timestep
                        check()
                        return values

                    raw_proposed = None
                    try:
                        try:
                            raw_proposed = run_isolated_inpaint(
                                inpaint_pipeline,
                                control_report=part_report,
                                hard_authorized_mask=hard_mask,
                                prompt=local_prompt,
                                negative_prompt=request.negative_prompt,
                                image=current,
                                mask_image=soft_mask,
                                ip_adapter_image=[reference.rgb],
                                width=current.width,
                                height=current.height,
                                strength=settings.inpaint_strength,
                                num_inference_steps=settings.inference_steps,
                                guidance_scale=settings.guidance_scale,
                                generator=torch.Generator(device="cpu").manual_seed(
                                    (request.seed + 1000 + index) % (2**32)),
                                callback_on_step_end=on_step_end,
                            ).images[0]
                        except ValueError as error:
                            if "output correction mask is empty at latent resolution" not in str(error):
                                raise
                            part_report.update(
                                status="unresolved_keep_original",
                                reason="output_mask_empty_at_latent_resolution",
                            )
                            part_report["trace"].update(
                                decision="keep_original",
                                latent_mask_error=str(error),
                            )
                    finally:
                        restoration = _restore_generation_scales(
                            generation_pipeline, visual_inputs, config)
                        report["pipeline_state_restore"] = restoration
                        part_report["trace"][
                            "pipeline_state_restore"] = restoration
                    if raw_proposed is None:
                        continue
                    proposed = raw_proposed.convert("RGB")
                    if proposed is not raw_proposed:
                        raw_proposed.close()
                    part_report["trace"]["inpaint_output"] = _image_trace(proposed)
                    if proposed.size != current.size:
                        part_report.update(
                            status="failed_keep_original",
                            reason="inpaint_output_size_mismatch",
                        )
                        part_report["trace"]["decision"] = "keep_original"
                        proposed.close()
                        continue
                    corrected = enforce_hard_paste(
                        current, proposed, hard_mask, part_report)
                    proposed.close()

            after = _similarity(
                generation_pipeline, corrected, target.source_mask, reference_feature)
            after_color = analyze_part_colors(
                corrected, target.source_mask, max_clusters=3, cancelled=lambda: (check() or False))
            after_color_distance = color_distribution_distance(source_color, after_color)
            similarity_improvement = after - before
            color_improvement = (
                None if before_color_distance is None or after_color_distance is None
                else before_color_distance - after_color_distance
            )
            similarity_ok = (
                similarity_improvement >= settings.minimum_improvement
                if shape_mismatch
                else similarity_improvement >= -settings.maximum_similarity_regression
            )
            color_absolute_ok = (
                after_color_distance is not None
                and after_color_distance <= settings.maximum_color_distance
            )
            color_improvement_ok = (
                color_improvement is not None
                and color_improvement >= settings.minimum_color_improvement
            )
            # 상대 개선량은 진단으로 남기되, 색상 오류가 있었던 결과는
            # 최종 절대 거리가 허용 범위 안에 들어온 경우에만 승인한다.
            color_ok = color_absolute_ok if color_mismatch else True
            part_report.update(
                after_similarity=after,
                similarity_improvement=similarity_improvement,
                after_color_distance=after_color_distance,
                color_improvement=color_improvement,
            )
            part_report["trace"]["after_verification"] = {
                "clip_similarity": after,
                "similarity_improvement": similarity_improvement,
                "color_evidence": _color_trace(after_color),
                "color_distance": after_color_distance,
                "color_improvement": color_improvement,
                "maximum_color_distance": settings.maximum_color_distance,
                "minimum_color_improvement": settings.minimum_color_improvement,
                "color_absolute_ok": color_absolute_ok,
                "color_improvement_ok": color_improvement_ok,
                "similarity_ok": similarity_ok,
                "color_ok": color_ok,
            }
            if similarity_ok and color_ok:
                if current is not generated_image:
                    current.close()
                current = corrected
                corrected_count += 1
                part_report.update(status="corrected_and_accepted",
                                   reason="all_required_checks_met")
                part_report["trace"]["decision"] = "accept_correction"
            else:
                corrected.close()
                failed_checks = []
                if not similarity_ok:
                    failed_checks.append("similarity_requirement_not_met")
                if color_mismatch and not color_absolute_ok:
                    failed_checks.append("absolute_color_target_not_met")
                part_report.update(
                    status="rejected_keep_original",
                    reason=",".join(failed_checks) or "verification_failed",
                )
                part_report["trace"]["decision"] = "keep_original"
        report.update(status="completed", corrected_count=corrected_count,
                      elapsed_seconds=round(perf_counter() - started, 3))
        return PartErrorCorrectionResult(current, report)
    except BaseException:
        if "current" in locals() and current is not generated_image:
            current.close()
        raise
    finally:
        for part in detected:
            part.close()
        if owns_analyzer:
            analyzer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
