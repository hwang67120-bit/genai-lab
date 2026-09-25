"""Visual garment conditioning, bounded candidates and advisory ROI similarity."""
from dataclasses import dataclass, field, fields, replace
from time import perf_counter
from pathlib import Path
import json
import tempfile
import numpy as np
from PIL import Image, ImageOps
from genai_lab.reference_diagnostics import (
    timed_stage, verify_and_save_ip_adapter_input,
)
from genai_lab.reference_contract import validate_reference_mode, validate_visual_inputs
from genai_lab.garment_reference_board import prepare_garment_board
from genai_lab.reference_experiment_approval import require_reference_experiment_approval
from genai_lab.reference_order import adapter_references, adapter_reference_names


def unresolved_cross_class_parts(
        report, approved_character_tags=(), overlap_threshold=.80):
    """Keep overlapping semantic candidates as diagnostics, not conditions."""
    normalized_tags = {
        str(tag).strip().lower().replace("_", " ")
        for tag in approved_character_tags
    }
    tail_approved = any(
        tag == "tail" or tag.endswith(" tail") for tag in normalized_tags)
    unresolved = set()
    conflicts = []
    overlaps = report.get("part_overlap", {}) if isinstance(report, dict) else {}
    for pair, value in overlaps.items():
        if not isinstance(value, dict) or ":" not in pair:
            continue
        left, right = pair.split(":", 1)
        names = {left, right}
        ratio = value.get("smaller_part_ratio")
        if ("tail" not in names
                or not names.intersection({"human_ears", "animal_ears", "ears"})
                or ratio is None
                or float(ratio) < overlap_threshold):
            continue
        action = "approved_tag_override" if tail_approved else "unresolved_skipped"
        conflicts.append({
            "classes": [left, right],
            "smaller_part_ratio": float(ratio),
            "threshold": overlap_threshold,
            "action": action,
            "raw_detection_preserved": True,
        })
        if not tail_approved:
            unresolved.add("tail")
    if conflicts:
        report["cross_class_conflicts"] = conflicts
    for name in unresolved:
        entry = report.get("parts", {}).get(name)
        if isinstance(entry, dict):
            entry.update(
                semantic_status="unresolved",
                automatic_conditioning=False,
                target_status="unresolved_cross_class_conflict",
                outcome_reason="cross_class_overlap_without_approved_tag",
            )
    return frozenset(unresolved)

def suppress_unapproved_optional_parts(
        report, approved_character_tags=()):
    # Optional anatomy is observable evidence until the user-approved tags
    # confirm that it belongs to the character.
    normalized_tags = {
        str(tag).strip().lower().replace("_", " ")
        for tag in approved_character_tags
    }
    approvals = {
        "tail": any(
            tag == "tail" or tag.endswith(" tail")
            for tag in normalized_tags
        ),
    }
    suppressed = set()
    for name, approved in approvals.items():
        entry = report.get("parts", {}).get(name)
        if approved or not isinstance(entry, dict):
            continue
        if not entry.get("automatic_conditioning", False):
            continue
        entry.update(
            semantic_status="unresolved",
            automatic_conditioning=False,
            target_status="unresolved_unapproved_character_part",
            outcome_reason="missing_approved_character_tag",
            raw_detection_preserved=True,
        )
        suppressed.add(name)
    if suppressed:
        report["unapproved_optional_parts"] = {
            "status": "unresolved_skipped",
            "parts": sorted(suppressed),
            "raw_detection_preserved": True,
            "automatic_conditioning": False,
        }
    return frozenset(suppressed)


def masked_crop(image, mask):
    if image.size != mask.size:
        raise ValueError('비교 이미지와 마스크 좌표가 다릅니다.')
    binary = Image.fromarray((np.asarray(mask.convert('L')) >= 128).astype('uint8') * 255)
    box = binary.getbbox()
    if box is None:
        raise ValueError('비교할 영역이 비어 있습니다.')
    return ImageOps.pad(Image.composite(image.convert('RGB'),
        Image.new('RGB', image.size, 'white'), binary).crop(box), (512, 512), color='white')


@dataclass
class VisualInputs:
    source: Image.Image
    identity: Image.Image
    garment: Image.Image
    identity_mask: Image.Image
    garment_mask: Image.Image
    vibe_reference: Image.Image | None = None
    full_character_mask: Image.Image | None = None
    scene_condition: object | None = None
    extra_references: tuple = ()
    analysis_record: dict | None = None
    color_evidence: dict | None = None
    reference_conditions: object | None = None
    approved_generation: object | None = None
    part_color_descriptions: tuple = ()
    hair_mask: Image.Image | None = None
    face_mask: Image.Image | None = None
    hair_reference: Image.Image | None = None
    hair_reference_scale: float = 0.60
    hair_source_mask: Image.Image | None = None
    source_identity_mask: Image.Image | None = None

    def close(self):
        for field in fields(self):
            image = getattr(self, field.name)
            if isinstance(image, Image.Image):
                image.close()
        if self.scene_condition is not None:
            self.scene_condition.close()
        for reference in self.extra_references:
            reference.close()
        if self.color_evidence is not None:
            self.color_evidence.clear()
        self.reference_conditions = None
        self.approved_generation = None
        self.part_color_descriptions = ()


def apply_reference_face_observation(image, regions, config, *, usage):
    """Add bounded face evidence while preserving parser failure diagnostics."""
    from genai_lab.reference_face_observation import (
        VERSION as FACE_OBSERVATION_VERSION,
        recover_reference_face_region,
    )
    try:
        report = recover_reference_face_region(
            image,
            regions.masks,
            config,
            exclude_from_garment=(
                usage == "source_generation_identity_reference"
            ),
        )
    except Exception as error:
        report = {
            "version": FACE_OBSERVATION_VERSION,
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "masks_modified": False,
            "automatic_generation_condition": False,
        }
    report["usage"] = usage
    regions.record["face_observation"] = report
    if report.get("masks_modified"):
        pixel_counts = regions.record.setdefault("pixel_counts", {})
        pixel_counts["face"] = int(
            np.count_nonzero(
                np.asarray(regions.masks["face"].convert("L")) >= 128
            )
        )
        pixel_counts["identity"] = int(
            np.count_nonzero(
                np.asarray(regions.masks["identity"].convert("L")) >= 128
            )
        )
        if report.get("garment_exclusion_applied"):
            pixel_counts["garment"] = int(
                np.count_nonzero(
                    np.asarray(regions.masks["garment"].convert("L")) >= 128
                )
            )
    return report


def apply_reference_hair_observation(image, regions, observer, *, usage):
    # Apply a measurement-only head/hair silhouette in current coordinates.
    if observer is None:
        report = {
            "version": "reference_hair_observation_v1",
            "status": "disabled",
            "reason": "observer_not_configured",
            "usage": usage,
            "semantic_mask": False,
            "automatic_generation_condition": False,
            "masks_modified": False,
        }
    else:
        try:
            report = observer.observe(image, regions.masks)
        except Exception as error:
            report = {
                "version": "reference_hair_observation_v1",
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                "usage": usage,
                "semantic_mask": False,
                "automatic_generation_condition": False,
                "masks_modified": False,
            }
            regions.record["hair_observation"] = report
            raise RuntimeError(
                f"measurement-only hair observation failed: {error}"
            ) from error
        report["usage"] = usage
    regions.record["hair_observation"] = report
    if report.get("masks_modified"):
        pixel_counts = regions.record.setdefault("pixel_counts", {})
        pixel_counts["hair"] = int(
            np.count_nonzero(
                np.asarray(regions.masks["hair"].convert("L")) >= 128
            )
        )
        if "identity" in regions.masks:
            pixel_counts["identity"] = int(
                np.count_nonzero(
                    np.asarray(regions.masks["identity"].convert("L")) >= 128
                )
            )
    return report


def prepare_visual_inputs(source, garment, config, root, run_log=None,
                          cancelled=lambda: False, status=lambda message: None):
    if config.get('scene_lineart', {}).get('enabled', False):
        from genai_lab.scene_generation import prepare_scene_inputs
        return prepare_scene_inputs(source, garment, config, run_log, cancelled, status)
    from genai_lab.reference_regions import analyze_reference_regions
    from genai_lab.hair_mask_refinement import (
        refine_hair_mask, resolve_hair_mask_refinement,
    )
    from genai_lab.part_color_analysis import analyze_part_colors
    from genai_lab.regional_reference import prepare_extra_references
    settings = config.get("reference_analysis", {})
    if settings.get("garment_reference_mode", "full_garment") != "full_garment":
        raise ValueError("의상 디테일 전용 추출기는 아직 연결되지 않았습니다.")
    status("캐릭터 참조 영역 추출 중 — 중립화·의상 제거 검증 없음")
    mask_processing_seconds = 0.0
    image_extraction_seconds = 0.0
    preparation_timing_logs_written = False
    active_extraction_started_at = None
    mask_started_at = perf_counter()
    try:
        regions = analyze_reference_regions(
            source, config, root, cancelled=cancelled, run_log=run_log,
            analysis_scope="character_input")
        face_observation = apply_reference_face_observation(
            source,
            regions,
            config,
            usage="source_generation_identity_reference",
        )
        if run_log is not None:
            run_log.write_stage("얼굴 보조 관측", str(face_observation))
    except BaseException:
        if run_log is not None:
            run_log.write_stage(
                "이미지 마스크 처리 시간",
                f"상태=failed, 소요={perf_counter() - mask_started_at:.3f}초",
            )
        raise
    mask_processing_seconds += perf_counter() - mask_started_at
    owned, extras, colors = [], (), {}
    human_ear_mask = None
    animal_ear_mask = None
    hair_accessory_mask = None
    condition_bundle = None
    try:
        active_extraction_started_at = perf_counter()
        identity_mask = regions.masks["identity"].copy()
        owned.append(identity_mask)
        source_identity_mask = identity_mask.copy()
        owned.append(source_identity_mask)
        garment_mask = regions.masks["garment"].copy()
        owned.append(garment_mask)
        if identity_mask.getbbox() is None:
            raise ValueError("얼굴·헤어 참조 영역이 비었습니다.")
        identity = masked_crop(source, identity_mask)
        owned.append(identity)
        garment_image, board_record = prepare_garment_board(garment)
        owned.append(garment_image)
        if run_log is not None:
            run_log.write_stage("의상 참조판 준비", str({
                "flux2_klein": board_record,
            }))
        # 전신 비교는 생성 조건이 아니다. 기존 설정의 비교 기능은 유지한다.
        foreground = regions.masks["foreground"]
        full_mask = foreground.copy() if foreground.getbbox() else None
        if full_mask is not None:
            owned.append(full_mask)
        vibe = masked_crop(source, full_mask) if full_mask is not None else None
        if vibe is not None:
            owned.append(vibe)
        image_extraction_seconds += perf_counter() - active_extraction_started_at
        active_extraction_started_at = None
        record = dict(regions.record)
        record["garment_reference_boards"] = {
            "flux2_klein": board_record,
        }
        record["additional_parts_status"] = "analyzer_not_connected"
        analyzer = settings.get("part_analyzer")
        if analyzer is None:
            from genai_lab.extra_parts_analysis import create_extra_parts_analyzer
            analyzer = create_extra_parts_analyzer(config, source_masks=regions.masks)
        if analyzer is not None:
            detected = []
            try:
                timeout = float(settings.get("part_analysis_timeout_seconds", 120))
                if not np.isfinite(timeout) or timeout <= 0:
                    raise ValueError("특수 부위 분석 제한 시간 오류")
                deadline = perf_counter() + timeout
                part_mask_started_at = perf_counter()
                try:
                    detected = analyzer.analyze(
                        source, source.size, cancelled=cancelled, deadline=deadline)
                finally:
                    mask_processing_seconds += perf_counter() - part_mask_started_at
                for part in detected:
                    if part.name == "human_ears":
                        human_ear_mask = part.source_mask.copy()
                    elif part.name in {"animal_ears", "ears"}:
                        animal_ear_mask = part.source_mask.copy()
                    elif part.name == "hair_accessory":
                        hair_accessory_mask = part.source_mask.copy()
                if cancelled():
                    raise InterruptedError("특수 부위 분석을 취소했습니다.")
                if perf_counter() > deadline:
                    raise TimeoutError("특수 부위 분석 제한 시간 초과")
                preview_only = getattr(analyzer, "preview_only", False)
                record["part_detection"] = getattr(analyzer, "report", {})
                from genai_lab.ear_contract import build_source_ear_contract
                record["ear_contract"] = build_source_ear_contract(
                    record["part_detection"],
                    source.size,
                    approved_character_tags=config.get(
                        "clothing_reference_generation", {}).get(
                            "approved_character_tags", ()),
                    explicit_states=settings.get("part_detection", {}).get(
                        "ear_presence_overrides", {}),
                )
                allowed_ear_parts = set(record["ear_contract"]["generation_parts"])
                approved_character_tags = config.get(
                    "clothing_reference_generation", {}
                ).get("approved_character_tags", ())
                unresolved_part_names = frozenset(
                    set(unresolved_cross_class_parts(
                        record["part_detection"],
                        approved_character_tags,
                    ))
                    | set(suppress_unapproved_optional_parts(
                        record["part_detection"],
                        approved_character_tags,
                    ))
                )
                if "human_ears" not in allowed_ear_parts and human_ear_mask is not None:
                    human_ear_mask.close()
                    human_ear_mask = None
                if "animal_ears" not in allowed_ear_parts and animal_ear_mask is not None:
                    animal_ear_mask.close()
                    animal_ear_mask = None
                if run_log is not None:
                    run_log.write_stage("귀·꼬리 연결 전 검출", str(record["part_detection"]))
                    run_log.write_stage(
                        "사람 귀·동물 귀 독립 계약",
                        str(record["ear_contract"]),
                    )
                part_mode = settings.get("part_detection", {}).get("conditioning_mode", "preview_only")
                if part_mode not in ("preview_only", "source_regions_experiment"):
                    raise ValueError("귀·꼬리 참조 연결 모드 오류")
                if preview_only and part_mode == "source_regions_experiment":
                    if not detected:
                        diagnostic = record["part_detection"]
                        diagnostic["no_confirmed_optional_parts"] = {
                            "status": "unresolved_skipped",
                            "reason": "no_candidate_passed_optional_part_gates",
                            "automatic_conditioning": False,
                            "continue_with_identity_and_garment": True,
                        }
                    from genai_lab.reference_conditions import (
                        ReferenceCondition, ReferenceConditions, source_part_conditions, SOURCE_LAYOUT,
                    )
                    # 부위별 원본 조건은 독립적인 불변 스냅샷이다.
                    generation_region_audit = {}
                    generation_parts = tuple(
                        part for part in detected
                        if part.name != "hair_accessory"
                        and part.name not in unresolved_part_names
                        and (part.name not in {"human_ears", "animal_ears", "ears"}
                             or ("animal_ears" if part.name == "ears" else part.name)
                             in allowed_ear_parts))
                    active_extraction_started_at = perf_counter()
                    try:
                        part_conditions = source_part_conditions(
                            source,
                            generation_parts,
                            settings.get("part_detection", {}).get(
                                "generation_region_strengthening", {}),
                            generation_region_audit,
                        )
                        condition_bundle = ReferenceConditions((
                            ReferenceCondition.capture("identity", identity, identity_mask),
                            ReferenceCondition.capture("garment", garment_image, garment_mask),
                            *part_conditions,
                        ), SOURCE_LAYOUT)
                        assembled, experiment = condition_bundle.materialize()
                    finally:
                        image_extraction_seconds += (
                            perf_counter() - active_extraction_started_at)
                        active_extraction_started_at = None
                    for old in (identity, garment_image, identity_mask, garment_mask):
                        old.close()
                    identity, identity_mask = assembled[0].rgb, assembled[0].region
                    garment_image, garment_mask = assembled[1].rgb, assembled[1].region
                    owned.extend((identity, garment_image, identity_mask, garment_mask))
                    extras = assembled[2:]
                    experiment["generation_region_strengthening"] = generation_region_audit
                    record["part_region_experiment"] = experiment
                    record["additional_parts_status"] = (
                        "source_regions_experiment"
                        if extras else "unresolved_skipped"
                    )
                    report = record["part_detection"]
                    report["mode"] = "source_regions_experiment"
                    generation_part_names = {
                        part.name for part in generation_parts
                    }
                    for part in detected:
                        entry = report.get("parts", {}).get(part.name)
                        if entry is not None:
                            if part.name in generation_part_names:
                                entry["target_status"] = "experimental_source_roi"
                            entry["generation_region"] = generation_region_audit.get(part.name)
                    if report.get("debug_dir"):
                        debug_dir = Path(report["debug_dir"])
                        for part in extras:
                            part.region.save(debug_dir / f"{part.name}_target_region.png")
                        (debug_dir / "parts.json").write_text(
                            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                elif preview_only:
                    # 원본 검출은 완료됐지만 생성 좌표 미확정. 추측해서 연결하지 않는다.
                    record["additional_parts_status"] = "detected_not_conditioned" if detected else "unresolved"
                else:
                    generation_parts = tuple(
                        part for part in detected
                        if part.name != "hair_accessory"
                        and part.name not in unresolved_part_names)
                    active_extraction_started_at = perf_counter()
                    try:
                        extras = prepare_extra_references(
                            source, generation_parts, (identity_mask, garment_mask))
                    finally:
                        image_extraction_seconds += (
                            perf_counter() - active_extraction_started_at)
                        active_extraction_started_at = None
                    record["additional_parts_status"] = "measured" if extras else "unresolved"
                record["additional_parts"] = [part.name for part in extras]
                if run_log is not None:
                    run_log.write_stage("귀·꼬리 검출 결과",
                        f"상태={record['additional_parts_status']}, 원본 검출={len(detected)}, 생성 참조={len(extras)}, 진단={record['part_detection']}")
                if settings.get("color_analysis_enabled", True):
                    for part in detected:
                        if (part.name == "hair_accessory"
                                or part.name in unresolved_part_names
                                or (part.name in {"human_ears", "animal_ears", "ears"}
                                    and ("animal_ears" if part.name == "ears" else part.name)
                                    not in allowed_ear_parts)):
                            continue
                        colors[part.name] = analyze_part_colors(
                            source, part.source_mask, max_clusters=3, cancelled=cancelled)
            finally:
                for part in detected:
                    part.close()
                analyzer.close()
        hair_mask_started_at = perf_counter()
        try:
            hair_result = refine_hair_mask(
                regions.masks["hair"],
                regions.masks["identity"],
                regions.masks["face"],
                animal_ear_mask,
                resolve_hair_mask_refinement(config),
                human_ears_mask=human_ear_mask,
                hair_accessory_mask=hair_accessory_mask,
            )
        finally:
            mask_processing_seconds += perf_counter() - hair_mask_started_at
        record["hair_mask_refinement"] = hair_result.record()
        if run_log is not None:
            run_log.write_stage("헤어 마스크 정밀화", str(record["hair_mask_refinement"]))
        if hair_result.mask is not None:
            old_hair_mask = regions.masks["hair"]
            regions.masks["hair"] = hair_result.mask
            old_hair_mask.close()
            if settings.get("color_analysis_enabled", True):
                status("검증된 헤어 영역 색 분포 측정 중")
                colors["hair"] = analyze_part_colors(
                    source, regions.masks["hair"], cancelled=cancelled)
        else:
            record["hair_mask_refinement"]["color_analysis_skipped"] = True
        refined_hair_mask = (
            regions.masks["hair"].copy() if hair_result.mask is not None else None)
        if refined_hair_mask is not None:
            owned.append(refined_hair_mask)
        face_mask = regions.masks["face"].copy()
        owned.append(face_mask)
        from genai_lab.hair_visual_condition import (
            hair_identity_separation_available,
            prepare_hair_visual_condition,
            resolve_hair_visual_condition,
        )
        hair_visual_settings = resolve_hair_visual_condition(config)
        hair_reference = None
        hair_separation_available = (
            refined_hair_mask is not None
            and hair_identity_separation_available(
                identity_mask, refined_hair_mask)
        )
        if hair_visual_settings.enabled and hair_separation_available:
            # 두 귀 원본은 헤어 반환값과 독립이다. 각 생성 영역만 헤어를
            # 침범하지 않도록 같은 원본 마스크에서 별도로 다시 계산한다.
            source_ear_masks = {
                "human_ears": human_ear_mask,
                "animal_ears": animal_ear_mask,
                "ears": animal_ear_mask,
            }
            for ear_name in ("human_ears", "animal_ears", "ears"):
                source_ear_mask = source_ear_masks[ear_name]
                if source_ear_mask is None or not any(
                        part.name == ear_name for part in extras):
                    continue
                from genai_lab.generation_region_strengthening import (
                    strengthen_generation_region,
                )
                from genai_lab.scene_reference import PartReference
                ear_region, ear_region_record = strengthen_generation_region(
                    ear_name,
                    source_ear_mask,
                    settings.get("part_detection", {}).get(
                        "generation_region_strengthening", {}),
                    blocked_mask=refined_hair_mask,
                )
                replacement_parts = []
                try:
                    for part in extras:
                        if part.name == ear_name:
                            replacement_parts.append(PartReference(
                                part.name, part.rgb.copy(),
                                ear_region, part.scale))
                            ear_region = None
                        else:
                            replacement_parts.append(PartReference(
                                part.name, part.rgb.copy(),
                                part.region.copy(), part.scale))
                except BaseException:
                    for part in replacement_parts:
                        part.close()
                    raise
                finally:
                    if ear_region is not None:
                        ear_region.close()
                for part in extras:
                    part.close()
                extras = tuple(replacement_parts)
                experiment_record = record.setdefault(
                    "part_region_experiment", {})
                experiment_record.setdefault(
                    "generation_region_strengthening", {})[
                        ear_name] = ear_region_record
                report_entry = record.get(
                    "part_detection", {}).get("parts", {}).get(ear_name)
                if report_entry is not None:
                    report_entry["generation_region"] = ear_region_record
                if run_log is not None:
                    run_log.write_stage(
                        "귀 생성 영역 재계산",
                        f"부위={ear_name}, 기준=헤어 정밀화 반환값, "
                        f"헤어 침범 제거={ear_region_record['blocked_pixels_removed']}, "
                        "헤어 마스크 변경=False",
                    )
            active_extraction_started_at = perf_counter()
            try:
                (separated_identity_mask,
                 hair_reference, hair_visual_record) = (
                    prepare_hair_visual_condition(
                        source, identity_mask, garment_mask,
                        refined_hair_mask,
                        tuple((part.name, part.region) for part in extras),
                        hair_visual_settings,
                    )
                )
                separated_identity = masked_crop(
                    source, separated_identity_mask)
            finally:
                image_extraction_seconds += (
                    perf_counter() - active_extraction_started_at)
                active_extraction_started_at = None
            identity.close()
            identity_mask.close()
            identity = separated_identity
            identity_mask = separated_identity_mask
            owned.extend((identity, identity_mask, hair_reference))
            record["hair_visual_condition"] = hair_visual_record
            if run_log is not None:
                run_log.write_stage(
                    "헤어 시각 참조 분리",
                    f"상태=prepared, 강도={hair_visual_settings.reference_scale}, "
                    f"identity 제거 픽셀="
                    f"{hair_visual_record['identity_hair_overlap_removed']}, "
                    "헤어 마스크 원본=정밀화 반환값, "
                    "귀 생성 영역=반환 마스크 기준 재계산, "
                    "원본 부위 마스크 변경=False")
        elif hair_visual_settings.enabled:
            record["hair_visual_condition"] = {
                "version": "returned_hair_mask_condition_v1",
                "status": "unresolved_skipped",
                "reason": (
                    "refined_hair_mask_unavailable"
                    if refined_hair_mask is None
                    else "face_identity_empty_after_hair_separation"
                ),
                "automatic_conditioning": False,
                "fallback": "approved_text_tags_only",
            }
            if run_log is not None:
                run_log.write_stage(
                    "헤어 시각 참조 분리",
                    "상태=unresolved_skipped, 정밀화 마스크 없음, "
                    "자동 텐서 주입=False, 승인 태그만 유지")
        else:
            record["hair_visual_condition"] = {
                "version": "returned_hair_mask_condition_v1",
                "status": "disabled",
            }
        from genai_lab.reference_conditions import (
            ReferenceCondition, ReferenceConditions, DISJOINT_LAYOUT,
        )
        if hair_reference is not None:
            # Hair separation was performed against the already materialized,
            # mutually disjoint masks. Seal those exact effective conditions;
            # re-running SOURCE_LAYOUT would subtract the same areas twice.
            condition_bundle = ReferenceConditions((
                ReferenceCondition.capture(
                    "identity", identity, identity_mask),
                ReferenceCondition.capture(
                    "garment", garment_image, garment_mask),
                ReferenceCondition.capture(
                    "hair", hair_reference, refined_hair_mask,
                    hair_visual_settings.reference_scale),
                *[ReferenceCondition.capture(
                    r.name, r.rgb, r.region, r.scale) for r in extras],
            ), DISJOINT_LAYOUT)
        elif condition_bundle is None:
            condition_bundle = ReferenceConditions((
                ReferenceCondition.capture(
                    "identity", identity, identity_mask),
                ReferenceCondition.capture(
                    "garment", garment_image, garment_mask),
                *[ReferenceCondition.capture(
                    r.name, r.rgb, r.region, r.scale) for r in extras],
            ), DISJOINT_LAYOUT)
        # 승인 계약에 들어갈 최종 반환 마스크를 기준으로 다시 검사한다.
        final_garment_region = np.asarray(garment_mask, dtype=np.uint8) >= 128
        final_part_overlaps = {
            part.name: int(np.count_nonzero(
                final_garment_region
                & (np.asarray(part.region, dtype=np.uint8) >= 128)
            ))
            for part in extras
        }
        nonzero_part_overlaps = {
            name: pixels for name, pixels in final_part_overlaps.items()
            if pixels
        }
        final_mask_audit = {
            "version": "returned_garment_part_overlap_audit_v1",
            "status": "passed" if not nonzero_part_overlaps else "failed",
            "garment_source": "materialized_visual_inputs",
            "parts": list(final_part_overlaps),
            "overlap_pixels": final_part_overlaps,
            "required_overlap_pixels": 0,
        }
        record["final_garment_part_overlap_audit"] = final_mask_audit
        if run_log is not None:
            run_log.write_stage(
                "최종 의상·부위 마스크 검증",
                f"상태={final_mask_audit['status']}, "
                f"중첩={final_part_overlaps}, 반환=VisualInputs.garment_mask",
            )
        if nonzero_part_overlaps:
            raise ValueError(
                "최종 의상 마스크가 귀·꼬리 생성 영역과 겹칩니다: "
                f"{nonzero_part_overlaps}"
            )
        record["condition_separation"] = condition_bundle.record()
        record["preparation_stage_timings"] = {
            "image_mask_processing_seconds": round(mask_processing_seconds, 3),
            "image_extraction_seconds": round(image_extraction_seconds, 3),
        }
        if run_log is not None:
            run_log.write_stage(
                "이미지 마스크 처리 시간",
                f"상태=completed, 소요={mask_processing_seconds:.3f}초",
            )
            run_log.write_stage(
                "이미지 추출 시간",
                f"상태=completed, 소요={image_extraction_seconds:.3f}초",
            )
            run_log.write_stage("부위 조건 분리",
                f"조건={[c.name for c in condition_bundle.conditions]}, 원본 조건 변경 없음, "
                f"최종 배치 규칙 유지={condition_bundle.layout_policy}, 공통 생성 모델 유지")
        preparation_timing_logs_written = True
        from genai_lab.part_color_descriptions import describe_input_part_colors, description_records
        color_descriptions = describe_input_part_colors(colors, settings.get("part_color_text", {}), root)
        record["part_color_descriptions"] = description_records(color_descriptions)
        if run_log is not None and color_descriptions:
            run_log.write_stage("부위 색상 설명 제안", str(record["part_color_descriptions"]))
        source_copy = source.copy()
        owned.append(source_copy)
        result = VisualInputs(source_copy, identity, garment_image, identity_mask,
                              garment_mask, vibe, full_mask, extra_references=extras,
                              analysis_record=record, color_evidence=colors,
                              reference_conditions=condition_bundle,
                              part_color_descriptions=color_descriptions,
                              hair_mask=refined_hair_mask,
                              face_mask=face_mask,
                              hair_reference=hair_reference,
                              hair_reference_scale=(
                                  hair_visual_settings.reference_scale),
                              hair_source_mask=refined_hair_mask,
                              source_identity_mask=source_identity_mask)
        validate_visual_inputs(result)
        if run_log is not None:
            run_log.write_stage("참조 분석 완료",
                f"헤어 색 측정={'hair' in colors}, 추가 부위={len(extras)}, 중립화/제거 검증=미실행")
        return result
    except BaseException:
        if active_extraction_started_at is not None:
            image_extraction_seconds += (
                perf_counter() - active_extraction_started_at)
            active_extraction_started_at = None
        if run_log is not None and not preparation_timing_logs_written:
            run_log.write_stage(
                "이미지 마스크 처리 시간",
                f"상태=failed, 소요={mask_processing_seconds:.3f}초",
            )
            run_log.write_stage(
                "이미지 추출 시간",
                f"상태=failed, 소요={image_extraction_seconds:.3f}초",
            )
        for image in owned:
            image.close()
        for reference in extras:
            reference.close()
        raise
    finally:
        if human_ear_mask is not None:
            human_ear_mask.close()
        if animal_ear_mask is not None:
            animal_ear_mask.close()
        if hair_accessory_mask is not None:
            hair_accessory_mask.close()
        regions.close()


def configure_visual_condition(
        pipeline, inputs, identity_scale=0.7, garment_scale=0.45,
        *, stage='all'):
    validate_visual_inputs(inputs)
    require_reference_experiment_approval(inputs)
    import torch
    scene = getattr(inputs, 'scene_condition', None)
    if scene is not None:
        from genai_lab.scene_generation import validate_scene_condition
        validate_scene_condition(scene, inputs.source.size)
    if stage == 'all':
        entries = adapter_references(inputs, identity_scale, garment_scale)
    else:
        from genai_lab.reference_order import adapter_references_for_stage
        entries = adapter_references_for_stage(
            inputs, stage, identity_scale, garment_scale)
    reference_images = [entry.image for entry in entries]
    scales = [entry.scale for entry in entries]
    from genai_lab.reference_order import unmasked_adapter_scale
    pipeline.set_ip_adapter_scale(unmasked_adapter_scale(scales))
    with torch.inference_mode():
        embeds = pipeline.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[reference_images],
            ip_adapter_image_embeds=None, device=pipeline._execution_device,
            num_images_per_prompt=1, do_classifier_free_guidance=True)
    # 분석 마스크는 부위별 JSON과 잘라낸 참조를 만드는 데까지만 사용한다.
    # 기준 이미지 좌표는 무작위 T2I 후보 좌표와 일치하지 않으므로 생성기의
    # IP-Adapter 공간 마스크로 재사용하지 않는다.
    return {'ip_adapter_image_embeds': embeds}


def image_feature(pipeline, image):
    import torch
    with torch.inference_mode():
        values = pipeline.feature_extractor(images=image, return_tensors='pt').pixel_values
        values = values.to(device=pipeline._execution_device, dtype=pipeline.image_encoder.dtype)
        feature = pipeline.image_encoder(values).image_embeds.detach().float().cpu().numpy().reshape(-1)
    norm = np.linalg.norm(feature)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('유효한 이미지 특징을 계산하지 못했습니다.')
    return feature / norm


def compare_candidate(pipeline, image, inputs, reference_features,
                      candidate_masks=None):
    output_coordinates = candidate_masks is not None
    scores = {
        'method': (
            'CLIP cosine / redetected output-coordinate ROI; not probability'
            if output_coordinates else
            'CLIP cosine / fixed spatial ROI; not probability'
        ),
        'character': None,
        'hair': None,
        'garment': None,
        'vibe': None,
        'warnings': [],
    }
    if output_coordinates:
        character_mask = candidate_masks.get(
            'face' if inputs.hair_reference is not None else 'identity')
        pairs = (
            ('character', character_mask),
            ('hair', candidate_masks.get('hair')),
            ('garment', candidate_masks.get('garment')),
            ('vibe', candidate_masks.get('foreground')),
        )
    else:
        pairs = (
            ('character', inputs.identity_mask),
            ('hair', None),
            ('garment', inputs.garment_mask),
            ('vibe', inputs.full_character_mask),
        )
    for key, mask in pairs:
        try:
            if key == 'hair' and not output_coordinates:
                continue
            if mask is None:
                raise ValueError('candidate output region is unavailable')
            if reference_features.get(key) is None:
                raise ValueError('reference feature is unavailable')
            with masked_crop(image, mask) as crop:
                feature = image_feature(pipeline, crop)
            scores[key] = float(np.clip(
                np.dot(feature, reference_features[key]), -1, 1))
        except (ValueError, AttributeError) as error:
            scores['warnings'].append(f'{key}: {error}')
    return scores


def detect_output_similarity_regions(
        image, config, root, directory, candidate_number, *,
        cancelled=lambda: False, run_log=None, suffix='base',
        hair_observer=None):
    """Redetect score masks in generated coordinates and persist the evidence."""
    from genai_lab.reference_regions import analyze_reference_regions
    scope_by_suffix = {
        "base": "base_output",
        "native_approved_base": "base_output",
        "final": "final_output",
        "flux2_klein": "final_output",
    }
    try:
        analysis_scope = scope_by_suffix[suffix]
    except KeyError as error:
        raise ValueError(
            f"unapproved output-coordinate analysis stage: {suffix}"
        ) from error
    regions = analyze_reference_regions(
        image, config, root, cancelled=cancelled, run_log=None,
        allow_parser_foreground_fallback=True,
        analysis_scope=analysis_scope)
    apply_reference_face_observation(
        image,
        regions,
        config,
        usage=f"{suffix}_measurement_only",
    )
    apply_reference_hair_observation(
        image,
        regions,
        hair_observer,
        usage=f"{suffix}_measurement_only",
    )
    target = (Path(directory) / 'output_regions' /
              f'candidate_{candidate_number}_{suffix}')
    target.mkdir(parents=True, exist_ok=True)
    try:
        for name, mask in regions.masks.items():
            mask.save(target / f'{name}.png')
        record = dict(regions.record)
        record.update({
            'coordinate_space': 'generated_candidate',
            'candidate_number': int(candidate_number),
            'stage': suffix,
            'source_reference_masks_reused': False,
        })
        (target / 'regions.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding='utf-8')
        regions.record = record
        if run_log is not None:
            run_log.write_stage(
                '출력 좌표 마스크 재검출',
                f'후보={candidate_number}, 단계={suffix}, '
                f"픽셀={record.get('pixel_counts')}, 저장={target}")
        return regions, target
    except BaseException:
        regions.close()
        raise


def detect_source_validation_regions(
        image, config, root, directory, *, cancelled=lambda: False,
        run_log=None, hair_observer=None):
    """Re-run source parsing after Base setup and persist fresh evidence."""
    from genai_lab.reference_regions import analyze_reference_regions

    regions = analyze_reference_regions(
        image,
        config,
        root,
        cancelled=cancelled,
        run_log=None,
        allow_parser_foreground_fallback=True,
        analysis_scope='character_input',
    )
    apply_reference_face_observation(
        image,
        regions,
        config,
        usage="source_revalidation_measurement_only",
    )
    apply_reference_hair_observation(
        image,
        regions,
        hair_observer,
        usage="source_revalidation_measurement_only",
    )
    target = Path(directory) / 'source_reference_revalidation'
    target.mkdir(parents=True, exist_ok=True)
    try:
        for name, mask in regions.masks.items():
            mask.save(target / f'{name}.png')
        record = dict(regions.record)
        record.update({
            'coordinate_space': 'source_reference',
            'stage': 'post_base_setup_revalidation',
            'source_reference_masks_reused_for_generation': False,
            'validation_only': True,
        })
        (target / 'regions.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding='utf-8')
        regions.record = record
        if run_log is not None:
            run_log.write_stage(
                '참조 이미지 재검증',
                f"픽셀={record.get('pixel_counts')}, 저장={target}, "
                '생성 마스크 주입=false',
            )
        return regions, target
    except BaseException:
        regions.close()
        raise


@dataclass
class CandidateBatch:
    candidates: list
    paths: list[Path]
    directory: Path
    warning: str = ''
    quarantined: list[dict] = field(default_factory=list)
    review_stage: str = 'final'

    def close(self):
        for candidate in self.candidates:
            candidate.image.close()


def _persist_candidate_record(path, record, candidate):
    """Persist the latest candidate decision without losing initial metadata."""
    from genai_lab.result import write_json

    saved_path = path.with_suffix('.json')
    saved = {}
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding='utf-8'))
    saved.update(record)
    saved.update(seed=candidate.seed, prompt=candidate.prompt)
    write_json(saved_path, saved)


def generate_visual_batch(
    pipeline,
    config,
    request,
    root,
    log,
    inputs,
    cancelled,
    status,
    *,
    defer_final_refinement: bool = False,
):
    validate_reference_mode(config)
    validate_visual_inputs(inputs)
    require_reference_experiment_approval(inputs)
    from genai_lab.generator import generate_character_candidate
    from genai_lab.scene_generation import scene_arguments
    # Validate pipeline and hint before any reference embedding or scoring work.
    if config.get('scene_lineart', {}).get('enabled', False) or inputs.scene_condition is not None:
        scene_arguments(pipeline, config, inputs, (request.width, request.height))
    reference_config = config['clothing_reference_generation']
    from genai_lab.native_pipeline_contract import native_pipeline_enabled
    native_character_base = native_pipeline_enabled(config)
    from genai_lab.approved_reference_run import require_reference_run, seal_encoded_condition
    approved_run = (require_reference_run(inputs, config, request)
                    if reference_config.get('enabled') else None)
    identity_scale = float(reference_config.get('identity_reference_scale', 0.7))
    configured_garment_scale = float(reference_config.get('garment_reference_scale', 0.45))
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in (identity_scale, configured_garment_scale)):
        raise ValueError('얼굴·헤어/의상 참조 강도는 0~1 사이의 유한한 값이어야 합니다.')
    if configured_garment_scale <= 0 and not native_character_base:
        raise ValueError('의상 이미지 참조 강도는 0보다 커야 합니다.')
    count = int(config['clothing_reference_generation']['candidate_count'])
    timeout = float(config.get('reference_analysis', {}).get('generation_timeout_seconds', 1800))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('참조 생성 시간 제한이 잘못됐습니다.')
    def start_timer(label, seconds):
        started = perf_counter()
        def check_running(*, include_cancel=True):
            if include_cancel and cancelled():
                raise InterruptedError('사용자가 참조 생성을 취소했습니다.')
            if perf_counter() - started >= seconds:
                raise TimeoutError(f'{label} 시간 제한 {seconds:g}초를 초과했습니다.')
        log.write_stage('시간 제한 시작', f'{label}, 제한={seconds:g}초, 독립 타이머')
        return check_running
    preparation_timeout = float(config.get('reference_analysis', {}).get('preparation_timeout_seconds', timeout))
    if not np.isfinite(preparation_timeout) or preparation_timeout <= 0:
        raise ValueError('참조 준비 시간 제한이 잘못됐습니다.')
    check_preparation = start_timer('공통 참조 준비', preparation_timeout)
    if not 2 <= count <= 100:
        raise ValueError('비교를 위해 후보 개수는 2~100개로 지정하세요.')
    directory = Path(tempfile.mkdtemp(prefix='genai-candidates-'))
    batch = CandidateBatch([], [], directory)
    references = {}
    condition = None
    common_prompt_condition = None
    common_prompt_record = None
    common_base_latents = None
    common_latent_settings = None
    similarity_retry_fallbacks = []
    semantic_gate_fallbacks = []
    reference_validation_fallbacks = []
    initial_gender_conflicts = []
    initial_quality_failures = []
    from genai_lab.generated_image_integrity import resolve_generated_image_integrity
    integrity_settings = resolve_generated_image_integrity(config)
    from genai_lab.candidate_pipeline import (
        resolve_candidate_pipeline,
        semantic_blocking_reasons,
        semantic_refinement_reasons,
        structure_blocking_reasons,
        structure_refinement_reasons,
    )
    candidate_pipeline_settings = resolve_candidate_pipeline(config, count)
    hair_observer = None
    if candidate_pipeline_settings.output_coordinate_similarity:
        from genai_lab.reference_hair_observation import ReferenceHairObserver
        hair_observer = ReferenceHairObserver(config)
    from genai_lab.candidate_semantic_gate import (
        CandidateSemanticAnalyzer,
        reset_candidate_runtime,
        resolve_candidate_semantic_gate,
    )
    semantic_gate_settings = resolve_candidate_semantic_gate(config)
    from genai_lab.candidate_structure_gate import (
        evaluate_candidate_structure,
        resolve_candidate_structure_gate,
    )
    structure_gate_settings = resolve_candidate_structure_gate(config)
    from genai_lab.body_morphology import request_body_morphology
    from genai_lab.body_morphology_similarity import (
        evaluate_body_morphology_similarity,
        resolve_body_morphology_similarity,
    )
    body_morphology = request_body_morphology(request)
    body_similarity_settings = resolve_body_morphology_similarity(config)
    semantic_analyzer = None
    semantic_analyzer_error = None
    source_validation_regions = None
    source_validation_directory = None
    source_validation_error = None
    source_validation_attempted = False
    attempt_limit = (
        candidate_pipeline_settings.maximum_attempts
        if candidate_pipeline_settings.enabled else count
    )
    maximum_attempt_limit = (
        attempt_limit + max(
            candidate_pipeline_settings.gender_retry_attempts,
            candidate_pipeline_settings.quality_retry_attempts,
        )
        if candidate_pipeline_settings.enabled else attempt_limit
    )
    retry_phase = None
    retry_attempt_limit = attempt_limit
    try:
        # Retain the approved references even if validation or encoding fails.
        for field in fields(inputs):
            image = getattr(inputs, field.name)
            if isinstance(image, Image.Image):
                image.save(directory / f'input_{field.name}.png')
        if inputs.reference_conditions is not None:
            inputs.reference_conditions.save(directory / 'independent_conditions')
        input_diagnostics = verify_and_save_ip_adapter_input(inputs.garment, directory, log)
        if inputs.hair_reference is not None:
            verify_and_save_ip_adapter_input(
                inputs.hair_reference, directory / 'hair', log,
                part_name='hair')
            inputs.hair_mask.save(directory / 'hair' / 'target_region.png')
        from genai_lab.part_color_analysis import save_color_evidence
        for name, evidence in (inputs.color_evidence or {}).items():
            save_color_evidence(evidence, directory / "part_colors", name)
        for part in inputs.extra_references:
            verify_and_save_ip_adapter_input(
                part.rgb, directory / part.name, log, part_name=part.name)
            part.region.save(directory / part.name / "target_region.png")
        if inputs.scene_condition is not None:
            inputs.scene_condition.hint.save(directory / 'scene_lineart.png')
            for index, part in enumerate(inputs.scene_condition.references):
                verify_and_save_ip_adapter_input(part.rgb, directory / f'reference_{index:02d}', log)
                part.region.save(directory / f'region_{index:02d}.png')
        from genai_lab.common_candidate_latents import (
            create_common_base_latents, resolve_common_candidate_latents,
        )
        common_latent_settings = resolve_common_candidate_latents(config)
        if common_latent_settings.disabled_reason:
            log.write_stage(
                '공통 후보 latent 비활성',
                'Img2Img는 승인 RGB를 VAE로 인코딩하도록 latents 인수를 전달하지 않음')
        if common_latent_settings.enabled:
            common_base_latents = create_common_base_latents(
                pipeline, request, common_latent_settings)
            log.write_stage(
                '공통 후보 latent 준비',
                f'상관계수={common_latent_settings.correlation}, 기준 시드={request.seed}, '
                f'형태={list(common_base_latents.shape)}, dtype={common_base_latents.dtype}, CPU 보관')
        check_preparation()
        log.write_stage('참조 전달 순서', ' → '.join(adapter_reference_names(inputs)))
        staged_generation = bool(
            config.get('staged_reference_generation', {}).get('enabled', False)
        )
        if native_character_base:
            if not staged_generation:
                raise ValueError(
                    'Native v2 캐릭터 Base에는 단계별 참조 격리가 필요합니다.'
                )
            if configured_garment_scale != 0:
                raise ValueError(
                    'Native v2 캐릭터 Base의 의상 어댑터 강도는 0이어야 합니다.'
                )
            prompt_record = reference_config.get('approved_prompt_record', {})
            positive_record = (
                prompt_record.get('positive', {})
                if isinstance(prompt_record, dict)
                else {}
            )
            if positive_record.get('garment_tags_included') is not False:
                raise ValueError(
                    'Native v2 캐릭터 Base 승인 기록에서 의상 프롬프트 '
                    '격리를 확인할 수 없습니다.'
                )
        if staged_generation:
            from genai_lab.reference_order import (
                adapter_references_for_stage,
            )
            base_reference_names = tuple(
                entry.name for entry in adapter_references_for_stage(
                    inputs, 'base', identity_scale, configured_garment_scale
                )
            )
            deferred_reference_names = tuple(
                name for name in adapter_reference_names(inputs)
                if name not in base_reference_names
            )
            if native_character_base and any(
                name == 'garment' for name in base_reference_names
            ):
                raise ValueError(
                    'Native v2 캐릭터 Base에 의상 시각 참조가 연결됐습니다.'
                )
            log.write_stage(
                'Base 시각 참조',
                f'전달={list(base_reference_names)}, '
                f'후속 국소 단계로 연기={list(deferred_reference_names)}',
            )
            condition = configure_visual_condition(
                pipeline, inputs, identity_scale, configured_garment_scale,
                stage='base',
            )
        else:
            base_reference_names = adapter_reference_names(inputs)
            condition = configure_visual_condition(
                pipeline, inputs, identity_scale, configured_garment_scale
            )
        log.write_stage(
            '생성 참조 격리',
            f'Base IP-Adapter 전달={list(base_reference_names)}, '
            f'전신 Identity 단일 참조={native_character_base}, '
            '국소 헤어·귀·꼬리·의상 참조=후속 단계로 연기, '
            '기준 이미지 좌표 IP-Adapter 공간 마스크=미전달')
        if candidate_pipeline_settings.enabled:
            from genai_lab.common_prompt_embeddings import (
                encode_common_prompt_embeddings,
            )
            approved_pair = reference_config.get('approved_prompt_pair')
            if (isinstance(approved_pair, (tuple, list))
                    and len(approved_pair) == 2):
                common_prompt_condition, common_prompt_record = (
                    encode_common_prompt_embeddings(
                        pipeline, approved_pair[0], approved_pair[1]))
                log.write_stage(
                    '공통 프롬프트 임베딩',
                    f"상태={common_prompt_record['status']}, "
                    f"shape={common_prompt_record.get('shapes')}, 후보 간 읽기 전용 공유")
        receipt = None
        if approved_run is not None:
            require_reference_run(inputs, config, request)
            receipt = seal_encoded_condition(inputs, condition)
            (directory / 'approved_generation.json').write_text(
                json.dumps(dict(approved_run.record(), approval_fingerprint=approved_run.fingerprint,
                                encoded_condition_fingerprint=receipt.condition_fingerprint),
                           ensure_ascii=False, indent=2), encoding='utf-8')
            from genai_lab.reference_condition_snapshot import (
                write_reference_condition_snapshot,
            )
            write_reference_condition_snapshot(
                directory / 'reference_condition_snapshot.json',
                inputs, reference_config)
            log.write_stage('승인 조건 봉인', approved_run.fingerprint)
        if semantic_gate_settings.enabled:
            try:
                semantic_analyzer = CandidateSemanticAnalyzer(config)
                log.write_stage(
                    '후보 의미 게이트 준비',
                    '상태=ready, 분석=WD14 원시 점수, 순위 계산 전 실행')
            except Exception as error:
                semantic_analyzer_error = f'{type(error).__name__}: {error}'
                log.write_stage(
                    '후보 의미 게이트 준비',
                    f'상태=unresolved, 오류={semantic_analyzer_error}, '
                    '후보 자동 승인 금지')
        for key, image in (
                ('character', inputs.identity),
                ('hair', inputs.hair_reference),
                ('garment', inputs.garment),
                ('vibe', inputs.vibe_reference)):
            try:
                if image is None:
                    raise ValueError('전신 참조 영역 없음')
                references[key] = image_feature(pipeline, image)
            except (ValueError, AttributeError) as error:
                references[key] = None
                log.write_stage('유사도 계산 불가', f'{key}: {error}')
        check_preparation()
        for index in range(maximum_attempt_limit):
            if index >= attempt_limit:
                if (len(batch.candidates)
                        >= candidate_pipeline_settings.target_valid_candidates):
                    break
                if retry_phase is None:
                    from genai_lab.candidate_pipeline import select_retry_phase
                    retry_phase, retry_count = select_retry_phase(
                        initial_gender_conflicts,
                        initial_quality_failures,
                        candidate_pipeline_settings,
                    )
                    if retry_phase is None:
                        break
                    retry_attempt_limit = attempt_limit + retry_count
                    if retry_phase == 'gender':
                        log.write_stage(
                            '성별 충돌 추가 후보 시작',
                            f'기본 시도={attempt_limit}회 소진, '
                            f'추가 시도={retry_count}회, '
                            '전체 기본 후보 성별 충돌, 새 변형 시드 사용')
                    else:
                        log.write_stage(
                            '일반 품질 추가 후보 시작',
                            f'기본 시도={attempt_limit}회 소진, '
                            f'추가 시도={retry_count}회, '
                            '구조 통과 후 재생성 가능한 의미·유사도 실패 후보 존재, '
                            '새 변형 시드 사용')
                if index >= retry_attempt_limit:
                    break
            if cancelled():
                batch.warning = '사용자가 중단했습니다. 완료된 후보만 표시합니다.'
                break
            status(
                f'후보 시도 {index + 1}/'
                f'{retry_attempt_limit if retry_phase else attempt_limit} 생성 중 '
                f'(정상 목표 {candidate_pipeline_settings.target_valid_candidates}개)'
            )
            local = dict(config)
            garment_scale = configured_garment_scale
            check_running = start_timer(f'후보 {index + 1} 생성', timeout)
            local['clothing_reference_generation'] = dict(config['clothing_reference_generation'],
                visual_inputs=inputs, visual_condition=condition, garment_reference_scale=garment_scale,
                identity_reference_scale=identity_scale, check_running=check_running,
                encoded_reference_receipt=receipt,
                common_prompt_condition=common_prompt_condition,
                common_prompt_record=common_prompt_record,
                integrity_debug_directory=str(directory / 'corrupted'),
                integrity_attempt=1)
            if config.get('scene_lineart', {}).get('enabled', False):
                local['scene_lineart'] = dict(config['scene_lineart'], cancelled=cancelled)
            current = replace(request, seed=(request.seed + index) % (2**32), candidate_number=index + 1)
            if common_latent_settings.enabled:
                from genai_lab.common_candidate_latents import correlated_candidate_latents
                candidate_latents, latent_record = correlated_candidate_latents(
                    common_base_latents, request.seed, index, common_latent_settings,
                    getattr(pipeline, '_execution_device', 'cpu'))
                local['clothing_reference_generation']['candidate_latents'] = candidate_latents
                local['clothing_reference_generation']['candidate_latent_record'] = latent_record
                log.write_stage(
                    '후보 공통 latent',
                    f"후보={index + 1}, 단계={latent_record['latent_phase']}, "
                    f"상관계수={latent_record['correlation']}, "
                    f"변형 시드={latent_record['variation_seed']}, sha256={latent_record['sha256']}")
            variant = (
                '캐릭터 전용 Base (의상 프롬프트·어댑터 OFF)'
                if native_character_base
                else '의상 이미지 참조 ON'
            )
            identity_label = (
                '전신 Identity'
                if native_character_base
                else (
                    '얼굴 전용' if inputs.hair_reference is not None
                    else '얼굴·헤어'
                )
            )
            log.write_stage(
                '참조 조건',
                f'후보={index + 1}, 조건={variant}, 시드={current.seed}, '
                f'{identity_label} 강도={identity_scale}, '
                f'헤어 전용 강도='
                f'{inputs.hair_reference_scale if inputs.hair_reference is not None else "미사용"}, '
                f'의상 강도={garment_scale}, '
                f'의상 태그 유지={not native_character_base}')
            timings = {}
            integrity_retry_count = 0
            runtime_reset_reports = []
            candidate = None
            while True:
                check_running()
                try:
                    local['clothing_reference_generation']['integrity_attempt'] = (
                        integrity_retry_count + 1)
                    with timed_stage(log, '후보 생성', index + 1, timings):
                        candidate = generate_character_candidate(
                            pipeline, local, current, root, log)
                        try:
                            # 반환된 완성 이미지는 취소 시 보존한다. 시간 초과만 검사한다.
                            check_running(include_cancel=False)
                        except Exception:
                            candidate.image.close()
                            raise
                    break
                except Exception as error:
                    from genai_lab.generated_image_integrity import GeneratedOutputIntegrityError
                    if not isinstance(error, GeneratedOutputIntegrityError):
                        raise
                    log.write_stage(
                        '후보 이미지 무결성 실패',
                        f'후보={index + 1}, 시도={integrity_retry_count + 1}, '
                        f"단계={error.report.get('failure_stage')}, "
                        f"재시도 가능={error.report.get('retryable', False)}, "
                        f'시드={current.seed}, 보고서={error.report}')
                    can_retry = (
                        bool(error.report.get('retryable', False))
                        and integrity_retry_count < integrity_settings.retry_count
                    )
                    if not can_retry:
                        quarantine = {
                            'candidate_number': index + 1,
                            'seed': current.seed,
                            'latent_sha256': local['clothing_reference_generation'].get(
                                'candidate_latent_record', {}).get('sha256'),
                            'failure_stage': error.report.get('failure_stage'),
                            'report': error.report,
                        }
                        batch.quarantined.append(quarantine)
                        log.write_stage(
                            '손상 후보 격리',
                            f"후보={index + 1}, 단계={quarantine['failure_stage']}, "
                            '다음 후보 생성 계속')
                        message = f'후보 {index + 1} 무결성 실패로 격리됨.'
                        batch.warning = f'{batch.warning} {message}'.strip()
                        if not integrity_settings.continue_after_corruption:
                            raise
                        candidate = None
                        break
                    integrity_retry_count += 1
                    reset_report = reset_candidate_runtime(
                        pipeline, inputs, identity_scale, garment_scale,
                        staged=staged_generation,
                    )
                    runtime_reset_reports.append(reset_report)
                    check_running = start_timer(
                        f'후보 {index + 1} 생성 재시도 {integrity_retry_count}', timeout)
                    local['clothing_reference_generation']['check_running'] = check_running
                    latent_hash = local['clothing_reference_generation'].get(
                        'candidate_latent_record', {}).get('sha256')
                    log.write_stage(
                        '후보 런타임 초기화 후 동일 조건 재시도',
                        f'후보={index + 1}, 재시도={integrity_retry_count}/'
                        f'{integrity_settings.retry_count}, 시드={current.seed}, '
                        f'latent_sha256={latent_hash}, 프롬프트참조강도 변경 없음, '
                        f'초기화={reset_report}')
            # 후보 호출에 전달한 GPU latent는 결과 성공 여부와 무관하게
            # 다음 시도 전에 지역 설정에서도 제거한다.
            stale_latents = local['clothing_reference_generation'].pop(
                'candidate_latents', None)
            local['clothing_reference_generation'].pop(
                'candidate_latent_record', None)
            if stale_latents is not None:
                del stale_latents
            if candidate is None:
                if index < attempt_limit:
                    initial_gender_conflicts.append(False)
                    # A quarantined or missing candidate can recover with a new
                    # variation latent after the initial pool is exhausted.
                    initial_quality_failures.append(True)
                continue
            semantic_report = {
                **semantic_gate_settings.record(),
                'status': 'DISABLED', 'action': 'pass', 'checks': {},
            }
            if semantic_gate_settings.enabled and semantic_analyzer is None:
                semantic_report = {
                    **semantic_gate_settings.record(),
                    'status': 'REVIEW', 'action': 'review',
                    'violations': [],
                    'unresolved': ['semantic_analyzer_unavailable'],
                    'checks': {},
                    'fallback_quality_score': -1e9,
                    'error': semantic_analyzer_error,
                    'absence_is_not_failure': True,
                }
            elif semantic_analyzer is not None:
                try:
                    with timed_stage(log, '후보 의미 검사', index + 1, timings):
                        approved_semantic_record = (
                            approved_run.record()
                            if approved_run is not None
                            else inputs.approved_generation.record()
                        )
                        import inspect
                        analyze_parameters = inspect.signature(
                            semantic_analyzer.analyze
                        ).parameters
                        if 'stage' in analyze_parameters:
                            semantic_report = semantic_analyzer.analyze(
                                candidate.image, approved_semantic_record,
                                semantic_gate_settings,
                                stage=(
                                    'base' if staged_generation else 'final'
                                ),
                            )
                        else:
                            semantic_report = semantic_analyzer.analyze(
                                candidate.image, approved_semantic_record,
                                semantic_gate_settings,
                            )
                except Exception as error:
                    semantic_report = {
                        **semantic_gate_settings.record(),
                        'status': 'REVIEW', 'action': 'review',
                        'violations': [],
                        'unresolved': ['semantic_analyzer_error'],
                        'checks': {},
                        'fallback_quality_score': -1e9,
                        'error': f'{type(error).__name__}: {error}',
                        'absence_is_not_failure': True,
                    }
            if index < attempt_limit:
                initial_gender_conflicts.append(any(
                    str(reason).startswith('gender_conflict:')
                    for reason in semantic_report.get('violations', ())))
            structure_report = evaluate_candidate_structure(
                candidate.image, structure_gate_settings)
            body_similarity_report = evaluate_body_morphology_similarity(
                candidate.image,
                body_morphology,
                body_similarity_settings,
                framing_type=getattr(request, 'framing_type', 'full_body'),
                structure_report=structure_report,
            )
            log.write_stage(
                '후보 체형 유사도',
                f"후보={index + 1}, 상태={body_similarity_report['status']}, "
                f"부분 유사도={body_similarity_report.get('overall_similarity_percentage')}, "
                f"관측 범위={body_similarity_report.get('coverage_percentage')}, "
                "반환 차단=false")
            semantic_blocking = semantic_blocking_reasons(
                semantic_report, candidate_pipeline_settings)
            semantic_refinement = semantic_refinement_reasons(
                semantic_report, candidate_pipeline_settings)
            structure_blocking = structure_blocking_reasons(
                structure_report, candidate_pipeline_settings)
            structure_refinement = structure_refinement_reasons(
                structure_report, candidate_pipeline_settings)
            log.write_stage(
                '후보 구조 게이트',
                f"후보={index + 1}, 결정={structure_report['action']}, "
                f"위반={structure_report.get('violations', [])}, "
                f"미확정={structure_report.get('unresolved', [])}, 적용 위치=유사도 계산 이전")
            scoring_error = None
            output_regions_record = None
            output_regions_directory = None
            reference_base_mask_validation = None
            if not semantic_blocking and not structure_blocking:
                output_regions = None
                try:
                    with timed_stage(log, '유사도 계산', index + 1, timings):
                        candidate_masks = None
                        if candidate_pipeline_settings.output_coordinate_similarity:
                            if not source_validation_attempted:
                                source_validation_attempted = True
                                try:
                                    (source_validation_regions,
                                     source_validation_directory) = (
                                        detect_source_validation_regions(
                                            inputs.source,
                                            config,
                                            root,
                                            directory,
                                            cancelled=cancelled,
                                            run_log=log,
                                            hair_observer=hair_observer,
                                        )
                                    )
                                except Exception as error:
                                    source_validation_error = (
                                        f'{type(error).__name__}: {error}'
                                    )
                                    log.write_stage(
                                        '참조 이미지 재검증',
                                        '상태=unresolved, 오류='
                                        f'{source_validation_error}, '
                                        'Base 생성 차단=false, 기존 마스크로 '
                                        '교차 검증 대체=false',
                                    )
                            output_regions, output_regions_directory = (
                                detect_output_similarity_regions(
                                    candidate.image,
                                    config,
                                    root,
                                    directory,
                                    index + 1,
                                    cancelled=cancelled,
                                    run_log=log,
                                    suffix='base',
                                    hair_observer=hair_observer,
                                )
                            )
                            candidate_masks = output_regions.masks
                            output_regions_record = output_regions.record
                            from genai_lab.reference_base_mask_validation import (
                                validate_reference_base_masks,
                            )
                            if source_validation_regions is None:
                                reference_base_mask_validation = {
                                    'version': (
                                        'reference_base_mask_cross_validation_v2'),
                                    'status': 'unresolved',
                                    'reason': 'source_revalidation_unavailable',
                                    'error': source_validation_error,
                                    'source_revalidation_directory': None,
                                    'parts': {},
                                    'blocking': True,
                                    'progression': 'blocked',
                                    'blocking_reasons': [
                                        'source_revalidation_unavailable'
                                    ],
                                    'generation_masks_modified': False,
                                    'source_masks_reused_for_generation': False,
                                }
                            else:
                                reference_base_mask_validation = (
                                    validate_reference_base_masks(
                                        source_validation_regions.masks,
                                        candidate_masks,
                                        settings=config.get(
                                            'reference_base_mask_validation', {}),
                                    )
                                )
                                reference_base_mask_validation[
                                    'source_revalidation_directory'
                                ] = str(source_validation_directory)
                                reference_base_mask_validation[
                                    'source_revalidation_record'
                                ] = source_validation_regions.record
                            log.write_stage(
                                '참조-Base 마스크 교차 검증',
                                '후보='
                                f'{index + 1}, 상태='
                                f"{reference_base_mask_validation['status']}, "
                                '진행 차단='
                                f"{str(bool(reference_base_mask_validation.get('blocking'))).lower()}, "
                                '원본 마스크 주입=false',
                            )
                        if candidate_masks is None:
                            scores = compare_candidate(
                                pipeline, candidate.image, inputs, references)
                        else:
                            scores = compare_candidate(
                                pipeline, candidate.image, inputs, references,
                                candidate_masks=candidate_masks)
                except Exception as error:
                    scoring_error = (
                        f'출력 좌표 재검출 또는 유사도 연산 중단: '
                        f'{type(error).__name__}: {error}')
                    scores = {
                        'method': 'output_coordinate_similarity_failed',
                        'character': None, 'hair': None,
                        'garment': None, 'vibe': None,
                        'warnings': [scoring_error],
                    }
                finally:
                    if output_regions is not None:
                        output_regions.close()
            else:
                scores = {'method': 'not_run_semantic_gate_failed',
                          'character': None, 'hair': None,
                          'garment': None, 'vibe': None,
                          'warnings': []}
            record = dict(candidate.design_reference_record or {}, similarity=scores,
                          candidate_number=index + 1,
                          identity_reference_scope=(
                              'full_character'
                              if native_character_base
                              else (
                                  'face_only'
                                  if inputs.hair_reference is not None
                                  else 'face_hair'
                              )
                          ),
                          base_adapter_reference_names=(
                              list(base_reference_names)
                          ),
                          reference_analysis=inputs.analysis_record,
                          peak_vram_bytes=getattr(candidate, 'peak_vram_bytes', None),
                          part_colors={name: evidence.record() for name, evidence in (inputs.color_evidence or {}).items()},
                          regional_reference_names=adapter_reference_names(inputs),
                          reference_variant=variant,
                          identity_scale=identity_scale, garment_scale=garment_scale,
                          garment_tags_enabled=not native_character_base,
                          ip_adapter_input_diagnostics=input_diagnostics,
                          stage_timings=timings,
                          generation_timeout_seconds=timeout,
                          generation_timeout_scope='per_candidate',
                          integrity_retry_count=integrity_retry_count,
                          runtime_reset_reports=runtime_reset_reports,
                          candidate_semantic_gate=semantic_report,
                          candidate_structure_gate=structure_report,
                          body_morphology=body_morphology.record(),
                          body_morphology_similarity=body_similarity_report,
                          output_coordinate_regions=output_regions_record,
                          output_coordinate_regions_directory=(
                              str(output_regions_directory)
                              if output_regions_directory is not None else None),
                          reference_base_mask_validation=(
                              reference_base_mask_validation),
                          scene_lineart=inputs.scene_condition is not None,
                          scene_debug_dir=(str(inputs.scene_condition.debug_dir) if inputs.scene_condition else None),
                          candidate_count=count, approval='pending')
            record['candidate_attempt_budget'] = {
                'approved_candidate_count': count,
                'initial_attempt_limit': attempt_limit,
                'gender_retry_attempts': (
                    candidate_pipeline_settings.gender_retry_attempts),
                'quality_retry_attempts': (
                    candidate_pipeline_settings.quality_retry_attempts),
                'retry_phase': retry_phase if index >= attempt_limit else None,
                'is_gender_retry': (
                    index >= attempt_limit and retry_phase == 'gender'),
                'is_quality_retry': (
                    index >= attempt_limit and retry_phase == 'quality'),
            }
            path = directory / f'candidate_{index + 1}.png'
            candidate.image.save(path)
            if (
                    reference_base_mask_validation is not None
                    and reference_base_mask_validation.get('blocking')
            ):
                reasons = list(
                    reference_base_mask_validation.get(
                        'blocking_reasons',
                        ('reference_base_mask_validation_failed',),
                    )
                )
                record['candidate_resolution'] = {
                    'version': 'candidate_resolution_v2',
                    'status': 'quarantined_before_native_refinement',
                    'returned_image': None,
                    'reasons': reasons,
                    'resolution_policy': (
                        'reference_base_validation_before_native_refinement'
                    ),
                }
                _persist_candidate_record(path, record, candidate)
                thumbnail = candidate.image.copy()
                thumbnail.thumbnail((192, 336))
                candidate.image.close()
                reference_validation_fallbacks.append((
                    replace(
                        candidate,
                        image=thumbnail,
                        design_reference_record=record,
                    ),
                    path,
                ))
                batch.quarantined.append({
                    'candidate_number': index + 1,
                    'seed': current.seed,
                    'failure_stage': 'reference_base_mask_validation',
                    'report': reference_base_mask_validation,
                })
                if index < attempt_limit:
                    initial_quality_failures.append(True)
                continue
            if structure_blocking:
                record['candidate_resolution'] = {
                    'status': 'quarantined_before_ranking',
                    'returned_image': None,
                    'reasons': list(structure_blocking),
                }
                _persist_candidate_record(path, record, candidate)
                batch.quarantined.append({
                    'candidate_number': index + 1, 'seed': current.seed,
                    'failure_stage': 'candidate_structure_gate',
                    'report': structure_report,
                })
                candidate.image.close()
                if index < attempt_limit:
                    # Keep the structural gate fail-closed, but allow a new
                    # variation latent to replace the rejected composition.
                    initial_quality_failures.append(True)
                continue
            if semantic_blocking:
                record['candidate_decision'] = {
                    'version': 'candidate_semantic_decision_v2',
                    'action': 'retry',
                    'reasons': list(semantic_blocking),
                }
                record['candidate_resolution'] = {
                    'version': 'candidate_resolution_v2',
                    'status': 'retry_with_new_variation_seed',
                    'returned_image': None,
                    'resolution_policy': 'semantic_gate_before_ranking',
                }
                _persist_candidate_record(path, record, candidate)
                thumbnail = candidate.image.copy()
                thumbnail.thumbnail((192, 336))
                candidate.image.close()
                semantic_gate_fallbacks.append((
                    replace(candidate, image=thumbnail,
                            design_reference_record=record), path))
                batch.quarantined.append({
                    'candidate_number': index + 1,
                    'seed': current.seed,
                    'failure_stage': 'candidate_semantic_gate',
                    'report': semantic_report,
                })
                log.write_stage(
                    '후보 의미 게이트',
                    f"후보={index + 1}, 결정={semantic_report['action']}, "
                    f"위반={semantic_report.get('violations')}, "
                    f"불명확={semantic_report.get('unresolved')}, "
                    'CLIP 순위 계산=생략, 다음 후보 생성')
                if index < attempt_limit:
                    initial_quality_failures.append(
                        bool(semantic_blocking)
                        and not structure_blocking
                    )
                continue
            from genai_lab.candidate_pipeline import (
                CandidateAction,
                preliminary_candidate_decision,
            )
            record['quality_contract'] = {
                'return_policy': candidate_pipeline_settings.return_policy,
                'semantic_refinement_reasons': list(semantic_refinement),
                'structure_refinement_reasons': list(structure_refinement),
                'hard_safety_passed': True,
            }
            preliminary = preliminary_candidate_decision(
                record,
                candidate_pipeline_settings,
                stage=('base' if staged_generation else 'final'),
            )
            record['candidate_decision'] = preliminary.report
            if preliminary.action != CandidateAction.KEEP:
                record['candidate_resolution'] = {
                    'version': 'candidate_resolution_v2',
                    'status': (
                        'retry_with_new_variation_seed'
                        if preliminary.action == CandidateAction.RETRY
                        else 'quarantined_before_ranking'),
                    'returned_image': None,
                    'reasons': list(preliminary.reasons),
                }
                (directory / f'candidate_{index + 1}.json').write_text(
                    json.dumps(
                        dict(record, seed=current.seed, prompt=candidate.prompt),
                        ensure_ascii=False, indent=2),
                    encoding='utf-8')
                batch.quarantined.append({
                    'candidate_number': index + 1,
                    'seed': current.seed,
                    'failure_stage': 'candidate_similarity_gate',
                    'report': preliminary.report,
                })
                retry_thumbnail = candidate.image.copy()
                retry_thumbnail.thumbnail((192, 336))
                candidate.image.close()
                similarity_retry_fallbacks.append((
                    replace(
                        candidate,
                        image=retry_thumbnail,
                        design_reference_record=record,
                    ),
                    path,
                ))
                if index < attempt_limit:
                    initial_quality_failures.append(True)
                remaining_in_current_phase = (
                    attempt_limit - index - 1
                    if index < attempt_limit
                    else retry_attempt_limit - index - 1
                )
                next_step = (
                    '현재 단계 내 다음 변형 latent 사용'
                    if remaining_in_current_phase > 0
                    else '현재 단계 소진, 통합 재시도 판단 대기'
                )
                log.write_stage(
                    '후보 자체 검증 재시도',
                    f'후보={index + 1}, 결정={preliminary.action.value}, '
                    f'사유={list(preliminary.reasons)}, {next_step}')
                continue
            if index < attempt_limit:
                initial_quality_failures.append(False)
            thumbnail = candidate.image.copy()
            thumbnail.thumbnail((192, 336))
            candidate.image.close()
            batch.candidates.append(replace(candidate, image=thumbnail, design_reference_record=record))
            batch.paths.append(path)
            (directory / f'candidate_{index + 1}.json').write_text(json.dumps(
                dict(record, seed=current.seed, prompt=candidate.prompt), ensure_ascii=False, indent=2), encoding='utf-8')
            log.write_stage('후보 비교', f'후보={index + 1}, 시드={current.seed}, 점수={scores}, 임시={path}')
            if (candidate_pipeline_settings.enabled
                    and len(batch.candidates)
                    >= candidate_pipeline_settings.target_valid_candidates):
                log.write_stage(
                    '정상 후보 목표 달성',
                    f'정상={len(batch.candidates)}개, 시도={index + 1}/{attempt_limit}, '
                    '남은 후보 생성 생략')
                break
            if scoring_error:
                benchmark_policy = config.get(
                    'candidate_similarity_benchmark', {})
                if benchmark_policy.get(
                        'continue_after_scoring_error', False):
                    batch.warning = (
                        scoring_error
                        + '. 벤치마크는 다음 Seed 생성을 계속합니다.')
                    log.write_stage(
                        '후보 유사도 분석 실패 후 계속',
                        f'후보={index + 1}, 제품 설정 변경 없음, '
                        '벤치마크 전용 다음 Seed 진행')
                else:
                    batch.warning = (
                        scoring_error
                        + '. 완료 이미지는 보존하고 추가 연산은 중단했습니다.')
                    break
    except Exception as error:
        if not batch.candidates:
            raise
        batch.warning = f'후속 생성 중단: {error}. 완료된 후보는 검토할 수 있습니다.'
        log.write_stage('후보 생성 중단', batch.warning)
    finally:
        if semantic_analyzer is not None:
            semantic_analyzer.close()
        if source_validation_regions is not None:
            source_validation_regions.close()
        if hair_observer is not None:
            hair_observer.close()
        condition = None
        common_prompt_condition = None
        pipeline.set_ip_adapter_scale(float(config.get('style', {}).get('scale', request.reference_image_strength)))

    if batch.candidates:
        for rejected_candidate, _ in similarity_retry_fallbacks:
            rejected_candidate.image.close()
        similarity_retry_fallbacks.clear()
        for rejected_candidate, _ in semantic_gate_fallbacks:
            rejected_candidate.image.close()
        semantic_gate_fallbacks.clear()
        for rejected_candidate, _ in reference_validation_fallbacks:
            rejected_candidate.image.close()
        reference_validation_fallbacks.clear()
    elif (
            similarity_retry_fallbacks
            or semantic_gate_fallbacks
            or reference_validation_fallbacks
            or batch.quarantined
    ):
        from genai_lab.candidate_pipeline import summarize_candidate_failures
        failure_summary = summarize_candidate_failures(
            batch.quarantined,
            candidate_pipeline_settings,
            retry_phase=retry_phase,
        )
        (directory / 'candidate_failure_summary.json').write_text(
            json.dumps(failure_summary, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        log.write_stage('후보 통합 진단', failure_summary)
        if (
                similarity_retry_fallbacks
                or semantic_gate_fallbacks
                or reference_validation_fallbacks
        ):
            for rejected_candidate, _ in (
                    similarity_retry_fallbacks
                    + semantic_gate_fallbacks
                    + reference_validation_fallbacks
            ):
                rejected_candidate.image.close()
            raise ValueError(
                '모든 후보가 필수 게이트에 실패했습니다. '
                f"통합 진단={failure_summary['next_action']}")

    if defer_final_refinement and batch.candidates:
        batch.review_stage = 'native_base'
        batch.warning = (
            '이 화면에는 픽셀 무결성·성별·명백한 구조 오염 검사를 통과한 '
            'Animagine Base가 표시됩니다. 얼굴·헤어 유사도는 탈락 조건이 '
            '아니며 퍼센트 진단과 미세 조정 대상으로 기록됩니다. 선택한 '
            'Base는 SDXL 국소 의상 정밀화를 실행합니다.'
        )
        log.write_stage(
            '로컬 네이티브 정밀화 인계',
            f'승인 가능한 Base={len(batch.candidates)}개, '
            '선택 후 SDXL 국소 정밀화 → 최종 게이트',
        )
        return batch

    if (candidate_pipeline_settings.enabled
            and candidate_pipeline_settings.repair_best_candidate_only
            and batch.candidates):
        from genai_lab.candidate_pipeline import (
            CandidateAction,
            candidate_rank_key,
            final_candidate_decision,
            semantic_blocking_reasons,
            semantic_refinement_reasons,
            structure_blocking_reasons,
            structure_refinement_reasons,
        )
        from genai_lab.generated_condition_audit import (
            build_generated_condition_audit,
        )
        from genai_lab.generated_image_integrity import (
            analyze_generated_image_integrity,
        )
        from genai_lab.part_error_correction import (
            PartCorrectionContractError,
            correct_detected_parts,
        )
        from genai_lab.hair_error_correction import (
            HairCorrectionContractError,
            correct_selected_hair,
        )
        from genai_lab.selected_garment_correction import (
            GarmentCorrectionContractError,
            correct_selected_garment,
        )
        selected_approval_record = (
            approved_run.record()
            if approved_run is not None
            else inputs.approved_generation.record()
        )

        ranked_indices = sorted(
            range(len(batch.candidates)),
            key=lambda item: candidate_rank_key(
                batch.candidates[item].design_reference_record),
            reverse=True,
        )
        selected_index = None
        for rank, batch_index in enumerate(ranked_indices, start=1):
            candidate = batch.candidates[batch_index]
            record = candidate.design_reference_record
            candidate_number = int(record['candidate_number'])
            selected_request = replace(
                request,
                seed=int(candidate.seed),
                candidate_number=candidate_number,
                prompt=candidate.prompt,
                negative_prompt=getattr(
                    candidate,
                    'negative_prompt',
                    selected_approval_record['negative_prompt'],
                ),
            )
            check_repair = start_timer(
                f'선택 후보 {candidate_number} 국소 보정', timeout)
            with Image.open(batch.paths[batch_index]) as stored:
                working_image = stored.convert('RGB')
            try:
                garment_image, garment_report = correct_selected_garment(
                    pipeline,
                    working_image,
                    inputs,
                    config,
                    selected_request,
                    root,
                    approved_run_record=selected_approval_record,
                    check_running=check_repair,
                    run_log=log,
                    output_regions_directory=record.get(
                        'output_coordinate_regions_directory'),
                )
                record['selected_garment_correction'] = garment_report
                garment_status = garment_report.get('status')
                garment_inputs = garment_report.get('ip_adapter_inputs')
                blocked_inputs = garment_report.get('blocked_ip_adapter_inputs')
                garment_plan = garment_report.get('garment_edit_plan', {})
                target_coverage = garment_report.get(
                    'target_garment_coverage', {})
                outside_change = garment_report.get('outside_change', {})
                log.write_stage(
                    '선택 후보 의상 보정',
                    f'후보={candidate_number}, '
                    f'상태={garment_status}, '
                    f'목표착용={target_coverage.get("status")}, '
                    f'마스크={garment_plan.get("status")}, '
                    f'외부변화율={outside_change.get("outside_changed_ratio")}, '
                    f'진단={garment_report.get("mask_diagnostic_manifest")}, '
                    f'전달={garment_inputs}, '
                    f'차단={blocked_inputs}',
                )
                if garment_image is not working_image:
                    working_image.close()
                    working_image = garment_image
            except (InterruptedError, GarmentCorrectionContractError):
                working_image.close()
                raise
            except TimeoutError as error:
                record['selected_garment_correction'] = {
                    'status': 'timeout_keep_original',
                    'error': str(error),
                }
            except Exception as error:
                record['selected_garment_correction'] = {
                    'status': 'failed_keep_original',
                    'error': f'{type(error).__name__}: {error}',
                }
            try:
                correction = correct_detected_parts(
                    pipeline,
                    working_image,
                    inputs,
                    config,
                    selected_request,
                    approved_run_record=selected_approval_record,
                    check_running=check_repair,
                )
                record['part_error_correction'] = correction.report
                if correction.image is not working_image:
                    working_image.close()
                    working_image = correction.image
            except (InterruptedError, PartCorrectionContractError):
                working_image.close()
                raise
            except TimeoutError as error:
                record['part_error_correction'] = {
                    'status': 'timeout_keep_original',
                    'parts': {},
                    'error': str(error),
                }
            except Exception as error:
                record['part_error_correction'] = {
                    'status': 'failed_keep_original',
                    'parts': {},
                    'error': f'{type(error).__name__}: {error}',
                }

            try:
                hair_image, hair_report = correct_selected_hair(
                    pipeline,
                    working_image,
                    inputs,
                    config,
                    selected_request,
                    approved_run_record=selected_approval_record,
                    check_running=check_repair,
                    output_regions_directory=record.get(
                        'output_coordinate_regions_directory'),
                )
                record['hair_error_correction'] = hair_report
                if hair_image is not working_image:
                    working_image.close()
                    working_image = hair_image
            except (InterruptedError, HairCorrectionContractError):
                working_image.close()
                raise
            except TimeoutError as error:
                record['hair_error_correction'] = {
                    'status': 'timeout_keep_original',
                    'error': str(error),
                }
            except Exception as error:
                record['hair_error_correction'] = {
                    'status': 'failed_keep_original',
                    'error': f'{type(error).__name__}: {error}',
                }

            hair_trace = record['hair_error_correction']
            adapter_policy = hair_trace.get('adapter_policy', {})
            hair_inpaint_executed = (
                hair_trace.get('pipeline_state_restore') is not None)
            hair_reference_delivery = (
                'hair_reference_only'
                if hair_inpaint_executed else 'not_executed'
            )
            log.write_stage(
                '헤어 IP-Adapter 격리',
                f'후보={candidate_number}, 상태={hair_trace.get("status")}, '
                f'사유={hair_trace.get("reason")}, '
                f'Inpaint 실행={hair_inpaint_executed}, '
                f'전달={hair_reference_delivery}, '
                f'헤어 강도={adapter_policy.get("hair_reference")}, '
                f'차단=identity:{adapter_policy.get("identity")},'
                f'human_ears:{adapter_policy.get("human_ears")},'
                f'animal_ears:{adapter_policy.get("animal_ears")},'
                f'tail:{adapter_policy.get("tail")},'
                f'garment:{adapter_policy.get("garment")}, '
                f'노이즈 강도={hair_trace.get("inpaint_strength")}, '
                f'보정 전={hair_trace.get("before_similarity")}, '
                f'보정 후={hair_trace.get("after_similarity")}, '
                f'메인 가중치 복구={hair_trace.get("pipeline_state_restore")}')

            boundary_trace = hair_trace.get('boundary_guard_result', {})
            if boundary_trace:
                seam_trace = boundary_trace.get('seam', {})
                raw_outside_trace = boundary_trace.get('raw_outside', {})
                log.write_stage(
                    '헤어 Inpaint 경계 게이트',
                    f'후보={candidate_number}, '
                    f"상태={boundary_trace.get('status')}, "
                    f"사유={boundary_trace.get('reason')}, "
                    f"Seam 평균={seam_trace.get('mean_difference')}, "
                    f"Seam P95={seam_trace.get('p95_difference')}, "
                    f"Seam 변경률={seam_trace.get('changed_ratio')}, "
                    f"원시 외부 변경률={raw_outside_trace.get('changed_ratio')}, "
                    f"최종 보호 변경 픽셀="
                    f"{boundary_trace.get('final_protected_changed_pixels')}, "
                    f"최종 외부 변경 픽셀="
                    f"{boundary_trace.get('final_outside_changed_pixels')}")

            promotion_trace = hair_trace.get('promotion_gate_result', {})
            if promotion_trace:
                promotion_metrics = promotion_trace.get('metrics', {})
                log.write_stage(
                    '헤어 승격 게이트',
                    f'후보={candidate_number}, '
                    f"상태={promotion_trace.get('status')}, "
                    f"사유={promotion_trace.get('reason')}, "
                    f"색상={promotion_metrics.get('color_similarity')}, "
                    f"형태 IoU={promotion_metrics.get('shape_iou')}, "
                    f"길이 변화율="
                    f"{promotion_metrics.get('length_change_ratio')}, "
                    f"면적 변화율="
                    f"{promotion_metrics.get('area_change_ratio')}")

            view_trace = hair_trace.get('view_compatibility', {})
            if view_trace:
                log.write_stage(
                    '헤어 시점 호환성',
                    f'후보={candidate_number}, '
                    f"상태={view_trace.get('status')}, "
                    f"기준={view_trace.get('reference', {}).get('view')}, "
                    f"후보={view_trace.get('candidate', {}).get('view')}, "
                    f"허용={view_trace.get('allowed_candidate_views')}, "
                    f"사유={view_trace.get('reason')}, "
                    f"Inpaint 허용={view_trace.get('inpaint_allowed')}")

            hair_view_retry_required = hair_trace.get('status') in {
                'skipped_view_incompatible', 'skipped_view_unresolved',
            }

            post_semantic_report = {
                **semantic_gate_settings.record(),
                'status': 'DISABLED',
                'action': 'pass',
                'checks': {},
            }
            if semantic_gate_settings.enabled:
                post_semantic_analyzer = None
                try:
                    post_semantic_analyzer = CandidateSemanticAnalyzer(config)
                    import inspect
                    post_parameters = inspect.signature(
                        post_semantic_analyzer.analyze
                    ).parameters
                    if 'stage' in post_parameters:
                        post_semantic_report = post_semantic_analyzer.analyze(
                            working_image,
                            selected_approval_record,
                            semantic_gate_settings,
                            stage='final',
                        )
                    else:
                        post_semantic_report = post_semantic_analyzer.analyze(
                            working_image,
                            selected_approval_record,
                            semantic_gate_settings,
                        )
                except Exception as error:
                    post_semantic_report = {
                        **semantic_gate_settings.record(),
                        'status': 'REVIEW',
                        'action': 'review',
                        'violations': [],
                        'unresolved': ['post_repair_semantic_analyzer_error'],
                        'checks': {},
                        'error': f'{type(error).__name__}: {error}',
                    }
                finally:
                    if post_semantic_analyzer is not None:
                        post_semantic_analyzer.close()
            record['post_repair_semantic_gate'] = post_semantic_report
            post_structure = evaluate_candidate_structure(
                working_image, structure_gate_settings)
            record['post_repair_structure_gate'] = post_structure
            post_regions = None
            post_regions_directory = None
            try:
                post_masks = None
                if candidate_pipeline_settings.output_coordinate_similarity:
                    post_regions, post_regions_directory = (
                        detect_output_similarity_regions(
                            working_image,
                            config,
                            root,
                            directory,
                            candidate_number,
                            cancelled=cancelled,
                            run_log=log,
                            suffix='final',
                            hair_observer=hair_observer,
                        )
                    )
                    post_masks = post_regions.masks
                    record['post_repair_output_coordinate_regions'] = (
                        post_regions.record)
                    record['post_repair_output_coordinate_regions_directory'] = (
                        str(post_regions_directory))
                if post_masks is None:
                    post_scores = compare_candidate(
                        pipeline, working_image, inputs, references)
                else:
                    post_scores = compare_candidate(
                        pipeline, working_image, inputs, references,
                        candidate_masks=post_masks)
            except Exception as error:
                post_scores = {
                    'method': 'output_coordinate_similarity_failed',
                    'character': None, 'hair': None,
                    'garment': None, 'vibe': None,
                    'warnings': [
                        f'{type(error).__name__}: {error}',
                    ],
                }
            finally:
                if post_regions is not None:
                    post_regions.close()
            record['post_repair_similarity'] = post_scores
            post_semantic_action = post_semantic_report.get('action')
            post_semantic_violations = post_semantic_report.get('violations')
            log.write_stage(
                '최종 의미 게이트',
                f'후보={candidate_number}, '
                f'결정={post_semantic_action}, '
                f'위반={post_semantic_violations}',
            )

            audit = build_generated_condition_audit(
                selected_approval_record, record)
            record['generated_condition_audit'] = audit
            gender_check = post_semantic_report.get('checks', {}).get('gender')
            if gender_check is not None:
                audit['checks']['gender'] = dict(gender_check)
            post_semantic_blocking = semantic_blocking_reasons(
                post_semantic_report, candidate_pipeline_settings)
            post_semantic_refinement = semantic_refinement_reasons(
                post_semantic_report, candidate_pipeline_settings)
            if post_semantic_blocking:
                audit['checks']['semantic_gate'] = {
                    'status': 'FAIL',
                    'reason': 'explicit_post_repair_hard_semantic_conflict',
                    'violations': list(post_semantic_blocking),
                }
            elif post_semantic_refinement:
                audit['checks']['semantic_quality'] = {
                    'status': 'NEEDS_REFINEMENT',
                    'reason': 'post_repair_semantic_difference',
                    'targets': list(post_semantic_refinement),
                }
            garment_trace = record.get('selected_garment_correction', {})
            garment_status = garment_trace.get('status')
            if garment_status == 'corrected_and_accepted':
                audit['checks']['garment'] = {
                    'status': 'PASS',
                    'reason': garment_status,
                    'similarity': garment_trace.get('after_similarity'),
                }
            elif garment_status in {
                    'rejected_keep_original', 'failed_keep_original'}:
                audit['checks']['garment'] = {
                    'status': 'FAIL',
                    'reason': garment_status,
                }
            from genai_lab.candidate_pipeline import preliminary_candidate_decision
            post_similarity = preliminary_candidate_decision(
                {'similarity': post_scores},
                candidate_pipeline_settings,
                stage='final',
            )
            post_structure_blocking = structure_blocking_reasons(
                post_structure, candidate_pipeline_settings)
            post_structure_refinement = structure_refinement_reasons(
                post_structure, candidate_pipeline_settings)
            if post_structure_blocking:
                audit['checks']['structure'] = {
                    'status': 'FAIL',
                    'reason': 'post_repair_hard_structure',
                    'violations': list(post_structure_blocking),
                }
            elif post_structure_refinement:
                audit['checks']['structure_quality'] = {
                    'status': 'NEEDS_REFINEMENT',
                    'reason': 'post_repair_structure_difference',
                    'targets': list(post_structure_refinement),
                }
            if post_similarity.report.get('refinement_required'):
                audit['checks']['similarity'] = {
                    'status': (
                        'FAIL'
                        if candidate_pipeline_settings.return_policy == 'strict_all'
                        else 'NEEDS_REFINEMENT'),
                    'reason': 'post_repair_similarity_below_target',
                    'percentages': post_similarity.report.get(
                        'similarity_percentages', {}),
                    'targets': post_similarity.report.get(
                        'refinement_targets', []),
                }
            if hair_view_retry_required:
                audit['checks']['hair_view'] = {
                    'status': (
                        'FAIL'
                        if candidate_pipeline_settings.return_policy == 'strict_all'
                        else 'NEEDS_REFINEMENT'),
                    'reason': hair_trace.get('reason'),
                    'correction_status': hair_trace.get('status'),
                    'view_compatibility': view_trace,
                }
            record['quality_diagnostic'] = {
                'version': 'quality_diagnostic_v1',
                'similarity_percentages': post_similarity.report.get(
                    'similarity_percentages', {}),
                'target_percentage': post_similarity.report.get(
                    'target_percentage'),
                'refinement_required': post_similarity.report.get(
                    'refinement_required', False),
                'refinement_targets': post_similarity.report.get(
                    'refinement_targets', []),
                'percentage_is_probability': False,
            }
            audit_states = {
                detail.get('status')
                for detail in audit.get('checks', {}).values()
            }
            audit['status'] = (
                'FAIL' if 'FAIL' in audit_states
                else 'UNRESOLVED' if 'UNRESOLVED' in audit_states
                else 'PASS'
            )
            post_repair_integrity = analyze_generated_image_integrity(
                working_image, integrity_settings)
            record['post_repair_image_integrity'] = post_repair_integrity
            decision = final_candidate_decision(
                audit, candidate_pipeline_settings, post_repair_integrity)
            record['candidate_decision'] = decision.report
            part_diagnostics = {
                name: {
                    'status': detail.get('status'),
                    'reason': detail.get('reason'),
                }
                for name, detail in record[
                    'part_error_correction'].get('parts', {}).items()
            }
            failed_audit_checks = {
                name: detail.get('reason')
                for name, detail in audit.get('checks', {}).items()
                if detail.get('status') == 'FAIL'
            }
            refinement_checks = {
                name: detail.get('reason')
                for name, detail in audit.get('checks', {}).items()
                if detail.get('status') in {
                    'NEEDS_REFINEMENT', 'UNRESOLVED'}
            }
            log.write_stage(
                '최상위 후보 국소 보정',
                f'순위={rank}, 후보={candidate_number}, '
                f"보정={record['part_error_correction'].get('status')}, "
                f"헤어={record['hair_error_correction'].get('status')}, "
                f"헤어 사유={record['hair_error_correction'].get('reason')}, "
                f'부위 진단={part_diagnostics}, '
                f'강제 차단={decision.report.get("hard_audit_failures")}, '
                f'미세 조정={refinement_checks}, '
                f"RGB={post_repair_integrity['status']}, "
                f"최종 감사={audit['status']}, 결정={decision.action.value}")

            if decision.action != CandidateAction.KEEP:
                record['candidate_resolution'] = {
                    'version': 'candidate_resolution_v1',
                    'status': 'quarantined_after_repair',
                    'returned_image': None,
                    'failed_audit_checks': failed_audit_checks,
                    'part_diagnostics': part_diagnostics,
                    'hair_diagnostic': {
                        'status': record['hair_error_correction'].get('status'),
                        'reason': record['hair_error_correction'].get('reason'),
                    },
                }
                _persist_candidate_record(
                    batch.paths[batch_index], record, candidate)
                batch.quarantined.append({
                    'candidate_number': candidate_number,
                    'seed': candidate.seed,
                    'failure_stage': 'generated_condition_post_audit',
                    'report': decision.report,
                    'failed_audit_checks': failed_audit_checks,
                    'part_diagnostics': part_diagnostics,
                    'hair_diagnostic': record[
                        'candidate_resolution']['hair_diagnostic'],
                })
                working_image.close()
                continue

            corrected_parts = any(
                detail.get('status') == 'corrected_and_accepted'
                for detail in record['part_error_correction'].get(
                    'parts', {}).values()
            )
            corrected_hair = (
                record['hair_error_correction'].get('status')
                == 'corrected_and_accepted')
            corrected_garment = (
                record.get('selected_garment_correction', {}).get('status')
                == 'corrected_and_accepted')
            correction_applied = (
                corrected_garment or corrected_parts or corrected_hair
            )
            working_image.save(batch.paths[batch_index])
            selected_thumbnail = working_image.copy()
            selected_thumbnail.thumbnail((192, 336))
            working_image.close()
            candidate.image.close()
            batch.candidates[batch_index] = replace(
                candidate,
                image=selected_thumbnail,
                design_reference_record=record,
            )
            if decision.action == CandidateAction.REVIEW:
                record['candidate_resolution'] = {
                    'version': 'candidate_resolution_v2',
                    'status': 'manual_review_required',
                    'resolution_policy': 'user_review_pending',
                    'returned_image': (
                        'corrected' if correction_applied
                        else 'original_pre_repair'),
                    'unresolved_checks': refinement_checks,
                    'quality_diagnostic': record.get(
                        'quality_diagnostic', {}),
                    'refinement_targets': decision.report.get(
                        'refinement_targets', []),
                }
                batch.warning = (
                    f'{batch.warning} 최상위 후보는 hard safety 검사를 '
                    '통과해 반환했습니다. 유사도·색상·부위 차이는 퍼센트 '
                    '진단과 미세 조정 대상으로 기록했습니다.'
                ).strip()
            else:
                record['candidate_resolution'] = {
                    'version': 'candidate_resolution_v2',
                    'status': 'selected_verified',
                    'returned_image': (
                        'corrected' if correction_applied else 'original'),
                }
            _persist_candidate_record(
                batch.paths[batch_index], record,
                batch.candidates[batch_index])
            selected_index = batch_index
            break

        if selected_index is None:
            batch.close()
            batch.candidates.clear()
            batch.paths.clear()
            raise ValueError('모든 보정 후보가 최종 게이트에 실패하여 원본도 반환하지 않습니다.')

        selected_candidate = batch.candidates[selected_index]
        selected_path = batch.paths[selected_index]
        for index, candidate in enumerate(batch.candidates):
            if index != selected_index:
                candidate.image.close()
        batch.candidates = [selected_candidate]
        batch.paths = [selected_path]
        log.write_stage(
            '최종 후보 선택',
            f"후보={selected_candidate.design_reference_record['candidate_number']}, "
            f'정상 기본 후보={len(ranked_indices)}개 중 1개 반환')
    if not batch.candidates:
        if batch.quarantined:
            from genai_lab.generated_image_integrity import GeneratedOutputIntegrityError
            raise GeneratedOutputIntegrityError(
                '모든 후보가 필수 품질 게이트에서 격리되어 반환할 수 없습니다.',
                {
                    'status': 'all_candidates_quarantined',
                    'candidate_count': count,
                    'quarantined': batch.quarantined,
                },
            )
        raise ValueError('생성된 후보가 없습니다. 취소 상태와 로그를 확인하세요.')
    return batch





