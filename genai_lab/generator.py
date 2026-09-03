"""생성 요청을 한 장씩 처리한다."""

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from genai_lab.clothing import (
    CatVTONLocalSettings,
    CharacterAgnosticApprovedInput,
    CharacterClothingProtectionError,
    ClothingReferenceInput,
    execute_catvton_clothing_try_on,
)
from genai_lab.detail import (
    CharacterDetailCorrectionError,
    correct_character_candidate_details,
)
from genai_lab.request import CharacterGenerationRequest
from genai_lab.pose_estimation import (
    PoseEstimationApprovedInput,
    prepare_pose_control_input,
)
from genai_lab.result import (
    CharacterGenerationCandidate,
    refresh_summary,
    update_request_result,
    write_json,
)
from genai_lab.run_log import GenerationRunLog
from genai_lab.style import (
    load_reference_image,
    prepare_ip_adapter_reference_image,
    prepare_original_image_canvas,
)


def generate_character_candidate(
    pipeline: Any,
    config: dict[str, Any],
    generation_request: CharacterGenerationRequest,
    project_root: Path,
    run_log: GenerationRunLog | None = None,
    clothing_reference_input: ClothingReferenceInput | None = None,
    catvton_settings: CatVTONLocalSettings | None = None,
    approved_agnostic_input: CharacterAgnosticApprovedInput | None = None,
    approved_pose_estimation: PoseEstimationApprovedInput | None = None,
) -> CharacterGenerationCandidate:
    """AI로 후보 한 장을 만들고 파일 저장 없이 메모리 객체로 반환한다.

    반환값:
        사용자 승인 전까지 메모리에만 존재하는 캐릭터 이미지 후보.

    오류:
        모델 실행 또는 GPU 메모리 부족 시 한글 오류를 발생시킨다.
    """
    import torch

    ip_adapter_reference_image = prepare_ip_adapter_reference_image(
        generation_request.reference_image
    )
    if run_log is not None:
        run_log.write_stage(
            "참조 이미지 준비",
            (
                f"IP-Adapter 입력 크기={ip_adapter_reference_image.width}x"
                f"{ip_adapter_reference_image.height}, 사용자 승인본 사용"
            ),
        )

    original_image_canvas = prepare_original_image_canvas(
        generation_request.reference_image,
        generation_request.width,
        generation_request.height,
    )
    if run_log is not None:
        run_log.write_stage(
            "원본 유지 화면 준비",
            (
                f"시작 화면={original_image_canvas.width}x"
                f"{original_image_canvas.height}, "
                f"원본 변경 강도="
                f"{generation_request.original_image_change_strength:.2f}, "
                "비율 유지 및 흰색 여백 사용"
            ),
        )

    prepared_pose_control = None
    pose_control_status = "not_requested"
    pose_control_model_id = None
    pose_control_conditioning_scale = None
    pose_control_guidance_start = None
    pose_control_guidance_end = None
    executed_image_change_strength = (
        generation_request.original_image_change_strength
    )
    if approved_pose_estimation is not None:
        pose_config = config.get("pose_control", {})
        pose_result_policy = config.get("pose_result_policy", {})
        if not pose_config.get("enabled", False):
            raise RuntimeError(
                "승인 자세가 있지만 설정 'pose_control.enabled'가 꺼져 있습니다."
            )
        prepared_pose_control = prepare_pose_control_input(
            approved_pose_estimation,
            generation_request.width,
            generation_request.height,
        )
        pose_control_status = "applied"
        pose_control_model_id = str(pose_config["model_id"])
        pose_control_conditioning_scale = float(
            pose_config["conditioning_scale"]
        )
        pose_control_guidance_start = float(pose_config["guidance_start"])
        pose_control_guidance_end = float(pose_config["guidance_end"])
        executed_image_change_strength = float(
            pose_config["original_image_change_strength"]
        )
        if run_log is not None:
            run_log.write_stage(
                "임시 자세 결과 정책",
                (
                    f"모드={pose_result_policy.get('mode', 'observe_only')}, "
                    f"목표 표본={pose_result_policy.get('target_sample_count', 3)}건, "
                    "자세 불일치 차단="
                    f"{int(bool(pose_result_policy.get('block_on_pose_mismatch', False)))}회, "
                    "Text2Img 전환="
                    f"{'사용' if pose_result_policy.get('switch_to_text_to_image', False) else '미사용'}, "
                    "IP-Adapter 크롭="
                    f"{'사용' if pose_result_policy.get('use_identity_crop', False) else '미사용'}, "
                    "기존 Img2Img·전체 참조 IP-Adapter 유지"
                ),
            )
            run_log.write_stage(
                "자세 제어 입력",
                (
                    f"DWPose 승인 관절={approved_pose_estimation.detected_joint_count}/18개, "
                    f"원본 지도={prepared_pose_control.source_width}x"
                    f"{prepared_pose_control.source_height}, "
                    f"생성 지도={prepared_pose_control.target_width}x"
                    f"{prepared_pose_control.target_height}, "
                    f"확대 비율={prepared_pose_control.resize_scale:.4f}, "
                    "검은 여백="
                    f"{prepared_pose_control.padding_left},"
                    f"{prepared_pose_control.padding_top},"
                    f"{prepared_pose_control.padding_right},"
                    f"{prepared_pose_control.padding_bottom}px, "
                    f"뼈대 픽셀={prepared_pose_control.non_black_pixel_count:,}px, "
                    f"ControlNet 강도={pose_control_conditioning_scale:.2f}, "
                    f"적용 구간={pose_control_guidance_start:.2f}~"
                    f"{pose_control_guidance_end:.2f}, "
                    f"이미지 변경 강도={executed_image_change_strength:.2f}"
                ),
            )

    torch.cuda.reset_peak_memory_stats()
    random_start = torch.Generator(device="cpu").manual_seed(
        generation_request.seed
    )
    model_arguments: dict[str, Any] = {
        "prompt": generation_request.prompt,
        "negative_prompt": generation_request.negative_prompt,
        "num_inference_steps": generation_request.inference_steps,
        "guidance_scale": generation_request.guidance_scale,
        "generator": random_start,
    }
    generation_mode = config["generation"].get("mode", "text_to_image")
    if generation_mode == "image_to_image":
        model_arguments["image"] = original_image_canvas
        model_arguments["strength"] = executed_image_change_strength
    else:
        model_arguments["width"] = generation_request.width
        model_arguments["height"] = generation_request.height
    model_arguments["ip_adapter_image"] = [ip_adapter_reference_image]
    if prepared_pose_control is not None:
        model_arguments["control_image"] = (
            prepared_pose_control.control_map_image
        )
        model_arguments["controlnet_conditioning_scale"] = (
            pose_control_conditioning_scale
        )
        model_arguments["control_guidance_start"] = pose_control_guidance_start
        model_arguments["control_guidance_end"] = pose_control_guidance_end

    generation_started_at = time.perf_counter()
    try:
        generated_image = pipeline(**model_arguments).images[0]
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        is_memory_error = isinstance(error, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(error).lower()
        )
        torch.cuda.empty_cache()
        if is_memory_error:
            raise RuntimeError(
                "GPU 메모리가 부족합니다. 다른 GPU 사용 프로그램을 닫은 뒤 "
                "같은 설정으로 다시 생성하세요."
            ) from error
        raise
    finally:
        ip_adapter_reference_image.close()
        original_image_canvas.close()
        if prepared_pose_control is not None:
            prepared_pose_control.close()

    elapsed_seconds = round(time.perf_counter() - generation_started_at, 3)
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    if run_log is not None:
        run_log.write_stage(
            "모델 반환",
            (
                f"이미지 크기={generated_image.width}x{generated_image.height}, "
                "파일 저장 없음, 사용자 검토용 메모리 보관"
            ),
        )

    before_clothing_image = None
    clothing_change_mask = None
    raw_clothing_try_on_image = None
    clothing_difference_image = None
    clothing_effect_metrics = None
    clothing_try_on_status = "not_requested"
    clothing_verification_warning_ko = None
    if clothing_reference_input is not None:
        clothing_try_on_status = "failed"
        if catvton_settings is None:
            clothing_verification_warning_ko = (
                "의상 참조가 있지만 CatVTON 실행 설정이 없습니다."
            )
        elif approved_agnostic_input is None:
            clothing_verification_warning_ko = (
                "의상 참조가 있지만 사용자 승인 Human-Agnostic 입력이 없습니다."
            )
        else:
            if run_log is not None:
                run_log.write_stage(
                    "의상 참조 합성",
                    "CatVTON 별도 프로세스와 의상 영역 보호 검사 시작",
                )
            try:
                if hasattr(pipeline, "maybe_free_model_hooks"):
                    pipeline.maybe_free_model_hooks()
                torch.cuda.empty_cache()
                clothing_try_on_result = execute_catvton_clothing_try_on(
                    base_character_image=generated_image,
                    clothing_reference_input=clothing_reference_input,
                    approved_agnostic_input=approved_agnostic_input,
                    settings=catvton_settings,
                    seed=generation_request.seed,
                )
                before_clothing_image = generated_image.copy()
                generated_image.close()
                generated_image = clothing_try_on_result.candidate.image
                clothing_change_mask = (
                    clothing_try_on_result.clothing_change_mask
                )
                raw_clothing_try_on_image = (
                    clothing_try_on_result.raw_try_on_image
                )
                clothing_difference_image = (
                    clothing_try_on_result.difference_image
                )
                clothing_effect_metrics = clothing_try_on_result.effect_metrics
                clothing_try_on_status = (
                    "no_effect"
                    if clothing_effect_metrics.no_effect
                    else "completed"
                )
                clothing_verification_warning_ko = (
                    "CatVTON 최종 합성의 승인 영역 안 변경이 0px입니다. "
                    "원시 출력과 차이맵을 확인하세요."
                    if clothing_effect_metrics.no_effect
                    else clothing_try_on_result.candidate.verification.reason_ko
                )
                if run_log is not None:
                    changed_pixel_count = (
                        clothing_try_on_result.candidate.verification
                        .changed_pixel_count_outside_clothing
                    )
                    run_log.write_stage(
                        "의상 참조 합성",
                        (
                            "완료, 의상 영역 밖 변경 픽셀="
                            f"{changed_pixel_count}, "
                            "마스크 출처="
                            f"{clothing_try_on_result.execution_metadata.mask_source}, "
                            "AutoMasker 실행="
                            f"{clothing_try_on_result.execution_metadata.automasker_run_count}회, "
                            "안전 검사="
                            f"{'활성화' if clothing_try_on_result.execution_metadata.safety_check_enabled else '비활성화'}, "
                            "승인 마스크 픽셀="
                            f"{clothing_try_on_result.execution_metadata.approved_mask_pixel_count:,}px, "
                            "원시 model_mask 안 변경="
                            f"{clothing_effect_metrics.raw_changed_inside_model_mask:,}px, "
                            "최종 승인 영역 안 변경="
                            f"{clothing_effect_metrics.final_changed_inside_approved_mask:,}px, "
                            "보호 합성 제거="
                            f"{clothing_effect_metrics.discarded_by_protection_pixels:,}px, "
                            "승인 영역 RGB L1 평균="
                            f"{clothing_effect_metrics.mean_rgb_l1_inside:.4f}, "
                            f"상태={clothing_try_on_status}"
                        ),
                    )
            except (CharacterClothingProtectionError, OSError) as error:
                clothing_verification_warning_ko = str(error)
                if run_log is not None:
                    run_log.write_stage(
                        "의상 참조 합성 실패",
                        f"{error} 기본 생성 후보를 유지합니다.",
                    )


    candidate_image = generated_image
    original_generated_image = None
    detail_correction_status = "disabled"
    detected_face_count = 0
    detected_hand_count = 0
    corrected_region_count = 0
    rejected_region_count = 0
    detail_verification_warning_ko = None
    detail_config = config.get("detail_correction", {})
    if detail_config.get("enabled", False):
        if run_log is not None:
            run_log.write_stage(
                "얼굴·손 부분 보정",
                "YOLO 탐지와 제한 영역 Inpaint 시작",
            )
        try:
            correction_result = correct_character_candidate_details(
                generation_pipeline=pipeline,
                generated_image=generated_image,
                approved_reference_image=generation_request.reference_image,
                prompt=generation_request.prompt,
                negative_prompt=generation_request.negative_prompt,
                seed=generation_request.seed,
                detail_config=detail_config,
                cache_dir=Path(config["model"]["cache_dir"]),
            )
            detected_face_count = correction_result.detected_face_count
            detected_hand_count = correction_result.detected_hand_count
            corrected_region_count = correction_result.corrected_region_count
            rejected_region_count = correction_result.rejected_region_count
            detail_verification_warning_ko = (
                correction_result.verification_warning_ko
            )
            if corrected_region_count:
                candidate_image = correction_result.corrected_image
                original_generated_image = (
                    correction_result.original_generated_image
                )
                generated_image.close()
                detail_correction_status = (
                    "warning"
                    if detail_verification_warning_ko
                    else "completed"
                )
            else:
                correction_result.corrected_image.close()
                correction_result.original_generated_image.close()
                detail_correction_status = "not_detected"
            if run_log is not None:
                run_log.write_stage(
                    "얼굴·손 부분 보정",
                    (
                        f"상태={detail_correction_status}, "
                        f"얼굴={detected_face_count}, 손={detected_hand_count}, "
                        f"보정={corrected_region_count}, 거절={rejected_region_count}, "
                        "마스크 밖 변경=0"
                    ),
                )
        except Exception as error:
            detail_correction_status = "failed"
            if isinstance(error, CharacterDetailCorrectionError):
                detail_verification_warning_ko = str(error)
            else:
                detail_verification_warning_ko = (
                    "예상하지 못한 세부 보정 오류가 발생했습니다: "
                    f"{type(error).__name__}: {error}"
                )
            if run_log is not None:
                run_log.write_stage(
                    "얼굴·손 부분 보정 실패",
                    (
                        f"{detail_verification_warning_ko} "
                        "보정 전 후보를 유지합니다."
                    ),
                )

    elapsed_seconds = round(time.perf_counter() - generation_started_at, 3)
    peak_vram_bytes = torch.cuda.max_memory_allocated()

    # CharacterGenerationCandidate(캐릭터 생성 후보)
    # - 포함: 생성 이미지, 시드, 화면 범위, 모델 설정과 실행 기록.
    # - 생성: AI 모델이 이미지 한 장을 반환한 직후 만든다.
    # - 처리: 사용자 승인 전 후보이며 AI가 품질을 확정하지 않는다.
    # - 저장: 현재 단계에서는 저장하지 않고 GUI 메모리에 전달한다.
    # - 다음 사용처: GUI 미리보기와 사용자 승인 후 저장에 사용한다.
    return CharacterGenerationCandidate(
        image=candidate_image,
        original_generated_image=original_generated_image,
        before_clothing_image=before_clothing_image,
        clothing_change_mask=clothing_change_mask,
        clothing_reference_name=(
            clothing_reference_input.image_path.name
            if clothing_reference_input is not None
            else None
        ),
        clothing_category=(
            clothing_reference_input.category.value
            if clothing_reference_input is not None
            else None
        ),
        clothing_try_on_status=clothing_try_on_status,
        clothing_verification_warning_ko=clothing_verification_warning_ko,
        raw_clothing_try_on_image=raw_clothing_try_on_image,
        clothing_difference_image=clothing_difference_image,
        clothing_effect_metrics=clothing_effect_metrics,
        reference_image_name=generation_request.reference_image_name,
        reference_enhancement_applied=(
            generation_request.reference_enhancement_applied
        ),
        reference_enhancement_model_id=(
            generation_request.reference_enhancement_model_id
        ),
        reference_quality_status=generation_request.reference_quality_status,
        framing_type=generation_request.framing_type.value,
        seed=generation_request.seed,
        candidate_number=generation_request.candidate_number,
        prompt=generation_request.prompt,
        negative_prompt=generation_request.negative_prompt,
        model_id=generation_request.model_id,
        reference_adapter_id=generation_request.reference_adapter_id,
        original_image_change_strength=executed_image_change_strength,
        reference_image_strength=generation_request.reference_image_strength,
        pose_control_status=pose_control_status,
        pose_control_model_id=pose_control_model_id,
        pose_control_conditioning_scale=pose_control_conditioning_scale,
        pose_control_guidance_start=pose_control_guidance_start,
        pose_control_guidance_end=pose_control_guidance_end,
        detail_correction_status=detail_correction_status,
        detected_face_count=detected_face_count,
        detected_hand_count=detected_hand_count,
        corrected_region_count=corrected_region_count,
        rejected_region_count=rejected_region_count,
        detail_verification_warning_ko=detail_verification_warning_ko,
        elapsed_seconds=elapsed_seconds,
        peak_vram_bytes=peak_vram_bytes,
        generated_at=datetime.now().astimezone().isoformat(),
    )


def apply_clothing_to_generated_candidate(
    base_candidate: CharacterGenerationCandidate,
    clothing_reference_input: ClothingReferenceInput,
    catvton_settings: CatVTONLocalSettings,
    approved_agnostic_input: CharacterAgnosticApprovedInput,
    run_log: GenerationRunLog | None = None,
) -> CharacterGenerationCandidate:
    """이미 생성된 같은 후보에 승인 마스크와 의상을 적용한다."""
    if run_log is not None:
        run_log.write_stage(
            "의상 참조 합성",
            "생성 후보 원본과 같은 좌표의 승인 마스크로 CatVTON 시작",
        )
    clothing_try_on_result = execute_catvton_clothing_try_on(
        base_character_image=base_candidate.image,
        clothing_reference_input=clothing_reference_input,
        approved_agnostic_input=approved_agnostic_input,
        settings=catvton_settings,
        seed=base_candidate.seed,
    )
    effect_metrics = clothing_try_on_result.effect_metrics
    clothing_try_on_status = (
        "no_effect" if effect_metrics.no_effect else "completed"
    )
    verification_message = (
        "CatVTON 최종 합성의 승인 영역 안 변경이 0px입니다. "
        "원시 출력과 차이맵을 확인하세요."
        if effect_metrics.no_effect
        else clothing_try_on_result.candidate.verification.reason_ko
    )
    if run_log is not None:
        metadata = clothing_try_on_result.execution_metadata
        run_log.write_stage(
            "의상 참조 합성",
            (
                "완료, 인물 입력=generated_candidate, "
                f"입력 크기={metadata.person_input_width}x"
                f"{metadata.person_input_height}, "
                "의상 조건="
                f"{metadata.clothing_input_width}x"
                f"{metadata.clothing_input_height}, "
                "의상 알파 점유율="
                f"{metadata.clothing_alpha_coverage_percent:.3f}%, "
                "의상 영역 밖 변경 픽셀="
                f"{clothing_try_on_result.candidate.verification.changed_pixel_count_outside_clothing}, "
                f"승인 마스크 픽셀={metadata.approved_mask_pixel_count:,}px, "
                f"model_mask 출처={metadata.model_mask_source}, "
                f"model_mask 픽셀={metadata.model_mask_pixel_count:,}px, "
                "원시 model_mask 안 변경="
                f"{effect_metrics.raw_changed_inside_model_mask:,}px, "
                "최종 승인 영역 안 변경="
                f"{effect_metrics.final_changed_inside_approved_mask:,}px, "
                "보호 합성 제거="
                f"{effect_metrics.discarded_by_protection_pixels:,}px, "
                "승인 영역 RGB L1 평균="
                f"{effect_metrics.mean_rgb_l1_inside:.4f}, "
                f"상태={clothing_try_on_status}"
            ),
        )
    return replace(
        base_candidate,
        image=clothing_try_on_result.candidate.image,
        before_clothing_image=base_candidate.image,
        clothing_change_mask=clothing_try_on_result.clothing_change_mask,
        raw_clothing_try_on_image=clothing_try_on_result.raw_try_on_image,
        clothing_difference_image=clothing_try_on_result.difference_image,
        clothing_effect_metrics=effect_metrics,
        clothing_reference_name=clothing_reference_input.image_path.name,
        clothing_category=clothing_reference_input.category.value,
        clothing_try_on_status=clothing_try_on_status,
        clothing_verification_warning_ko=verification_message,
    )


def generate_images(
    pipeline,
    config: dict[str, Any],
    prompts: list[Any],
    run_directory: Path,
    result: dict[str, Any],
    project_root: Path,
    run_log: GenerationRunLog | None = None,
) -> None:
    import torch

    generation = config["generation"]
    reference_image = load_reference_image(config, project_root)
    if reference_image is not None and run_log is not None:
        run_log.write_stage(
            "참조 이미지 준비",
            (
                f"IP-Adapter 입력 크기={reference_image.width}x"
                f"{reference_image.height}, 전체 이미지 여백 보존"
            ),
        )
    default_negative = generation.get("default_negative_prompt", "")
    result_path = run_directory / "result.json"

    for index, item in enumerate(prompts, start=1):
        output_path = run_directory / "images" / item.filename
        if output_path.is_file():
            print(f"[{index}/{len(prompts)}] 이미 완료되어 건너뜀: {item.filename}")
            update_request_result(result, item.request_id, status="skipped")
            refresh_summary(result)
            write_json(result_path, result)
            continue

        print(f"[{index}/{len(prompts)}] 생성 중: {item.description_ko or item.request_id}")
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        random_start = torch.Generator(device="cpu").manual_seed(item.seed)
        arguments: dict[str, Any] = {
            "prompt": item.prompt,
            "negative_prompt": item.negative_prompt or default_negative,
            "width": generation["width"],
            "height": generation["height"],
            "num_inference_steps": generation["steps"],
            "guidance_scale": generation["guidance_scale"],
            "generator": random_start,
        }
        if reference_image is not None:
            arguments["ip_adapter_image"] = [reference_image] if not isinstance(reference_image, list) else reference_image

        try:
            image = pipeline(**arguments).images[0]
            if run_log is not None:
                run_log.write_stage(
                    "모델 반환",
                    (
                        f"이미지 크기={image.width}x{image.height}, "
                        "저장 전 자르기 없음"
                    ),
                )
            image.save(output_path)
            elapsed = round(time.perf_counter() - started, 3)
            peak_vram = torch.cuda.max_memory_allocated()
            update_request_result(
                result,
                item.request_id,
                status="completed",
                output=str(output_path.relative_to(project_root)),
                elapsed_seconds=elapsed,
                peak_vram_bytes=peak_vram,
            )
            refresh_summary(result)
            write_json(result_path, result)
            print(f"저장 완료: {output_path.name} ({elapsed}초)")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            is_memory_error = isinstance(error, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(error).lower()
            )
            torch.cuda.empty_cache()
            update_request_result(
                result,
                item.request_id,
                status="failed",
                error="GPU 메모리 부족" if is_memory_error else str(error),
            )
            result["status"] = "failed"
            refresh_summary(result)
            write_json(result_path, result)
            if is_memory_error:
                raise RuntimeError(
                    "GPU 메모리가 부족합니다. 해상도와 설정은 자동으로 바꾸지 않았습니다.\n"
                    "다른 GPU 사용 프로그램을 닫고 같은 결과 폴더로 이어서 실행하세요.\n"
                    f"결과 폴더: {run_directory}"
                ) from error
            raise

    result["status"] = "completed"
    result["finished_at"] = datetime.now().astimezone().isoformat()
    refresh_summary(result)
    write_json(result_path, result)
