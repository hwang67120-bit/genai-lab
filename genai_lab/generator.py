"""생성 요청을 한 장씩 처리한다."""

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from genai_lab.clothing import (
    CatVTONLocalSettings,
    CharacterAgnosticApprovedInput,
    ClothingReferenceInput,
)
from genai_lab.detail import (
    CharacterDetailCorrectionError,
    correct_character_candidate_details,
)
from genai_lab.request import CharacterGenerationRequest
from genai_lab.clothing_reference_generation import prepare_design_reference_request
from genai_lab.hair_detail_analysis import hair_prompt_delivery_report
from genai_lab.reference_contract import validate_reference_mode, validate_visual_condition
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
    if any(value is not None for value in (clothing_reference_input, catvton_settings, approved_agnostic_input)):
        raise ValueError('기존 CatVTON 의상 합성 기능은 제거되었습니다. 의상 디자인 참조 생성을 사용하세요.')
    validate_reference_mode(config, approved_pose_estimation is not None)
    reference_section = config.get('clothing_reference_generation', {})
    if reference_section.get('enabled'):
        if reference_section.get('visual_inputs') is None:
            raise ValueError('태그 전용 의상 생성 경로는 제거되었습니다. 승인된 얼굴·의상 참조가 필요합니다.')
        if getattr(reference_section['visual_inputs'], 'initial', None) is not None:
            raise ValueError('폐기된 초기 이미지가 생성 입력에 남아 있습니다.')
        validate_visual_condition(reference_section.get('visual_condition'))
    import torch

    reference_config = config.get('clothing_reference_generation', {})
    reference_mode = bool(reference_config.get('enabled', False))
    design_record = None
    approved_run = None
    if reference_mode:
        if any(value is not None for value in (
            clothing_reference_input, catvton_settings, approved_agnostic_input, approved_pose_estimation,
        )):
            raise ValueError('디자인 참조 생성에는 의상 합성·신체 복원·외부 자세 입력을 연결하지 않습니다.')
        from genai_lab.native_pipeline_contract import (
            native_pipeline_enabled,
            prepare_character_only_base_request,
        )
        native_character_base = native_pipeline_enabled(config)
        if native_character_base:
            generation_request, design_record = prepare_character_only_base_request(
                generation_request,
                (pipeline.tokenizer, pipeline.tokenizer_2),
                character_tags=reference_config.get(
                    'approved_character_tags', ()),
                character_gender=reference_config.get(
                    'character_gender', 'unspecified'),
            )
        else:
            generation_request, design_record = prepare_design_reference_request(
                generation_request, reference_config.get('approved_tags', ()),
                (pipeline.tokenizer, pipeline.tokenizer_2),
                character_tags=reference_config.get(
                    'approved_character_tags', ()),
                character_gender=reference_config.get(
                    'character_gender', 'unspecified'),
                approved_detail_tags=reference_config.get(
                    'approved_detail_tags', ()),
                part_color_descriptions=reference_config.get(
                    'part_color_descriptions', ()),
                approved_part_color_descriptions=reference_config.get(
                    'approved_part_color_descriptions', ()),
                garment_prompt_policy=reference_config.get(
                    'garment_prompt_policy', {}),
                long_prompt_settings=config.get('long_prompt_embedding', {}),
            )
        design_record['garment_detail_analysis'] = reference_config.get('garment_detail_report')
        design_record['hair_detail_analysis'] = reference_config.get('hair_detail_report')
        design_record['hair_prompt_delivery'] = hair_prompt_delivery_report(
            design_record['approved_character_tags'], generation_request.prompt,
            design_record['hair_detail_analysis'])
        hair_visual = getattr(
            reference_config.get('visual_inputs'), 'hair_reference', None)
        design_record['hair_prompt_delivery'].update(
            visual_reference_scope=(
                'text_only_in_base'
                if native_character_base
                else (
                    'hair_only' if hair_visual is not None
                    else 'identity(face_hair)'
                )
            ),
            separate_hair_adapter=(
                False if native_character_base
                else hair_visual is not None
            ),
        )
        design_record['eye_color_analysis'] = reference_config.get('eye_color_report')
        design_record['character_feature_routing'] = reference_config.get('character_feature_routing')
        if reference_config.get('require_prompt_approval'):
            expected = reference_config.get('approved_prompt_pair')
            actual = (generation_request.prompt, generation_request.negative_prompt)
            if not isinstance(expected, (list, tuple)) or tuple(expected) != actual:
                raise ValueError('승인한 생성 조건과 실제 프롬프트가 다릅니다. 다시 검토하세요.')
            design_record['prompt_preflight_approved'] = True
        from genai_lab.approved_reference_run import require_reference_run
        approved_run = require_reference_run(
            reference_config['visual_inputs'], config, generation_request,
            prepared=True)
        if run_log is not None:
            hair_delivery = design_record['hair_prompt_delivery']
            run_log.write_stage(
                '헤어 생성 조건 전달',
                f"승인={list(hair_delivery['approved_hair_tags'])}, "
                f"실제 프롬프트 전달={list(hair_delivery['delivered_hair_tags'])}, "
                f"승인 후 누락={list(hair_delivery['missing_approved_hair_tags'])}, "
                f"분석 후 미전달={list(hair_delivery['analyzed_but_not_delivered'])}, "
                f"상세 자동후보={list(hair_delivery['optional_hair_candidates'])}, "
                f"시각 참조={hair_delivery['visual_reference_scope']}, "
                f"별도 헤어 어댑터={hair_delivery['separate_hair_adapter']}")
            run_log.write_stage('캐릭터 성별 지정',
                f"지정={design_record['character_gender']}, "
                f"제외 태그={design_record['removed_character_gender_tags']}, "
                f"충돌 네거티브 제거={design_record['removed_gender_negative_terms']}")
            run_log.write_stage('의상 프롬프트 정리',
                f"제거={design_record['removed_conflicting_negative_terms']}, "
                f"실제 부정 프롬프트={generation_request.negative_prompt}, "
                f"실제 부정 토큰={design_record['negative']['effective_token_counts']}")
            run_log.write_stage('의상 성별 조건 차단',
                f"정책={design_record['garment_gender_policy']}, "
                f"의상 제외 태그={design_record['excluded_non_design_tags']}, "
                f"성별 네거티브={design_record['gender_negative_guard']}, "
                '이미지 내부 체형 분리=미구현, 결과 승인 필요')
            run_log.write_stage('의상 디자인 참조 생성',
                f"방식={'영역별 이미지+태그' if reference_config.get('visual_inputs') is not None else '승인 태그'}, 신체 복원/외부 자세/부분 보정=미사용, "
                f"크기={generation_request.width}x{generation_request.height}, "
                f"prompt={generation_request.prompt}, 잘림={design_record['positive']['truncated']}")

    visual_inputs = reference_config.get('visual_inputs') if reference_mode else None
    if visual_inputs is not None:
        # Base는 승인된 캐릭터 원본에서 시작한다. 분석 마스크나 의상 RGB를
        # 시작 캔버스에 합성하지 않아 원본 좌표 오염을 만들지 않는다.
        original_image_canvas = prepare_original_image_canvas(
            visual_inputs.source,
            generation_request.width,
            generation_request.height,
        )
        ip_adapter_reference_image = visual_inputs.identity.copy()
        garment_scale = float(reference_config.get('garment_reference_scale', 0.45))
        identity_scale = float(reference_config.get('identity_reference_scale', 0.7))
        design_record.update(
                             mode=(
                                 'animagine_character_only_base'
                                 if native_character_base
                                 else 'visual_reference_regeneration'
                             ),
                             garment_image_adapter=(
                                 False
                                 if native_character_base
                                 else garment_scale > 0
                             ),
                             garment_tags_enabled=not native_character_base,
                             clothing_neutralized=original_image_canvas is not None, body_restoration=False,
                             initial_image_used=original_image_canvas is not None,
                             identity_reference_scope=(
                                 'full_character'
                                 if native_character_base
                                 else (
                                     'face_only'
                                     if visual_inputs.hair_reference is not None
                                     else 'face_hair'
                                 )
                             ),
                             hair_image_adapter=(
                                 False
                                 if native_character_base
                                 else visual_inputs.hair_reference is not None
                             ),
                             hair_reference_scale=(
                                 None
                                 if native_character_base
                                 else (
                                     visual_inputs.hair_reference_scale
                                     if visual_inputs.hair_reference is not None
                                     else None
                                 )
                             ),
                             identity_scale=identity_scale, garment_scale=garment_scale)
        if run_log is not None:
            scene_condition = getattr(visual_inputs, 'scene_condition', None)
            from genai_lab.reference_order import adapter_reference_names
            reference_count = (len(scene_condition.references)
                               if scene_condition else
                               len(adapter_reference_names(visual_inputs)))
            run_log.write_stage(
                '시각 참조 입력',
                f'잘라낸 부위별 참조 {reference_count}개, '
                f'의상 참조 강도={garment_scale}, '
                f'캐릭터 전용 Base={native_character_base}, '
                '분석 마스크는 크롭·JSON에만 사용, 생성 공간 마스크=미사용, '
                '신체 복원=0회')
    else:
        ip_adapter_reference_image = prepare_ip_adapter_reference_image(generation_request.reference_image)
        original_image_canvas = prepare_original_image_canvas(
            generation_request.reference_image, generation_request.width, generation_request.height)
        if run_log is not None:
            run_log.write_stage('참조 이미지 준비',
                f'IP-Adapter 입력 크기={ip_adapter_reference_image.width}x{ip_adapter_reference_image.height}, 사용자 승인본 사용')
    if run_log is not None and original_image_canvas is not None:
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
    from genai_lab.reference_step_schedule import effective_denoising_steps
    base_denoising_steps = effective_denoising_steps(
        generation_request.inference_steps,
        generation_mode,
        executed_image_change_strength,
    )
    if visual_inputs is not None:
        if generation_mode != 'image_to_image':
            raise ValueError('참조 생성 Base는 승인된 캐릭터 RGB를 사용하는 image_to_image여야 합니다.')
        if design_record is not None:
            design_record.update(
                generation_mode=generation_mode,
                effective_strength=executed_image_change_strength,
                effective_denoising_steps=base_denoising_steps,
            )
        if run_log is not None:
            scene_active = getattr(visual_inputs, 'scene_condition', None) is not None
            run_log.write_stage('편집 기반 Base 입력',
                f'pipeline={type(pipeline).__name__}, 승인 캐릭터 RGB 사용, '
                f'strength={executed_image_change_strength:.2f}, '
                f'실제 디노이징={base_denoising_steps}회, '
                f'구조 선화={scene_active}, 잘라낸 부위별 참조 유지, '
                '기준 이미지 좌표 공간 마스크 미전달')
    if generation_mode == "image_to_image":
        model_arguments["image"] = original_image_canvas
        model_arguments["strength"] = executed_image_change_strength
    else:
        model_arguments["width"] = generation_request.width
        model_arguments["height"] = generation_request.height
    model_arguments["ip_adapter_image"] = [ip_adapter_reference_image]
    if visual_inputs is not None:
        model_arguments.pop('ip_adapter_image')
        model_arguments.update(reference_config['visual_condition'])
        model_arguments['width'] = generation_request.width
        model_arguments['height'] = generation_request.height
        candidate_latents = reference_config.get('candidate_latents')
        if generation_mode == "image_to_image" and candidate_latents is not None:
            raise ValueError(
                "Img2Img Base에 사전 생성 latent를 전달하면 승인 RGB 시작 이미지를 우회합니다."
            )
        if candidate_latents is not None:
            # 파이프라인이 작업 텐서를 변경해도 재시도 원본은 그대로 보존한다.
            model_arguments['latents'] = candidate_latents.clone()
            design_record['common_candidate_latent'] = reference_config.get(
                'candidate_latent_record')
        from genai_lab.scene_generation import scene_arguments
        model_arguments.update(scene_arguments(
            pipeline, config, visual_inputs,
            (generation_request.width, generation_request.height), run_log))
        if getattr(visual_inputs, 'scene_condition', None) is not None:
            design_record.update(scene_lineart=True, initial_rgb_image=True)
            if run_log is not None:
                run_log.write_stage(
                    '장면 선화 조건',
                    'control_image=구조 선화, image=승인 캐릭터 RGB, 인페인팅/원본 픽셀 덧씌우기 없음',
                )
    if prepared_pose_control is not None:
        model_arguments["control_image"] = (
            prepared_pose_control.control_map_image
        )
        model_arguments["controlnet_conditioning_scale"] = (
            pose_control_conditioning_scale
        )
        model_arguments["control_guidance_start"] = pose_control_guidance_start
        model_arguments["control_guidance_end"] = pose_control_guidance_end

    refinement_settings = None
    integrity_settings = None
    defer_part_correction = False
    final_latent_audit = {}
    if visual_inputs is not None:
        from genai_lab.latent_refinement import resolve_latent_refinement
        refinement_settings = resolve_latent_refinement(config, generation_request)
        from genai_lab.generated_image_integrity import resolve_generated_image_integrity
        integrity_settings = resolve_generated_image_integrity(config)
        from genai_lab.candidate_pipeline import resolve_candidate_pipeline
        candidate_pipeline_settings = resolve_candidate_pipeline(
            config,
            int(reference_config.get("candidate_count", 2)),
        )
        defer_part_correction = (
            candidate_pipeline_settings.enabled
            and candidate_pipeline_settings.repair_best_candidate_only
        )
        if refinement_settings.enabled:
            if getattr(visual_inputs, "scene_condition", None) is not None:
                raise ValueError(
                    "현재 latent 정밀화는 장면 선화 조건과 동시에 사용할 수 없습니다."
                )
            model_arguments["output_type"] = "latent"

    generation_started_at = time.perf_counter()
    try:
        reference_schedule = None
        if visual_inputs is not None:
            from genai_lab.reference_order import (
                adapter_references,
                adapter_references_for_stage,
            )
            from genai_lab.reference_step_schedule import build_reference_scale_schedule
            staged_generation = bool(
                config.get('staged_reference_generation', {}).get('enabled', False)
            )
            reference_entries = (
                adapter_references_for_stage(
                    visual_inputs, 'base', identity_scale, garment_scale
                )
                if staged_generation
                else adapter_references(
                    visual_inputs, identity_scale, garment_scale
                )
            )
            reference_schedule, schedule_record = build_reference_scale_schedule(
                config, reference_entries, base_denoising_steps)
            if reference_schedule is not None:
                pipeline.set_ip_adapter_scale(reference_schedule.initial_scale)
                model_arguments['callback_on_step_end'] = reference_schedule
                design_record['reference_step_schedule'] = schedule_record
                if run_log is not None:
                    run_log.write_stage(
                        '단계별 참조 강도',
                        f"순서={schedule_record['reference_order']}, "
                        f"단계={schedule_record['phases']}")
        check_running = reference_config.get('check_running') if reference_mode else None
        if check_running is not None:
            check_running()
            previous_callback = model_arguments.get('callback_on_step_end')
            def on_step_end(pipe, step, timestep, callback_kwargs):
                check_running()
                if previous_callback is not None:
                    return previous_callback(pipe, step, timestep, callback_kwargs)
                return callback_kwargs
            on_step_end.reference_step_schedule_record = getattr(
                previous_callback, 'reference_step_schedule_record', None)
            model_arguments['callback_on_step_end'] = on_step_end
        if integrity_settings is not None and integrity_settings.enabled:
            from genai_lab.generated_image_integrity import wrap_final_latent_audit
            model_arguments['callback_on_step_end'] = wrap_final_latent_audit(
                model_arguments.get('callback_on_step_end'),
                base_denoising_steps,
                final_latent_audit,
                integrity_settings.record(),
            )
        if visual_inputs is not None:
            from genai_lab.approved_reference_run import verify_pipeline_boundary
            verification_started = time.perf_counter()
            fingerprint = verify_pipeline_boundary(visual_inputs, config, generation_request, model_arguments)
            if run_log is not None:
                run_log.write_stage('최종 승인 조건 검증',
                    f"후보={generation_request.candidate_number}, 완료, "
                    f"소요={time.perf_counter() - verification_started:.3f}초")
            # 단계 스케줄이 없을 때도 이전 후보의 최종 강도가 남지 않게 승인값을 복원한다.
            if reference_schedule is None:
                from genai_lab.reference_order import (
                    adapter_references,
                    adapter_references_for_stage,
                    unmasked_adapter_scale,
                )
                restore_entries = (
                    adapter_references_for_stage(
                        visual_inputs, 'base', identity_scale, garment_scale
                    )
                    if config.get('staged_reference_generation', {}).get(
                        'enabled', False
                    )
                    else adapter_references(
                        visual_inputs, identity_scale, garment_scale
                    )
                )
                pipeline.set_ip_adapter_scale(unmasked_adapter_scale(
                    entry.scale for entry in restore_entries))
            design_record['approved_generation_fingerprint'] = fingerprint
            long_plan = design_record['positive'].get('long_prompt', {})
            common_prompt_condition = reference_config.get(
                'common_prompt_condition')
            common_prompt_record = reference_config.get('common_prompt_record')
            if (common_prompt_condition is not None
                    and long_plan.get('chunk_count', 1) <= 1):
                from genai_lab.common_prompt_embeddings import (
                    validate_common_prompt_embeddings,
                )
                validate_common_prompt_embeddings(
                    common_prompt_condition,
                    common_prompt_record,
                    generation_request.prompt,
                    generation_request.negative_prompt,
                )
                model_arguments.pop('prompt')
                model_arguments.pop('negative_prompt')
                model_arguments.update(common_prompt_condition)
                design_record['common_prompt_embeddings'] = common_prompt_record
                if run_log is not None:
                    run_log.write_stage(
                        '공통 프롬프트 임베딩 전달',
                        f"후보={generation_request.candidate_number}, "
                        "승인 텍스트 해시 검증=통과, 재인코딩=없음")
            elif long_plan.get('chunk_count', 1) > 1:
                if refinement_settings is not None and refinement_settings.enabled:
                    raise ValueError(
                        "2청크 프롬프트와 latent 정밀화의 동시 사용은 아직 승인되지 않았습니다.")
                from genai_lab.sdxl_long_prompt import build_long_prompt_embeddings
                embedding_arguments, embedding_record = build_long_prompt_embeddings(
                    pipeline, long_plan, generation_request.negative_prompt)
                model_arguments.pop('prompt')
                model_arguments.pop('negative_prompt')
                model_arguments.update(embedding_arguments)
                design_record['long_prompt_embedding'] = embedding_record
                if run_log is not None:
                    run_log.write_stage(
                        '긴 프롬프트 임베딩',
                        f"정책={embedding_record['policy_version']}, "
                        f"청크={long_plan['chunk_count']}, "
                        f"shape={embedding_record['positive_shape']}, "
                        "추론 단계 변경 없음")
        base_started_at = time.perf_counter()
        from genai_lab.provenance import observe_pipeline
        observe_pipeline(config, pipeline, "base_call", model_arguments)
        first_stage_images = pipeline(**model_arguments).images
        # Diffusers는 PIL 출력일 때는 이미지 목록을, output_type="latent"일 때는
        # [B, 4, H, W] 텐서 자체를 images에 담는다. latent의 [0]을 먼저 꺼내면
        # 배치 축이 사라져 [4, H, W]가 되므로 정밀화 경로에서는 그대로 유지한다.
        first_stage_output = (
            first_stage_images
            if refinement_settings is not None and refinement_settings.enabled
            else first_stage_images[0]
        )
        base_elapsed_seconds = time.perf_counter() - base_started_at

        if refinement_settings is not None and refinement_settings.enabled:
            from genai_lab.latent_refinement import (
                prepare_refinement_latents,
                refinement_pipeline_from,
            )
            refinement_latents = prepare_refinement_latents(
                first_stage_output,
                generation_request,
                refinement_settings,
                vae_scale_factor=getattr(pipeline, "vae_scale_factor", 8),
            )
            refinement_record = refinement_settings.record()
            if run_log is not None:
                run_log.write_stage(
                    "1차 latent 검증",
                    f"후보={generation_request.candidate_number}, "
                    f"shape={tuple(first_stage_output.shape)}, "
                    f"확대배율={refinement_settings.latent_scale_factor:.2f}, "
                    f"소요={base_elapsed_seconds:.3f}초, 의미 품질 판정=미수행",
                )

            if check_running is not None:
                check_running()
            if hasattr(pipeline, "maybe_free_model_hooks"):
                pipeline.maybe_free_model_hooks()
            torch.cuda.empty_cache()
            refinement_pipeline = refinement_pipeline_from(pipeline)

            from genai_lab.reference_order import (
                adapter_references,
                unmasked_adapter_scale,
            )
            refinement_entries = adapter_references(
                visual_inputs, identity_scale, garment_scale)
            refinement_pipeline.set_ip_adapter_scale(
                unmasked_adapter_scale(
                    entry.scale for entry in refinement_entries)
            )

            def on_refinement_step_end(pipe, step, timestep, callback_kwargs):
                del pipe, step, timestep
                if check_running is not None:
                    check_running()
                return callback_kwargs

            on_refinement_step_end.latent_refinement_record = refinement_record
            refinement_arguments = {
                "prompt": generation_request.prompt,
                "negative_prompt": generation_request.negative_prompt,
                "image": refinement_latents,
                "strength": refinement_settings.strength,
                "num_inference_steps": refinement_settings.inference_steps,
                "guidance_scale": refinement_settings.guidance_scale,
                "generator": torch.Generator(device="cpu").manual_seed(
                    generation_request.seed
                ),
                "callback_on_step_end": on_refinement_step_end,
            }
            refinement_arguments.update(reference_config["visual_condition"])

            from genai_lab.approved_reference_run import verify_refinement_boundary
            verify_refinement_boundary(
                visual_inputs,
                config,
                generation_request,
                refinement_arguments,
                refinement_latents,
            )
            refinement_started_at = time.perf_counter()
            generated_image = refinement_pipeline(**refinement_arguments).images[0]
            refinement_elapsed_seconds = time.perf_counter() - refinement_started_at
            design_record["latent_refinement"] = dict(
                refinement_record,
                status="completed",
                base_elapsed_seconds=round(base_elapsed_seconds, 3),
                refinement_elapsed_seconds=round(refinement_elapsed_seconds, 3),
                input_latent_shape=list(first_stage_output.shape),
                refined_latent_shape=list(refinement_latents.shape),
                output_size=list(generated_image.size),
            )
            if run_log is not None:
                run_log.write_stage(
                    "2차 Img2Img 정밀화",
                    f"후보={generation_request.candidate_number}, "
                    f"strength={refinement_settings.strength:.2f}, "
                    f"steps={refinement_settings.inference_steps}, "
                    f"guidance={refinement_settings.guidance_scale:.2f}, "
                    f"소요={refinement_elapsed_seconds:.3f}초, "
                    f"출력={generated_image.width}x{generated_image.height}",
                )
            del refinement_latents
            del first_stage_output
            torch.cuda.empty_cache()
        else:
            generated_image = first_stage_output
            if design_record is not None:
                design_record["latent_refinement"] = {
                    "enabled": False,
                    "status": "disabled",
                    "base_elapsed_seconds": round(base_elapsed_seconds, 3),
                }
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
        # 콜백 closure와 후보 latent 복사본이 다음 후보까지 GPU tensor를
        # 붙잡지 않도록 파이프라인 호출 직후 참조를 끊는다.
        model_arguments.pop("callback_on_step_end", None)
        model_arguments.pop("latents", None)
        ip_adapter_reference_image.close()
        if original_image_canvas is not None:
            original_image_canvas.close()
    if prepared_pose_control is not None:
        prepared_pose_control.close()

    if integrity_settings is not None and integrity_settings.enabled:
        from genai_lab.generated_image_integrity import (
            GeneratedOutputIntegrityError,
            analyze_generated_image_integrity,
        )
        if not final_latent_audit:
            generated_image.close()
            raise GeneratedOutputIntegrityError(
                "최종 denoising latent 감사 기록이 생성되지 않았습니다.",
                {
                    "status": "invalid",
                    "failure_stage": "final_latent",
                    "reason": "final_latent_audit_missing",
                    "retryable": integrity_settings.retry_final_latent_failure,
                },
            )
        image_integrity = analyze_generated_image_integrity(
            generated_image, integrity_settings)
        design_record["final_latent_integrity"] = dict(final_latent_audit)
        design_record["generated_image_integrity"] = image_integrity
        from genai_lab.provenance import observe_gate
        observe_gate(config, "integrity", image_integrity,
                     f"base:{generation_request.candidate_number}")
        if run_log is not None:
            run_log.write_stage(
                "최종 latent 무결성",
                f"후보={generation_request.candidate_number}, 결과={final_latent_audit}",
            )
            run_log.write_stage(
                "생성 이미지 무결성",
                f"후보={generation_request.candidate_number}, 결과={image_integrity}",
            )
        if image_integrity["status"] == "corrupted":
            debug_directory = reference_config.get("integrity_debug_directory")
            attempt = int(reference_config.get("integrity_attempt", 1))
            if (integrity_settings.preserve_corrupted_image
                    and debug_directory is not None):
                debug_path = (
                    Path(debug_directory)
                    / (f"candidate_{generation_request.candidate_number}"
                       f"_attempt_{attempt}_corrupted.png")
                )
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                generated_image.save(debug_path)
                image_integrity["debug_image_path"] = str(debug_path)
                if run_log is not None:
                    run_log.write_stage(
                        "손상 후보 보존",
                        f"후보={generation_request.candidate_number}, "
                        f"시도={attempt}, 경로={debug_path}",
                    )
            generated_image.close()
            raise GeneratedOutputIntegrityError(
                "생성 이미지에서 기계적 픽셀 손상이 감지됐습니다.", image_integrity)

    if visual_inputs is not None and not defer_part_correction:
        from genai_lab.part_error_correction import (
            PartCorrectionContractError,
            correct_detected_parts,
        )
        if run_log is not None:
            run_log.write_stage(
                "후보 부위 오류 검사",
                "완성 후보에서 귀·꼬리 위치 재검출 및 원본 부위 특징 비교 시작",
            )
        try:
            correction = correct_detected_parts(
                pipeline,
                generated_image,
                visual_inputs,
                config,
                generation_request,
                approved_run_record=approved_run.record(),
                check_running=check_running,
            )
            design_record["part_error_correction"] = correction.report
            if correction.image is not generated_image:
                generated_image.close()
                generated_image = correction.image
            if run_log is not None:
                run_log.write_stage(
                    "후보 부위 오류 보정",
                    f"상태={correction.report['status']}, "
                    f"보정 채택={correction.report.get('corrected_count', 0)}, "
                    f"부위={correction.report.get('parts', {})}",
                )
        except InterruptedError:
            generated_image.close()
            raise
        except PartCorrectionContractError:
            generated_image.close()
            raise
        except TimeoutError as error:
            design_record["part_error_correction"] = {
                "status": "timeout_keep_original",
                "error": str(error),
            }
            if run_log is not None:
                run_log.write_stage(
                    "후보 부위 오류 보정 시간 초과",
                    f"{error}; 완성된 1차 후보 유지",
                )
        except Exception as error:
            design_record["part_error_correction"] = {
                "status": "failed_keep_original",
                "error": f"{type(error).__name__}: {error}",
            }
            if run_log is not None:
                run_log.write_stage(
                    "후보 부위 오류 보정 실패",
                    f"{type(error).__name__}: {error}; 1차 후보 유지",
                )
        from genai_lab.generated_condition_audit import (
            build_generated_condition_audit,
        )
        audit = build_generated_condition_audit(
            visual_inputs.approved_generation.record(), design_record)
        design_record["generated_condition_audit"] = audit
        if run_log is not None:
            run_log.write_stage(
                "생성 결과 조건 교차 검증",
                f"상태={audit['status']}, 검사={audit['checks']}")

    if visual_inputs is not None and defer_part_correction:
        design_record["part_error_correction"] = {
            "status": "deferred_for_batch_selection",
            "parts": {},
            "reason": "repair_best_candidate_only",
        }
        from genai_lab.generated_condition_audit import (
            build_generated_condition_audit,
        )
        audit = build_generated_condition_audit(
            visual_inputs.approved_generation.record(), design_record)
        design_record["generated_condition_audit"] = audit
        if run_log is not None:
            run_log.write_stage(
                "후보 부위 오류 보정",
                "상태=deferred_for_batch_selection, 정상 후보 비교 후 최상위 1개만 보정",
            )

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



    candidate_image = generated_image
    original_generated_image = None
    detail_correction_status = "disabled"
    detected_face_count = 0
    detected_hand_count = 0
    corrected_region_count = 0
    rejected_region_count = 0
    detail_verification_warning_ko = None
    detail_config = {} if reference_mode else config.get("detail_correction", {})
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
            else reference_config.get('source_name') if reference_mode else None
        ),
        design_reference_record=design_record,
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
    """오래된 호출자가 무거운 합성 경로로 진입하지 않도록 명시적으로 거부한다."""
    raise ValueError("기존 CatVTON 의상 합성 기능은 제거되었습니다. 의상 디자인 참조 생성을 사용하세요.")


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
