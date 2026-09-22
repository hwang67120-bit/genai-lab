"""사용자 승인 시점의 조건을 봉인한다. 해시는 모델 품질이나 보안 서명이 아니다."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from genai_lab.reference_contract import validate_visual_inputs, validate_visual_condition
from genai_lab.reference_experiment_approval import (
    experiment_fingerprint, require_reference_experiment_approval,
)
from genai_lab.reference_order import (
    adapter_references,
    adapter_references_for_stage,
)
from genai_lab.reference_tag_policy import validate_character_gender
from genai_lab.native_pipeline_contract import (
    BASE_PROFILE, native_pipeline_enabled, resolve_refinement_mode,
)
from genai_lab.body_morphology import request_body_morphology
from genai_lab.body_morphology_similarity import (
    resolve_body_morphology_similarity,
)
from genai_lab.garment_topology import resolve_garment_topology
from genai_lab.hair_structure import resolve_hair_structure
from genai_lab.hair_transfer_contract import build_hair_transfer_contract


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def image_digest(image):
    digest = hashlib.sha256(canonical((image.mode, image.size)))
    digest.update(image.tobytes())
    return digest.hexdigest()


def run_payload(inputs, config, request, *, prepared=False):
    section = config["clothing_reference_generation"]
    native_character_base = native_pipeline_enabled(config)
    if (native_character_base
            and section.get("native_base_profile") != BASE_PROFILE):
        raise ValueError("Native Base 실행 프로필이 승인 계약과 다릅니다.")
    pair = ((request.prompt, request.negative_prompt) if prepared
            else section.get("approved_prompt_pair"))
    if not isinstance(pair, (tuple, list)) or len(pair) != 2 or not all(isinstance(v, str) for v in pair):
        raise ValueError("사전 승인된 실제 프롬프트가 필요합니다.")
    identity_scale = float(section.get("identity_reference_scale", .7))
    garment_scale = float(section.get("garment_reference_scale", .45))
    entries = adapter_references(inputs, identity_scale, garment_scale)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("승인 조건 부위 이름이 중복됩니다.")
    invalid_garment_scale = (
        garment_scale != 0 if native_character_base else garment_scale <= 0
    )
    if (any(not np.isfinite(e.scale) or not 0 <= e.scale <= 1 for e in entries)
            or invalid_garment_scale):
        raise ValueError("승인 조건 참조 강도 오류")
    parts = tuple({"name": e.name, "rgb": image_digest(e.image),
                   "mask": image_digest(e.mask), "scale": e.scale} for e in entries)
    if (type(section.get("candidate_count")) is not int or not 2 <= section["candidate_count"] <= 100
            or type(request.seed) is not int or not 0 <= request.seed < 2**32):
        raise ValueError("승인 조건 후보 개수/시드 오류")
    if ((request.width, request.height) != inputs.source.size
            or type(request.inference_steps) is not int or request.inference_steps < 1
            or not np.isfinite(request.guidance_scale) or request.guidance_scale <= 0):
        raise ValueError("승인 조건 출력 크기/추론 설정 오류")
    from genai_lab.part_color_descriptions import description_records
    descriptions = getattr(inputs, "part_color_descriptions", ())
    if section.get("part_color_descriptions", ()) != descriptions:
        raise ValueError("검토한 부위 색상 설명과 생성 조건이 다릅니다.")
    color_records = description_records(descriptions)
    approved_color_records = description_records(tuple(
        section.get("approved_part_color_descriptions", ())))
    if any(record not in color_records for record in approved_color_records):
        raise ValueError("승인한 귀꼬리 색상 조건이 원본 관찰값에 없습니다.")
    scene = getattr(inputs, "scene_condition", None)
    scene_config = config.get("scene_lineart", {})
    generation_mode = config.get("generation", {}).get("mode", "image_to_image")
    edit_strength = float(getattr(
        request, "original_image_change_strength",
        config.get("generation", {}).get("original_image_change_strength", .25),
    ))
    if generation_mode != "image_to_image" or not 0 < edit_strength < 1:
        raise ValueError("참조 생성 Base에는 0과 1 사이 강도의 image_to_image 설정이 필요합니다.")
    from genai_lab.reference_step_schedule import (
        build_reference_scale_schedule,
        effective_denoising_steps,
    )
    denoising_steps = effective_denoising_steps(
        request.inference_steps, generation_mode, edit_strength)
    staged_generation = config.get('staged_reference_generation', {})
    schedule_entries = (
        adapter_references_for_stage(
            inputs, 'base', identity_scale, garment_scale
        )
        if staged_generation.get('enabled', False)
        else entries
    )
    _, step_schedule = build_reference_scale_schedule(
        config, schedule_entries, denoising_steps)
    from genai_lab.latent_refinement import resolve_latent_refinement
    refinement = resolve_latent_refinement(config, request).record()
    from genai_lab.common_candidate_latents import resolve_common_candidate_latents
    common_latents = resolve_common_candidate_latents(config).record()
    from genai_lab.part_error_correction import resolve_part_error_correction
    part_correction = resolve_part_error_correction(config, request).record()
    from genai_lab.hair_error_correction import resolve_hair_error_correction
    hair_correction = resolve_hair_error_correction(config).record()
    from genai_lab.generated_image_integrity import resolve_generated_image_integrity
    image_integrity = resolve_generated_image_integrity(config).record()
    from genai_lab.candidate_pipeline import resolve_candidate_pipeline
    candidate_pipeline = resolve_candidate_pipeline(
        config, section["candidate_count"]).record()
    from genai_lab.candidate_semantic_gate import resolve_candidate_semantic_gate
    candidate_semantic_gate = resolve_candidate_semantic_gate(config).record()
    from genai_lab.candidate_structure_gate import resolve_candidate_structure_gate
    candidate_structure_gate = resolve_candidate_structure_gate(config).record()
    body_morphology_similarity = resolve_body_morphology_similarity(config).record()
    from genai_lab.body_proportion_presets import body_proportion_contract
    body_proportion = body_proportion_contract(config, request)
    from genai_lab.style import prepare_original_image_canvas
    edit_canvas = prepare_original_image_canvas(inputs.source, request.width, request.height)
    try:
        base_edit = {
            "version": "approved_base_edit_v2",
            "mode": generation_mode,
            "source_canvas_sha256": image_digest(edit_canvas),
            "strength": edit_strength,
            "effective_denoising_steps": denoising_steps,
        }
    finally:
        edit_canvas.close()
    from genai_lab.ear_contract import (
        ANIMAL_EARS, HUMAN_EARS, build_source_ear_contract,
    )
    analysis_record = getattr(inputs, "analysis_record", None) or {}
    source_ear_contract = analysis_record.get("ear_contract", {})
    explicit_ear_states = {
        HUMAN_EARS: source_ear_contract.get("human", {}).get("state"),
        ANIMAL_EARS: source_ear_contract.get("animal", {}).get("state"),
    }
    explicit_ear_states = {
        name: state for name, state in explicit_ear_states.items()
        if state is not None
    }
    ear_contract = build_source_ear_contract(
        analysis_record.get("part_detection", {}),
        inputs.source.size,
        approved_character_tags=section.get("approved_character_tags", ()),
        explicit_states=explicit_ear_states,
    )
    garment_topology = resolve_garment_topology(
        section.get("approved_tags", ()),
        section.get("approved_detail_tags", ()),
        section.get("approved_garment_topology"),
    )
    hair_structure = resolve_hair_structure(
        section.get("approved_character_tags", ()),
        section.get("hair_detail_report"),
    )
    hair_transfer_contract = build_hair_transfer_contract(
        section.get("approved_character_tags", ()),
        section.get("hair_detail_report"),
    )
    result = {
        "version": "approved_reference_run_v22",
        "visual_fingerprint": experiment_fingerprint(inputs),
        "parts": parts, "prompt": pair[0], "negative_prompt": pair[1],
        "gender": validate_character_gender(section.get("character_gender", "unspecified")),
        "approved_tags": tuple(section.get("approved_tags", ())),
        "approved_character_tags": tuple(section.get("approved_character_tags", ())),
        "ear_contract": ear_contract,
        "approved_detail_tags": tuple(section.get("approved_detail_tags", ())),
        "garment_detail_analysis": section.get("garment_detail_report"),
        "garment_topology": garment_topology,
        "hair_detail_analysis": section.get("hair_detail_report"),
        "hair_structure": hair_structure,
        "hair_transfer_contract": hair_transfer_contract,
        "part_color_descriptions": color_records,
        "approved_part_color_descriptions": approved_color_records,
        "width": request.width, "height": request.height,
        "inference_steps": request.inference_steps, "guidance_scale": request.guidance_scale,
        "model_id": request.model_id, "reference_adapter_id": request.reference_adapter_id,
        "model_settings": {key: config.get("model", {}).get(key) for key in ("id", "family", "dtype")},
        "adapter_settings": {key: config.get("style", {}).get(key)
                             for key in ("enabled", "adapter_repository", "adapter_subfolder", "adapter_weight")},
        "base_seed": request.seed, "candidate_count": section["candidate_count"],
        "body_morphology": request_body_morphology(request).record(),
        "base_edit": base_edit,
        "body_proportion_control": body_proportion,
        "maximum_candidate_attempts": (
            section["candidate_count"]
            + max(
                candidate_pipeline["gender_retry_attempts"],
                candidate_pipeline["quality_retry_attempts"],
            )),
        "seed_rule": "(base_seed + candidate_number - 1) modulo 2**32",
        "common_candidate_latents": common_latents,
        "reference_step_schedule": step_schedule,
        "latent_refinement": refinement,
        "part_error_correction": part_correction,
        "hair_error_correction": hair_correction,
        "generated_image_integrity": image_integrity,
        "candidate_pipeline": candidate_pipeline,
        "candidate_semantic_gate": candidate_semantic_gate,
        "candidate_structure_gate": candidate_structure_gate,
        "body_morphology_similarity": body_morphology_similarity,
        "refinement_execution": {
            "enabled": native_character_base,
            "mode": resolve_refinement_mode(config),
        },
        "native_pipeline_v2": {
            "enabled": native_character_base,
            "base_profile": config.get("native_pipeline_v2", {}).get(
                "base_profile"
            ),
            "base_garment_prompt_enabled": config.get(
                "native_pipeline_v2", {}
            ).get("base_garment_prompt_enabled"),
            "base_garment_adapter_enabled": config.get(
                "native_pipeline_v2", {}
            ).get("base_garment_adapter_enabled"),
        },
        "scene_hint": image_digest(scene.hint) if scene is not None else None,
        "scene_settings": {key: scene_config.get(key) for key in
                           ("enabled", "model_id", "conditioning_scale", "guidance_end")},
    }
    result['staged_reference_generation'] = staged_generation
    from genai_lab.reference_condition_snapshot import (
        build_reference_condition_snapshot,
    )
    result["condition_snapshot"] = build_reference_condition_snapshot(
        inputs, section)
    return result


@dataclass(frozen=True)
class ApprovedReferenceRun:
    payload: bytes
    fingerprint: str

    def __post_init__(self):
        if not isinstance(self.payload, bytes) or hashlib.sha256(self.payload).hexdigest() != self.fingerprint:
            raise ValueError("승인 조건 해시가 일치하지 않습니다.")

    def record(self):
        return json.loads(self.payload)


def approve_reference_run(inputs, config, request):
    """실제 UI 승인 이벤트에서만 호출한다. 조건 작성만으로 자동 승인하지 않는다."""
    validate_visual_inputs(inputs)
    require_reference_experiment_approval(inputs)
    payload = canonical(run_payload(inputs, config, request))
    inputs.approved_generation = ApprovedReferenceRun(payload, hashlib.sha256(payload).hexdigest())


def require_reference_run(inputs, config, request, *, prepared=False):
    approval = getattr(inputs, "approved_generation", None)
    if not isinstance(approval, ApprovedReferenceRun):
        raise ValueError("생성 조건 전체가 승인되지 않았습니다. 입력을 다시 검토하세요.")
    validate_visual_inputs(inputs)
    require_reference_experiment_approval(inputs)
    actual = run_payload(inputs, config, request, prepared=prepared)
    expected = approval.record()
    if prepared:
        number = request.candidate_number
        if (type(number) is not int
                or not 1 <= number <= expected["maximum_candidate_attempts"]
                or request.seed != (expected["base_seed"] + number - 1) % (2**32)):
            raise ValueError("승인된 후보별 시드 규칙과 다릅니다.")
        actual["base_seed"] = expected["base_seed"]
        # 구형 요청 객체에는 체형 벡터 필드가 없다. 이 경우에만 승인 시점
        # 벡터를 상속하며, 새 요청 객체의 명시적 벡터는 그대로 비교한다.
        if getattr(request, "body_morphology", None) is None:
            actual["body_morphology"] = expected["body_morphology"]
    if canonical(actual) != approval.payload:
        raise ValueError("승인 이후 생성 조건이 바뀌었습니다. 다시 검토하세요.")
    return approval


def encoded_condition_fingerprint(condition):
    """인코딩 직후/후보 실행 직전에만 검사. 확산 단계별 CPU 복사나 추가 추론 없음."""
    import torch
    validate_visual_condition(condition)
    digest = hashlib.sha256()
    digest.update(b"cropped_visual_embeddings_only_v1")
    values = condition["ip_adapter_image_embeds"]
    digest.update(canonical(("embeds", len(values))))
    for tensor in values:
        if not isinstance(tensor, torch.Tensor) or not tensor.numel():
            raise ValueError("인코딩 조건에 유효한 텐서가 필요합니다.")
        if not torch.isfinite(tensor).all().item():
            raise ValueError("인코딩 조건에 유한하지 않은 값이 있습니다.")
        cpu = tensor.detach().contiguous().cpu()
        digest.update(canonical((str(cpu.dtype), tuple(cpu.shape))))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EncodedReferenceReceipt:
    approval_fingerprint: str
    condition_fingerprint: str


def seal_encoded_condition(inputs, condition):
    approval = inputs.approved_generation
    if not isinstance(approval, ApprovedReferenceRun):
        raise ValueError("승인 전에 참조 인코딩을 봉인할 수 없습니다.")
    return EncodedReferenceReceipt(approval.fingerprint, encoded_condition_fingerprint(condition))


def verify_pipeline_boundary(inputs, config, request, arguments):
    """프롬프트 재조립과 인코딩 후에도 승인 조건과 실제 호출 인수가 같은지 검사한다."""
    approval = require_reference_run(inputs, config, request, prepared=True)
    section = config["clothing_reference_generation"]
    receipt = section.get("encoded_reference_receipt")
    if not isinstance(receipt, EncodedReferenceReceipt) or receipt.approval_fingerprint != approval.fingerprint:
        raise ValueError("승인 입력에 연결된 참조 인코딩 기록이 없습니다.")
    cross = arguments.get("cross_attention_kwargs")
    if isinstance(cross, dict) and cross.get("ip_adapter_masks") is not None:
        raise ValueError("기준 이미지 좌표의 IP-Adapter 공간 마스크가 생성 경계에 전달됐습니다.")
    condition = {"ip_adapter_image_embeds": arguments.get("ip_adapter_image_embeds")}
    if encoded_condition_fingerprint(condition) != receipt.condition_fingerprint:
        raise ValueError("인코딩 이후 참조 텐서가 바뀌었습니다. 다시 준비하세요.")
    expected = {"prompt": request.prompt, "negative_prompt": request.negative_prompt,
                "width": request.width, "height": request.height,
                "num_inference_steps": request.inference_steps, "guidance_scale": request.guidance_scale}
    if any(arguments.get(key) != value for key, value in expected.items()):
        raise ValueError("승인 요청과 실제 파이프라인 인수가 다릅니다.")
    if arguments.get("generator") is None or arguments["generator"].initial_seed() != request.seed:
        raise ValueError("실제 파이프라인 시드가 승인 규칙과 다릅니다.")
    from genai_lab.common_candidate_latents import validate_candidate_latent
    validate_candidate_latent(
        arguments, request, approval.record()["common_candidate_latents"],
        section.get("candidate_latent_record"))
    expected_schedule = approval.record()["reference_step_schedule"]
    callback = arguments.get("callback_on_step_end")
    actual_schedule = getattr(callback, "reference_step_schedule_record", None)
    if expected_schedule.get("enabled"):
        if actual_schedule != expected_schedule:
            raise ValueError("승인된 단계별 참조 강도와 실제 생성 콜백이 다릅니다.")
    elif actual_schedule is not None:
        raise ValueError("승인되지 않은 단계별 참조 강도 콜백이 연결됐습니다.")
    expected_integrity = approval.record()["generated_image_integrity"]
    actual_integrity = getattr(callback, "generated_image_integrity_record", None)
    if expected_integrity.get("enabled"):
        if actual_integrity != expected_integrity:
            raise ValueError("승인된 생성 이미지 무결성 콜백이 실제 호출과 다릅니다.")
    elif actual_integrity is not None:
        raise ValueError("승인되지 않은 생성 이미지 무결성 콜백이 연결됐습니다.")
    refinement = approval.record()["latent_refinement"]
    if refinement["enabled"]:
        if arguments.get("output_type") != "latent":
            raise ValueError("승인된 1차 생성은 latent 출력이어야 합니다.")
    elif arguments.get("output_type", "pil") != "pil":
        raise ValueError("승인되지 않은 latent 출력이 연결됐습니다.")
    approval_record = approval.record()
    base_edit = approval_record.get("base_edit", {})
    from PIL import Image
    actual_initial = arguments.get("image")
    if (
        base_edit.get("mode") != "image_to_image"
        or not isinstance(actual_initial, Image.Image)
        or image_digest(actual_initial) != base_edit.get("source_canvas_sha256")
        or arguments.get("strength") != base_edit.get("strength")
    ):
        raise ValueError("승인된 캐릭터 RGB 시작 이미지 또는 편집 강도와 실제 호출이 다릅니다.")
    body_control = approval_record.get("body_proportion_control", {})
    body_control_active = body_control.get("status") == "active"
    if body_control_active:
        if inputs.scene_condition is not None:
            raise ValueError("체형 프리셋과 장면 선화 조건을 동시에 전달할 수 없습니다.")
        from genai_lab.body_proportion_presets import PROJECT_ROOT, prepare_body_proportion_control
        expected_control, runtime_record = prepare_body_proportion_control(
            config, request, PROJECT_ROOT
        )
        try:
            actual_control = arguments.get("control_image")
            settings = body_control.get("settings", {})
            if (
                expected_control is None
                or not isinstance(actual_control, Image.Image)
                or image_digest(actual_control) != image_digest(expected_control)
                or arguments.get("controlnet_conditioning_scale")
                != float(settings.get("conditioning_scale", .42))
                or arguments.get("control_guidance_start")
                != float(settings.get("guidance_start", 0.0))
                or arguments.get("control_guidance_end")
                != float(settings.get("guidance_end", .72))
                or runtime_record.get("control_sha256")
                != body_control.get("control_sha256")
            ):
                raise ValueError("승인된 체형 프리셋 ControlNet 조건과 실제 호출이 다릅니다.")
        finally:
            if expected_control is not None:
                expected_control.close()
    elif "control_image" in arguments and inputs.scene_condition is None:
        raise ValueError("승인되지 않은 ControlNet 조건이 참조 생성에 연결됐습니다.")
    if inputs.scene_condition is not None:
        hint = arguments.get("control_image")
        scene_config = config.get("scene_lineart", {})
        if (not isinstance(hint, Image.Image) or image_digest(hint) != image_digest(inputs.scene_condition.hint)
                or arguments.get("controlnet_conditioning_scale") != float(scene_config.get("conditioning_scale", .5))
                or arguments.get("control_guidance_end") != float(scene_config.get("guidance_end", .8))):
            raise ValueError("승인된 장면 선화 조건과 실제 호출이 다릅니다.")
    return approval.fingerprint


def verify_refinement_boundary(inputs, config, request, arguments, starting_latents):
    """2차 Img2Img가 승인된 1차 latent와 동일한 조건만 사용하는지 검사한다."""
    approval = require_reference_run(inputs, config, request, prepared=True)
    refinement = approval.record()["latent_refinement"]
    if not refinement["enabled"]:
        raise ValueError("승인되지 않은 latent 정밀화 호출입니다.")
    if inputs.scene_condition is not None:
        raise ValueError("현재 latent 정밀화는 장면 선화 조건과 동시에 사용할 수 없습니다.")

    section = config["clothing_reference_generation"]
    receipt = section.get("encoded_reference_receipt")
    if not isinstance(receipt, EncodedReferenceReceipt) or receipt.approval_fingerprint != approval.fingerprint:
        raise ValueError("승인 입력에 연결된 참조 인코딩 기록이 없습니다.")
    cross = arguments.get("cross_attention_kwargs")
    if isinstance(cross, dict) and cross.get("ip_adapter_masks") is not None:
        raise ValueError("기준 이미지 좌표의 IP-Adapter 공간 마스크가 정밀화 경계에 전달됐습니다.")
    condition = {"ip_adapter_image_embeds": arguments.get("ip_adapter_image_embeds")}
    if encoded_condition_fingerprint(condition) != receipt.condition_fingerprint:
        raise ValueError("2차 정밀화의 참조 텐서가 승인 조건과 다릅니다.")

    expected = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "num_inference_steps": refinement["inference_steps"],
        "guidance_scale": refinement["guidance_scale"],
        "strength": refinement["strength"],
    }
    if any(arguments.get(key) != value for key, value in expected.items()):
        raise ValueError("승인된 latent 정밀화 설정과 실제 호출 인수가 다릅니다.")
    if arguments.get("image") is not starting_latents:
        raise ValueError("2차 정밀화가 검증된 1차 latent를 사용하지 않습니다.")
    if "width" in arguments or "height" in arguments or arguments.get("output_type", "pil") != "pil":
        raise ValueError("2차 정밀화의 출력 형식 또는 크기 인수가 승인 범위를 벗어났습니다.")
    generator = arguments.get("generator")
    if generator is None or generator.initial_seed() != request.seed:
        raise ValueError("2차 정밀화 시드가 승인 규칙과 다릅니다.")
    callback = arguments.get("callback_on_step_end")
    if getattr(callback, "latent_refinement_record", None) != refinement:
        raise ValueError("승인된 latent 정밀화 콜백이 연결되지 않았습니다.")
    return approval.fingerprint

