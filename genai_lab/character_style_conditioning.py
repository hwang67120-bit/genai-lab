"""SDXL에 전달할 캐릭터/구조/그림체 조건을 CPU에서 조립한다.

build_conditioning_call은 GPU·네트워크·모델 로딩을 하지 않는다.
dispatch_conditioning_once는 이미 준비한 파이프라인을 별도 승인 실행기에서만 호출한다.
"""
from copy import deepcopy
from dataclasses import dataclass
import inspect
import math

from PIL import Image
from genai_lab.character_style_inputs import CharacterInputError, read_verified_bundle


@dataclass(frozen=True)
class ConditioningOptions:
    """강도는 모델 제어값이지 동일성/그림체 달성 비율이 아니다."""
    identity_scale: float
    style_scale: float
    structure_scale: float
    identity_visual: bool = True
    structure: bool = True
    steps: int = 50
    guidance: float = 5.0
    seed: int = 42
    outfit_scale: float | None = None   # 의상 참조(세 번째 어댑터, D-195 0.8). 묶음에 의상 참조가 있을 때만 준다.

    def __post_init__(self):
        scales = (self.identity_scale, self.style_scale, self.structure_scale)
        for value in scales + (() if self.outfit_scale is None else (self.outfit_scale,)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise CharacterInputError("conditioning_scale_out_of_range")
        if type(self.identity_visual) is not bool or type(self.structure) is not bool:
            raise CharacterInputError("condition_switch_must_be_bool")
        if type(self.steps) is not int or not 1 <= self.steps <= 100:
            raise CharacterInputError("steps_out_of_range")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise CharacterInputError("seed_out_of_range")
        if type(self.guidance) not in (int, float) or not math.isfinite(self.guidance) or self.guidance <= 0:
            raise CharacterInputError("guidance_invalid")


@dataclass
class ConditioningCall:
    kwargs: dict
    adapter_scales: list
    evidence: dict
    seed: int


def build_feature_prompts(profile):
    """특징 문장을 빠짐없이 공유 프롬프트로 전달한다. 문장 이름만으로 독립 제어를 주장하지 않는다."""
    traits = "; ".join(row["text"].strip() for row in profile["traits"] if row["text"].strip())
    positive = profile["scene_prompt"].strip() + ". " + (traits + ". " if traits else "") + profile["style_prompt"].strip()
    negative = "photograph, skin pores, flat cel shading, latex, felt, watermark"
    if profile.get("negative_terms", "").strip():          # 예: 성별을 알 때 반대 성별 낱말(D-199)
        negative += ", " + profile["negative_terms"].strip()
    return positive, negative


def build_conditioning_call(folder, options):
    """시각 참조 두 개(의상 참조가 있으면 세 개)와 구조 입력을 조립한다. 이미지를 생성하지 않는다."""
    from genai_lab.character_style_inputs import outfit_roles
    manifest, images = read_verified_bundle(folder)
    try:
        outfits = outfit_roles(manifest["profile"])
        has_outfit = bool(outfits)
        if has_outfit != (options.outfit_scale is not None):
            raise CharacterInputError("outfit_reference_and_scale_must_match")
        from diffusers.image_processor import IPAdapterMaskProcessor
        width, height = manifest["canvas"]
        processor = IPAdapterMaskProcessor()
        style_mask = processor.preprocess(images["style_mask"], height=height, width=width)
        identity_mask_image = images["identity_mask"] if options.identity_visual else Image.new("L", (width, height), 0)
        identity_mask = processor.preprocess(identity_mask_image, height=height, width=width)
        face = images["identity"].copy() if options.identity_visual else Image.new("RGB", images["identity"].size, 0)
        pose = images["pose"].copy() if options.structure else Image.new("RGB", (width, height), 0)
        prompt, negative = build_feature_prompts(manifest["profile"])
        # 목표 참조는 SDXL up.block_0에만 연결한다. 실제 그림체 효과는 GPU 미검증.
        scales = [
            {"up": {"block_0": [0.0, options.style_scale, 0.0]}},
            options.identity_scale if options.identity_visual else 0.0,
        ]
        references = [images["style"].copy(), face]
        masks = [style_mask, identity_mask]
        order = ["style", "identity"]
        for row, (role, mask_role) in zip(manifest["profile"].get("outfits", []), outfits):
            # 영역 참조(의상·머리)는 자기 영역에만, 모든 블록에 적용한다(D-195, 윗옷·아래옷 나눔·머리 참조 D-200).
            scales.append(row.get("scale", options.outfit_scale))
            references.append(images[role].copy())
            masks.append(processor.preprocess(images[mask_role], height=height, width=width))
            order.append(role)
        kwargs = {
            "prompt": prompt, "prompt_2": prompt, "negative_prompt": negative,
            "image": pose, "ip_adapter_image": references,
            "cross_attention_kwargs": {"ip_adapter_masks": masks},
            "width": width, "height": height, "num_inference_steps": options.steps,
            "guidance_scale": options.guidance, "num_images_per_prompt": 1,
            "controlnet_conditioning_scale": options.structure_scale if options.structure else 0.0,
            "control_guidance_start": 0.0, "control_guidance_end": 1.0,
        }
        evidence = {
            "state": "conditioning_prepared_not_generated", "model_calls": 0,
            "character_traits": deepcopy(manifest["profile"]["traits"]),
            "prompt": prompt, "negative_prompt": negative,
            "adapter_order": order, "adapter_scales": deepcopy(scales),
            "identity_visual_enabled": options.identity_visual,
            "structure_enabled": options.structure,
            "structure_origin": manifest["profile"]["layout"]["origin"],
            "style_reference_sha256": manifest["assets"]["style"]["sha256"],
            "identity_reference_sha256": manifest["assets"]["identity"]["sha256"] if options.identity_visual else None,
            "pose_sha256": manifest["assets"]["pose"]["sha256"] if options.structure else None,
            "outfit_reference_sha256": {role: manifest["assets"][role]["sha256"] for role, _ in outfits} or None,
            "mask_shapes": [list(m.shape) for m in masks],
            "mask_devices": [str(m.device) for m in masks],
            "original_rgb_initialization": False, "postprocessing": [],
            "tokenizer_delivery": "not_checked_until_loaded_pipeline",
            "quality_approval": None,
        }
        return ConditioningCall(kwargs, scales, evidence, options.seed)
    finally:
        for image in images.values():
            image.close()


def validate_pipeline_contract(pipeline, adapter_count=2):
    """img2img 오접속을 거부하고 시각 참조 수(그림체·얼굴[·의상])를 갖춘 SDXL ControlNet 계약을 검사한다."""
    parameters = inspect.signature(pipeline.__call__).parameters
    required = {"image", "ip_adapter_image", "cross_attention_kwargs", "controlnet_conditioning_scale", "prompt_2"}
    if not required.issubset(parameters) or "strength" in parameters:
        raise CharacterInputError("requires_sdxl_controlnet_text_to_image_contract")
    layers = getattr(getattr(getattr(pipeline, "unet", None), "encoder_hid_proj", None),
                     "image_projection_layers", ())
    if len(layers) != adapter_count or not callable(getattr(pipeline, "set_ip_adapter_scale", None)):
        raise CharacterInputError("two_ordered_ip_adapters_required_style_then_identity" if adapter_count == 2
                                  else f"{adapter_count}_ordered_ip_adapters_required_style_identity_outfits")
    if getattr(pipeline, "controlnet", None) is None:
        raise CharacterInputError("pose_controlnet_required")


def check_prompt_delivery(pipeline, call):
    """양쪽 토크나이저의 잘림을 실행 전에 거부해 특징 문장 손실을 막는다."""
    counts = []
    for tokenizer in (pipeline.tokenizer, pipeline.tokenizer_2):
        limit = tokenizer.model_max_length
        if type(limit) is not int or not 1 <= limit <= 512:
            raise CharacterInputError("tokenizer_limit_unknown")
        row = {}
        for name in ("prompt", "negative_prompt"):
            ids = tokenizer(call.kwargs[name], add_special_tokens=True, truncation=False)["input_ids"]
            if len(ids) > limit:
                raise CharacterInputError(f"prompt_would_truncate: {name}: {len(ids)} > {limit}")
            row[name] = len(ids)
        counts.append(row)
    return counts


def dispatch_conditioning_once(pipeline, call, *, execution_authorized=False, callback=None):
    """별도 실행 승인 시 한 번만 호출한다. 실패 시 원본 반환·재시도·후처리 없음."""
    if execution_authorized is not True:
        raise PermissionError("generation_not_authorized")
    if call.evidence["model_calls"]:
        raise CharacterInputError("conditioning_call_already_dispatched")
    validate_pipeline_contract(pipeline, len(call.adapter_scales))
    token_counts = check_prompt_delivery(pipeline, call)
    from torch import Generator
    arguments = dict(call.kwargs, generator=Generator(device="cpu").manual_seed(call.seed))
    if callback is not None:
        arguments["callback_on_step_end"] = callback
    pipeline.set_ip_adapter_scale(deepcopy(call.adapter_scales))
    call.evidence.update(model_calls=1, state="running", tokenizer_delivery=token_counts)
    try:
        result = pipeline(**arguments)
        if len(result.images) != 1:
            raise CharacterInputError("expected_one_generated_image")
    except BaseException as error:
        call.evidence.update(state="failed", error_type=type(error).__name__, error=str(error))
        raise
    call.evidence.update(state="generated_unreviewed", quality_approval=None,
                         character_preservation="review_needed", style_application="review_needed")
    return result.images[0], deepcopy(call.evidence)
