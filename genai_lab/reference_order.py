"""IP-Adapter 전달 순서만 결정한다. 원본 조건·배치·강도는 변경하지 않는다."""
from dataclasses import dataclass
from enum import Enum
import math
from PIL import Image


@dataclass(frozen=True)
class AdapterReference:
    """이미지·마스크·강도를 함께 이동시키는 묶음. 이미지 소유권은 inputs에 있다."""
    name: str
    image: Image.Image
    mask: Image.Image
    scale: float


class ReferenceStage(str, Enum):
    '''Visual-reference stage boundary.'''

    BASE = 'base'
    GARMENT = 'garment'
    HAIR = 'hair'
    EARS = 'ears'
    TAIL = 'tail'


def adapter_references(inputs, identity_scale=.7, garment_scale=.45):
    scene = getattr(inputs, "scene_condition", None)
    if scene is not None:
        # 이번 실험은 일반 참조 경로만 변경한다. 선화 전용 계약은 그대로다.
        from genai_lab.scene_generation import reference_scales
        scales = reference_scales(inputs, identity_scale, garment_scale)
        return tuple(AdapterReference(part.name, part.rgb, part.region, scale)
                     for part, scale in zip(scene.references, scales))
    extras = sorted(
        getattr(inputs, "extra_references", ()),
        key=lambda part: {
            "human_ears": 0, "animal_ears": 1, "ears": 1, "tail": 2,
        }.get(part.name, 3),
    )
    hair_image = getattr(inputs, "hair_reference", None)
    hair_mask = getattr(inputs, "hair_mask", None)
    # hair_mask is also retained for diagnostics/correction when the dedicated
    # visual adapter is disabled. Only a reference image requires the mask.
    if hair_image is not None and hair_mask is None:
        raise ValueError("헤어 참조 이미지에는 헤어 마스크가 필요합니다.")
    hair_scale = float(getattr(inputs, "hair_reference_scale", .60))
    return (
        AdapterReference("identity", inputs.identity, inputs.identity_mask, identity_scale),
        *(() if hair_image is None else (
            AdapterReference("hair", hair_image, hair_mask, hair_scale),
        )),
        *(AdapterReference(part.name, part.rgb, part.region, part.scale) for part in extras),
        AdapterReference("garment", inputs.garment, inputs.garment_mask, garment_scale),
    )


def adapter_reference_names(inputs):
    return [entry.name for entry in adapter_references(inputs)]


def adapter_references_for_stage(
        inputs, stage, identity_scale=.7, garment_scale=.45):
    '''Return only references approved for one isolated render stage.'''
    try:
        resolved = stage if isinstance(stage, ReferenceStage) else ReferenceStage(stage)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Unknown visual reference stage: {stage}') from error
    entries = adapter_references(inputs, identity_scale, garment_scale)
    if resolved is ReferenceStage.BASE:
        # Base Img2Img의 편집 결과 좌표는 아직 없다. 얼굴/헤어 크롭을
        # global image prompts can reproduce detached fragments on the canvas.
        # Use one approved full-character image prompt and keep local crops for
        # the later output-coordinate stages.
        base_image = (
            getattr(inputs, "vibe_reference", None)
            or getattr(inputs, "source", None)
            or inputs.identity
        )
        base_mask = (
            getattr(inputs, "full_character_mask", None)
            or inputs.identity_mask
        )
        selected = (
            AdapterReference(
                "identity", base_image, base_mask, float(identity_scale)
            ),
        )
    else:
        allowed = {
            ReferenceStage.GARMENT: {'garment'},
            ReferenceStage.HAIR: {'hair'},
            ReferenceStage.EARS: {'human_ears', 'animal_ears', 'ears'},
            ReferenceStage.TAIL: {'tail'},
        }[resolved]
        selected = tuple(
            entry for entry in entries if entry.name in allowed
        )
    if resolved is ReferenceStage.BASE and not any(
            entry.name == 'identity' for entry in selected):
        raise ValueError('Base generation requires an identity reference.')
    maximum = 2 if resolved is ReferenceStage.EARS else 1
    if resolved is not ReferenceStage.BASE and len(selected) > maximum:
        raise ValueError(
            f'{resolved.value} stage accepts at most {maximum} visual references.')
    return selected


def unmasked_adapter_scale(scales):
    """Collapse per-crop strengths for one unmasked Diffusers IP-Adapter.

    Diffusers accepts a per-image scale list only together with spatial masks.
    Without masks the attention processor requires one scalar for the adapter.
    The arithmetic mean preserves the approved set's overall strength without
    silently promoting every crop to the strongest reference.
    """
    values = tuple(float(value) for value in scales)
    if (not values or any(not math.isfinite(value) or not 0 <= value <= 1
                          for value in values)):
        raise ValueError("마스크 없는 IP-Adapter 참조 강도 오류")
    return sum(values) / len(values)
