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
    """시각 참조의 단계 경계를 관리한다."""

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
    # 전용 시각 어댑터가 꺼져 있어도 머리 마스크는 진단과 보정을 위해
    # 유지한다. 참조 이미지가 있을 때만 마스크를 필수로 요구한다.
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
    """분리된 생성 단계 하나에 승인된 참조만 반환한다."""
    try:
        resolved = stage if isinstance(stage, ReferenceStage) else ReferenceStage(stage)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Unknown visual reference stage: {stage}') from error
    entries = adapter_references(inputs, identity_scale, garment_scale)
    if resolved is ReferenceStage.BASE:
        # Base Img2Img의 편집 결과 좌표는 아직 없다. 얼굴/헤어 크롭을
        # 전체 이미지 조건은 화면에 떨어진 조각을 재현할 수 있다.
        # 승인한 전신 이미지 조건 하나만 사용하고 부분 크롭은
        # 후속 결과 좌표 처리 단계에 남긴다.
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
    """마스크 없는 Diffusers 얼굴 참조용으로 크롭별 강도를 하나로 합친다. 이미지별 강도 목록은 공간 마스크와 함께만 지원된다. 마스크가 없으면 단일 강도가
    필요하다. 산술 평균으로 승인된 전체 강도를 유지하며 모든 크롭을 가장 강한 값으로 올리지 않는다.
    """
    values = tuple(float(value) for value in scales)
    if (not values or any(not math.isfinite(value) or not 0 <= value <= 1
                          for value in values)):
        raise ValueError("마스크 없는 IP-Adapter 참조 강도 오류")
    return sum(values) / len(values)
