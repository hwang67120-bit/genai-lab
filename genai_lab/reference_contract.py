"""Hard input contracts for reference-only generation; no image-quality threshold."""
import numpy as np


def validate_reference_mode(config, pose_control_enabled=False):
    from genai_lab.scene_generation import scene_settings
    scene_settings(config)
    section = config.get('clothing_reference_generation', {})
    if not section.get('enabled', False):
        return
    if section.get('without_initial_image', False) is not False:
        raise ValueError('참조 생성 Base에는 승인된 캐릭터 RGB 시작 이미지가 필요합니다.')
    if pose_control_enabled:
        raise ValueError('디자인 참조 생성에는 외부 자세/ControlNet을 연결하지 않습니다.')


def validate_visual_inputs(inputs):
    scene = getattr(inputs, 'scene_condition', None)
    if scene is not None:
        if getattr(inputs, 'extra_references', ()):
            raise ValueError('선화 부위 참조와 별도 부위 참조를 동시에 연결하지 않습니다.')
        from genai_lab.scene_generation import validate_scene_condition
        validate_scene_condition(scene, inputs.source.size)
        by_name = {part.name: part for part in scene.references}
        for name, image, mask in (('identity', inputs.identity, inputs.identity_mask),
                                  ('garment', inputs.garment, inputs.garment_mask)):
            part = by_name[name]
            if (image.mode != part.rgb.mode or image.size != part.rgb.size
                    or image.tobytes() != part.rgb.tobytes()
                    or mask.mode != part.region.mode or mask.size != part.region.size
                    or mask.tobytes() != part.region.tobytes()):
                raise ValueError('비교/미리보기와 실제 생성 참조가 일치하지 않습니다.')
    if getattr(inputs, 'initial', None) is not None:
        raise ValueError('폐기된 초기 이미지가 입력에 남아 있습니다.')
    regions = []
    for name in ('identity_mask', 'garment_mask'):
        mask = getattr(inputs, name)
        if mask.size != inputs.source.size or mask.mode != 'L':
            raise ValueError(f'{name}: 기준 이미지와 같은 크기의 L 모드 마스크가 필요합니다.')
        pixels = np.asarray(mask)
        if not np.isin(pixels, (0, 255)).all():
            raise ValueError(f'{name}: 0/255 하드 마스크가 필요합니다. 자동 반전/보정하지 않습니다.')
        selected = pixels == 255
        if not selected.any():
            raise ValueError(f'{name}: 선택 영역이 비어 있습니다.')
        regions.append(selected)
    if np.any(regions[0] & regions[1]):
        raise ValueError('얼굴·헤어와 의상 영향 영역이 겹칩니다. 승인 마스크를 다시 확인하세요.')
    hair_reference = getattr(inputs, 'hair_reference', None)
    hair_mask = getattr(inputs, 'hair_mask', None)
    # A refined hair mask may exist solely for diagnostics or local repair.
    # It becomes a generation condition only when hair_reference is present.
    if hair_reference is not None and hair_mask is None:
        raise ValueError('헤어 시각 참조 이미지에는 헤어 마스크가 필요합니다.')
    if hair_reference is not None:
        if (hair_reference.mode != 'RGB' or min(hair_reference.size) <= 0
                or hair_mask.mode != 'L' or hair_mask.size != inputs.source.size):
            raise ValueError('헤어 시각 참조 이미지 또는 마스크 형식 오류')
        hair_region = np.asarray(hair_mask) == 255
        if (not np.isin(np.asarray(hair_mask), (0, 255)).all()
                or not hair_region.any()
                or np.any(hair_region & (regions[0] | regions[1]))):
            raise ValueError('헤어 시각 참조 영역이 비었거나 얼굴·의상과 겹칩니다.')
        if (not np.isfinite(inputs.hair_reference_scale)
                or not 0 <= inputs.hair_reference_scale <= 1):
            raise ValueError('헤어 시각 참조 강도 오류')
        regions.append(hair_region)
    from genai_lab.regional_reference import valid_part_name
    from genai_lab.reference_regions import read_binary_mask
    extras = getattr(inputs, 'extra_references', ())
    if len(extras) > 7:
        raise ValueError('추가 부위 참조는 최대 7개입니다.')
    occupied = np.logical_or.reduce(regions)
    names = set()
    for part in extras:
        if not valid_part_name(part.name) or part.name in names:
            raise ValueError('추가 부위 이름 오류 또는 중복')
        names.add(part.name)
        region = read_binary_mask(part.region, inputs.source.size)
        if not region.any() or np.any(region & occupied):
            raise ValueError('추가 부위 영역이 비었거나 다른 참조와 겹칩니다.')
        if part.rgb.mode != 'RGB' or not np.isfinite(part.scale) or not 0 <= part.scale <= 1:
            raise ValueError('추가 부위 참조 이미지 또는 강도 오류')
        occupied |= region
    for name in ('identity', 'garment'):
        image = getattr(inputs, name)
        if image.mode != 'RGB' or min(image.size) <= 0:
            raise ValueError(f'{name}: 유효한 RGB 참조 이미지가 필요합니다.')

    bundle = getattr(inputs, 'reference_conditions', None)
    if (getattr(inputs, 'analysis_record', None) or {}).get('condition_separation') and bundle is None:
        raise ValueError('부위 원본 조건이 누락되었습니다. 다시 준비하고 승인하세요.')
    if bundle is not None:
        bundle.validate_materialized(inputs)


def validate_visual_condition(condition):
    if not isinstance(condition, dict) or set(condition) != {'ip_adapter_image_embeds'}:
        raise ValueError('잘라낸 부위별 시각 참조 임베딩만 생성기에 전달할 수 있습니다. 기준 이미지 좌표 마스크는 금지됩니다.')
    embeds = condition['ip_adapter_image_embeds']
    if embeds is None or not isinstance(embeds, (list, tuple)) or not embeds:
        raise ValueError('부위별 시각 참조 임베딩이 비어 있습니다.')
