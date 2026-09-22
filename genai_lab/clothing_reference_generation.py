"""Design-reference regeneration, without garment erasure or body reconstruction."""

from dataclasses import replace
from genai_lab.request import CharacterFramingType
from scripts.generation_inputs import prepare_prompt_for_clip
from genai_lab.reference_prompt_budget import build_reference_prompt
from genai_lab.reference_tag_policy import (
    excluded_garment_tag, resolve_character_gender, gender_tag_kind, CHARACTER_GENDERS, normalize_tag,
)


BASE_GENDER_CONDITION_TAGS = {
    'male': ('male focus', 'masculine silhouette'),
    'female': ('female focus', 'feminine silhouette'),
}


def _matches_garment_base(tag, base):
    tag = normalize_tag(tag)
    base = normalize_tag(base)
    return tag == base or tag.endswith(f" {base}")


def resolve_redundant_ensemble_tags(tags, policy=None):
    """성별 추론 없이 상세 구성품과 중복되는 포괄 의상 태그만 제거한다."""
    if not policy:
        return tuple(tags), ()
    if not isinstance(policy, dict):
        raise ValueError("의상 프롬프트 정책은 객체여야 합니다.")
    remaining = list(tags)
    suppressed = []
    rules = policy.get("redundant_ensemble_rules", ())
    if not isinstance(rules, (list, tuple)):
        raise ValueError("포괄 의상 태그 규칙은 목록이어야 합니다.")
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("포괄 의상 태그 규칙 항목은 객체여야 합니다.")
        broad_tag = normalize_tag(rule.get("remove", ""))
        groups = rule.get("when_groups_present", ())
        if not broad_tag or not groups:
            continue
        if not all(isinstance(group, (list, tuple)) and group for group in groups):
            raise ValueError("의상 구성 그룹은 비어 있지 않은 목록이어야 합니다.")
        if broad_tag not in remaining:
            continue
        all_groups_present = all(
            any(_matches_garment_base(tag, allowed)
                for tag in remaining for allowed in group)
            for group in groups
        )
        if not all_groups_present:
            continue
        remaining = [tag for tag in remaining if normalize_tag(tag) != broad_tag]
        suppressed.append(broad_tag)
    return tuple(remaining), tuple(suppressed)


def prepare_design_reference_request(request, approved_tags, tokenizers, character_tags=(),
                                     character_gender='unspecified', approved_detail_tags=(),
                                     part_color_descriptions=(), garment_prompt_policy=None,
                                     long_prompt_settings=None,
                                     approved_part_color_descriptions=()):
    character_tags, removed_gender_tags = resolve_character_gender(character_tags, character_gender)
    tags = tuple(dict.fromkeys(normalize_tag(term)
                              for tag in approved_tags for term in str(tag).split(',') if term.strip()))
    excluded_tags = tuple(tag for tag in tags if excluded_garment_tag(tag))
    tags = tuple(tag for tag in tags if tag not in excluded_tags)
    tags, suppressed_ensemble_tags = resolve_redundant_ensemble_tags(
        tags, garment_prompt_policy)
    if not tags:
        raise ValueError('의상 디자인 태그가 없습니다. 분석 결과에서 반영할 태그를 승인하세요.')
    framing = {
        CharacterFramingType.FULL_BODY: 'full body, head to toe, feet visible',
        CharacterFramingType.UPPER_BODY: 'upper body, head to waist',
        CharacterFramingType.FACE: 'portrait, head and shoulders',
    }[request.framing_type]
    from genai_lab.garment_detail_analysis import detail_group, LOCAL_DETAIL_GROUPS
    detail_tags = tuple(dict.fromkeys(normalize_tag(term)
        for tag in approved_detail_tags for term in str(tag).split(',') if term.strip()))
    excluded_details = tuple(tag for tag in detail_tags if detail_group(tag) not in LOCAL_DETAIL_GROUPS)
    detail_tags = tuple(tag for tag in detail_tags if tag not in excluded_details)
    from genai_lab.part_color_descriptions import description_records
    # 측정과 승인 기록은 보존하되 파생 색상 이름은 공통 프롬프트에 넣지 않는다.
    replaced_generic_parts = ()
    selected_gender_tag = CHARACTER_GENDERS[character_gender]
    base_gender_condition_tags = BASE_GENDER_CONDITION_TAGS.get(
        character_gender, ())
    positive, positive_record = build_reference_prompt(
        tokenizers,
        core_character=(*character_tags, *base_gender_condition_tags),
        core_outfit=tags,
        framing=framing.split(','),
        optional=(*detail_tags, 'natural fabric folds', 'coherent anatomy', 'best quality'),
        long_prompt_settings=long_prompt_settings)
    terms = [term.strip() for term in request.negative_prompt.split(',') if term.strip()]
    conflicts = {'different outfit', 'mismatched colors'}
    removed = [term for term in terms if term.casefold() in conflicts]
    retained = [term for term in terms if term.casefold() not in conflicts]
    removed_gender_negative = [
        term for term in retained
        if selected_gender_tag and gender_tag_kind(term) == character_gender
    ]
    retained = [term for term in retained if term not in removed_gender_negative]
    # 사용자 지정값만 사용한다. 의상/체형으로 반대 성별을 추정하지 않는다.
    gender_guard = {'male': '1girl', 'female': '1boy'}.get(character_gender)
    if gender_guard:
        retained = [gender_guard] + [term for term in retained if normalize_tag(term) != gender_guard]
    negative, negative_record = prepare_prompt_for_clip(', '.join(retained), tokenizers)
    if gender_guard and negative.split(',')[0].strip() != gender_guard:
        raise ValueError('사용자 지정 성별의 네거티브 조건이 토큰 한도로 누락됐습니다.')
    negative_record['source_before_conflict_removal'] = request.negative_prompt
    if not positive or 'wearing ' not in positive:
        raise ValueError('의상 조건을 프롬프트에 담지 못했습니다. 태그를 줄여주세요.')
    if selected_gender_tag and positive.split(',')[0].strip() != selected_gender_tag:
        raise ValueError('사용자가 지정한 성별 태그가 실행 프롬프트에서 누락됐습니다.')
    return replace(request, prompt=positive, negative_prompt=negative), {
        'mode': 'design_tags_regeneration', 'approved_tags': tags,
        'suppressed_redundant_ensemble_tags': suppressed_ensemble_tags,
        'garment_conflict_policy': 'config_driven_component_consistency_v1',
        'approved_character_tags': character_tags,
        'part_color_descriptions': description_records(part_color_descriptions),
        'approved_part_color_descriptions': description_records(
            approved_part_color_descriptions),
        'applied_part_color_tags': (),
        'generic_part_tags_replaced_by_color_description': replaced_generic_parts,
        'approved_detail_tags': detail_tags,
        'excluded_non_detail_tags': excluded_details,
        'omitted_detail_tags': tuple(tag for tag in detail_tags if tag in positive_record['omitted_optional']),
        'character_gender': character_gender,
        'gender_source': 'user' if selected_gender_tag else 'approved_tags',
        'base_gender_condition_tags': base_gender_condition_tags,
        'base_gender_condition_policy': 'explicit_user_gender_base_prompt_v1',
        'removed_character_gender_tags': removed_gender_tags,
        'removed_gender_negative_terms': removed_gender_negative,
        'gender_negative_guard': gender_guard,
        'garment_gender_policy': 'exclude_gender_and_body_tags_v1',
        'garment_visual_gender_isolated': False,
        'excluded_non_design_tags': excluded_tags,
        'removed_conflicting_negative_terms': removed,
        'positive': positive_record, 'negative': negative_record,
        'garment_image_adapter': False, 'body_restoration': False,
        'pixel_preservation_guaranteed': False,
    }
