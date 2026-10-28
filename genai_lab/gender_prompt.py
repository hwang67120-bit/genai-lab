"""사용자가 명시한 성별 조건을 공유한다. 외형으로 성별을 추정하지 않는다."""
from genai_lab.reference_tag_policy import (
    CHARACTER_GENDERS, gender_tag_kind, normalize_tag, resolve_character_gender,
    validate_character_gender,
)


BASE_GENDER_CONDITION_TAGS = {
    'male': ('male focus', 'masculine silhouette'),
    'female': ('female focus', 'feminine silhouette'),
}


def prepare_gender_character_tags(tags, character_gender):
    """성별 미지정을 포함해 기존 승인 태그 동작을 유지한다."""
    resolved, removed = resolve_character_gender(tags, character_gender)
    support = BASE_GENDER_CONDITION_TAGS.get(character_gender, ())
    return resolved, removed, support


def prepare_gender_negative_terms(terms, character_gender):
    """같은 성별의 부정 태그를 제거하고 반대 성별 제한을 앞에 둔다. 관련 없는 태그의 표기·순서·중복은 유지한다. 의상 필터와 토큰 예산은 생성 경로별 호출부가
    관리한다.
    """
    validate_character_gender(character_gender)
    terms = list(terms)
    selected = CHARACTER_GENDERS[character_gender]
    removed = [term for term in terms
               if selected and gender_tag_kind(term) == character_gender]
    retained = [term for term in terms if term not in removed]
    guard = {'male': '1girl', 'female': '1boy'}.get(character_gender)
    if guard:
        retained = [guard] + [term for term in retained
                              if normalize_tag(term) != guard]
    return retained, removed, guard
