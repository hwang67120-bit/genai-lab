"""선택적 1회 생성의 성별 요청만 준비한다. 모델·토큰 예산·생성 경로는 연결하지 않는다. 후보 태그는 사용자 선택이 아니며 이미지별 저장 선택이 필요하다.
미지정도 명시적으로 선택해야 한다.
"""
from dataclasses import dataclass
from pathlib import Path

from genai_lab.character_preferences import load_character_gender
from genai_lab.gender_prompt import (
    BASE_GENDER_CONDITION_TAGS, prepare_gender_character_tags,
    prepare_gender_negative_terms,
)
from genai_lab.reference_tag_policy import (
    CHARACTER_GENDERS, gender_tag_kind, normalize_tag,
)


@dataclass(frozen=True)
class OnePassGenderConditions:
    character_gender: str
    prefix_tags: tuple[str, ...]
    appearance_tags: tuple[str, ...]
    negative_prompt: str
    removed_candidate_gender_tags: tuple[str, ...]

    def to_record(self):
        return {
            'character_gender': self.character_gender,
            'gender_source': 'user_preference',
            'gender_policy': 'onepass_explicit_user_gender_v1',
            'gender_prefix_tags': self.prefix_tags,
            'appearance_tags': self.appearance_tags,
            'removed_candidate_gender_tags': self.removed_candidate_gender_tags,
            'negative_prompt': self.negative_prompt,
        }


def _is_gender_condition(tag):
    return gender_tag_kind(tag) is not None or any(
        tag in support for support in BASE_GENDER_CONDITION_TAGS.values())


def validate_onepass_negative_template(negative_template):
    """사용자 성별 제한을 넣기 전에 1·3단계 공통 문구를 검증한다."""
    terms = [term.strip() for term in negative_template.split(',') if term.strip()]
    if any(_is_gender_condition(normalize_tag(term)) for term in terms):
        raise ValueError('1회 생성 부정 프롬프트 템플릿에는 성별 조건을 넣을 수 없습니다.')
    return terms


def prepare_onepass_gender(source_path: str | Path, character_tags,
                           negative_template: str, *, settings=None):
    """기존 선택 저장소와 제품 성별 규칙을 재사용한다. 선택이 없으면 입력 확인으로 돌아가며 검출값이나 미지정을 자동 선택하지 않는다. 1회 생성의 부정 문구는
    제한 적용 전 성별 중립이어야 한다. 구형 호출부의 문구 동작은 유지한다.
    """
    gender = load_character_gender(source_path, settings=settings)
    if gender is None:
        raise ValueError('캐릭터 성별 선택이 없습니다. 남성·여성·지정 안 함 중 선택하세요.')
    if isinstance(character_tags, str):
        raise TypeError('캐릭터 태그 목록이 필요합니다.')
    terms = validate_onepass_negative_template(negative_template)
    candidates = tuple(dict.fromkeys(
        normalize_tag(term) for tag in character_tags
        for term in str(tag).split(',') if term.strip()))
    removed = tuple(tag for tag in candidates if _is_gender_condition(tag))
    appearance = tuple(tag for tag in candidates if not _is_gender_condition(tag))
    resolved, _, support = prepare_gender_character_tags(appearance, gender)
    selected = CHARACTER_GENDERS[gender]
    prefix = (selected, *support) if selected else ()
    # 선택 태그는 승인 의상 태그보다 앞의 시작 문구에 넣는다.
    appearance = tuple(tag for tag in resolved if tag != selected)
    negative, _, _ = prepare_gender_negative_terms(terms, gender)
    return OnePassGenderConditions(
        gender, prefix, appearance, ', '.join(negative), removed)
