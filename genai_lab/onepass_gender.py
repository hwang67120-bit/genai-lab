"""Gender-only request preparation for the opt-in one-pass path.

No pipeline, model loading, token budgeting or generation is connected here.
Candidate tags are not user gender choices. The saved per-image preference is
required, including an explicit 'unspecified' choice.
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
    """Validate the shared stage 1/3 template before adding a user gender guard."""
    terms = [term.strip() for term in negative_template.split(',') if term.strip()]
    if any(_is_gender_condition(normalize_tag(term)) for term in terms):
        raise ValueError('1회 생성 부정 프롬프트 템플릿에는 성별 조건을 넣을 수 없습니다.')
    return terms


def prepare_onepass_gender(source_path: str | Path, character_tags,
                           negative_template: str, *, settings=None):
    """Read the existing preference store and reuse the product gender rules.

    A missing selection must go back to input approval. It must not inherit a
    detector label or silently become the user's explicit unspecified choice.
    The one-pass negative template must be gender-neutral before applying the
    shared guard. Legacy callers keep their existing template behavior.
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
    # The selected tag belongs in the prefix, before approved outfit tags.
    appearance = tuple(tag for tag in resolved if tag != selected)
    negative, _, _ = prepare_gender_negative_terms(terms, gender)
    return OnePassGenderConditions(
        gender, prefix, appearance, ', '.join(negative), removed)
