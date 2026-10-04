"""Shared explicit-user gender conditions; no appearance-based inference."""
from genai_lab.reference_tag_policy import (
    CHARACTER_GENDERS, gender_tag_kind, normalize_tag, resolve_character_gender,
    validate_character_gender,
)


BASE_GENDER_CONDITION_TAGS = {
    'male': ('male focus', 'masculine silhouette'),
    'female': ('female focus', 'feminine silhouette'),
}


def prepare_gender_character_tags(tags, character_gender):
    """Keep legacy approved-tag behavior, including unspecified gender."""
    resolved, removed = resolve_character_gender(tags, character_gender)
    support = BASE_GENDER_CONDITION_TAGS.get(character_gender, ())
    return resolved, removed, support


def prepare_gender_negative_terms(terms, character_gender):
    """Remove same-gender negatives and put the opposite guard first.

    Preserve spelling, order and duplicates of unrelated terms. Callers own
    outfit filtering and token budgeting, which differ between generation paths.
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
