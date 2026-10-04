"""Keep garment descriptions separate from user-approved character descriptions."""
import re


def normalize_tag(tag):
    return ' '.join(str(tag).strip().replace('_', ' ').casefold().split())


def excluded_garment_tag(tag):
    tag = normalize_tag(tag)
    # 의상 출처의 인물/체형 설명은 캐릭터 조건으로 승격하지 않는다.
    if ',' in tag and any(excluded_garment_tag(term) for term in tag.split(',')):
        return True
    if gender_tag_kind(tag) is not None or garment_body_tag(tag):
        return True
    return bool(re.fullmatch(r'\d+\+?(?:girls?|boys?|others?)', tag)) or tag in {
        'girl', 'boy', 'girls', 'boys', 'man', 'woman', 'men', 'women',
        'male', 'female', 'male focus', 'female focus', 'multiple girls',
        'multiple boys', 'multiple others', 'androgynous', 'gender swap',
        'genderswap', 'solo', 'solo focus', 'group', 'couple',
        'watermark', 'text', 'signature', 'head out of frame', 'out of frame',
        'cropped', 'unworn shoes', 'white background', 'simple background', 'breasts',
    }


def garment_body_tag(tag):
    """정해진 태그 어휘만 판정한다. 옷 종류로 성별을 추론하지 않는다."""
    tag = normalize_tag(tag)
    return bool(re.fullmatch(
        r'(?:(?:very |huge |large |medium |small |flat |wide |narrow )*)'
        r'(?:breasts|chest|hips|waist)', tag)) or tag in {
        'cleavage', 'bust', 'curvy', 'hourglass figure', 'slender', 'muscular',
        'muscles', 'abs', 'broad shoulders', 'beard', 'facial hair',
        'male body', 'female body', 'male physique', 'female physique',
        'feminine', 'masculine', 'feminine male', 'masculine female',
    }


def suggested_character_tag(tag):
    tag = normalize_tag(tag)
    # Suggestions only: users approve/clear every checkbox, including gender tags.
    return tag.endswith(' hair') or tag.endswith(' eyes') or tag in {'bangs', 'ponytail', 'twintails', 'braid', 'bob cut'}


CHARACTER_GENDERS = {'unspecified': None, 'male': '1boy', 'female': '1girl'}
GENDER_LABELS = {'unspecified': '지정 안 함', 'male': '남성', 'female': '여성'}


def validate_character_gender(value):
    if not isinstance(value, str) or value not in CHARACTER_GENDERS:
        raise ValueError('캐릭터 성별은 unspecified / male / female 중 하나여야 합니다.')
    return value


def gender_tag_kind(tag):
    """Only explicit gender/count tags; do not infer gender from clothing or build."""
    tag = normalize_tag(tag)
    if re.fullmatch(r'\d+\+?boys?', tag) or tag in {
            'boy', 'boys', 'man', 'men', 'male', 'male focus', 'multiple boys'}:
        return 'male'
    if re.fullmatch(r'\d+\+?girls?', tag) or tag in {
            'girl', 'girls', 'woman', 'women', 'female', 'female focus', 'multiple girls'}:
        return 'female'
    if tag in {'gender swap', 'genderswap'}:
        return 'conflict'
    return None


def resolve_character_gender(tags, gender='unspecified'):
    validate_character_gender(gender)
    normalized = tuple(dict.fromkeys(
        normalize_tag(term) for tag in tags for term in str(tag).split(',')
        if term.strip()))
    if gender == 'unspecified':
        return normalized, ()
    removed = tuple(tag for tag in normalized if gender_tag_kind(tag) is not None)
    # The explicit user choice replaces detector gender labels, not appearance.
    retained = tuple(tag for tag in normalized if gender_tag_kind(tag) is None)
    return (CHARACTER_GENDERS[gender], *retained), removed


# Input-field scope only: exposure-v2/rule_v2.md A/B/C + recorded exposure terms,
# plus the existing negative template's nsfw/buruma. This is NOT an output gate:
# even legitimate garment exposure belongs in garment approval, not a tail/ear field.
APPENDAGE_EXPOSURE_TAGS = frozenset({
    "nsfw", "panties", "underwear", "bra", "lingerie", "nude", "completely nude",
    "topless", "bottomless", "nipples", "leotard", "swimsuit", "one-piece swimsuit",
    "bikini", "school swimsuit", "midriff", "navel", "ass", "cleavage", "sideboob",
    "underboob", "midriff peek", "bare legs", "bare shoulders", "bare arms", "buruma",
})


def excluded_appendage_tag(tag):
    """Tail/ear descriptions cannot inject known character/body/exposure tags.

    Reuse the garment field's exclusion contract without broadening garment
    policy. This is a finite tag check, not a natural-language safety classifier.
    """
    normalized = normalize_tag(tag)
    return excluded_garment_tag(normalized) or normalized in APPENDAGE_EXPOSURE_TAGS
