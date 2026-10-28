"""의상 설명을 사용자 승인 캐릭터 설명과 분리한다."""
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
    # 제안만 표시한다. 성별을 포함한 모든 선택란은 사용자가 승인하거나 해제한다.
    return tag.endswith(' hair') or tag.endswith(' eyes') or tag in {'bangs', 'ponytail', 'twintails', 'braid', 'bob cut'}


CHARACTER_GENDERS = {'unspecified': None, 'male': '1boy', 'female': '1girl'}
GENDER_LABELS = {'unspecified': '지정 안 함', 'male': '남성', 'female': '여성'}


def validate_character_gender(value):
    if not isinstance(value, str) or value not in CHARACTER_GENDERS:
        raise ValueError('캐릭터 성별은 unspecified / male / female 중 하나여야 합니다.')
    return value


def gender_tag_kind(tag):
    """명시적인 성별·개수 태그만 사용하며 의상이나 체격으로 성별을 추정하지 않는다."""
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
    # 사용자의 명시적 선택은 검출 성별만 교체하고 외형은 바꾸지 않는다.
    retained = tuple(tag for tag in normalized if gender_tag_kind(tag) is None)
    return (CHARACTER_GENDERS[gender], *retained), removed


# 입력 칸에만 적용하는 규칙이다. exposure-v2/rule_v2.md A/B/C와 기록된 노출 단어,
# 기존 부정 문구의 nsfw/buruma를 포함한다. 결과 이미지 검사는 아니다.
# 정상적인 의상 노출 설명도 의상 승인에서 다루며 꼬리·귀 칸에는 넣지 않는다.
APPENDAGE_EXPOSURE_TAGS = frozenset({
    "nsfw", "panties", "underwear", "bra", "lingerie", "nude", "completely nude",
    "topless", "bottomless", "nipples", "leotard", "swimsuit", "one-piece swimsuit",
    "bikini", "school swimsuit", "midriff", "navel", "ass", "cleavage", "sideboob",
    "underboob", "midriff peek", "bare legs", "bare shoulders", "bare arms", "buruma",
})


def excluded_appendage_tag(tag):
    """꼬리·귀 설명에 알려진 캐릭터·체형·노출 태그를 넣지 못하게 한다. 의상 칸의 제외 규칙을 재사용하되 의상 정책을 넓히지 않는다. 유한한 태그 검사이며 자연어
    안전 분류기는 아니다.
    """
    normalized = normalize_tag(tag)
    return excluded_garment_tag(normalized) or normalized in APPENDAGE_EXPOSURE_TAGS
