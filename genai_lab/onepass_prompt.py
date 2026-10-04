"""Pure one-pass strings and CLIP chunk plans; no encoders or preferences I/O.

One-pass intentionally puts gender support immediately after 1boy/1girl,
before clothing/appearance (gender-contract U_*). Legacy product order differs.
Unspecified gender does not restore discarded detector gender candidates.
"""
from dataclasses import dataclass, field
from genai_lab.gender_prompt import BASE_GENDER_CONDITION_TAGS, prepare_gender_negative_terms
from genai_lab.reference_tag_policy import CHARACTER_GENDERS, normalize_tag, excluded_appendage_tag
from genai_lab.onepass_gender import OnePassGenderConditions, validate_onepass_negative_template
from genai_lab.onepass_garment_vocabulary import garment_nouns, uncovered_parts, filter_pocket_tags
from genai_lab.onepass_prompt_settings import OnePassPromptSettings

@dataclass(frozen=True)
class CharacterTagGroups:
    appearance: tuple[str, ...] = ()
    body: tuple[str, ...] = ()
    fixed: tuple[str, ...] = ()

@dataclass(frozen=True)
class PartAppearance:
    """User text is a draft until explicitly confirmed; edits need re-confirmation."""
    text: str = ""
    confirmed: bool = False

    def __post_init__(self):
        if not isinstance(self.text, str) or type(self.confirmed) is not bool:
            raise TypeError("외형 문구와 확인 상태가 올바르지 않습니다.")
        if self.confirmed:
            self.tags()  # Reject invalid confirmed data even outside the GUI.

    def tags(self):
        values = tuple(t.strip() for t in self.text.split(","))
        if not self.text.strip() or any(not t for t in values):
            raise ValueError("확인할 영어 외형 문구를 쉼표로 구분해 입력해 주세요.")
        if not self.text.isascii() or any(ord(c) < 32 for c in self.text):
            raise ValueError("외형 문구는 한 줄의 영어 태그로 입력해 주세요.")
        tags = _tags(values)
        if any(excluded_appendage_tag(t) or
               any(normalize_tag(t) in support for support in BASE_GENDER_CONDITION_TAGS.values())
               for t in tags):
            raise ValueError("꼬리·귀 외형 문구에는 성별·체형·노출·다른 의상 지시를 넣을 수 없습니다.")
        return tags


@dataclass(frozen=True)
class AppearanceOverrides:
    tail: PartAppearance = field(default_factory=PartAppearance)
    ears: PartAppearance = field(default_factory=PartAppearance)

    def __post_init__(self):
        if not isinstance(self.tail, PartAppearance) or not isinstance(self.ears, PartAppearance):
            raise TypeError("꼬리·귀 외형 확인 데이터가 필요합니다.")


def appendage_tags(fixed):
    """Match whole tag boundaries, never hairstyle words such as twintails."""
    groups = {"tail": [], "ears": []}
    for tag in fixed:
        value = normalize_tag(tag)
        if value == "tail" or value.endswith(" tail"):
            groups["tail"].append(tag)
        elif value == "animal ears" or value.endswith(" ears"):
            groups["ears"].append(tag)
    return {part: tuple(tags) for part, tags in groups.items()}


def replace_appendage_appearance(fixed, appearance, *, enable_ear_override):
    """Apply approved descriptions after validating the original gender contract."""
    found = appendage_tags(fixed)
    result = list(fixed)
    records = {}
    for part in ("tail", "ears"):
        choice = getattr(appearance, part)
        if part == "ears" and choice.confirmed and not enable_ear_override:
            raise ValueError("귀 외형 변경은 현재 적용하지 않습니다.")
        if choice.confirmed and not found[part]:
            raise ValueError("감지되지 않은 부위의 외형 문구는 적용할 수 없습니다.")
        replacements = choice.tags() if choice.confirmed else ()
        if replacements:
            removed = set(found[part])
            position = next(i for i, tag in enumerate(result) if tag in removed)
            result = [t for t in result[:position] if t not in removed] + list(replacements) + [
                t for t in result[position:] if t not in removed]
        records[part] = {
            "detected_tags": found[part], "draft": choice.text,
            "confirmed": choice.confirmed, "applied": bool(replacements),
            "removed_tags": found[part] if replacements else (),
            "replacement_tags": replacements,
            "reason": "user_confirmed_appearance" if replacements else "not_confirmed",
        }
    return tuple(result), records


@dataclass(frozen=True)
class PromptChunk:
    kind: str
    text: str | None
    token_ids: tuple[int, ...]  # BOS + <=75 content + EOS + pad, always 77.

@dataclass(frozen=True)
class EncoderChunkPlan:
    positive_tokens: int
    negative_tokens: int
    positive: tuple[PromptChunk, ...]
    negative: tuple[PromptChunk, ...]

@dataclass(frozen=True)
class OnePassPrompt:
    positive: str
    negative: str
    encoders: tuple[EncoderChunkPlan, ...]
    rules: dict


def _tags(values):
    if isinstance(values, str):
        raise TypeError('태그 문자열이 아니라 태그 목록이 필요합니다.')
    out = tuple(t.replace('_', ' ').strip() for t in values)
    if any(not t or ',' in t for t in out):
        raise ValueError('태그는 쉼표 없이 비어 있지 않은 항목이어야 합니다.')
    return out


def plan_prompt_chunks(positive, negative, tokenizers):
    """Match run_int.py pieces/ids_for without invoking text encoders.

    Historical long chunks use tokenizer 1 content IDs for both encoders.
    Reject divergent long-prompt tokenizations instead of silently misencoding.
    Short prompts remain text chunks. Longer negative expands both sides to
    prevent truncation/length mismatch; blank chunks are explicitly recorded.
    """
    if len(tokenizers) != 2:
        raise ValueError('SDXL 토크나이저 두 개가 필요합니다.')
    pos = [
        list(t(positive, add_special_tokens=False)['input_ids'])
        for t in tokenizers
    ]
    neg = [
        list(t(negative, add_special_tokens=False)['input_ids'])
        for t in tokenizers
    ]
    for seq in (pos, neg):
        if max(map(len, seq)) > 75 and seq[0] != seq[1]:
            raise ValueError('긴 프롬프트의 두 토크나이저 ID가 달라 기존 청크 계약을 재현할 수 없습니다.')
    count = max(1, *((len(ids) + 74) // 75 for ids in pos + neg))
    result = []
    for i, tokenizer in enumerate(tokenizers):
        def chunks(ids, text):
            short = len(ids) <= 75
            groups = (
                [ids] if short
                else [ids[k:k + 75] for k in range(0, len(ids), 75)]
            )
            built = []
            for k in range(count):
                content = groups[k] if k < len(groups) else []
                kind = 'text' if short or k >= len(groups) else 'ids'
                chunk_text = (
                    (text if k == 0 and short else '')
                    if kind == 'text' else None
                )
                tokens = [
                    tokenizer.bos_token_id, *content, tokenizer.eos_token_id
                ]
                tokens += [tokenizer.pad_token_id] * (77 - len(tokens))
                built.append(PromptChunk(kind, chunk_text, tuple(tokens)))
            return tuple(built)
        result.append(EncoderChunkPlan(
            len(pos[i]), len(neg[i]),
            chunks(pos[i], positive), chunks(neg[i], negative),
        ))
    return tuple(result)


def assemble_onepass_prompt(gender: OnePassGenderConditions, groups: CharacterTagGroups,
                            garment_tags, pose_tags, *, slim, tokenizers,
                            settings=OnePassPromptSettings(), appearance=AppearanceOverrides()):
    """Ordered assembly from approved inputs. Never infer/correct user gender.

    Character classification must partition the stage 1 appearance tags. No tag
    invented here may bypass that contract. Garment approval is not reinterpreted
    from images; source detection mistakes remain visible to user review.
    """
    if slim is not None and type(slim) is not bool:
        raise ValueError('slim은 True/False/미측정(None)이어야 합니다.')
    terms = validate_onepass_negative_template(settings.negative_template)
    expected_negative = ', '.join(
        prepare_gender_negative_terms(terms, gender.character_gender)[0]
    )
    tag = CHARACTER_GENDERS[gender.character_gender]
    prefix = (
        (tag, *BASE_GENDER_CONDITION_TAGS.get(gender.character_gender, ()))
        if tag else ()
    )
    if gender.prefix_tags != prefix or gender.negative_prompt != expected_negative:
        raise ValueError('단계 1 성별 조건과 프롬프트 설정이 일치하지 않습니다.')
    app, body, fixed = (
        _tags(g) for g in (groups.appearance, groups.body, groups.fixed)
    )
    classified = tuple(normalize_tag(t) for t in (*app, *body, *fixed))
    if (len(classified) != len(set(classified))
            or set(classified) != set(gender.appearance_tags)):
        raise ValueError('캐릭터 분류는 단계 1 외형 태그와 정확히 대응해야 합니다.')
    garments, poses = _tags(garment_tags), _tags(pose_tags)
    uncovered = uncovered_parts(garments)
    poses, removed = filter_pocket_tags(poses, garments)
    policy = settings.character.slim_policy
    body_before = body
    if slim is True and policy != 'off':
        additions = ('slender', 'skinny') if policy == 'ps' else ('slender',)
        body = tuple(t for t in additions if t not in body) + tuple(
            'small breasts' if t == 'medium breasts' else t for t in body
        )
    fixed_before = fixed
    fixed, appearance_record = replace_appendage_appearance(
        fixed, appearance, enable_ear_override=settings.enable_ear_override)
    positive = ', '.join((
        *prefix, *garments, *uncovered, *app, *body, *fixed, *poses, *settings.tail
    ))
    negative = gender.negative_prompt
    encoders = plan_prompt_chunks(positive, negative, tokenizers)
    before_encoders = encoders
    if fixed != fixed_before:
        original = ', '.join((
            *prefix, *garments, *uncovered, *app, *body, *fixed_before, *poses, *settings.tail))
        before_encoders = plan_prompt_chunks(original, negative, tokenizers)
    return OnePassPrompt(positive, negative, encoders, {
        'appendage_review_required': any(appendage_tags(fixed_before).values()),
        'appendage_appearance': appearance_record,
        'appearance_token_counts_before': tuple(e.positive_tokens for e in before_encoders),
        'appearance_token_counts_after': tuple(e.positive_tokens for e in encoders),
        'appearance_chunks_before': len(before_encoders[0].positive),
        'appearance_chunks_after': len(encoders[0].positive),
        'gender_source': 'stage1_user_preference', 'prefix': prefix,
        'uncovered_policy': 'torso_noun_blocks_automatic_midriff',
        'garment_nouns': tuple(n.name for n in garment_nouns(garments)),
        'uncovered_added': uncovered, 'pocket_removed': removed,
        'slim_measured': slim, 'slim_policy': policy, 'slim_policy_decision': 'pending',
        'body_before': body_before, 'body_after': body,
        'negative_template_matched': True,
        'chunks': len(encoders[0].positive),
    })
