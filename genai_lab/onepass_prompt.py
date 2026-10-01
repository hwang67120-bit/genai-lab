"""Pure one-pass strings and CLIP chunk plans; no encoders or preferences I/O.

One-pass intentionally puts gender support immediately after 1boy/1girl,
before clothing/appearance (gender-contract U_*). Legacy product order differs.
Unspecified gender does not restore discarded detector gender candidates.
"""
from dataclasses import dataclass
from genai_lab.gender_prompt import BASE_GENDER_CONDITION_TAGS, prepare_gender_negative_terms
from genai_lab.reference_tag_policy import CHARACTER_GENDERS, normalize_tag
from genai_lab.onepass_gender import OnePassGenderConditions, validate_onepass_negative_template
from genai_lab.onepass_garment_vocabulary import garment_nouns, uncovered_parts, filter_pocket_tags
from genai_lab.onepass_prompt_settings import OnePassPromptSettings

@dataclass(frozen=True)
class CharacterTagGroups:
    appearance: tuple[str, ...] = ()
    body: tuple[str, ...] = ()
    fixed: tuple[str, ...] = ()

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
                            settings=OnePassPromptSettings()):
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
    positive = ', '.join((
        *prefix, *garments, *uncovered, *app, *body, *fixed, *poses, *settings.tail
    ))
    negative = gender.negative_prompt
    encoders = plan_prompt_chunks(positive, negative, tokenizers)
    return OnePassPrompt(positive, negative, encoders, {
        'gender_source': 'stage1_user_preference', 'prefix': prefix,
        'uncovered_policy': 'torso_noun_blocks_automatic_midriff',
        'garment_nouns': tuple(n.name for n in garment_nouns(garments)),
        'uncovered_added': uncovered, 'pocket_removed': removed,
        'slim_measured': slim, 'slim_policy': policy, 'slim_policy_decision': 'pending',
        'body_before': body_before, 'body_after': body,
        'negative_template_matched': True,
        'chunks': len(encoders[0].positive),
    })
