"""Reserve all approved core conditions before optional finishing tags."""
from scripts.generation_inputs import _token_count, _tokenizer_limit
from genai_lab.reference_tag_policy import normalize_tag


class PromptBudgetError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            f"필수 캐릭터·의상 조건이 토큰 한도를 초과했습니다. "
            f"필요={report['required_token_counts']}, 한도={report['tokenizer_limits']}. "
            "자동 삭제하지 않았습니다. 상세 수정에서 조건을 검토하세요.")


def unique_tags(values):
    if isinstance(values, str):
        raise TypeError('태그 목록이 필요합니다.')
    return list(dict.fromkeys(normalize_tag(part)
        for value in values for part in value.split(',') if part.strip()))


def _character_priority(tag):
    words = normalize_tag(tag).split()
    last = words[-1] if words else ""
    if (last.endswith("boy") or last.endswith("girl")) and last[:1].isdigit():
        return 0
    return {
        "eyes": 1,
        "hair": 2,
        "ears": 3,
        "tail": 4,
        "tails": 4,
    }.get(last, 5)


def build_reference_prompt(tokenizers, *, core_character, core_outfit, framing=(),
                           optional=(), long_prompt_settings=None):
    tokenizers = tuple(tokenizers)
    if not tokenizers or any(t is None for t in tokenizers):
        raise ValueError('실제 생성용 토크나이저가 필요합니다.')
    character, outfit, frame = map(
        unique_tags, (core_character, core_outfit, framing))
    character = sorted(character, key=_character_priority)
    if not outfit:
        raise ValueError('승인된 의상 조건이 없습니다.')
    base = ', '.join([*unique_tags([*character, *frame]), 'wearing ' + ', '.join(outfit)])
    limits = tuple(_tokenizer_limit(t) for t in tokenizers)
    def counts(text):
        return tuple(_token_count(t, text) for t in tokenizers)
    def fits(text):
        return all(n <= limit for n, limit in zip(counts(text), limits))
    report = {'required_prompt': base, 'required_token_counts': counts(base),
              'tokenizer_limits': limits, 'omitted_optional': []}
    report["priority_order"] = (
        "gender", "eyes", "hair", "ears", "tail", "other_character",
        "framing", "garment", "optional")
    long_settings = None
    if long_prompt_settings is not None:
        from genai_lab.sdxl_long_prompt import resolve_long_prompt_settings
        long_settings = resolve_long_prompt_settings(long_prompt_settings)
    if not fits(base) and not (long_settings and long_settings["enabled"]):
        raise PromptBudgetError(report)
    kept, omitted = [], []
    seen = set(character + outfit + frame)
    extras = []
    for tag in unique_tags(optional):
        if tag in seen:
            continue
        seen.add(tag)
        extras.append(tag)
        candidate = ', '.join([base, *kept, tag])
        long_fits = False
        if long_settings and long_settings["enabled"]:
            try:
                from genai_lab.sdxl_long_prompt import plan_phrase_chunks
                plan_phrase_chunks(candidate, tokenizers, long_settings)
                long_fits = True
            except ValueError:
                long_fits = False
        if fits(candidate) or long_fits:
            kept.append(tag)
        else:
            omitted.append(tag)
    original = ', '.join([base, *extras])
    effective = ', '.join([base, *kept])
    report.update(original=original, effective=effective, truncated=bool(omitted),
                  original_token_counts=counts(original), effective_token_counts=counts(effective),
                  omitted_optional=omitted, retained_segment_count=len(effective.split(',')),
                  source_segment_count=len(original.split(',')))
    if not fits(effective):
        from genai_lab.sdxl_long_prompt import plan_phrase_chunks
        report["long_prompt"] = plan_phrase_chunks(
            effective, tokenizers, long_settings)
    else:
        report["long_prompt"] = {
            "policy_version": "single_clip_context_v1",
            "chunk_count": 1,
        }
    return effective, report


def load_reference_tokenizers(config):
    """Load tokenizer assets only; never load SDXL weights or allocate CUDA."""
    from transformers import CLIPTokenizer
    model = config['model']
    options = {'cache_dir': str(model['cache_dir'])}
    if model.get('revision'):
        options['revision'] = model['revision']
    return tuple(CLIPTokenizer.from_pretrained(model['id'], subfolder=folder, **options)
                 for folder in ('tokenizer', 'tokenizer_2'))
