"""진단 B의 품질 문구를 명시적으로 적용한다. 기본 부정 문구는 변경하지 않는다."""
from dataclasses import replace
from genai_lab.onepass_prompt import plan_prompt_chunks

IDENTITY_TERMS = frozenset(("different character", "different hairstyle", "different hair color",
                            "different eye color", "missing character features"))
QUALITY_POSITIVE = ("masterpiece", "best quality", "very aesthetic", "absurdres")
QUALITY_NEGATIVE = ("error", "extra", "missing", "jpeg artifacts", "unfinished", "displeasing",
                    "oldest", "early", "artistic error", "username", "scan")


def quality_strings(positive, negative):
    pos = [t.strip() for t in positive.split(",") if t.strip() not in ("best quality", "coherent anatomy")]
    neg = [t.strip() for t in negative.split(",") if t.strip() not in IDENTITY_TERMS]
    pos.extend(QUALITY_POSITIVE)
    neg.extend(t for t in QUALITY_NEGATIVE if t not in neg)
    return ", ".join(pos), ", ".join(neg)


def apply_quality_format(prompt, tokenizers):
    positive, negative = quality_strings(prompt.positive, prompt.negative)
    encoders = plan_prompt_chunks(positive, negative, tokenizers)
    return replace(prompt, positive=positive, negative=negative,
                   encoders=encoders,
                   rules={**prompt.rules, "chunks":len(encoders[0].positive), "quality_format":"diagnostic_2b", "quality_default_decision":"pending",
                          "negative_template_matched":False, "negative_change":"explicit_quality_option"})
