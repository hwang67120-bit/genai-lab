"""입력별 검사 계약과 보정 후보 재검증. 점수는 자동 관측이며 사람의 품질 승인을 대신하지 않는다."""
import math

from genai_lab.identity_autofix import GONE, PRESENT, garment_targets, to_fix

CONTAMINATION_TAGS = ("watermark", "signature", "text", "logo")
FOOT_TAGS = {"feet", "foot", "barefoot", "footwear", "shoes", "boots", "sneakers",
             "sandals", "loafers", "high_heels", "socks", "kneehighs", "thighhighs"}
LOWER_TAGS = FOOT_TAGS | {"pants", "shorts", "skirt", "jeans", "leggings", "pantyhose",
                         "legs", "leg", "thighs", "thigh", "knees", "knee", "full_body"}


def checked_scores(scores):
    """태그 관측을 검사한다. 빈 관측·비정상 수치를 '모두 정상'으로 해석하지 않는다."""
    if not isinstance(scores, dict) or not scores:
        raise ValueError("태그 관측이 없어 자동 재검증을 진행할 수 없습니다.")
    for tag, value in scores.items():
        if not isinstance(tag, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("태그 관측의 이름 또는 수치가 잘못되었습니다.")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("태그 관측 수치는 0~1 사이의 유한한 값이어야 합니다.")
    return scores


def outside_frame(tag, framing):
    """구도 밖의 옷·발을 누락으로 판단하여 화면 안으로 강제하지 않는다."""
    excluded = LOWER_TAGS if framing.startswith("Upper-body") else FOOT_TAGS if framing.startswith("Cowboy") else set()
    return any(tag == word or tag.endswith("_" + word) for word in excluded)


def build_contract(original, garment, kinds, *, with_garment=False, framing=""):
    """원본에서 지킬 특징과 교체 의상에서 보일 항목을 분리해 고정한다."""
    checked_scores(original)
    if with_garment:
        checked_scores(garment)
    sure = [tag for tag, value in original.items() if value >= PRESENT]
    # 종류 미확인 항목은 장면/노출을 동일성으로 잘못 강조할 수 있어 자동 보정에서 제외한다.
    known = [tag for tag in sure if tag in kinds]
    retained = to_fix(known, kinds, with_garment)
    identity = [tag for tag in retained if kinds[tag]["kind"] == "body"]
    outfit = garment_targets(garment, kinds) if with_garment else [
        tag for tag in retained if kinds[tag]["kind"] == "outfit"]
    excluded = sorted(tag for tag in identity + outfit if outside_frame(tag, framing))
    identity = sorted(tag for tag in identity if tag not in excluded)
    outfit = sorted(tag for tag in outfit if tag not in excluded)
    source = dict(original)
    if with_garment:
        source.update({tag: garment[tag] for tag in outfit})
    return dict(version=1, framing=framing, with_garment=with_garment,
                identity=identity, garment=outfit,
                target_scores={tag: source[tag] for tag in identity + outfit},
                excluded_out_of_frame=excluded,
                unclassified=sorted(tag for tag in sure if tag not in kinds),
                evidence="automatic_tags", user_quality_approved=False)


def diagnose_candidate(contract, scores, *, style_weight=None):
    """관측된 누락·문자 오염 의심을 분리한다. 그림체·성별 인상·해부학은 태그로 합격 처리하지 않는다."""
    checked_scores(scores)
    missing = {group: [tag for tag in contract[group] if scores.get(tag, 0) < GONE]
               for group in ("identity", "garment")}
    contamination = [tag for tag in CONTAMINATION_TAGS if scores.get(tag, 0) >= PRESENT]
    unknown = bool(contract["unclassified"]) or not contract["identity"]
    flagged = any(missing.values()) or bool(contamination)
    return dict(state="issues_detected" if flagged else "inconclusive" if unknown else "no_tag_issue_detected",
                missing=missing, contamination_suspected=contamination,
                style=dict(state="not_applied" if style_weight == 0 else "user_review_required",
                           adapter_weight=style_weight),
                unclassified=contract["unclassified"], user_quality_approved=False,
                unassessed=["visual_identity", "style_similarity", "gender_appearance", "anatomy", "texture"])


def correction_regressions(contract, before, after):
    """보정 전에 지켰던 요구가 사라지거나 문자 오염 의심이 새로 생겼는지 반환한다."""
    checked_scores(before)
    checked_scores(after)
    lost = [tag for tag in contract["target_scores"]
            if before.get(tag, 0) >= GONE and after.get(tag, 0) < GONE]
    contamination = [tag for tag in CONTAMINATION_TAGS
                     if after.get(tag, 0) >= PRESENT and before.get(tag, 0) < PRESENT]
    return dict(newly_missing=lost, new_contamination_suspected=contamination)


def choose_correction(contract, before, after, watch):
    """같은 seed 보정 후보는 요구 손상 없이 목적 항목이 실제 감소했을 때만 선택한다."""
    regressions = correction_regressions(contract, before, after)
    required = set(contract["target_scores"])
    watch = list(dict.fromkeys(tag for tag in watch if tag in required))
    counts = [sum(scores.get(tag, 0) < GONE for tag in watch) for scores in (before, after)]
    regressed = any(regressions.values())
    improved = counts[1] < counts[0]
    return dict(use_correction=improved and not regressed,
                reason="regression" if regressed else "improved" if improved else "no_improvement",
                before_missing=counts[0], after_missing=counts[1], **regressions)


def audit_flux_prompt(tokenizer, prompt, limit):
    """FLUX Qwen 채팅 템플릿까지 포함해 실제 잘림 여부를 CPU에서 측정한다."""
    if not isinstance(prompt, str) or not prompt.strip() or limit < 1:
        raise ValueError("생성 문장과 입력 한도가 필요합니다.")
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, truncation=False)["input_ids"]
    return dict(state="blocked" if len(ids) > limit else "ready", token_count=len(ids), limit=limit,
                truncated=len(ids) > limit, delivered_text=tokenizer.decode(ids[:limit]),
                omitted_text=tokenizer.decode(ids[limit:]), evidence="local_tokenizer_chat_template")
