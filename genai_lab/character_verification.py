"""결과 검증기(2단계 ③, D-121): 기준 캐릭터 특징표가 결과에서 유지됐는지 항목별로 kept / changed / missing / added / unknown.

같은 추출(태그 + 픽셀)을 결과에 돌려 기준과 비교한다. 자동 결과는 사람 판정(same_character)을 돕는 근거이지 대체가 아니다.
픽셀 비교는 결과를 원본 좌표로 되돌린 그림(compose/aligned.png)에 원본 마스크를 그대로 대는 방식이라 자세·구도가 바뀌지 않은
편집 단계에서만 뜻이 있다. 새 구도(자유 생성)에서는 결과 쪽 분할이 필요하며, 그때까지 픽셀 항목은 unknown으로 남긴다.
문턱은 공통 하나씩(캐릭터별 조정 없음).
"""
import numpy as np

from genai_lab.character_features import (BODY_PARTS, TAG_GROUPS, TAG_THRESHOLD, WEAK_THRESHOLD, dominant_colors,
                                          eye_color, tag_values)

SCHEMA = "character_verification/1"
HAIR_DE = 15                    # 머리색 Lab 거리(outfit_edit.HAIR_COLOR_DE와 같은 값)
EYE_DE = 20                     # 눈 색은 마스크가 작아 조금 느슨하게
SKIN_DE = 12
NOT_ANIMAL_EARS = ("pointy_ears", "long_pointy_ears", "fake_animal_ears")


def reference_values(item, key):
    """기준 특징표 항목에서 비교할 값 목록(확정값 또는 측정값). unknown이면 None."""
    if not item or item.get("state") in (None, "unknown", "weak"):
        return None
    if key == "hair_length":
        return [item["value"]] if item.get("value") else []
    if key in ("hair_color", "eye_color", "skin"):
        return list(item.get("tags") or [])
    return list(item.get("values") or [])


def compare_tags(reference, result_scores, names):
    """기준 값(태그 목록) vs 결과 태그. 기준이 비어 있으면(없다고 확정) 결과에 강한 태그가 생겼을 때 added."""
    strong = {n for n, _ in tag_values(result_scores, names, TAG_THRESHOLD)}
    weak = {n for n, _ in tag_values(result_scores, names, WEAK_THRESHOLD)}
    if reference is None:
        return dict(status="unknown", reason="reference unknown", result=sorted(strong))
    if not reference:
        return dict(status="added" if strong else "kept", result=sorted(strong))
    present = [v for v in reference if v in weak]
    lost = [v for v in reference if v not in weak]
    if not lost:
        return dict(status="kept", result=sorted(strong))
    replaced = sorted(strong - set(reference))
    return dict(status="changed" if replaced else "missing", lost=lost, result=sorted(strong), present=present)


MIN_CLUSTER = 0.2               # 머리색은 군집 여러 개(바탕색·그늘·하이라이트) — 이 비율 이상인 기준 군집 중 가장 가까운 것과 비교


def compare_color(reference_colors, result_color, limit):
    """reference_colors: 기준 색 하나 또는 군집 목록. 결과 주색이 기준 군집(비율 MIN_CLUSTER 이상) 중 하나와 limit 안이면 kept."""
    if isinstance(reference_colors, dict):
        reference_colors = [reference_colors]
    candidates = [c for c in (reference_colors or []) if c and c.get("fraction", 1.0) >= MIN_CLUSTER] or list(reference_colors or [])
    if not candidates or not result_color:
        return dict(status="unknown", reason="color not measurable")
    deltas = [float(np.linalg.norm(np.array(c["lab"]) - np.array(result_color["lab"]))) for c in candidates]
    best = int(np.argmin(deltas))
    return dict(status="kept" if deltas[best] <= limit else "changed", delta_e=round(deltas[best], 1), limit=limit,
                reference_rgb=candidates[best]["rgb"], result_rgb=result_color["rgb"])


def body_parts_in_result(scores):
    """결과 태그에서 동물 귀·꼬리·날개(guessed_body_parts와 같은 규칙)."""
    def hit(exact, suffix):
        return any(v >= TAG_THRESHOLD and (k == exact or k.endswith(suffix)) and k not in NOT_ANIMAL_EARS for k, v in scores.items())
    return tuple(p for p, (exact, suffix) in zip(BODY_PARTS, (("animal_ears", "_ears"), ("tail", "_tail"), ("wings", "_wings")))
                 if hit(exact, suffix))


GENDER_CONFIDENT = 0.7          # 결과 성별이 이 점수 이상일 때만 '바뀜'을 말한다(태그 성별은 틀릴 수 있다, D-106)


def gender_in_result(scores):
    boy, girl = float(scores.get("1boy", 0.0)), float(scores.get("1girl", 0.0))
    if max(boy, girl) < GENDER_CONFIDENT:
        return "unspecified"
    return "male" if boy > girl else "female"


def verify_features(reference, result_scores, result_image=None, parts=None):
    """reference: 특징표(features.json), result_scores: 결과 태그 점수, result_image/parts: 픽셀 비교용(원본 좌표 결과 + 원본 마스크).
    반환: 항목별 상태와 요약."""
    items = {}
    ref_items = (reference or {}).get("items", {})
    for key, names in TAG_GROUPS.items():
        items[key] = compare_tags(reference_values(ref_items.get(key), key), result_scores, names)
    # 픽셀: 머리색·눈 색·피부색
    for key, limit, pick in (("hair_color", HAIR_DE, "colors"), ("eye_color", EYE_DE, "color"), ("skin", SKIN_DE, "color")):
        ref = ref_items.get(key) or {}
        reference_color = ref.get("colors") if pick == "colors" else ref.get("color")
        result_color = None
        if result_image is not None and parts is not None:
            if key == "hair_color" and "hair" in parts:
                from genai_lab.character_features import hair_mask_without_background
                found = dominant_colors(result_image, hair_mask_without_background(result_image, parts["hair"]))
                result_color = found[0] if found else None
            elif key == "eye_color" and "eyes" in parts:
                result_color = eye_color(result_image, parts["eyes"])
            elif key == "skin":
                from genai_lab.character_features import skin_color
                result_color = skin_color(result_image, parts)
        items[key]["pixels"] = compare_color(reference_color, result_color, limit)
        if items[key]["pixels"]["status"] != "unknown":                      # 색은 픽셀 비교가 우선(태그는 노이즈가 많다)
            items[key]["tags_status"] = items[key]["status"]
            items[key]["status"] = items[key]["pixels"]["status"]
    # 부위·성별
    parts_ref = ref_items.get("body_parts") or {}
    found = body_parts_in_result(result_scores)
    if parts_ref.get("state") != "measured" or parts_ref.get("values") is None:
        items["body_parts"] = dict(status="unknown", reason="reference unknown", result=list(found))
    else:
        expected = set(parts_ref["values"])
        missing, added = sorted(expected - set(found)), sorted(set(found) - expected)
        items["body_parts"] = dict(status="missing" if missing else ("added" if added else "kept"), missing=missing, added=added,
                                   result=list(found), note="개수·모양 변화는 태그로 잡지 못한다(사람 판정)")
    gender_ref = ref_items.get("gender") or {}
    result_gender = gender_in_result(result_scores)
    if gender_ref.get("state") != "measured" or not gender_ref.get("value") or gender_ref["value"] == "unspecified":
        items["gender"] = dict(status="unknown", result=result_gender)
    else:
        items["gender"] = dict(status="kept" if result_gender == gender_ref["value"] else ("unknown" if result_gender == "unspecified" else "changed"),
                               result=result_gender)
    summary = {s: sorted(k for k, v in items.items() if v["status"] == s) for s in ("kept", "changed", "missing", "added", "unknown")}
    return dict(schema=SCHEMA, items=items, summary=summary,
                note="자동 비교는 same_character 사람 판정의 근거다. unknown은 비교하지 못한 것이지 변화가 아니다.")


def measure_result_scores(image, runtime):
    """결과 그림의 태그 점수(CPU 태거). 실패하면 None."""
    from PIL import Image
    try:
        from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
        settings = ClothingDesignAnalysisSettings(cache_dir=str(runtime.model_cache), local_files_only=True)
        with WdTagSession(settings) as tagger:
            return dict(tagger.analyze(Image.fromarray(np.asarray(image))).raw_general_scores)
    except Exception:
        return None


def verify_result(reference, image_path, parts_path, runtime):
    """파이프라인용: 결과(원본 좌표) 그림 + 원본 마스크로 검증. 태거가 없으면 status unmeasured."""
    from PIL import Image
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    scores = measure_result_scores(image, runtime)
    if scores is None:
        return dict(schema=SCHEMA, status="unmeasured", reason="tagger unavailable", items={}, summary={})
    try:
        parts = {k: v.astype(bool) for k, v in np.load(parts_path).items()}
    except (OSError, ValueError):
        parts = None
    return dict(verify_features(reference, scores, image, parts), status="measured")
