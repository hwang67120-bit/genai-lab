"""결과 검증기(2단계 ③, D-121): 기준 캐릭터 특징표가 결과에서 유지됐는지 항목별로 kept / changed / missing / added / unknown.

같은 추출(태그 + 픽셀)을 결과에 돌려 기준과 비교한다. 자동 결과는 사람 판정(same_character)을 돕는 근거이지 대체가 아니다.
편집 단계는 결과를 원본 좌표로 되돌린 그림(compose/aligned.png)에 원본 마스크를 대고, 자유 생성은 결과 자체를 분할해 비교한다.
문턱은 공통 하나씩(캐릭터별 조정 없음).

D-126(요청 20 후속): 재시도는 확신 있는 위반(`retry`)만. 경계값·태그와 픽셀의 엇갈림·감긴 눈·구체→일반 태그는 unknown으로 둔다.
문턱(색조 35°/50°, 새 머리색 ΔE 30·비율 0.03·채도 20)은 요청 20의 10시도에서 정했다 — 처음 보는 결과로 다시 확인해야 한다.
"""
import numpy as np

from genai_lab.character_features import (BODY_PARTS, TAG_GROUPS, TAG_THRESHOLD, WEAK_THRESHOLD, dominant_colors,
                                          eye_color, eye_colors, tag_values)

SCHEMA = "character_verification/2"
HAIR_DE = 15                    # 머리색 Lab 거리(outfit_edit.HAIR_COLOR_DE와 같은 값)
EYE_DE = 20                     # 눈 색은 마스크가 작아 조금 느슨하게
SKIN_DE = 12
BORDERLINE = 1.25               # 문턱~1.25배는 경계값(unknown) — N1-2 눈 ΔE 20.1 vs 20 재시도 오판
NOT_ANIMAL_EARS = ("pointy_ears", "long_pointy_ears", "fake_animal_ears")
CONFIDENT_TAG = 0.7             # 태그 changed는 바뀐 결과 태그가 이 점수 이상일 때만 재시도 근거
EYES_HIDDEN_TAGS = ("closed_eyes", "covered_eyes", "from_behind", "facing_away")
EYE_HUE_KEPT = 35               # 두 눈 각각 색조(Lab a·b 각도) 차이. 밝기·채도는 그림체·조명으로 흔히 바뀐다
EYE_HUE_CHANGED = 50
EYE_MIN_CHROMA = 12             # 이보다 무채색이면 색조가 뜻이 없어 ΔE로 비교
NEW_HAIR_DE = 30                # 결과 머리 군집이 기준 군집 어느 것과도 이만큼 멀고
NEW_HAIR_FRACTION = 0.03        # 이 비율 이상이고
NEW_HAIR_CHROMA = 20            # 채도가 있으면 새 머리색(added) — N2-2 청록·노랑·초록 추가


def reference_values(item, key):
    """기준 특징표 항목에서 비교할 값 목록(확정값 또는 측정값). unknown이면 None."""
    if not item or item.get("state") in (None, "unknown", "weak"):
        return None
    if key == "hair_length":
        return [item["value"]] if item.get("value") else []
    if key in ("hair_color", "eye_color", "skin"):
        return list(item.get("tags") or [])
    return list(item.get("values") or [])


def generic_of(tag, others):
    """others 중 tag의 일반형(x_hair_ornament → hair_ornament). 없으면 None."""
    return next((o for o in others if o != tag and tag.endswith("_" + o)), None)


def compare_tags(reference, result_scores, names):
    """기준 값(태그 목록) vs 결과 태그. 기준이 비어 있으면(없다고 확정) 결과에 강한 태그가 생겼을 때 added.
    결과에 기준 태그의 일반형만 있으면 unknown(kept_generic): 구체 형태는 태그로 확인 불가이지 바뀜이 아니다."""
    strong = {n for n, _ in tag_values(result_scores, names, TAG_THRESHOLD)}
    weak = {n for n, _ in tag_values(result_scores, names, WEAK_THRESHOLD)}
    if reference is None:
        return dict(status="unknown", reason="reference unknown", result=sorted(strong))
    if not reference:
        return dict(status="added" if strong else "kept", result=sorted(strong), confident=bool(strong))
    present = [v for v in reference if v in weak]
    lost = [v for v in reference if v not in weak]
    if not lost:
        return dict(status="kept", result=sorted(strong))
    generic = {v: generic_of(v, weak) for v in lost}
    replaced = sorted(strong - set(reference) - {g for g in generic.values() if g})
    if all(generic.values()) and not replaced:
        return dict(status="unknown", reason="kept_generic", lost=lost, generic=generic, result=sorted(strong), present=present)
    confident = any(float(result_scores.get(n, 0.0)) >= CONFIDENT_TAG for n in replaced)
    return dict(status="changed" if replaced else "missing", lost=lost, result=sorted(strong), present=present,
                confident=confident)


MIN_CLUSTER = 0.2               # 머리색은 군집 여러 개(바탕색·그늘·하이라이트) — 이 비율 이상인 기준 군집 중 가장 가까운 것과 비교


def compare_color(reference_colors, result_color, limit):
    """reference_colors: 기준 색 하나 또는 군집 목록. 결과 주색이 기준 군집(비율 MIN_CLUSTER 이상) 중 하나와 limit 안이면 kept,
    limit×BORDERLINE 밖이면 changed, 그 사이는 경계값(unknown)."""
    if isinstance(reference_colors, dict):
        reference_colors = [reference_colors]
    candidates = [c for c in (reference_colors or []) if c and c.get("fraction", 1.0) >= MIN_CLUSTER] or list(reference_colors or [])
    if not candidates or not result_color:
        return dict(status="unknown", reason="color not measurable")
    deltas = [float(np.linalg.norm(np.array(c["lab"]) - np.array(result_color["lab"]))) for c in candidates]
    best = int(np.argmin(deltas))
    delta = deltas[best]
    status = "kept" if delta <= limit else ("changed" if delta > limit * BORDERLINE else "unknown")
    return dict(status=status, delta_e=round(delta, 1), limit=limit, reason="borderline" if status == "unknown" else None,
                reference_rgb=candidates[best]["rgb"], result_rgb=result_color["rgb"], confident=status == "changed")


def new_hair_colors(reference_palette, result_palette):
    """기준 군집 어느 것과도 먼, 채도 있는 결과 군집(새 머리색 후보)."""
    if not reference_palette or not result_palette:
        return []
    found = []
    for c in result_palette:
        if c["fraction"] < NEW_HAIR_FRACTION or float(np.hypot(c["lab"][1], c["lab"][2])) < NEW_HAIR_CHROMA:
            continue
        nearest = min(float(np.linalg.norm(np.array(c["lab"]) - np.array(r["lab"]))) for r in reference_palette)
        if nearest > NEW_HAIR_DE:
            found.append(dict(rgb=c["rgb"], fraction=c["fraction"], nearest_delta_e=round(nearest, 1)))
    return found


def hue_gap(a, b):
    """두 눈 색의 차이(도). 둘 다 채도가 있으면 색조 각도, 아니면 ΔE를 같은 눈금으로 쓴다."""
    if a["chroma"] >= EYE_MIN_CHROMA and b["chroma"] >= EYE_MIN_CHROMA:
        gap = abs(a["hue"] - b["hue"]) % 360
        return min(gap, 360 - gap)
    return float(np.linalg.norm(np.array(a["lab"]) - np.array(b["lab"])))


def heterochromia(eyes, tags=()):
    measured = [e for e in eyes or () if e]
    return "heterochromia" in (tags or ()) or (len(measured) == 2 and hue_gap(*measured) > EYE_HUE_CHANGED)


def compare_eyes(reference_eyes, reference_tags, result_eyes):
    """두 눈 각각 비교(좌우 짝은 가까운 쪽으로 — 시점 반전 대비). 오드아이가 한 색이 되면 changed(heterochromia_lost)."""
    ref = [e for e in reference_eyes or () if e]
    res = [e for e in result_eyes or () if e]
    if not ref or not res:
        return dict(status="unknown", reason="eye not measurable")
    hetero = heterochromia(reference_eyes, reference_tags)
    if hetero and len(res) == 2 and hue_gap(*res) <= EYE_HUE_KEPT:
        return dict(status="changed", reason="heterochromia_lost", confident=True,
                    reference_rgb=[e["rgb"] for e in ref], result_rgb=[e["rgb"] for e in res])
    if len(ref) == 2 and len(res) == 2:
        gap = min(max(hue_gap(ref[0], res[0]), hue_gap(ref[1], res[1])), max(hue_gap(ref[0], res[1]), hue_gap(ref[1], res[0])))
    else:
        gap = max(min(hue_gap(r, e) for r in ref) for e in res)
    status = "kept" if gap <= EYE_HUE_KEPT else ("changed" if gap > EYE_HUE_CHANGED else "unknown")
    if status == "kept" and hetero and len(res) < 2:
        status = "unknown"            # 한쪽만 보이면 오드아이 유지는 확인 불가
    return dict(status=status, hue_gap=round(gap, 1), reason="borderline" if status == "unknown" else None,
                confident=status == "changed", reference_rgb=[e["rgb"] for e in ref], result_rgb=[e["rgb"] for e in res])


def combine(tags, pixels):
    """눈 색: 픽셀이 태그를 덮어쓰지 않는다. 한쪽이 근거 없음(unknown·태그 missing)이면 다른 쪽, 엇갈리면 unknown."""
    t, p = tags["status"], pixels["status"]
    if p == "unknown":
        return t, tags.get("confident", False)
    if t in ("unknown", "missing") or t == p:
        return p, pixels.get("confident", False) or (t == p and tags.get("confident", False))
    return "unknown", False


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


def verify_features(reference, result_scores, result_image=None, parts=None, skip=None):
    """reference: 특징표(features.json), result_scores: 결과 태그 점수, result_image/parts: 픽셀 비교용.
    skip: {항목: 이유} — 요청과 충돌해 검사하지 않는 항목(눈 감기 요청이면 눈 색). 반환: 항목별 상태, 요약, 재시도 근거(retry)."""
    items = {}
    ref_items = (reference or {}).get("items", {})
    for key, names in TAG_GROUPS.items():
        items[key] = compare_tags(reference_values(ref_items.get(key), key), result_scores, names)
    hidden = [t for t in EYES_HIDDEN_TAGS if float(result_scores.get(t, 0.0)) >= TAG_THRESHOLD]
    # 픽셀: 머리색·눈 색·피부색
    for key, limit in (("hair_color", HAIR_DE), ("eye_color", EYE_DE), ("skin", SKIN_DE)):
        ref = ref_items.get(key) or {}
        measurable = result_image is not None and parts is not None
        if key == "eye_color":
            if ref.get("eyes") is not None:          # 두 눈 따로(D-126). 이전 특징표는 한 색 비교로
                pixels = (compare_eyes(ref["eyes"], ref.get("tags"), eye_colors(result_image, parts["eyes"]))
                          if measurable and "eyes" in parts else dict(status="unknown", reason="color not measurable"))
            else:
                pixels = compare_color(ref.get("color"), eye_color(result_image, parts["eyes"]) if measurable and "eyes" in parts else None, limit)
            items[key]["pixels"] = pixels
            items[key]["tags_status"] = items[key]["status"]
            items[key]["status"], items[key]["confident"] = combine(items[key], pixels)
            if hidden:
                items[key].update(status="unknown", confident=False, reason="eyes hidden: " + ", ".join(hidden))
            continue
        result_color, result_palette = None, None
        if measurable:
            if key == "hair_color" and "hair" in parts:
                from genai_lab.character_features import hair_mask_without_background
                mask = hair_mask_without_background(result_image, parts["hair"])
                found = dominant_colors(result_image, mask)
                result_color = found[0] if found else None
                result_palette = dominant_colors(result_image, mask, k=5)
            elif key == "skin":
                from genai_lab.character_features import skin_color
                result_color = skin_color(result_image, parts)
        reference_color = ref.get("colors") if key == "hair_color" else ref.get("color")
        items[key]["pixels"] = compare_color(reference_color, result_color, limit)
        if items[key]["pixels"]["status"] != "unknown":                      # 머리·피부 색은 픽셀이 우선(태그는 노이즈가 많다)
            items[key]["tags_status"] = items[key]["status"]
            items[key]["status"] = items[key]["pixels"]["status"]
            items[key]["confident"] = items[key]["pixels"]["confident"]
        if key == "hair_color":
            added = new_hair_colors(ref.get("palette") or ref.get("colors"), result_palette)
            items[key]["new_colors"] = added
            if added and items[key]["status"] != "changed":
                items[key].update(status="added", confident=True)
    # 부위·성별
    parts_ref = ref_items.get("body_parts") or {}
    found = body_parts_in_result(result_scores)
    if parts_ref.get("state") != "measured" or parts_ref.get("values") is None:
        items["body_parts"] = dict(status="unknown", reason="reference unknown", result=list(found))
    else:
        expected = set(parts_ref["values"])
        missing, added = sorted(expected - set(found)), sorted(set(found) - expected)
        items["body_parts"] = dict(status="missing" if missing else ("added" if added else "kept"), missing=missing, added=added,
                                   result=list(found), confident=bool(added) and not missing,
                                   note="개수·모양 변화는 태그로 잡지 못한다(사람 판정)")
    gender_ref = ref_items.get("gender") or {}
    result_gender = gender_in_result(result_scores)
    if gender_ref.get("state") != "measured" or not gender_ref.get("value") or gender_ref["value"] == "unspecified":
        items["gender"] = dict(status="unknown", result=result_gender)
    else:
        changed = result_gender not in (gender_ref["value"], "unspecified")
        items["gender"] = dict(status="kept" if result_gender == gender_ref["value"] else ("changed" if changed else "unknown"),
                               result=result_gender, confident=changed)
    for key, reason in (skip or {}).items():
        if key in items:
            items[key] = dict(status="unknown", reason=reason, skipped=True, observed=items[key])
    summary = {s: sorted(k for k, v in items.items() if v["status"] == s) for s in ("kept", "changed", "missing", "added", "unknown")}
    retry = sorted(k for k, v in items.items() if v["status"] in ("changed", "added") and v.get("confident"))
    return dict(schema=SCHEMA, items=items, summary=summary, retry=retry,
                note="자동 비교는 same_character 사람 판정의 근거다. unknown은 비교하지 못한 것이지 변화가 아니다. "
                     "retry는 확신 있는 위반만(경계값·엇갈림·가림 제외).")


def retry_reasons(report):
    """재시도 근거 항목. retry가 없는 보고(측정 실패 등)는 요약의 changed·added로 본다."""
    if "retry" in report:
        return list(report["retry"])
    summary = report.get("summary") or {}
    return sorted(set(summary.get("changed") or ()) | set(summary.get("added") or ()))


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
