"""기준 캐릭터 특징표 추출(2단계 1번, D-118 / docs/CHARACTER_FEATURES_DESIGN.md).

입력: 캐릭터 캔버스(RGB), 부위 분할(parts.npz: hair·face·eyes·headwear·eyewear·…), 태그 점수, 이미 확정된 성별·부위.
출력: 항목마다 값·출처(tags / pixels / user)·상태(measured / unknown)를 적은 특징표. 추측해서 채우지 않는다 — 재지 못하면 unknown.
여기서는 추출만 한다. 사람 확정·저장(확인 창), 생성 조건, 결과 검증은 다음 단계에서 이 구조 위에 올린다.
캐릭터별 문턱 조정은 없다(D-104).
"""
import json
from pathlib import Path

import numpy as np

SCHEMA = "character_features/1"
TAG_THRESHOLD = 0.5             # 부위 미리 채움(guessed_body_parts)과 같은 문턱 = measured
WEAK_THRESHOLD = 0.35           # 이 구간(0.35~0.5)은 약한 후보로만 기록한다(조건에는 안 쓰고 사람 확정 때 보여 줌).
                                # 근거: 10명 CPU 측정(rules18-cpu/character-tag-tops.json) — N2 ahoge 0.35, S07 white_hair 0.36, N4 very_long_hair 0.39 등 맞는 값이 이 구간에 있었다
MIN_PIXELS = 50                 # 이보다 적으면 색을 재지 않는다

HAIR_LENGTH_TAGS = ("very_short_hair", "short_hair", "medium_hair", "long_hair", "very_long_hair", "absurdly_long_hair")
HAIR_STYLE_TAGS = ("twintails", "low_twintails", "short_twintails", "ponytail", "side_ponytail", "high_ponytail", "braid",
                   "twin_braids", "single_braid", "french_braid", "drill_hair", "bob_cut", "hime_cut", "ahoge", "bangs",
                   "blunt_bangs", "swept_bangs", "parted_bangs", "hair_between_eyes", "hair_over_one_eye", "sidelocks",
                   "hair_bun", "double_bun", "messy_hair", "wavy_hair", "curly_hair", "straight_hair", "hair_intakes",
                   "antenna_hair", "spiked_hair", "asymmetrical_hair", "hair_flaps", "half_updo", "low_ponytail")
HAIR_COLOR_TAGS = ("blonde_hair", "brown_hair", "black_hair", "blue_hair", "light_blue_hair", "aqua_hair", "green_hair",
                   "light_green_hair", "pink_hair", "purple_hair", "light_purple_hair", "red_hair", "white_hair", "silver_hair",
                   "grey_hair", "orange_hair", "dark_blue_hair", "light_brown_hair", "multicolored_hair", "gradient_hair",
                   "streaked_hair", "two-tone_hair", "colored_inner_hair", "colored_tips")
EYE_COLOR_TAGS = ("blue_eyes", "aqua_eyes", "red_eyes", "green_eyes", "yellow_eyes", "purple_eyes", "brown_eyes",
                  "orange_eyes", "pink_eyes", "grey_eyes", "black_eyes", "white_eyes", "heterochromia", "light_blue_eyes",
                  "golden_eyes", "amber_eyes")
HEAD_ACCESSORY_TAGS = ("hair_ornament", "hairclip", "hair_ribbon", "hair_bow", "hairband", "headband", "hair_flower",
                       "headphones", "headset", "animal_ear_headphones", "hat", "beret", "cap", "baseball_cap", "witch_hat",
                       "hood_up", "horns", "halo", "maid_headdress", "glasses", "eyewear_on_head", "sunglasses", "earrings",
                       "crown", "tiara", "hair_scrunchie", "x_hair_ornament", "hairpin", "hair_tubes", "veil", "mask",
                       "eyepatch", "goggles", "headgear", "helmet", "flower")
SKIN_TAGS = ("dark_skin", "very_dark_skin", "pale_skin", "tan", "colored_skin", "blue_skin", "grey_skin", "green_skin")
BODY_PARTS = ("ears", "tail", "wings")
TAG_GROUPS = {"hair_length": HAIR_LENGTH_TAGS, "hair_style": HAIR_STYLE_TAGS, "hair_color": HAIR_COLOR_TAGS,
              "eye_color": EYE_COLOR_TAGS, "head_accessories": HEAD_ACCESSORY_TAGS, "skin": SKIN_TAGS}
TAG_ITEMS = tuple(TAG_GROUPS)
LABELS = {"hair_length": "머리 길이", "hair_style": "머리 모양", "hair_color": "머리색", "eye_color": "눈 색",
          "head_accessories": "머리 장식", "skin": "피부"}
# 고유 형태 메모(D-126 ④): 태그·픽셀이 못 재는 형태를 사람이 짧게 적는다("heart-shaped tail tip"). 자동 추출·자동 판정 없음.
# 태그 항목이 아니므로 TAG_ITEMS에 넣지 않는다. 글 상자에서는 ';'로 나눈다(메모 안에 쉼표가 들어갈 수 있다).
DISTINCTIVE = "distinctive"
DISTINCTIVE_LABEL = "고유 형태"
MAX_DISTINCTIVE = 4


UNKNOWN_WORDS = ("모름", "확인 불가", "?", "unknown")
NONE_WORDS = ("없음", "none", "-")


def tag_features(scores):
    """확인 창 미리 채움용: 태그만으로 뽑은 항목(값 목록 + 약한 후보). 분할이 없는 시점(시작 전)에 쓴다.
    태그가 아무것도 못 찾은 항목은 None(모름)이다 — 자동 분석이 '없다'고 확정하지 않는다(검토 5번)."""
    items = {}
    for key, names in TAG_GROUPS.items():
        strong = [n for n, _ in tag_values(scores, names)]
        if key == "hair_length":
            strong = strong[:1]
        items[key] = dict(values=strong or None, weak=[n for n, _ in weak_values(scores, names)])
    return items


def features_text(items):
    """사람이 고치는 글 상자 내용: 한 줄에 한 항목, '라벨: 값, 값'. 확정 없음은 '없음', 확인 불가는 '모름'.
    마지막 줄은 고유 형태 메모('; '로 나눔)."""
    lines = []
    for key in TAG_ITEMS:
        values = (items.get(key) or {}).get("values")
        text = "모름" if values is None else (", ".join(values) if values else "없음")
        lines.append(f"{LABELS[key]}: {text}")
    notes = (items.get(DISTINCTIVE) or {}).get("values")
    lines.append(f"{DISTINCTIVE_LABEL}: " + ("모름" if notes is None else ("; ".join(notes) if notes else "없음")))
    return "\n".join(lines)


def parse_features_text(text):
    """글 상자 → {항목: [값] | None}. 모르는 라벨이나 빠진 항목은 오류.
    '없음' = [] (사용자가 없다고 확정), '모름'·빈 칸 = None (확인 불가, 비교·조건에 쓰지 않음). 두 창(편집·자유 생성)이 같은 뜻으로 쓴다."""
    by_label = {v: k for k, v in LABELS.items()}
    items = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"'라벨: 값' 형식이 아닙니다: {line}")
        label, _, rest = line.partition(":")
        if label.strip() == DISTINCTIVE_LABEL:
            notes = [v.strip() for v in rest.split(";") if v.strip()]
            if not notes or all(v in UNKNOWN_WORDS for v in notes):
                items[DISTINCTIVE] = None
            elif any(v in UNKNOWN_WORDS for v in notes):
                raise ValueError(f"'{DISTINCTIVE_LABEL}'에 '모름'과 메모를 함께 쓸 수 없습니다.")
            else:
                notes = [v for v in notes if v not in NONE_WORDS]
                if len(notes) > MAX_DISTINCTIVE or any(len(v) > 60 for v in notes):
                    raise ValueError(f"'{DISTINCTIVE_LABEL}'은 {MAX_DISTINCTIVE}개까지, 하나에 60자 이내로 써 주세요.")
                items[DISTINCTIVE] = notes
            continue
        key = by_label.get(label.strip())
        if key is None:
            raise ValueError(f"모르는 항목입니다: {label.strip()}")
        raw_values = [v.strip() for v in rest.replace("，", ",").split(",") if v.strip()]
        if not raw_values or all(v in UNKNOWN_WORDS for v in raw_values):
            items[key] = None
            continue
        if any(v in UNKNOWN_WORDS for v in raw_values):
            raise ValueError(f"'{LABELS[key]}'에 '모름'과 값을 함께 쓸 수 없습니다.")
        values = [v.replace(" ", "_") for v in raw_values if v not in NONE_WORDS]
        items[key] = values
    missing = [LABELS[k] for k in TAG_ITEMS if k not in items]
    if missing:
        raise ValueError("빠진 항목: " + ", ".join(missing))
    items.setdefault(DISTINCTIVE, None)       # 메모 줄이 없던 이전 글 상자 = 모름
    return items


def weak_note(items):
    """확실하지 않은 후보를 한 줄로(없으면 빈 문자열)."""
    parts = [f"{LABELS[k]} {', '.join(v['weak'])}" for k, v in items.items() if v.get("weak")]
    return ("확실하지 않은 후보(맞으면 위에 적어 주세요): " + " · ".join(parts)) if parts else ""


def tag_values(scores, names, threshold=TAG_THRESHOLD):
    """문턱 이상인 태그를 점수 내림차순으로. [(이름, 점수)]"""
    found = [(n, round(float(scores.get(n, 0.0)), 3)) for n in names if float(scores.get(n, 0.0)) >= threshold]
    return sorted(found, key=lambda kv: -kv[1])


def weak_values(scores, names):
    return [(n, v) for n, v in tag_values(scores, names, WEAK_THRESHOLD) if v < TAG_THRESHOLD]


def to_lab(rgb):
    import cv2
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def hair_mask_without_background(image, hair):
    """분할기가 머리카락에 넣은 원본 배경(그림 테두리 색, D-115와 같은 규칙)을 뺀 머리 마스크."""
    from genai_lab.outfit_edit import original_background
    try:
        return hair & ~original_background(np.ascontiguousarray(image), frame_fallback=True)
    except Exception:
        return hair


def dominant_colors(image, mask, k=3):
    """마스크 안 픽셀의 Lab k-means 군집(비율 내림차순). 픽셀이 적으면 []."""
    import cv2
    pixels = to_lab(image)[mask]
    if len(pixels) < MIN_PIXELS:
        return []
    k = min(k, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    fractions = np.bincount(labels.ravel(), minlength=k) / len(labels)
    order = np.argsort(-fractions)
    return [dict(lab=[round(float(v), 1) for v in centers[i]], rgb=lab_to_rgb(centers[i]), fraction=round(float(fractions[i]), 3))
            for i in order]


def lab_to_rgb(lab):
    import cv2
    raw = np.array([[[lab[0] * 255 / 100, lab[1] + 128, lab[2] + 128]]], np.float32).clip(0, 255).astype(np.uint8)
    return [int(v) for v in cv2.cvtColor(raw, cv2.COLOR_LAB2RGB)[0, 0]]


def eye_color(image, eyes):
    """눈 영역에서 흰자(밝고 무채색)와 동공·선(어두움)을 뺀 홍채 후보 픽셀의 중앙값. 부족하면 None."""
    lab = to_lab(image)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    iris = eyes & (lab[..., 0] > 15) & (lab[..., 0] < 90) & (chroma > 10)
    if int(iris.sum()) < 20:
        return None
    center = np.median(lab[iris], 0)
    return dict(lab=[round(float(v), 1) for v in center], rgb=lab_to_rgb(center), pixels=int(iris.sum()))


COLORED_CLUSTER = 0.08          # 머리 군집 중 이 비율 이상이고
COLORED_CHROMA = 20             # 채도가 있는 색(N2의 옅은 분홍)은 결과에서 비중이 줄었는지 본다(D-127)
SHARE_DE = 15                   # 군집 중심에서 이 거리 안의 머리 픽셀 비중


def color_share(image, mask, lab):
    """마스크 안 픽셀 중 lab에서 SHARE_DE 안인 비중. 픽셀이 적으면 None."""
    pixels = to_lab(image)[mask]
    if len(pixels) < MIN_PIXELS:
        return None
    return round(float((np.linalg.norm(pixels - np.array(lab, np.float32), axis=1) <= SHARE_DE).mean()), 3)


def colored_hair(image, mask, palette):
    """기준 머리의 '색 있는' 군집과 그 비중(감소 관찰용)."""
    return [dict(lab=c["lab"], rgb=c["rgb"], share=color_share(image, mask, c["lab"])) for c in palette
            if c["fraction"] >= COLORED_CLUSTER and float(np.hypot(c["lab"][1], c["lab"][2])) >= COLORED_CHROMA]


EYE_TOP_CHROMA = 60             # 눈 하나 안에서 채도 상위 40%만 홍채로 본다(피부·눈꺼풀이 섞이면 색이 탁해진다, D-126 ①)


def eye_colors(image, eyes):
    """두 눈을 따로(마스크의 큰 연결 요소 두 개, 왼쪽부터). 눈마다 홍채 색(Lab 중앙값·색조·채도) 또는 None(픽셀 부족)."""
    import cv2
    lab = to_lab(image)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(eyes).astype(np.uint8), connectivity=8)
    biggest = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]
    found = []
    for i in sorted(biggest, key=lambda i: stats[i, cv2.CC_STAT_LEFT]):
        iris = (labels == i) & (lab[..., 0] > 15) & (lab[..., 0] < 90) & (chroma > 10)
        if int(iris.sum()) < 20:
            found.append(None)
            continue
        iris &= chroma >= np.percentile(chroma[iris], EYE_TOP_CHROMA)
        center = np.median(lab[iris], 0)
        found.append(dict(lab=[round(float(v), 1) for v in center], rgb=lab_to_rgb(center), pixels=int(iris.sum()),
                          hue=round(float(np.degrees(np.arctan2(center[2], center[1]))), 1),
                          chroma=round(float(np.hypot(center[1], center[2])), 1)))
    return found


def skin_color(image, parts):
    """얼굴에서 눈·코·입을 뺀 밝은 픽셀 군집 중 가장 큰 것. 부족하면 None."""
    face = parts.get("face")
    if face is None:
        return None
    region = face.copy()
    for key in ("eyes", "nose", "mouth", "eyewear"):
        if key in parts:
            region &= ~parts[key]
    region &= to_lab(image)[..., 0] > 30
    colors = dominant_colors(image, region, k=3)
    return dict(colors[0], pixels=int(region.sum())) if colors else None


def _item(state, source, **values):
    return dict(state=state, source=source, confirmed=False, **values)


def extract_features(image, parts, scores, *, gender="unknown", body_parts=None, provenance=None, confirmed=None):
    """특징표. image: RGB uint8, parts: {이름: bool 마스크}, scores: 태그 점수. gender·body_parts는 이미 확정된 값(없으면 unknown).
    confirmed: 확인 창에서 사람이 확정한 태그 항목({항목: [값]}) — 있으면 태그 측정값 대신 그 값을 쓴다(출처 user, confirmed True)."""
    provenance = provenance or {}
    items = {}
    weak = {name: weak_values(scores, names) for name, names in
            (("hair_length", HAIR_LENGTH_TAGS), ("hair_style", HAIR_STYLE_TAGS), ("hair_color", HAIR_COLOR_TAGS),
             ("eye_color", EYE_COLOR_TAGS), ("head_accessories", HEAD_ACCESSORY_TAGS), ("skin", SKIN_TAGS))}
    def state(strong, pixels=None):
        return "measured" if (strong or pixels) else "unknown"
    length = tag_values(scores, HAIR_LENGTH_TAGS)
    items["hair_length"] = (_item("measured", "tags", value=length[0][0], score=length[0][1], candidates=length)
                            if length else _item("unknown", "tags", value=None, candidates=[]))
    style = tag_values(scores, HAIR_STYLE_TAGS)
    items["hair_style"] = _item(state(style), "tags", values=[n for n, _ in style], scores=dict(style))
    hair_tags = tag_values(scores, HAIR_COLOR_TAGS)
    hair_colors = dominant_colors(image, hair_mask_without_background(image, parts["hair"])) if "hair" in parts else []
    hair_mask = hair_mask_without_background(image, parts["hair"]) if "hair" in parts else None
    palette = dominant_colors(image, hair_mask, k=5) if hair_mask is not None else []
    from genai_lab.hair_color_restore import hair_ab_quantiles
    items["hair_color"] = _item(state(hair_tags, hair_colors), "tags+pixels", tags=[n for n, _ in hair_tags], colors=hair_colors,
                                palette=palette, colored=colored_hair(image, hair_mask, palette) if hair_mask is not None else [],
                                ab_quantiles=hair_ab_quantiles(image, hair_mask) if hair_mask is not None else None)
    eye_tags = tag_values(scores, EYE_COLOR_TAGS)
    eye = eye_color(image, parts["eyes"]) if "eyes" in parts else None
    eyes = eye_colors(image, parts["eyes"]) if "eyes" in parts else []
    items["eye_color"] = _item(state(eye_tags, eye), "tags+pixels", tags=[n for n, _ in eye_tags], color=eye, eyes=eyes)
    accessories = tag_values(scores, HEAD_ACCESSORY_TAGS)
    masks = {k: int(parts[k].sum()) for k in ("headwear", "eyewear", "earwear") if k in parts}
    items["head_accessories"] = _item("measured", "tags+parts", values=[n for n, _ in accessories], scores=dict(accessories),
                                      mask_px=masks)
    skin_tags = tag_values(scores, SKIN_TAGS)
    skin = skin_color(image, parts)
    items["skin"] = _item(state(skin_tags, skin), "tags+pixels", tags=[n for n, _ in skin_tags], color=skin)
    for key, candidates in weak.items():          # 약한 후보: 조건에는 쓰지 않고 확정 때 보여 준다
        items[key]["weak"] = candidates
        if items[key]["state"] == "unknown" and candidates:
            items[key]["state"] = "weak"
    notes = (confirmed or {}).get(DISTINCTIVE)
    items[DISTINCTIVE] = (_item("unknown", "user", values=None) if notes is None
                          else dict(_item("measured", "user", values=list(notes)), confirmed=True))
    for key, values in (confirmed or {}).items():  # 사람이 확정한 값이 태그 측정값을 대신한다(픽셀 측정은 그대로 둔다)
        if key not in TAG_ITEMS or values is None:   # 모름: 자동 측정값을 그대로 두고 확정 표시하지 않는다
            continue
        item = items[key]
        if key == "hair_length":
            item.update(value=values[0] if values else None, candidates=[])
        elif key in ("hair_color", "eye_color", "skin"):
            item["tags"] = list(values)
        else:
            item.update(values=list(values), scores={})
        item.update(state="measured", source="user", confirmed=True, weak=[])
    if body_parts is None:
        items["body_parts"] = _item("unknown", "user", values=None)
    else:
        items["body_parts"] = dict(_item("measured", provenance.get("body_parts", "user"), values=sorted(body_parts)), confirmed=True)
    items["gender"] = (_item("unknown", "user", value=None) if gender in (None, "unknown", "unspecified")
                       else dict(_item("measured", provenance.get("gender", "user"), value=gender), confirmed=True))
    unknown = [k for k, v in items.items() if v["state"] in ("unknown", "weak") and k != DISTINCTIVE]
    return dict(schema=SCHEMA, items=items, unknown=unknown,
                note="측정값은 자동 추출이며 사람 확정 전이다. unknown은 재지 못한 것이지 '없음'이 아니고, weak은 0.35~0.5 후보만 있는 것이다.")


def write_features(directory, features):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "features.json"
    path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summary_line(features):
    """사람이 한눈에 보는 한 줄."""
    it = features["items"]
    hair = "/".join(it["hair_color"]["tags"][:2]) or ("rgb" + str(it["hair_color"]["colors"][0]["rgb"]) if it["hair_color"]["colors"] else "?")
    eyes = "/".join(it["eye_color"]["tags"][:2]) or ("rgb" + str(it["eye_color"]["color"]["rgb"]) if it["eye_color"]["color"] else "?")
    return (f"머리 {it['hair_length']['value'] or '?'} {' '.join(it['hair_style']['values'][:4])} 색 {hair} · 눈 {eyes}"
            f" · 장식 {' '.join(it['head_accessories']['values'][:4]) or '없음'} · 부위 {it['body_parts']['values']} · 성별 {it['gender']['value']}")
