"""모드 B(FLUX, D-210) 준비 단계의 입력 도우미: 얼굴 상자와 특징 문장(특징표·태그에서 자동으로 만든다). 생성은 하지 않는다.

예전 SDXL 경로(그림체 참조·의상 참조·표준 자세·입력 묶음)는 D-216에서 지웠다 — 기록은 git 역사와 DECISIONS.md.
"""
import numpy as np

FACE_BOX_SCALE = 1.6
POINT_SCORE = 0.3
MAX_GARMENT_WORDS = 6
GARMENT_DETAIL = r"sleeves|collared|buttons?$|^open |^dress shirt$|^sleeves "
# 피부 낱말: 얼굴 피부 Lab L* 기준. 45·60은 자유 생성(D-136)과 같다. 60~72는 밝은 면을 재 짙은 피부도 들어와(V006) 낱말 없이
# 얼굴 참조에 맡기고, 72 이상만 fair skin(이 경로에서 새로 둔 값, S4 판정으로 확인).
SKIN_WORDS = ((45, "dark brown skin"), (60, "brown skin"), (72, ""), (101, "fair skin"))
# 특징표 머리 모양 목록(자유 생성과 공유)에 없는 드문 머리 모양 태그. 이 경로에서만 문장에 더한다(D-196 N004 투블럭).
EXTRA_HAIR_STYLE_TAGS = ("dreadlocks", "undercut", "afro", "cornrows", "mohawk", "buzz_cut")
TAG_THRESHOLD = 0.5
GENDER_WORDS = {"male": "adult man", "female": "adult woman"}
EVIDENCE = "자동 측정(특징표·태그, style_redraw_inputs). 사람 확인 전."


class StyleRedrawInputError(ValueError):
    """모드 B 준비를 멈추고 이유를 알려야 하는 입력 문제."""


def face_box(face_points, face_scores, size, scale=FACE_BOX_SCALE):
    """DWPose 얼굴 점(점수 0.3 이상)을 감싸는 정사각형 x1.6. 시험(detect_pose_v001)과 같은 규칙."""
    points = np.asarray(face_points, dtype=float).reshape(-1, 2)
    scores = np.asarray(face_scores, dtype=float).reshape(-1)
    points = points[scores >= POINT_SCORE]
    if len(points) < 5:
        raise StyleRedrawInputError("얼굴을 찾지 못했습니다. 얼굴이 잘 보이는 그림을 넣어 주세요.")
    cx, cy = points.mean(0)
    side = scale * max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    w, h = size
    return [int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(w, cx + side / 2)), int(min(h, cy + side / 2))]


def _plain(tag):
    return tag.replace("_", " ")


def _measured(items, key):
    item = items.get(key) or {}
    return item if item.get("state") == "measured" else None


def garment_words(character_scores, threshold=0.35):    # 0.35 = ClothingDesignAnalysisSettings.score_threshold
    """캐릭터 그림의 옷 태그(점수순, 자세·머리 장식 태그 제외)를 낱말로. 지금 앱의 옷 태그 거르기와 같은 규칙."""
    from genai_lab.character_features import HEAD_ACCESSORY_TAGS
    from genai_lab.free_generation import POSE_TAG
    from genai_lab.reference_features import tag_scope
    from genai_lab.reference_tag_policy import excluded_garment_tag
    import re
    found = [(_plain(tag), float(score)) for tag, score in (character_scores or {}).items()
             if float(score) >= threshold and tag_scope(tag) == "garment" and not excluded_garment_tag(tag)
             and not re.search(POSE_TAG, tag) and tag not in HEAD_ACCESSORY_TAGS]
    words = [w for w, _ in sorted(found, key=lambda kv: -kv[1])]
    # 색·종류가 붙은 낱말(white shirt)이 있으면 같은 이름의 일반 낱말(shirt)은 뺀다 — 6개 제한에서 색 있는 옷이 잘리지 않게.
    words = [w for w in words if not any(o != w and o.endswith(" " + w) for o in words)]
    # 소매·깃·단추 같은 세부 낱말은 옷 이름 뒤로(T014: 세부 낱말이 자리를 차지해 조끼가 잘림).
    words = sorted(words, key=lambda w: bool(re.search(GARMENT_DETAIL, w)))
    return words[:MAX_GARMENT_WORDS]


def _row(name, text, description_ko):
    if text:
        return dict(name=name, text=text, description_ko=description_ko, evidence=EVIDENCE, status="automatic_measurement")
    return dict(name=name, text="", description_ko=description_ko + " — 재지 못함(참조 그림에 맡김)", evidence=EVIDENCE,
                status="unmeasured")


def trait_rows(features, character_scores, confirmed_gender=None):
    """특징표(character_features)와 태그 점수로 눈·피부·머리·옷·체형 문장을 만든다. 재지 못한 항목은 빈 문장(unmeasured)으로
    두고 참조 그림에 맡긴다 — 추측해서 채우지 않는다(시각 조건 우선).
    성별 낱말은 사람이 확정한 성별(캐릭터 기억·확인)이 있을 때만 쓴다. 자동 태그 추정은 넣지 않는다 — S4에서 태그가 남성 둘을
    1girl 0.97·1.00으로 읽었고, 그 낱말로 성별이 바뀌어 사용자 판정 0%·50%가 됐다(D-198)."""
    items = (features or {}).get("items", {})
    eye = _measured(items, "eye_color")
    eyes = ", ".join(_plain(t) for t in eye.get("tags", [])) if eye else ""
    skin = _measured(items, "skin")
    lab = (((skin or {}).get("color") or {}).get("lab")) if skin else None
    skin_text = next((word for limit, word in SKIN_WORDS if float(lab[0]) < limit), "") if lab else ""
    hair = []
    length = _measured(items, "hair_length")
    if length and length.get("value"):
        hair.append(_plain(length["value"]).replace(" hair", ""))
    colour = _measured(items, "hair_color")
    if colour:
        hair.extend(_plain(t).replace(" hair", "") for t in colour.get("tags", []))
    hair_text = (" ".join(hair) + " hair") if hair else ""
    style = _measured(items, "hair_style")
    styles = list(style.get("values") or []) if style else []
    styles += [t for t in EXTRA_HAIR_STYLE_TAGS
               if float((character_scores or {}).get(t, 0.0)) >= TAG_THRESHOLD and t not in styles]
    if styles:
        hair_text = ", ".join(([hair_text] if hair_text else []) + [_plain(v) for v in styles])
    clothes = ", ".join(garment_words(character_scores))
    build = GENDER_WORDS.get(confirmed_gender, "")
    return [_row("eyes", eyes, "눈"), _row("skin", skin_text, "피부"), _row("hair", hair_text, "머리"),
            _row("clothes", clothes, "옷"), _row("build", build, "체형·성별")]
