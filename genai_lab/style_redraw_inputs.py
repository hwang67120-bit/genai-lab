"""모드 B(그림체로 다시 그리기, docs/STYLE_MODE_APP_DESIGN.md S3)의 CPU 준비 단계.

D-195 시험 실행기(local_batch_v5s.py)의 단계를 제품 함수로 옮긴다: 얼굴 상자, 의상 참조, 그림체 참조 색 맞춤 계획,
특징 문장(시험은 사람이 관찰해 썼고, 여기서는 특징표·태그에서 자동으로 만든다), 입력 묶음 주석. 생성은 하지 않는다.
"""
import numpy as np
from PIL import Image

CANVAS = (768, 1024)
# D-195 고정값(시험 10명 판정 근거). 바꾸면 사용자 판정을 다시 받는다.
SCENE_PROMPT = "Full body, standing by a window"
STYLE_PROMPT = "Painted illustration, blended shadows, nuanced colors"
SCALES = dict(identity_scale=0.4, style_scale=0.7, structure_scale=0.6, outfit_scale=0.8)
OUTFIT_BOX = [230, 240, 540, 1000]          # 표준 자세의 몸 영역(목 아래), 캔버스 768x1024
# D-200: 의상 참조 한 장이면 넓은 옷(V002 회색 바지)이 작은 옷(초록 조끼)을 덮었다 → 윗옷(목~엉덩이)·아래옷(엉덩이 아래)을
# 나눠 표준 자세의 각 영역에만 적용한다. 엉덩이 관절이 없으면 예전처럼 한 장(body).
OUTFIT_SPLIT_BOXES = {"upper": [230, 240, 540, 575], "lower": [230, 545, 540, 1000]}
OUTFIT_BACKGROUND = (214, 210, 202)
OUTFIT_NECK_MARGIN = 0.02                   # 목 관절보다 그림 높이의 2% 위에서 자른다
OUTFIT_HIP_BELOW = 0.05                     # 윗옷 조각은 엉덩이 관절보다 5% 아래까지
OUTFIT_HIP_ABOVE = 0.03                     # 아래옷 조각은 엉덩이 관절보다 3% 위부터
OUTFIT_MIN_PIXELS = 2000                    # 조각에 사람 픽셀이 이보다 적으면 그 조각은 없는 것으로
# D-200 머리 참조: 그림체 참조(target-02)의 단발 모양이 결과 머리로 넘어오고, 원본 머리 모양을 담은 참조가 없었다(얼굴 참조는
# 얼굴만, 의상 참조는 목 아래). 원본 머리카락+얼굴을 잘라 표준 자세 머리 영역에만 0.7로 준다(V002·T009·V007·T014·N004 시험).
HEAD_BOX = [264, 30, 504, 320]
HAIR_SCALE = 0.7
HAIR_MARGIN = 0.04                          # 머리+얼굴 영역 둘레 여유(그림 높이 비율)
# 후보 3장(사용자 10/29 (가)): 몸선은 seed에 묶여 문장으로 못 움직인다(D-199) → seed 셋으로 만들고 사용자가 1장 고른다.
SEEDS = (42, 43, 44)
FACE_BOX_SCALE = 1.6
POINT_SCORE = 0.3
# 그림체 참조는 명암·재질 표현만 맡는다: 머리·눈은 원본 색으로 칠하고, 옷은 무채색(잔무늬 누름). 옷 색·형태는 의상 참조가 맡는다.
STYLE_PLAN = {"hair": dict(reference=["hair"], source=["hair"], keep_skin=True),
              "eyes": dict(reference=["eyes"], source=["eyes"], split="left_right"),
              "upper": dict(reference=["topwear", "handwear"], source=["topwear", "handwear"], flatten=True, neutral=True),
              "lower": dict(reference=["bottomwear"], source=["bottomwear", "legwear"], flatten=True, neutral=True)}
# 결과 자세는 하나로 고정한다(사용자 10/29: 인체만 맞으면 자세는 상관없다). 원본 관절은 옷 색 읽기와 의상 참조 자르기에만 쓴다.
STANDARD_POSE = {"nose": [384, 175], "neck": [384, 255], "right_shoulder": [304, 268], "right_elbow": [278, 410],
                 "right_wrist": [266, 540], "left_shoulder": [464, 268], "left_elbow": [490, 410], "left_wrist": [502, 540],
                 "right_hip": [340, 560], "right_knee": [334, 760], "right_ankle": [330, 950], "left_hip": [428, 560],
                 "left_knee": [434, 760], "left_ankle": [438, 950], "right_eye": [366, 160], "left_eye": [402, 160],
                 "right_ear": [346, 170], "left_ear": [422, 170]}
# D-201 원본 비율 맞춘 표준 자세: 고정 뼈대는 머리/어깨 0.62·엉덩이/어깨 0.55·다리/몸통 1.28로 원본 10명(0.70~1.16·0.60~0.70·
# 1.43~1.68)보다 머리가 작고 어깨가 넓고 다리가 짧았다(사용자: 인체 비율이 안 맞음). 자세(정면, 팔 내림)는 그대로 두고 길이·폭만
# 원본에서 잰 비율로 만든다. 재지 못한 비율은 아래 기본값(원본 10명의 가운데쯤), 터무니없는 값은 범위 안으로 자른다.
PROPORTION_DEFAULTS = dict(head=0.8, torso=1.6, hip=0.65, leg=1.5, neck_drop=0.55)
PROPORTION_LIMITS = dict(head=(0.5, 1.3), torso=(1.2, 2.2), hip=(0.5, 0.9), leg=(1.2, 1.9), neck_drop=(0.3, 0.9))
POSE_TOP, POSE_BOTTOM = 40, 960             # 머리 위·발목 높이(캔버스 768x1024)
MAX_GARMENT_WORDS = 6
GARMENT_DETAIL = r"sleeves|collared|buttons?$|^open |^dress shirt$|^sleeves "
# 피부 낱말: 얼굴 피부 Lab L* 기준. 45·60은 자유 생성(D-136)과 같다. 60~72는 밝은 면을 재 짙은 피부도 들어와(V006) 낱말 없이
# 얼굴 참조에 맡기고, 72 이상만 fair skin(이 경로에서 새로 둔 값, S4 판정으로 확인).
SKIN_WORDS = ((45, "dark brown skin"), (60, "brown skin"), (72, ""), (101, "fair skin"))
# 특징표 머리 모양 목록(자유 생성과 공유)에 없는 드문 머리 모양 태그. 이 경로에서만 문장에 더한다(D-196 N004 투블럭).
EXTRA_HAIR_STYLE_TAGS = ("dreadlocks", "undercut", "afro", "cornrows", "mohawk", "buzz_cut")
TAG_THRESHOLD = 0.5
GENDER_WORDS = {"male": "adult man", "female": "adult woman"}
# 성별 낱말만으로는 짧은 곱슬머리·짙은 피부의 여성(V002·T014)이 남성처럼 나왔다(D-199) → 반대 성별을 부정 문장에.
GENDER_NEGATIVE = {"male": "woman, female, feminine", "female": "man, male, masculine"}
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


def _clip(value, key):
    low, high = PROPORTION_LIMITS[key]
    return min(high, max(low, value))


def measured_proportions(joints, face):
    """원본 관절·얼굴 상자에서 어깨 폭 기준 비율. 잴 수 없는 항목은 기본값, 출처를 함께 돌려준다."""
    found, source = dict(PROPORTION_DEFAULTS), {k: "default" for k in PROPORTION_DEFAULTS}
    if not {"left_shoulder", "right_shoulder", "neck"} <= set(joints):
        return found, source
    s = abs(joints["left_shoulder"][0] - joints["right_shoulder"][0])
    if s < 10:
        return found, source
    side = face[3] - face[1]
    values = dict(head=side / s, neck_drop=(joints["neck"][1] - (face[1] + face[3]) / 2) / side)
    if {"left_hip", "right_hip"} <= set(joints):
        hip_y = (joints["left_hip"][1] + joints["right_hip"][1]) / 2
        torso = hip_y - joints["neck"][1]
        values.update(torso=torso / s, hip=abs(joints["left_hip"][0] - joints["right_hip"][0]) / s)
        if {"left_ankle", "right_ankle"} <= set(joints) and torso > 0:
            values["leg"] = ((joints["left_ankle"][1] + joints["right_ankle"][1]) / 2 - hip_y) / torso
    for key, value in values.items():
        found[key], source[key] = round(_clip(value, key), 3), "measured"
    return found, source


def fitted_pose(proportions, canvas=CANVAS):
    """표준 자세(정면, 팔 내림)를 주어진 비율로 만들어 캔버스에 맞춘다. 반환: 관절, 얼굴 마스크 상자, 머리·윗옷·아래옷·몸 영역.
    기준 단위는 어깨 폭 1. 팔·발 모양은 고정 뼈대(STANDARD_POSE)와 같은 비율을 쓴다."""
    p = proportions
    f = p["head"]                                   # 얼굴 상자 한 변
    torso, leg = p["torso"], p["leg"] * p["torso"]
    face_cy = -p["neck_drop"] * f                   # 목(0,0) 기준 얼굴 중심
    top, bottom = face_cy - 1.1 * f, torso + leg    # 머리 위(머리카락 여유) ~ 발목
    k = (POSE_BOTTOM - POSE_TOP) / (bottom - top)
    k = min(k, (canvas[0] / 2 - 40) / 0.75)         # 팔 바깥(어깨 폭 0.75)이 캔버스 안에
    cx, y0 = canvas[0] / 2, POSE_TOP - top * k
    def at(x, y):
        return [round(cx + x * k, 1), round(y0 + y * k, 1)]
    joints = {"neck": at(0, 0), "nose": at(0, face_cy + 0.083 * f),
              "right_eye": at(-0.15 * f, face_cy - 0.042 * f), "left_eye": at(0.15 * f, face_cy - 0.042 * f),
              "right_ear": at(-0.317 * f, face_cy + 0.042 * f), "left_ear": at(0.317 * f, face_cy + 0.042 * f)}
    drop = 0.081                                    # 어깨는 목보다 어깨 폭의 8% 아래(고정 뼈대 13/160)
    for side, sign in (("right", -1), ("left", 1)):
        joints[f"{side}_shoulder"] = at(sign * 0.5, drop)
        joints[f"{side}_elbow"] = at(sign * 0.6625, drop + 0.466 * torso)
        joints[f"{side}_wrist"] = at(sign * 0.7375, drop + 0.892 * torso)
        joints[f"{side}_hip"] = at(sign * p["hip"] / 2, torso)
        joints[f"{side}_knee"] = at(sign * (p["hip"] / 2 + 0.0375), torso + 0.513 * leg)
        joints[f"{side}_ankle"] = at(sign * (p["hip"] / 2 + 0.0625), torso + leg)
    def box(x0, y0_, x1, y1):
        a, b = at(x0, y0_), at(x1, y1)
        return [int(max(0, a[0])), int(max(0, a[1])), int(min(canvas[0], b[0])), int(min(canvas[1], b[1]))]
    half = 0.96
    boxes = dict(face=box(-f / 2, face_cy - f / 2, f / 2, face_cy + f / 2),
                 head=box(-f, face_cy - 1.125 * f, f, 0.54 * f),
                 upper=box(-half, -0.094, half, torso + 0.094),
                 lower=box(-half, torso - 0.094, half, torso + leg + 0.31),
                 body=box(-half, -0.094, half, torso + leg + 0.31))
    return joints, boxes


def outfit_references(image, alpha, neck_y, hip_y=None, boxes=None):
    """의상 참조 목록 [(이름, 그림, 자른 상자, 적용 영역)]. 엉덩이 높이를 알면 윗옷·아래옷 두 장, 모르면 몸 전체 한 장.
    아래옷 조각에 사람이 거의 없으면(허벅지 위에서 잘린 그림) 윗옷만 몸 전체 영역에 쓴다.
    boxes: 적용 영역(fitted_pose). 없으면 고정 뼈대의 영역."""
    boxes = boxes or dict(OUTFIT_SPLIT_BOXES, body=list(OUTFIT_BOX))
    h = np.asarray(alpha).shape[0]
    top = neck_y - OUTFIT_NECK_MARGIN * h
    if hip_y is None:
        piece, crop = outfit_reference(image, alpha, top)
        return [("body", piece, crop, list(boxes["body"]))]
    upper, upper_crop = outfit_reference(image, alpha, top, hip_y + OUTFIT_HIP_BELOW * h)
    try:
        lower, lower_crop = outfit_reference(image, alpha, hip_y - OUTFIT_HIP_ABOVE * h)
    except StyleRedrawInputError:
        return [("body", upper, upper_crop, list(boxes["body"]))]
    return [("upper", upper, upper_crop, list(boxes["upper"])),
            ("lower", lower, lower_crop, list(boxes["lower"]))]


def hair_reference(image, alpha, hair, face):
    """머리 참조: 원본 머리카락+얼굴 영역(둘레 4% 여유)의 높이만 남기고 배경을 지워 회색 바탕에 둔다. (그림, 자른 상자)."""
    head = np.asarray(hair, bool) | np.asarray(face, bool)
    ys, xs = np.nonzero(head)
    if len(xs) < OUTFIT_MIN_PIXELS:
        raise StyleRedrawInputError("머리를 찾지 못했습니다. 머리가 잘 보이는 그림을 넣어 주세요.")
    h, w = head.shape
    margin = int(HAIR_MARGIN * h)
    box = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
           min(w, int(xs.max()) + margin), min(h, int(ys.max()) + margin))
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    a = a / 255.0 if a.max() > 1 else a
    comp = rgb * a[..., None] + np.array(OUTFIT_BACKGROUND, np.float32) * (1 - a[..., None])
    return Image.fromarray(np.clip(comp + 0.5, 0, 255).astype(np.uint8)).crop(box), list(box)


def outfit_reference(image, alpha, top, bottom=None):
    """의상 참조 한 조각: 배경을 지우고 top~bottom 높이만 남겨 회색 바탕에 몸만 둔다. (그림, 자른 상자)."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    if a.max() > 1:
        a = a / 255.0
    a = a.copy()
    a[:int(max(0, top))] = 0
    if bottom is not None:
        a[int(min(rgb.shape[0], bottom)):] = 0
    ys, xs = np.nonzero(a > 0.5)
    if len(xs) < OUTFIT_MIN_PIXELS:
        raise StyleRedrawInputError("목 아래에서 옷을 찾지 못했습니다. 몸이 보이는 그림을 넣어 주세요.")
    comp = rgb * a[..., None] + np.array(OUTFIT_BACKGROUND, np.float32) * (1 - a[..., None])
    box = (max(0, int(xs.min()) - 8), max(0, int(ys.min()) - 8),
           min(rgb.shape[1], int(xs.max()) + 8), min(rgb.shape[0], int(ys.max()) + 8))
    return Image.fromarray(np.clip(comp + 0.5, 0, 255).astype(np.uint8)).crop(box), list(box)


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


def build_annotations(source_sha256, style_sha256, outfits, face, traits, gender=None, joints=None, face_mask=None):
    """입력 묶음 주석(character_style_inputs.validate_annotations 형식).
    outfits: [(이름, SHA-256, 적용 영역)] 또는 [(이름, SHA-256, 적용 영역, 강도)] — 참조 순서 그대로(의상 다음 머리).
    joints·face_mask: 원본 비율로 만든 표준 자세(fitted_pose)와 그 얼굴 자리. 없으면 고정 뼈대와 원본 얼굴 상자."""
    evidence = ("원본 비율로 만든 표준 서 있는 자세(D-201; 사용자 10/29: 자세 무관, 인체만 맞으면 됨). 원본 관절은 비율·옷 색 읽기에만"
                if joints else "고정 표준 서 있는 자세(사용자 10/29: 자세 무관, 인체만 맞으면 됨). 원본 관절은 옷 색 읽기에만")
    extra = {"identity_mask_box": list(face_mask)} if face_mask else {}
    return dict(schema_version=1, negative_terms=GENDER_NEGATIVE.get(gender, ""), source_sha256=source_sha256, style_sha256=style_sha256, canvas=list(CANVAS),
                coordinate_system="image_xy", height_cm=None, face_box=list(face), traits=traits,
                scene_prompt=SCENE_PROMPT, style_prompt=STYLE_PROMPT, **extra,
                layout=dict(origin="manual_image_annotation", evidence=evidence,
                            joints={k: list(v) for k, v in (joints or STANDARD_POSE).items()}),
                outfits=[dict(name=row[0], sha256=row[1], box=list(row[2]), **({"scale": row[3]} if len(row) > 3 else {}))
                         for row in outfits],
                cpu_probe_options={k: SCALES[k] for k in ("identity_scale", "style_scale", "structure_scale")})
