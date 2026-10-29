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
OUTFIT_BACKGROUND = (214, 210, 202)
OUTFIT_NECK_MARGIN = 0.02                   # 목 관절보다 그림 높이의 2% 위에서 자른다
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


def outfit_reference(image, alpha, neck_y):
    """의상 참조: 배경을 지우고 목 위(그림 높이 2% 여유)를 잘라 회색 바탕에 몸만 남긴다. (그림, 자른 상자)."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    if a.max() > 1:
        a = a / 255.0
    a = a.copy()
    a[:int(max(0, neck_y - OUTFIT_NECK_MARGIN * rgb.shape[0]))] = 0
    ys, xs = np.nonzero(a > 0.5)
    if len(xs) == 0:
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


def trait_rows(features, character_scores, gender):
    """특징표(character_features)와 태그 점수로 눈·피부·머리·옷·체형 문장을 만든다. 재지 못한 항목은 빈 문장(unmeasured)으로
    두고 참조 그림에 맡긴다 — 추측해서 채우지 않는다(시각 조건 우선)."""
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
    build = GENDER_WORDS.get(gender, "")
    return [_row("eyes", eyes, "눈"), _row("skin", skin_text, "피부"), _row("hair", hair_text, "머리"),
            _row("clothes", clothes, "옷"), _row("build", build, "체형·성별")]


def build_annotations(source_sha256, style_sha256, outfit_sha256, face, traits):
    """입력 묶음 주석(character_style_inputs.validate_annotations 형식). 자세는 표준 자세, 의상 참조는 몸 영역."""
    return dict(schema_version=1, source_sha256=source_sha256, style_sha256=style_sha256, canvas=list(CANVAS),
                coordinate_system="image_xy", height_cm=None, face_box=list(face), traits=traits,
                scene_prompt=SCENE_PROMPT, style_prompt=STYLE_PROMPT,
                layout=dict(origin="manual_image_annotation",
                            evidence="고정 표준 서 있는 자세(사용자 10/29: 자세 무관, 인체만 맞으면 됨). 원본 관절은 옷 색 읽기에만",
                            joints={k: list(v) for k, v in STANDARD_POSE.items()}),
                outfit_sha256=outfit_sha256, outfit_box=list(OUTFIT_BOX),
                cpu_probe_options={k: SCALES[k] for k in ("identity_scale", "style_scale", "structure_scale")})
