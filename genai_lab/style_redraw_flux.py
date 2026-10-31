"""모드 B 본체 = FLUX.2-klein-4B 편집(D-210, 사용자 10/30).

원본 그림을 참조로 넣고 S3b 그림체 문장으로 같은 캐릭터를 다시 그린다. RealVisXL + 스타일 LoRA는 본체가 선·칠을 정해 셀 그림체에
머물렀다(D-209). FLUX는 학습 자료를 만든 모델이라 S3b를 그대로 그리고, IP-Adapter(ViT-H)가 필요 없다(Apache-2.0).
- 자세: 원본과 같은 자세·팔·손을 지시한다("팔을 내리고 서 있음"을 넣으면 팔짱 원본에서 팔이 늘어났다).
- 색: 편집은 색을 밝고 선명한 쪽으로 옮긴다(짙은 피부가 밝아짐, 와인→빨강, 청록→파랑) → 사람 영역 안에서 부위별(피부·머리·윗옷·아래옷)
  Lab 평균·퍼짐을 원본 쪽으로 옮긴다. 명암(그림체)은 퍼짐 비율을 0.8~1.25로 묶어 유지하고, 한 번에 옮기는 양에 상한을 둔다.
- 손: 결과의 손 개수가 원본보다 많으면 표시한다(후보 고를 때 참고, 자동 탈락은 아님).
"""
import numpy as np

SEEDS = (42, 43, 44)
STEPS = 4
# 생성 크기(D-230, 사용자 10/31 "그림체는 이걸로 고정"): 참조 캔버스 768x1024는 그대로, 결과는 처음부터 1024x1360으로 그린다.
# 같은 문장·seed에서 눈 반사광·머리 가닥·홍조가 촘촘하고 마감이 매끈해짐(세이버 참고 3번 쪽), 후보 1장 +7초, 최고 4.55GiB.
# 768x1024로 그린 뒤 키워 다시 그리는 2단계는 차이가 작고 귀걸이를 잃었다.
OUTPUT_SIZE = (1024, 1360)
GUIDANCE = 1.0
MAX_SEQUENCE_LENGTH = 256
# 그림체 = S4w(사용자 10/30): 목표(고밀도 반실사 애니메이션 일러스트)의 그리는 방식을 문장으로 — 목표 그림의 내용은 따라 하지 않는다.
STYLE = ("highly detailed semi-realistic anime illustration, dense digital painting: delicate thin line art, anime face with realistic soft "
         "skin rendering, subtle glossy highlights and light blush, detailed eyes with reflections, very fine individual hair strands with "
         "loose flyaway wisps and strong backlight, deep realistic fabric folds and creases, detailed material texture, strong directional "
         "sunlight with sharp deep shadows and soft bounce light, high detail density, cinematic colour grading")
BACKGROUND = "Weathered concrete wall background with soft texture, strong warm key light from upper left."
COLOUR_SHIFT_CAP = np.array([15.0, 12.0, 12.0])     # L, a, b — 이보다 크게 옮기지 않는다(잘못 잡은 영역의 피해를 줄임)
SPREAD_CLIP = (0.8, 1.25)
HIGHLIGHT_L = 94.0                  # 밝힐 때 부위의 밝은 쪽(99번째 백분위)이 이 L을 넘지 않게
HIGHLIGHT_PERCENTILE = 99
MIN_PART_PIXELS = 200
PARTS = ("skin", "hair", "upper", "lower")
IDENTITY_PARTS = ("skin", "hair")             # 옷을 바꿀 때는 캐릭터 원본에 맞출 부위가 피부·머리뿐(옷은 옷 그림에서 온다)
IDENTITY_SHIFT_CAP = np.array([40.0, 20.0, 20.0])   # 옷 그림의 흰 마네킹·배경에 피부가 밝게 끌릴 수 있어 폭을 넓힌다
# 장면 문장 공통 규칙(D-225): 황금빛 역광에서 검은 머리가 회색으로(U003), '걷기'가 서 있는 자세로, 거리 간판에 가짜 글자.
SCENE_RULES = ("The scene lighting may add highlights and shadows but must not change the character's own colours: the hair, "
               "eyes, skin and clothes keep their colours from image 1. Show the pose and action of the scene clearly with the whole "
               "body, arms and legs. ")
# 간판 문장("Signs and screens … no letters")은 시험에서 가짜 글자를 못 막고 공원에 빈 간판을 새로 세워 뺐다(D-225).


def flux_prompt(traits, with_garment=False, body_parts=None, scene=None):
    """같은 캐릭터를 목표 그림체로 다시 그리라는 문장. 특징 문장(자동 측정)과 피부색 유지 지시를 넣는다.
    with_garment: 그림 2의 옷으로 갈아입히고 그린다(D-212, 모드 A·B 통합). 원래 옷 낱말은 빼야 새 옷과 다투지 않는다.
    body_parts: 확인된 동물 귀·꼬리·날개(튜플, 없으면 빈 튜플, 모르면 None) — Qwen 경로와 같은 부위 문장(D-108·D-111, D-221).
    scene: 새 자세·장면 문장(scene_composer 또는 검사를 지난 사용자 문장, D-222). 없으면 원본 자세 + 콘크리트 벽 배경."""
    from genai_lab.outfit_edit import body_parts_sentence
    parts = body_parts_sentence(None if body_parts is None else tuple(body_parts))
    if scene:
        # D-219·D-221 장면 시험 문장(처음 보는 8명 동일성 유지)과 같은 짜임. 옷 그림이 있으면 그 옷을 입고 새 장면에.
        outfit = ("wearing the complete outfit shown in image 2 (replace all of the character's current clothing with it, matching its "
                  "shape, length, colours and details; nothing from the original outfit may remain)" if with_garment else "")
        described = "; ".join(row["text"].strip() for row in traits if row.get("text", "").strip()
                              and not (with_garment and row.get("name") == "clothes"))
        who = f" ({described})" if described else ""
        keep = "body proportions" if with_garment else "body proportions and outfit"
        return (f"Show the same adult character as in image 1{who}{' ' + outfit if outfit else ''} in a new scene. Scene: {scene.strip()} "
                f"Keep the same face, hairstyle, hair colour, eye colour, skin tone and {keep} exactly as in image 1; do not lighten "
                f"or darken the skin. {parts}{SCENE_RULES}Full body, the whole character inside the frame, face clearly visible. Render as a new "
                f"{STYLE}. {'Do not copy the mannequin or the clothing photo background. ' if with_garment else ''}"
                f"No lettering, logo, watermark or multiple people.")
    if with_garment:
        described = "; ".join(row["text"].strip() for row in traits
                              if row.get("text", "").strip() and row.get("name") != "clothes")
        who = f" ({described})" if described else ""
        return (f"Dress the same adult character from image 1{who} in the complete outfit shown in image 2: replace all of the "
                f"character's current clothing with the outfit from image 2, matching its shape, length, colours and details; nothing "
                f"from the original outfit may remain. Keep the same face, hairstyle, hair colour, eye colour, skin tone and body "
                f"proportions as in image 1; do not lighten or darken the skin. {parts}Keep the same pose and framing as image 1. Render as a "
                f"new {STYLE}. {BACKGROUND} Do not copy the mannequin or the clothing photo background. No lettering, logo, watermark "
                f"or multiple people.")
    described = "; ".join(row["text"].strip() for row in traits if row.get("text", "").strip())
    keep = f" ({described})" if described else ""
    return (f"Redraw the same adult character as in the reference image as a new {STYLE}. Keep the same face, hairstyle, hair colour, "
            f"eye colour, skin tone and outfit exactly as in the reference{keep}. Keep the exact same skin tone as the reference; do not "
            f"lighten or darken the skin. {parts}Keep the same pose, arms, hands and framing as in the reference. {BACKGROUND} "
            f"No lettering, logo, watermark or multiple people.")


def part_masks(rgb, parts, person):
    """분할 결과(figure_parts)와 사람 영역(anime-seg)에서 색 맞춤할 네 부위. 모두 사람 영역 안으로 자른다."""
    from genai_lab.style_reference_colors import skin_like
    shape = np.asarray(rgb).shape[:2]
    empty = np.zeros(shape, bool)
    def any_of(*keys):
        return np.logical_or.reduce([np.asarray(parts.get(k, empty), bool) for k in keys])
    person = np.asarray(person, bool)
    clothes = any_of("topwear", "handwear", "bottomwear", "legwear", "footwear")
    features = any_of("hair", "eyes", "mouth", "eyewear")
    skin = ((skin_like(rgb, parts["face"]) & ~clothes) | any_of("face", "neck")) & ~features & person
    return dict(skin=skin, hair=any_of("hair") & person, upper=any_of("topwear", "handwear") & ~skin & person,
                lower=any_of("bottomwear", "legwear") & ~skin & person)


def colour_match(result_rgb, original_rgb, result_masks, original_masks, feather=2.0, parts=PARTS, cap=None, scene_lighting=False):
    """부위마다 결과 색을 원본 쪽으로(Lab 평균 이동 + 퍼짐 비율). 반환: (RGB uint8, {부위: 옮긴 양 또는 None}).
    scene_lighting: 새 장면(조명이 바뀜)이면 밝기는 조명 몫 — 밝게 올리지 않고(어둡게 되돌리기만) 명암 폭도 그대로(D-225:
    황금빛 역광의 검은 머리를 밝은 셀 원본 평균 쪽으로 +13 올려 회색이 됐다, U003). 색조(a·b)는 그대로 맞춘다."""
    from scipy.ndimage import gaussian_filter
    from genai_lab.style_reference_colors import lab_to_rgb, rgb_to_lab
    lab, source = rgb_to_lab(np.asarray(result_rgb)), rgb_to_lab(np.asarray(original_rgb))
    cap = COLOUR_SHIFT_CAP if cap is None else np.asarray(cap, float)
    shifts = {}
    for part in parts:
        mr, ms = np.asarray(result_masks[part], bool), np.asarray(original_masks[part], bool)
        if mr.sum() < MIN_PART_PIXELS or ms.sum() < MIN_PART_PIXELS:
            shifts[part] = None
            continue
        a, b = lab[mr], source[ms]
        mu_r, sd_r, mu_s, sd_s = a.mean(0), a.std(0) + 1e-3, b.mean(0), b.std(0) + 1e-3
        shift = np.clip(mu_s - mu_r, -cap, cap)
        spread = np.clip(sd_s / sd_r, *SPREAD_CLIP)
        if scene_lighting:
            shift[0], spread[0] = min(shift[0], 0.0), 1.0
        if shift[0] > 0:              # 밝히는 쪽은 밝은 곳이 하얗게 날아가지 않을 만큼만(T017·V001: 흰 셀 피부 쪽으로 +20 → 얼굴 윤곽 사라짐)
            top = (np.percentile(a[:, 0], HIGHLIGHT_PERCENTILE) - mu_r[0]) * spread[0] + mu_r[0]
            shift[0] = min(shift[0], max(HIGHLIGHT_L - top, 0.0))
        fixed = (lab - mu_r) * spread + mu_r + shift
        w = gaussian_filter(mr.astype(np.float32), feather)[..., None] if feather else mr[..., None].astype(np.float32)
        lab = lab * (1 - w) + fixed * w
        shifts[part] = [round(float(x), 1) for x in shift]
    return np.clip(lab_to_rgb(lab), 0, 255).astype(np.uint8), shifts


PART_TAG_MIN = 0.2      # 기록할 부위 태그(꼬리 종류 등) 하한


def missing_body_parts(body_parts, result_scores):
    """확인된 동물 귀·꼬리·날개가 결과 그림의 태그(WD 태거, 준비 단계와 같은 규칙 guessed_body_parts)에서 안 보이면 그 이름 목록.
    부위 분할기는 원본에서도 꼬리를 못 잡아(U003·U007·U008 원본 0픽셀) 쓰지 않는다(D-221). 자동 탈락이 아니라 표시."""
    from genai_lab.studio_controller import guessed_body_parts
    found = guessed_body_parts(dict(character_scores=result_scores or {}))
    return [p for p in tuple(body_parts or ()) if p not in found]


def body_part_tags(scores):
    """결과의 귀·꼬리·날개 종류 태그(기록용, 예: mechanical_tail → striped_tail로 바뀐 것 확인)."""
    return {k: round(float(v), 2) for k, v in (scores or {}).items()
            if (k == "animal_ears" or k.endswith(("_ears", "tail", "wings"))) and float(v) >= PART_TAG_MIN}


def hand_flag(original_count, result_count):
    """손이 원본보다 많으면 팔·손이 늘어난 후보로 표시. 셀 수 없으면 None."""
    if original_count is None or result_count is None:
        return None
    return result_count > original_count
