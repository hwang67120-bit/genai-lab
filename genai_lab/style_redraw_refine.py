"""모드 B 다듬기(D-203): 사용자가 고른 후보 1장을 목표 그림체 쪽으로 한 번 더 다듬고, 색은 고른 후보에서 되돌린다.

D-204 이후 그림체는 스타일 LoRA가 맡는다. 이 다듬기는 그림체 참조(target-02)를 다시 쓰므로 기본 흐름에서는 부르지 않는다
(그림체 참조 방식의 기록·비교용으로 남긴다).

1단계(후보 3장)는 동일성·구도·옷을 잡는다. 다듬기는 원본 유지 50%로 다시 그리며 깎지 않은 그림체 참조(target-02)를 1.0으로 걸어
선·명암·하이라이트·머리 가닥을 더한다. 그림체 참조가 옷 무늬(그래피티)를 넘기므로 옷 영역에서는 끄고(R1), target의 색(빨간 머리
가닥, 바지 색)이 넘어오므로 밝기만 다듬기 결과에서 가져오고 색(Lab a·b)은 고른 후보에서 가져온다(C2). V002·T009 시험 10/30.
"""
import numpy as np
from PIL import Image

REFINE_STRENGTH = 0.5
REFINE_STEPS = 30
REFINE_STYLE_SCALE = 1.0
REFINE_FACE_SCALE = 0.4
REFINE_STRUCTURE_SCALE = 0.6
COLOUR_BACK_SIGMA = 3                   # 고른 후보의 색을 살짝 흐려 다듬기에서 움직인 경계에 색이 번지지 않게
STYLE_WORDS = ("highly detailed anime illustration, strong directional sunlight, dramatic soft shadows, glossy highlights on hair and skin, "
               "fine hair strands, detailed fabric folds and texture, rich colors")
# 목표 그림(target-02)의 노출·내용을 따라 하지 않는다(10/26 검토 기준).
CONTENT_NEGATIVE = "crop top, midriff, navel, bare shoulders, tattoo, shorts, choker, cleavage, nsfw"
BASE_NEGATIVE = "photograph, skin pores, flat cel shading, latex, felt, watermark"
GARMENT_REGIONS = ("upper", "lower", "body")


def colour_back(chosen_rgb, refined_rgb, sigma=COLOUR_BACK_SIGMA):
    """밝기(L)는 다듬기 결과, 색(a·b)은 고른 후보(흐림 sigma)에서. 같은 크기의 RGB uint8 배열 → RGB uint8."""
    from scipy.ndimage import gaussian_filter
    from genai_lab.style_reference_colors import lab_to_rgb, rgb_to_lab
    chosen, refined = np.asarray(chosen_rgb), np.asarray(refined_rgb)
    if chosen.shape != refined.shape:
        raise ValueError("고른 후보와 다듬기 결과의 크기가 다릅니다.")
    a, b = rgb_to_lab(chosen), rgb_to_lab(refined)
    ab = a[..., 1:] if not sigma else np.stack([gaussian_filter(a[..., i], sigma) for i in (1, 2)], -1)
    return np.clip(lab_to_rgb(np.concatenate([b[..., :1], ab], -1)), 0, 255).astype(np.uint8)


def style_mask_without_garments(profile, canvas):
    """그림체 참조를 적용할 영역: 캔버스 전체에서 옷 영역(윗옷·아래옷·몸)을 뺀 곳(머리·얼굴·배경)."""
    mask = Image.new("L", tuple(canvas), 255)
    for row in profile.get("outfits", []):
        if row["name"] in GARMENT_REGIONS:
            mask.paste(0, tuple(row["box"]))
    return mask


def refine_prompts(profile):
    """다듬기 문장: 1단계 장면·특징 문장 + 그림체 낱말. 부정 문장: 기본 + 목표 그림 내용 + 성별 부정(있으면)."""
    traits = "; ".join(row["text"].strip() for row in profile["traits"] if row["text"].strip())
    positive = profile["scene_prompt"].strip() + ". " + (traits + ". " if traits else "") + STYLE_WORDS
    extra = profile.get("negative_terms", "").strip()
    negative = BASE_NEGATIVE + ", " + CONTENT_NEGATIVE + (", " + extra if extra else "")
    return positive, negative
