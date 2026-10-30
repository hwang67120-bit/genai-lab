"""모드 B 얼굴 다듬기(D-206): 고른 후보 1장의 얼굴만 원본 쪽으로 다시 그리고, 눈 색은 고른 후보에서 되돌린다.

스타일 LoRA가 얼굴 그리는 방식(턱선·눈매)까지 끌어 V001·T017 얼굴이 원본과 달랐다. 얼굴 영역만 잘라 키워(1024) LoRA는 약하게(0.3),
원본 얼굴 참조는 강하게(0.8) 다시 그리면(원본 유지 50%) 얼굴이 원본 쪽으로 온다. 10명 시험에서 캐릭터끼리 얼굴 거리는 거의 그대로
(다양성 유지), 눈 색이 흔들려(T014·V001) 눈 영역만 색을 되돌린다. 사용자 0.5 선택(10/30).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FACE_STRENGTH = 0.5
FACE_STEPS = 30
FACE_LORA_WEIGHT = 0.3
FACE_REF_SCALE = 0.8
FACE_BOX_SCALE = 1.8
FACE_SIZE = 1024
EYE_DILATE = 1                          # 다듬기에서 눈이 조금 움직여도 덮도록(3이면 눈꺼풀까지 덮었다)
SCLERA_L, SCLERA_CHROMA = 78, 12        # 이보다 밝고 색이 없는 픽셀(흰자·반사광)은 칠하지 않는다 — V001 흰자가 보라로 번졌다(10/30)


def face_crop_box(face_box, canvas, scale=FACE_BOX_SCALE):
    """생성 자세의 얼굴 자리(identity_mask_box)를 가운데에 둔 정사각형 x1.8, 캔버스 안으로 자름."""
    x0, y0, x1, y1 = face_box
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, scale * max(x1 - x0, y1 - y0)
    w, h = canvas
    return [int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(w, cx + side / 2)), int(min(h, cy + side / 2))]


def feather_mask(size):
    """얼굴 조각을 붙일 때 쓰는 부드러운 타원 마스크(가장자리는 원래 그림이 남음)."""
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse((size[0] * .12, size[1] * .08, size[0] * .88, size[1] * .92), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(size[0] * .06))


def paste_face(full, redrawn, box):
    """다시 그린 얼굴(정사각형, 임의 크기)을 상자 크기로 줄여 부드럽게 붙인다. 원래 그림은 바꾸지 않고 새 그림을 돌려준다."""
    size = (box[2] - box[0], box[3] - box[1])
    small = redrawn.resize(size, Image.LANCZOS)
    merged = full.copy()
    merged.paste(small, tuple(box[:2]), feather_mask(size))
    return merged


def eye_colour_back(detailed_rgb, chosen_rgb, eyes_mask, face_mask=None, dilate=EYE_DILATE):
    """눈 영역(분할의 eyes, 조금 넓힘)의 홍채만 색(a·b)을 고른 후보에서 가져오고 밝기는 다듬은 그림 그대로.
    다듬은 그림에서 밝고 색이 없는 픽셀(흰자·반사광)과 얼굴 피부색 픽셀(눈꺼풀)은 칠하지 않는다."""
    from scipy.ndimage import binary_dilation
    from genai_lab.style_reference_colors import lab_to_rgb, rgb_to_lab, skin_like
    detailed, chosen = np.asarray(detailed_rgb), np.asarray(chosen_rgb)
    mask = np.asarray(eyes_mask, bool)
    if dilate:
        mask = binary_dilation(mask, iterations=dilate)
    a, b = rgb_to_lab(detailed), rgb_to_lab(chosen)
    sclera = (a[..., 0] > SCLERA_L) & (np.hypot(a[..., 1], a[..., 2]) < SCLERA_CHROMA)
    mask &= ~sclera
    if face_mask is not None:
        mask &= ~skin_like(detailed, face_mask)
    if not mask.any():
        return detailed.copy(), 0
    a[mask, 1:] = b[mask, 1:]
    out = detailed.copy()
    out[mask] = np.clip(lab_to_rgb(a), 0, 255).astype(np.uint8)[mask]
    return out, int(mask.sum())


def face_prompts(profile):
    """얼굴 다듬기 문장: 호출어 + 얼굴 확대 + 특징 문장. 부정 문장은 기본 + 성별 부정(있으면)."""
    from genai_lab.style_redraw_inputs import STYLE_TRIGGER
    traits = "; ".join(row["text"].strip() for row in profile["traits"] if row["text"].strip())
    positive = f"{STYLE_TRIGGER}, close-up portrait of the face, {traits}, detailed anime illustration"
    extra = profile.get("negative_terms", "").strip()
    return positive, "photograph, skin pores, flat cel shading, watermark" + (", " + extra if extra else "")
