"""얼굴 다듬기(D-243, 사용자 11/1 "남은 크레딧 다 쓰고 이미지 생성 마무리" — 전신 구도에서 얼굴이 작아 눈·얼굴 세부가 뭉개짐).

후보 그림에서 머리를 찾아(deepghs anime_head_detection) 머리 상자의 1.9배 정사각형을 잘라 1024로 키우고, 같은 FLUX + 그림체 LoRA로
'원래 얼굴에서 출발해 뒤쪽 단계만 다시 그리기'(flux_img2img, 세기 0.5)를 한 뒤 원래 크기로 줄여 위상상관으로 맞추고 가장자리를
부드럽게 한 타원으로 붙인다. 편집으로 새로 그리면 구도·표정이 바뀌어 붙일 수 없었다(S41 1판 6/6 어긋남). 어긋남이 크거나 머리를
못 찾으면 그대로 둔다. S41 2판: 6/6 붙음(어긋남 0.2px 이하), 후보 1장 +18초, 최고 6.0GiB.
"""
import numpy as np

CROP_SCALE = 1.9            # 머리 상자의 1.9배 정사각형(귀·머리카락 끝까지)
FACE_SIZE = 1024
STRENGTH = 0.5              # 4단계 중 뒤쪽 2단계만(0.75는 세부가 더 바뀜)
MAX_SHIFT = 12.0            # 원래 크기에서 이보다 어긋나면 붙이지 않는다
FEATHER = 0.18              # 타원 가장자리 부드럽게(자른 변 대비)
MIN_SIDE = 96               # 이보다 작은 머리 상자는 다듬지 않는다(검출 잡음)
SEED = 7
PROMPT = ("Redraw image 1, a close-up of the character's head, as the same picture with much finer detail: keep exactly the same "
          "face, expression, head angle, hair shape, hair colour, eye colour, ears, accessories, lighting and framing as image 1. "
          "The character is the one shown in image 2. Sharp detailed eyes with clear reflections and eyelashes, fine individual "
          "hair strands, clean crisp line art, smooth soft shading. No text, no new objects.")


def head_box(rgb, head_cache):
    """가장 확실한 머리 상자 [x0, y0, x1, y1]과 점수. 없으면 None."""
    import huggingface_hub.constants as hub
    from PIL import Image
    hub.HF_HUB_CACHE = str(head_cache)
    from imgutils.detect import detect_heads
    found = detect_heads(Image.fromarray(np.asarray(rgb, np.uint8)))
    if not found:
        return None
    (x0, y0, x1, y1), _, score = max(found, key=lambda d: d[2])
    return [int(x0), int(y0), int(x1), int(y1)], float(score)


def square(box, w, h):
    """머리 상자 → 그림 안에 들어가는 정사각형 (left, top, side)."""
    x0, y0, x1, y1 = box
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, CROP_SCALE * max(x1 - x0, y1 - y0)
    side = int(min(side, w, h))
    left = int(min(max(cx - side / 2, 0), w - side))
    top = int(min(max(cy - side / 2, 0), h - side))
    return left, top, side


def paste(base, detail, left, top, side):
    """detail(side×side)을 위상상관으로 맞춰 타원 가중치로 base에 붙인다. 반환 (그림, dict)."""
    import cv2
    from scipy.ndimage import gaussian_filter
    base = np.asarray(base, np.uint8)
    detail = np.asarray(detail, np.uint8)
    region = base[top:top + side, left:left + side]
    gray = lambda a: cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(gray(detail), gray(region), cv2.createHanningWindow((side, side), cv2.CV_32F))
    info = dict(shift=[round(float(dx), 1), round(float(dy), 1)], response=round(float(response), 3))
    if max(abs(dx), abs(dy)) > MAX_SHIFT:
        return base, dict(info, pasted=False)
    moved = cv2.warpAffine(detail, np.float32([[1, 0, dx], [0, 1, dy]]), (side, side), borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.mgrid[0:side, 0:side]
    r = np.sqrt(((xx - side / 2) / (side / 2)) ** 2 + ((yy - side / 2) / (side / 2)) ** 2)
    weight = gaussian_filter(np.clip((1.0 - r) / FEATHER, 0, 1).astype(np.float32), side * 0.02)[..., None]
    out = base.copy()
    out[top:top + side, left:left + side] = np.clip(moved * weight + region * (1 - weight), 0, 255).astype(np.uint8)
    return out, dict(info, pasted=True)


def detail_face(pipe, rgb, reference, head_cache, prompt_prefix="", steps=4, guidance=1.0, max_sequence_length=256):
    """후보 하나(RGB 배열)의 얼굴을 다듬는다. 반환 (RGB 배열, dict). 머리를 못 찾거나 작거나 어긋나면 원래 그림."""
    from PIL import Image
    from genai_lab.flux_img2img import redraw
    found = head_box(rgb, head_cache)
    if not found:
        return rgb, dict(pasted=False, reason="머리를 찾지 못함")
    box, score = found
    if max(box[2] - box[0], box[3] - box[1]) < MIN_SIDE / CROP_SCALE:
        return rgb, dict(pasted=False, reason="머리가 너무 작음", box=box, score=round(score, 3))
    h, w = rgb.shape[:2]
    left, top, side = square(box, w, h)
    crop = Image.fromarray(np.asarray(rgb, np.uint8)[top:top + side, left:left + side]).resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS)
    detail = redraw(pipe, crop, [crop, reference], f"{prompt_prefix} {PROMPT}".strip(), STRENGTH, SEED, steps=steps, guidance=guidance,
                    max_sequence_length=max_sequence_length)
    out, info = paste(rgb, np.asarray(detail.resize((side, side), Image.LANCZOS)), left, top, side)
    return out, dict(info, box=box, score=round(score, 3), crop=[left, top, side])
