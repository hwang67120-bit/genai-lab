"""모드 B 부위별 고르기(D-214, 사용자 10/31): 후보 하나의 얼굴·머리 + 다른 후보의 몸·옷.

후보 3장은 같은 원본·같은 자세로 그려져 머리 위치가 거의 같다(10명 30짝에서 어긋남 4.3px 이하). 두 후보의 머리 마스크
(머리카락·얼굴·귀·눈·코·입)를 합쳐 넓히고 부드럽게 한 뒤, 머리 상자 안의 위상상관으로 잰 작은 어긋남을 되돌려 섞는다.
사람 영역(얼굴 후보) 밖은 몸 후보 그대로 둔다 — 후보마다 배경 빛이 달라 머리 둘레에 테두리가 생겼다(V002).
어긋남이 크면 섞지 않는다(겹친 머리 = 구조 실패). 상관이 약하면(머리 모양이 많이 다름) 옮기지 않고 섞고 표시한다. CPU만 쓴다.
"""
import numpy as np

HEAD_KEYS = ("hair", "face", "ears", "eyes", "nose", "mouth")
DILATE = 6
FEATHER = 5.0
MAX_SHIFT_PX = 8.0
PERSON_GROW = 2                 # 사람 영역 경계 여유(머리카락 끝 반투명)
EDGE_FEATHER = 1.5
BOX_MARGIN = 16                 # 머리 상자 둘레 여유(가장자리에서 잰 값이 흔들리지 않게)
MIN_RESPONSE = 0.15             # 위상상관 응답이 이보다 약하면 어긋남을 믿지 않는다(10명 같은 캐릭터 후보 짝 최소 0.23, 다른 캐릭터 짝 최대 0.09)


def head_mask(parts, shape):
    """분할 결과(figure_parts)에서 머리 전체 마스크."""
    empty = np.zeros(shape, bool)
    return np.logical_or.reduce([np.asarray(parts.get(k, empty), bool) for k in HEAD_KEYS])


def combine_parts(face_rgb, body_rgb, face_head, body_head, face_person=None):
    """face_rgb의 머리를 body_rgb에 얹는다. face_person: 얼굴 후보의 사람 영역(있으면 그 밖은 몸 후보 배경).
    반환: (RGB uint8, dict(shift=[dx, dy], response, aligned))."""
    import cv2
    from scipy.ndimage import binary_dilation, gaussian_filter
    face_rgb, body_rgb = np.asarray(face_rgb, np.uint8), np.asarray(body_rgb, np.uint8)
    if face_rgb.shape != body_rgb.shape:
        raise ValueError("두 후보의 크기가 다릅니다.")
    union = np.asarray(face_head, bool) | np.asarray(body_head, bool)
    if not union.any():
        raise ValueError("후보에서 머리 영역을 찾지 못했습니다.")
    h, w = body_rgb.shape[:2]
    ys, xs = np.nonzero(union)
    y0, y1 = max(ys.min() - BOX_MARGIN, 0), min(ys.max() + 1 + BOX_MARGIN, h)
    x0, x1 = max(xs.min() - BOX_MARGIN, 0), min(xs.max() + 1 + BOX_MARGIN, w)
    gray = lambda a: cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)[y0:y1, x0:x1]
    (dx, dy), response = cv2.phaseCorrelate(gray(face_rgb), gray(body_rgb), cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F))
    aligned = response >= MIN_RESPONSE
    if not aligned:                     # 두 머리가 많이 달라 상관이 약하면 잰 값은 잡음 — 옮기지 않고 그대로 섞는다
        dx = dy = 0.0
    elif max(abs(dx), abs(dy)) > MAX_SHIFT_PX:
        raise ValueError(f"두 후보의 머리 위치가 너무 다릅니다(어긋남 {dx:.1f}, {dy:.1f}px). 다른 짝을 골라 주세요.")
    move = np.float32([[1, 0, dx], [0, 1, dy]])
    face_moved = cv2.warpAffine(face_rgb, move, (w, h), borderMode=cv2.BORDER_REPLICATE)
    head_moved = cv2.warpAffine(np.asarray(face_head, np.uint8), move, (w, h)) > 0
    weight = gaussian_filter(binary_dilation(head_moved | np.asarray(body_head, bool), iterations=DILATE).astype(np.float32),
                             FEATHER)
    if face_person is not None:
        person_moved = cv2.warpAffine(np.asarray(face_person, np.uint8), move, (w, h)) > 0
        weight = weight * gaussian_filter(binary_dilation(person_moved, iterations=PERSON_GROW).astype(np.float32), EDGE_FEATHER)
    weight = weight[..., None]
    out = np.clip(face_moved * weight + body_rgb * (1 - weight), 0, 255).astype(np.uint8)
    return out, dict(shift=[round(float(dx), 2), round(float(dy), 2)], response=round(float(response), 3), aligned=bool(aligned))
