"""얼굴 다듬기(D-243): 자를 정사각형·붙이기(위상상관 맞춤, 크게 어긋나면 안 붙임). 모델 불필요."""
import numpy as np

from genai_lab.face_detail import MAX_SHIFT, paste, square


def test_square_is_inside_the_picture():
    left, top, side = square([300, 0, 500, 180], 1024, 1360)
    assert side == int(1.9 * 200) and top == 0 and 0 <= left and left + side <= 1024     # 위쪽 끝에 붙은 머리는 위로 넘치지 않음
    left, top, side = square([900, 1200, 1024, 1360], 1024, 1360)
    assert left + side <= 1024 and top + side <= 1360


def _picture(seed=0):
    rng = np.random.default_rng(seed)
    base = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(base, (3, 3, 0)).astype(np.uint8)


def test_paste_keeps_position_and_refuses_large_shifts():
    base = _picture()
    left, top, side = 64, 64, 128
    region = base[top:top + side, left:left + side].copy()
    sharper = np.clip(region.astype(int) + 5, 0, 255).astype(np.uint8)
    out, info = paste(base, sharper, left, top, side)
    assert info["pasted"] and max(abs(v) for v in info["shift"]) < 1
    assert out[top + side // 2, left + side // 2].tolist() == sharper[side // 2, side // 2].tolist()   # 가운데는 다듬은 것
    assert (out[:top] == base[:top]).all()                                                       # 상자 밖은 그대로
    moved = np.roll(sharper, int(MAX_SHIFT) + 20, axis=1)
    out2, info2 = paste(base, moved, left, top, side)
    assert not info2["pasted"] and (out2 == base).all()
