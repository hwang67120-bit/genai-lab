"""색 공간 도우미(CIE Lab, sRGB D65)와 피부처럼 보이는 픽셀 찾기. 모드 B 색 맞춤(style_redraw_flux)이 쓴다.

예전 그림체 참조 색 맞춤(recolor_reference 등, D-187~D-195)은 참조 그림 경로와 함께 D-216에서 지웠다 — 기록은 git 역사.
"""
import numpy as np

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    c = xyz @ np.linalg.inv(_M).T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c * 255.0 + 0.5, 0, 255).astype(np.uint8)


def skin_like(rgb, face):
    """얼굴 부위의 색상·채도 범위(5~95%)에 드는 픽셀. 얼굴이 너무 작으면 빈 마스크."""
    import PIL.Image
    hsv = np.asarray(PIL.Image.fromarray(np.asarray(rgb, np.uint8)).convert("HSV")).astype(np.int16)
    face = np.asarray(face, bool) & (hsv[..., 2] > 60)
    if face.sum() < 50:
        return np.zeros(face.shape, bool)
    h5, h95 = np.percentile(hsv[..., 0][face], [5, 95])
    s5, s95 = np.percentile(hsv[..., 1][face], [5, 95])
    return ((hsv[..., 0] >= h5 - 8) & (hsv[..., 0] <= h95 + 8) & (hsv[..., 1] >= max(s5 - 20, 18))
            & (hsv[..., 1] <= s95 + 60) & (hsv[..., 2] > 35))
