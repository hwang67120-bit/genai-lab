"""색 공간 도우미 시험: Lab 왕복과 피부처럼 보이는 픽셀(모델 불필요)."""
import numpy as np

from genai_lab.style_reference_colors import lab_to_rgb, rgb_to_lab, skin_like


def test_lab_round_trip():
    rgb = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert np.abs(lab_to_rgb(rgb_to_lab(rgb)).astype(int) - rgb).max() <= 1


def test_skin_like_follows_face_colour_and_skips_other_hues():
    rgb = np.zeros((40, 40, 3), np.uint8); rgb[:] = (60, 120, 200)          # 파란 옷
    rgb[:20] = (230, 190, 160)                                              # 피부
    face = np.zeros((40, 40), bool); face[2:12, 5:30] = True
    mask = skin_like(rgb, face)
    assert mask[15, 10] and not mask[30, 10]
    assert not skin_like(rgb, np.zeros_like(face)).any()                     # 얼굴이 없으면 빈 마스크
