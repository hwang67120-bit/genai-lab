"""생성 과정 미리보기: 묶음 풀기가 diffusers 배치와 같고, 변환식이 맞고, 파일을 원자적으로 쓴다."""
import numpy as np
import pytest

from genai_lab.generation_preview import latest_preview, load_factors, preview_rgb, unpack, write_preview


def pack(grid):
    """(16, h, w) → (1, 묶음, 64). diffusers QwenImage `_pack_latents`와 같은 배치."""
    c, h, w = grid.shape
    return grid.reshape(c, h // 2, 2, w // 2, 2).transpose(1, 3, 0, 2, 4).reshape(1, (h // 2) * (w // 2), c * 4)


def test_unpack_inverts_diffusers_packing():
    grid = np.random.default_rng(0).normal(size=(16, 6, 8)).astype(np.float32)
    assert np.array_equal(unpack(pack(grid), 48, 64), grid)
    with pytest.raises(ValueError):
        unpack(pack(grid), 64, 64)


def test_preview_uses_the_fitted_linear_map():
    weights, bias = load_factors()
    grid = np.zeros((16, 4, 4), np.float32)
    rgb = preview_rgb(pack(grid), 32, 32)
    assert rgb.shape == (4, 4, 3) and (rgb == (np.clip(bias, 0, 1) * 255).round()).all()
    grid[3] = 1.0
    expected = (np.clip(weights[3] + bias, 0, 1) * 255).round()
    assert (preview_rgb(pack(grid), 32, 32)[0, 0] == expected).all()


def test_write_preview_uses_new_names_and_keeps_the_last_few(tmp_path):
    rgb = np.zeros((4, 6, 3), np.uint8)
    path = write_preview(tmp_path, rgb, 0)
    from PIL import Image
    with Image.open(path) as image:
        assert image.size == (24, 16)
    for step in range(1, 6):
        write_preview(tmp_path, rgb + step, step)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["preview-003.png", "preview-004.png", "preview-005.png"]
    assert latest_preview([tmp_path]).name == "preview-005.png"


def test_old_preview_held_open_does_not_stop_new_steps(tmp_path):
    # 요청 34: 작업실이 읽는 중인 파일을 교체하려다 Windows 접근 거부로 미리보기가 멈췄다 — 새 이름이면 막히지 않는다
    rgb = np.zeros((4, 6, 3), np.uint8)
    first = write_preview(tmp_path, rgb, 0)
    with open(first, "rb"):
        for step in range(1, 5):
            write_preview(tmp_path, rgb, step)
    assert latest_preview([tmp_path]).name == "preview-004.png"
