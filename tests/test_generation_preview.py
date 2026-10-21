"""생성 과정 미리보기: 묶음 풀기가 diffusers 배치와 같고, 변환식이 맞고, 파일을 원자적으로 쓴다."""
import numpy as np
import pytest

from genai_lab.generation_preview import load_factors, preview_rgb, unpack, write_preview


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


def test_write_preview_replaces_the_file(tmp_path):
    rgb = np.zeros((4, 6, 3), np.uint8)
    path = write_preview(tmp_path, rgb)
    from PIL import Image
    with Image.open(path) as image:
        assert image.size == (24, 16)
    write_preview(tmp_path, rgb + 10)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["preview.png"]
