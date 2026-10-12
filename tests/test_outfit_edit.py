"""제품 공통 모듈: 고정 지시문과 캔버스 규칙(GPU 없이)."""
import numpy as np
import pytest
from PIL import Image

from genai_lab import outfit_edit


def test_product_prompt_has_no_garment_description_and_only_accessory_choice_differs():
    keep, remove = outfit_edit.outfit_prompt("keep"), outfit_edit.outfit_prompt("remove")
    assert "coat" not in keep and "coat" not in remove          # 의상은 그림 2를 따른다(시험 지시문의 의상 묘사 제거)
    assert "head accessories" in keep and "same person" in remove
    assert keep.startswith(outfit_edit.OUTFIT_BASE) and remove.startswith(outfit_edit.OUTFIT_BASE)
    with pytest.raises(ValueError):
        outfit_edit.outfit_prompt("maybe")


@pytest.mark.parametrize("size", [(400, 1200), (1200, 400), (768, 1024)])
def test_canvas_keeps_whole_figure_on_white(size):
    image = Image.new("RGB", size, (10, 20, 30))
    out = np.asarray(outfit_edit.canvas(image))
    assert out.shape == (1024, 768, 3)
    painted = (out != 255).any(axis=2)
    ys, xs = np.nonzero(painted)
    w, h = size
    assert abs((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) - w / h) < 0.02   # 비율 유지, 자르지 않음
