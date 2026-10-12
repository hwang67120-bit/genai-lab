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


def test_garment_terms_keep_clothing_and_drop_pose_tags():
    # 제품 의상 분석이 이 정장에서 실제로 낸 태그(outputs/studio-runs/179d…/inputs/analysis.json)
    tags = ['shirt', 'suspenders', 'pants', 'hands in pockets', 'shoes', 'black ribbon', 'coat', 'Coat', 'black_socks']
    terms = outfit_edit.clean_garment_terms(tags)
    assert "hands in pockets" not in terms and terms.count("coat") == 1
    assert {"shoes", "black ribbon", "black socks"} <= set(terms)
    prompt = outfit_edit.outfit_prompt("keep", terms)
    assert "consists of: shirt, suspenders" in prompt and prompt.startswith(outfit_edit.OUTFIT_BASE)
    with pytest.raises(ValueError):
        outfit_edit.outfit_prompt("keep", ("shirt", "hands in pockets"))   # 정리되지 않은 목록은 거부


def test_base_prompt_asks_to_replace_all_current_clothing():
    # 요청 12 W3: "그림 2의 옷을 입혀라"만으로는 몸에 붙는 원래 옷이 남았다
    assert "Replace all of the character's current clothing" in outfit_edit.outfit_prompt("keep")
