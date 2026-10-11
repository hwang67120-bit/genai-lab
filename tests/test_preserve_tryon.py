"""보존형 의상 교체 시험 스크립트: 캔버스 규칙과 '자동 마스크 - 보존 영역' 계산을 GPU 없이 확인한다."""
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canvas_keeps_whole_image_with_white_padding():
    prepare = load("preserve_tryon_prepare")
    tall = Image.new("RGB", (100, 400), (10, 20, 30))
    out = prepare.canvas(tall)
    arr = np.asarray(out)
    assert out.size == (768, 1024)
    assert (arr[:, :256] == 255).all() and (arr[:, -256:] == 255).all()   # 좌우 흰 여백(그림 폭 256px)
    assert (arr[:, 384] == (10, 20, 30)).all()                              # 위아래는 잘리지 않음


def test_change_mask_never_touches_protected_area_plus_margin():
    tryon = load("preserve_tryon")
    auto = np.zeros((1024, 768), np.uint8); auto[100:900, 100:700] = 255   # 얼굴까지 덮은 자동 마스크
    protect = np.zeros((1024, 768), bool); protect[100:250, 300:500] = True  # 얼굴·머리
    change = tryon.final_mask(auto, protect)
    margin = tryon.PROTECT_MARGIN
    assert not change[100:250 + margin, 300 - margin:500 + margin].any()
    assert change[400:900, 100:700].all()
