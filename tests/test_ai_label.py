"""저장 결과 AI 생성 표시(D-152): 픽셀 그대로, PNG 텍스트 + XMP(IPTC 디지털 출처 유형)."""
import io

import numpy as np
import pytest
from PIL import Image

from genai_lab.ai_label import SOURCE_TYPES, label_record, labeled_png, read_label


def png(mode="RGB"):
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(12, 9, 3 if mode == "RGB" else 4), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(array, mode).save(out, format="PNG")
    return out.getvalue(), array


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_label_keeps_every_pixel(tmp_path, mode):
    data, array = png(mode)
    labeled = labeled_png(data, label_record("generated"))
    path = tmp_path / "saved.png"
    path.write_bytes(labeled)
    with Image.open(path) as image:
        assert image.mode == mode and np.array_equal(np.asarray(image), array)
        assert SOURCE_TYPES["generated"] in image.text["XML:com.adobe.xmp"]
    assert read_label(path) == dict(ai_generated=True, kind="generated", software="GenAI Lab",
                                    description="AI로 생성한 이미지입니다 (GenAI Lab)")


def test_composite_kind_and_no_label(tmp_path):
    data, _ = png()
    path = tmp_path / "plain.png"
    path.write_bytes(data)
    assert read_label(path) is None
    path.write_bytes(labeled_png(data, label_record("composite")))
    assert read_label(path)["kind"] == "composite"


def test_only_png_and_known_kinds():
    out = io.BytesIO()
    Image.new("RGB", (4, 4)).save(out, format="JPEG")
    with pytest.raises(ValueError):
        labeled_png(out.getvalue(), label_record("generated"))
    with pytest.raises(ValueError):
        label_record("photo")
