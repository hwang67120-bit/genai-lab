from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from genai_lab.clothing_analysis import (
    ClothingTagLabel,
    build_general_tag_candidates,
    load_wd14_tag_labels,
    prepare_wd14_image,
)


@pytest.mark.parametrize('tag', ['1girl', '1boy', '6+girls', '2boys', '1other', 'male_focus', 'FEMALE', 'multiple_girls'])
def test_garment_person_tags_excluded(tag):
    from genai_lab.reference_tag_policy import excluded_garment_tag
    assert excluded_garment_tag(tag)


@pytest.mark.parametrize('tag', ['blue_jacket', 'high_heels', 'skirt', 'pants', 'white_shirt'])
def test_garment_design_tags_remain(tag):
    from genai_lab.reference_tag_policy import excluded_garment_tag
    assert not excluded_garment_tag(tag)


def test_generic_wd_image_analysis_accepts_character_source(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    import genai_lab.clothing_analysis as module
    labels = [module.ClothingTagLabel('1boy', 0), module.ClothingTagLabel('blue_hair', 0)]
    monkeypatch.setattr(module, 'download_wd14_model_files', lambda settings: (tmp_path/'model.onnx', tmp_path/'labels.csv'))
    monkeypatch.setattr(module, 'load_wd14_tag_labels', lambda path: labels)
    class Session:
        def __init__(self, path, providers):
            assert providers == ['CPUExecutionProvider']
        def get_inputs(self):
            return [SimpleNamespace(name='image', shape=[1, 32, 32, 3])]
        def run(self, outputs, feed):
            assert feed['image'].shape == (1, 32, 32, 3)
            return [np.array([[.9, .8]], dtype=np.float32)]
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        InferenceSession=Session, get_available_providers=lambda: ['CPUExecutionProvider']))
    image = Image.new('RGB', (64, 96), 'blue')
    result = module.analyze_image_tags(image, module.ClothingDesignAnalysisSettings())
    assert (result.input_width, result.input_height) == (64, 96)
    assert [tag.tag_name for tag in result.tag_candidates] == ['1boy', 'blue_hair']
    image.close()


def test_wd14_labels_keep_numeric_categories(tmp_path: Path) -> None:
    label_path = tmp_path / "selected_tags.csv"
    label_path.write_text(
        "tag_id,name,category,count\n"
        "0,safe,9,1\n"
        "1,white_shirt,0,1\n"
        "2,character_name,4,1\n",
        encoding="utf-8",
    )

    labels = load_wd14_tag_labels(label_path)

    assert [(label.name, label.category) for label in labels] == [
        ("safe", 9),
        ("white_shirt", 0),
        ("character_name", 4),
    ]


def test_wd14_preparation_crops_alpha_and_returns_bgr_square() -> None:
    extracted_image = Image.new("RGBA", (6, 4), (0, 0, 0, 0))
    for x in range(2, 4):
        for y in range(1, 3):
            extracted_image.putpixel((x, y), (10, 20, 30, 255))

    prepared_image = prepare_wd14_image(
        extracted_image,
        model_input_size=4,
    )

    extracted_image.close()
    assert prepared_image.shape == (1, 4, 4, 3)
    assert prepared_image.dtype == np.float32
    assert np.all(prepared_image[0] == np.array([30, 20, 10]))


def test_general_tags_exclude_rating_and_character_then_limit_count() -> None:
    labels = (
        ClothingTagLabel(name="safe", category=9),
        ClothingTagLabel(name="white_shirt", category=0),
        ClothingTagLabel(name="long_sleeves", category=0),
        ClothingTagLabel(name="character_name", category=4),
        ClothingTagLabel(name="buttons", category=0),
    )
    scores = np.array([0.99, 0.90, 0.80, 0.95, 0.20], dtype=np.float32)

    candidates = build_general_tag_candidates(
        tag_labels=labels,
        model_scores=scores,
        score_threshold=0.35,
        maximum_tag_count=2,
    )

    assert [candidate.tag_name for candidate in candidates] == [
        "white_shirt",
        "long_sleeves",
    ]
    assert candidates[0].display_name == "white shirt"
    assert candidates[0].score == pytest.approx(0.90)
