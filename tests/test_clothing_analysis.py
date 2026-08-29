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
