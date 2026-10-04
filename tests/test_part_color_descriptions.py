"""색상 명명/프롬프트 연결 검증. 모델의 색상 재현 정확도 검증은 아니다."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image
from genai_lab.part_color_descriptions import (
    PartColorNamer, PartColorDescription, color_prompt_tags, bind_color_descriptions,
    describe_input_part_colors, color_description_summary,
)
from genai_lab.part_color_analysis import analyze_part_colors
from genai_lab.reference_prompt_budget import build_reference_prompt, PromptBudgetError

ROOT = Path(__file__).resolve().parents[1]


def namer(**kwargs):
    return PartColorNamer(json.loads((ROOT/"configs/part_color_vocabulary.json").read_text(encoding="utf-8")), **kwargs)


def evidence(colors, ratios=None):
    return SimpleNamespace(palette_rgb=colors, area_ratios=ratios or [1/len(colors)]*len(colors), pixel_count=100)


def proposal(name="tail", colors=("blue",)):
    return PartColorDescription(name, colors, "proposed", (), b"{}", "test")


@pytest.mark.parametrize("rgb,family", [
    ((0,0,255), "blue"), ((255,0,0), "red"), ((0,128,0), "green"),
    ((255,255,0), "yellow"), ((128,0,128), "purple"),
    ((0,255,255), "cyan"), ((255,255,255), "white"), ((0,0,0), "black"),
])
def test_general_colors_not_blue_specific(rgb, family):
    result = namer().describe("tail", evidence([rgb]))
    assert result.status == "proposed"
    assert any(color.split()[-1] == family for color in result.colors)


def test_multicolor_and_brightness_preserved_without_pattern_inference():
    result = namer().describe("tail", evidence([(0,0,255),(255,0,0)], [.6,.4]))
    assert len(result.colors) == 2
    assert "blue" in result.prompt_text and "red" in result.prompt_text
    assert not any(word in result.prompt_text for word in ("striped", "gradient", "tip", "girl", "boy"))
    tones = proposal(colors=("dark blue","light blue","blue"))
    assert tones.prompt_text == "dark and light blue tail"


def test_ambiguous_families_do_not_force_single_color():
    vocabulary = {"families": {"blue": {"samples":["gray"]}, "purple": {"samples":["gray"]}}}
    result = PartColorNamer(vocabulary).describe("ears", evidence([(128,128,128)]))
    assert result.status == "unresolved" and result.prompt_text == ""
    assert result.record()["evidence"]["clusters"][0]["margin"] == 0


@pytest.mark.parametrize("kind", ["missing", "empty", "complex", "far"])
def test_unresolved_is_explicit(kind):
    n = namer(maximum_colors=1) if kind == "complex" else namer(max_distance=.001) if kind == "far" else namer()
    ev = None if kind == "missing" else evidence([(0,0,255)])
    if kind == "empty": ev.pixel_count = 0
    if kind == "complex": ev = evidence([(0,0,255),(255,0,0)])
    if kind == "far": ev = evidence([(1,2,3)])
    result = n.describe("ears", ev)
    assert result.status == "unresolved" and not color_prompt_tags((result,))
    assert "미전달" in color_description_summary((result,))


def test_source_mask_and_other_parts_are_not_modified_or_used():
    with Image.new("RGB", (20,10), "red") as source, Image.new("L", (20,10)) as mask:
        source.paste((0,0,255), (0,0,10,10))
        mask.paste(255, (0,0,10,10))
        before = (source.tobytes(), mask.tobytes())
        measured = analyze_part_colors(source, mask)
        result = namer().describe("tail", measured)
        assert result.colors == ("dark blue",)
        assert before == (source.tobytes(), mask.tobytes())
        before_json = result.evidence_json
        measured.palette_rgb[0][:] = [255,0,0]
        assert result.evidence_json == before_json


def test_only_ear_tail_are_routed_and_missing_measurement_is_not_absence():
    result = describe_input_part_colors({"hair": evidence([(255,0,0)]),
        "garment": evidence([(0,255,0)]), "tail": evidence([(0,0,255)])}, {"enabled":True}, ROOT)
    assert tuple(d.part_name for d in result) == ("ears","tail")
    assert result[0].status == "unresolved"
    assert result[1].colors == ("dark blue",)
    assert describe_input_part_colors({}, {}, ROOT) == ()


def test_descriptions_bind_only_generic_part_tags_not_species_or_outfit():
    original = ("1boy", "animal ears", "tail", "raccoon ears", "raccoon tail", "blue hair")
    desc = (proposal("ears", ("blue",)), proposal())
    combined, replaced = bind_color_descriptions(original, desc)
    assert replaced == ("animal ears","tail")
    assert "raccoon ears" in combined and "raccoon tail" in combined
    assert "blue animal ears" in combined and "blue tail" in combined
    assert original[1:3] == ("animal ears", "tail")


class Tokenizer:
    model_max_length = 77
    def __call__(self, text, **kwargs):
        return {"input_ids": [0, *range(len(text.split())), 1]}


def test_color_conditions_are_required_and_optional_quality_goes_first():
    core, _ = bind_color_descriptions(("1boy","tail"), (proposal(),))
    tokenizer = Tokenizer()
    tokenizer.model_max_length = 9
    text, report = build_reference_prompt((tokenizer,), core_character=core,
        core_outfit=("skirt",), optional=("natural fabric folds","best quality"))
    assert "blue tail" in text and "wearing skirt" in text
    assert "natural fabric folds" in report["omitted_optional"]
    tokenizer.model_max_length = 4
    with pytest.raises(PromptBudgetError):
        build_reference_prompt((tokenizer,), core_character=core, core_outfit=("skirt",))


def test_ui_requires_explicit_color_approval_and_keeps_unresolved_unselectable():
    from PySide6.QtWidgets import QApplication
    from genai_lab.character_tag_review import CharacterTagReview
    app = QApplication.instance() or QApplication([])
    desc = (proposal(), namer().describe("ears", None))
    with Image.new("RGB", (32,32)) as image:
        dialog = CharacterTagReview(image, SimpleNamespace(tag_candidates=()), part_color_descriptions=desc)
        try:
            assert "꼬리" in dialog.part_color_summary.text()
            assert "판단 보류" in dialog.part_color_summary.text()
            assert "선택한 항목만 생성 조건에 전달" in dialog.part_color_summary.text()
            assert len(dialog.part_color_options) == 1
            assert dialog.approved_part_color_names == ()
            dict(dialog.part_color_options)["tail"].click()
            assert dialog.approved_part_color_names == ("tail",)
            assert len(dialog.tag_checkboxes) == 0
            assert not dialog.result()
        finally:
            dialog.close()
