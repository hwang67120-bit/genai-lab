from dataclasses import dataclass
from types import SimpleNamespace
import pytest
from PIL import Image
from genai_lab.request import CharacterFramingType
from genai_lab.reference_tag_policy import resolve_character_gender
from genai_lab.clothing_reference_generation import prepare_design_reference_request


@dataclass
class Request:
    prompt: str = "unchanged"
    negative_prompt: str = "bad anatomy, different eye color, different outfit"
    framing_type: CharacterFramingType = CharacterFramingType.FULL_BODY


class Tokenizer:
    model_max_length = 77
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}


@pytest.mark.parametrize("gender,expected", [("male", "1boy"), ("female", "1girl")])
def test_explicit_gender_overrides_detected_tags_without_changing_appearance(gender, expected):
    tags = ("1girl", "1boy", "female_focus", "male", "gender_swap",
            "androgynous", "long_hair", "slender", "purple_eyes", "cat_ears")
    resolved, removed = resolve_character_gender(tags, gender)
    assert resolved == (expected, "androgynous", "long hair", "slender", "purple eyes", "cat ears")
    assert set(removed) == {"1girl", "1boy", "female focus", "male", "gender swap"}
    assert tags[0] == "1girl"


def test_unspecified_preserves_approved_gender_and_no_guess():
    assert resolve_character_gender(("1boy", "blue_hair")) == (("1boy", "blue hair"), ())
    assert resolve_character_gender(())[0] == ()


@pytest.mark.parametrize("value", [None, "", "auto", "unknown", 1])
def test_invalid_gender_rejected(value):
    with pytest.raises(ValueError):
        resolve_character_gender(("blue hair",), value)


def test_comma_group_cannot_reintroduce_conflicting_tag():
    assert resolve_character_gender(("1girl, blue_hair",), "male")[0] == ("1boy", "blue hair")


@pytest.mark.parametrize("gender,expected,blocked", [
    ("male", "1boy", "1girl"), ("female", "1girl", "1boy"),
])
def test_prompt_gender_and_negative_conflicts(gender, expected, blocked):
    request = Request(negative_prompt=f"{expected}, bad anatomy, different eye color, different outfit")
    before = request.negative_prompt
    prepared, record = prepare_design_reference_request(
        request, ("blue_jacket", "white_shirt", "skirt"),
        (Tokenizer(), Tokenizer()),
        character_tags=(blocked, "androgynous", "blue_hair"), character_gender=gender)
    assert prepared.prompt.startswith(expected + ", ")
    assert blocked not in prepared.prompt
    assert prepared.prompt.index("blue hair") < prepared.prompt.index("androgynous")
    assert "wearing blue jacket, white shirt, skirt" in prepared.prompt
    assert "muscular" not in prepared.prompt and "beard" not in prepared.prompt
    assert prepared.negative_prompt == f"{blocked}, bad anatomy, different eye color"
    assert request.negative_prompt == before
    assert record["character_gender"] == gender and record["gender_source"] == "user"
    assert record["removed_gender_negative_terms"] == [expected]


@pytest.mark.parametrize('gender,guard', [('male', '1girl'), ('female', '1boy'), ('unspecified', None)])
def test_garment_gender_and_body_tags_do_not_enter_character_conditions(gender, guard):
    prepared, record = prepare_design_reference_request(
        Request(), ('1girl, BLUE_JACKET', 'male_focus', 'large_breasts', 'wide_hips',
                    'muscular', 'female_body', 'skirt', 'high_heels'), (Tokenizer(),),
        character_tags=('androgynous', 'slender'), character_gender=gender)
    assert record['approved_tags'] == ('blue jacket', 'skirt', 'high heels')
    assert 'androgynous, slender' in prepared.prompt
    assert record['gender_negative_guard'] == guard
    assert record['garment_visual_gender_isolated'] is False
    if guard:
        assert prepared.negative_prompt.startswith(guard + ',')
    else:
        assert '1girl' not in prepared.prompt and '1boy' not in prepared.prompt


def test_gender_guard_survives_negative_truncation():
    prepared, record = prepare_design_reference_request(
        Request(negative_prompt=', '.join(['bad anatomy'] * 100)), ('jacket',),
        (Tokenizer(),), character_gender='male')
    assert prepared.negative_prompt.startswith('1girl,')
    assert record['negative']['truncated'] is True


def test_only_garment_body_gender_tags_cannot_generate():
    with pytest.raises(ValueError, match='의상 디자인 태그가 없습니다'):
        prepare_design_reference_request(Request(), ('1girl, large breasts',), (Tokenizer(),))


def test_automatic_garment_summary_excludes_body_but_keeps_skirt():
    from genai_lab.reference_features import automatic_feature_selection
    candidates = [SimpleNamespace(tag_name=name, score=.9)
                  for name in ('large_breasts', 'female_focus', 'muscular', 'skirt', 'high_heels')]
    summary = automatic_feature_selection(candidates, 'garment')
    assert summary['selected'] == ('skirt', 'high_heels')
    assert summary['excluded'] == ('large_breasts', 'female_focus', 'muscular')
    assert not summary['unresolved']


def test_gender_only_is_valid_without_detected_character_tags():
    prepared, record = prepare_design_reference_request(
        Request(), ("jacket",), (Tokenizer(),), character_gender="male")
    assert prepared.prompt.startswith("1boy, ")
    assert record["approved_character_tags"] == ("1boy",)


def test_gender_ui_and_detector_checkbox_priority():
    from PySide6.QtWidgets import QApplication
    from genai_lab.clothing_reference_generation_review import ClothingReferenceGenerationDialog
    from genai_lab.character_tag_review import CharacterTagReview
    app = QApplication.instance() or QApplication([])
    settings = ClothingReferenceGenerationDialog((768, 1344), ("jacket",))
    assert settings.character_gender is None
    settings.character_gender_input.setCurrentIndex(settings.character_gender_input.findData("male"))
    assert settings.character_gender == "male"
    image = Image.new("RGB", (32, 48), "white")
    result = SimpleNamespace(tag_candidates=[
        SimpleNamespace(tag_name=tag, display_name=tag, score=.9)
        for tag in ("1girl", "1boy", "blue_hair", "androgynous")])
    review = CharacterTagReview(image, result, character_gender=settings.character_gender)
    boxes = dict(review.tag_checkboxes)
    assert not boxes["1girl"].isEnabled() and not boxes["1boy"].isEnabled()
    assert boxes["blue_hair"].isEnabled() and boxes["blue_hair"].isChecked()
    assert boxes["androgynous"].isEnabled()
    assert review.approved_tags == ("blue_hair", "androgynous")
    review.close()
    settings.close()
    image.close()


@pytest.mark.parametrize("gender", ["male", "female", "unspecified"])
def test_approved_garments_not_changed_by_gender_or_broad_category(gender):
    outfit = ("skirt", "high heels", "suit", "formal", "uniform", "blue jacket")
    prepared, record = prepare_design_reference_request(
        Request(), outfit, (Tokenizer(),), character_gender=gender)
    assert record["approved_tags"] == outfit
    assert "pants" not in prepared.prompt
    assert "trousers" not in prepared.prompt
    assert record["excluded_non_design_tags"] == ()
