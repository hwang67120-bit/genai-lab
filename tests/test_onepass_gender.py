from dataclasses import dataclass

import pytest
from PySide6.QtCore import QSettings

from genai_lab.character_preferences import (
    load_character_gender, preference_key, save_character_gender,
)
from genai_lab.clothing_reference_generation import prepare_design_reference_request
from genai_lab.gender_prompt import prepare_gender_negative_terms
from genai_lab.onepass_gender import prepare_onepass_gender
from genai_lab.request import CharacterFramingType


@dataclass
class Request:
    prompt: str = "unchanged"
    negative_prompt: str = "nsfw, bad anatomy"
    framing_type: CharacterFramingType = CharacterFramingType.FULL_BODY


class Tokenizer:
    model_max_length = 77

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 2))}


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "gender.ini"), QSettings.Format.IniFormat)


@pytest.mark.parametrize("gender,prefix,guard", [
    ("male", ("1boy", "male focus", "masculine silhouette"), "1girl"),
    ("female", ("1girl", "female focus", "feminine silhouette"), "1boy"),
    ("unspecified", (), "nsfw"),
])
def test_onepass_matches_product_gender_without_accepting_detector_gender(
        tmp_path, store, gender, prefix, guard):
    source = tmp_path / "character.png"
    save_character_gender(source, gender, store)
    candidates = ("1boy, 1girl", "female_focus", "male_focus",
                  "feminine_silhouette", "masculine_silhouette", "gender_swap",
                  "blue_hair", "muscular", "raccoon_ears", "blue hair")
    original = tuple(candidates)
    condition = prepare_onepass_gender(source, candidates, "nsfw, bad anatomy", settings=store)
    assert condition.prefix_tags == prefix
    assert condition.appearance_tags == ("blue hair", "muscular", "raccoon ears")
    assert condition.negative_prompt.split(", ")[0] == guard
    assert condition.removed_candidate_gender_tags == (
        "1boy", "1girl", "female focus", "male focus",
        "feminine silhouette", "masculine silhouette", "gender swap")
    assert candidates == original
    assert load_character_gender(source, store) == gender
    # 기존 제품 진입점에서 승인 외형을 비교한다.
    # 구형 미지정 호출부는 명시적으로 승인한 성별 태그를 유지할 수 있다.
    prepared, _ = prepare_design_reference_request(
        Request(), ("skirt", "high heels"), (Tokenizer(), Tokenizer()),
        character_tags=condition.appearance_tags, character_gender=gender)
    onepass_terms = (*condition.prefix_tags, *condition.appearance_tags)
    assert onepass_terms[0] == prepared.prompt.split(", ")[0]
    assert condition.negative_prompt == prepared.negative_prompt
    record = condition.to_record()
    assert record["gender_source"] == "user_preference"
    assert record["character_gender"] == gender
    assert record["removed_candidate_gender_tags"] == condition.removed_candidate_gender_tags


@pytest.mark.parametrize("tag", ["1boy", "1girl", "2boys", "female_focus",
                                    "masculine silhouette", "gender_swap"])
def test_onepass_rejects_gendered_negative_template(tmp_path, store, tag):
    source = tmp_path / "character.png"
    save_character_gender(source, "unspecified", store)
    with pytest.raises(ValueError, match="템플릿에는 성별"):
        prepare_onepass_gender(source, ("blue hair",), f"nsfw, {tag}, bad anatomy", settings=store)


def test_onepass_requires_explicit_choice_without_writing_preferences(tmp_path, store):
    source = tmp_path / "unset.png"
    keys = store.allKeys()
    with pytest.raises(ValueError, match="성별 선택이 없습니다"):
        prepare_onepass_gender(source, ("1girl",), "nsfw", settings=store)
    assert store.allKeys() == keys


def test_invalid_saved_choice_does_not_fall_back_to_detector(tmp_path, store):
    source = tmp_path / "invalid.png"
    store.setValue(preference_key(source), "auto")
    with pytest.raises(ValueError, match="성별 선택이 없습니다"):
        prepare_onepass_gender(source, ("1boy",), "nsfw", settings=store)


def test_preference_isolation_and_changed_selection(tmp_path, store):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    save_character_gender(first, "male", store)
    save_character_gender(second, "female", store)
    for path, expected in ((first, "1boy"), (second, "1girl")):
        result = prepare_onepass_gender(path, ("blue hair",), "nsfw", settings=store)
        assert result.prefix_tags[0] == expected
    save_character_gender(first, "unspecified", store)
    result = prepare_onepass_gender(first, ("1boy", "blue hair"), "nsfw", settings=store)
    assert result.prefix_tags == ()
    assert result.appearance_tags == ("blue hair",)
    assert result.negative_prompt == "nsfw"
    assert load_character_gender(second, store) == "female"


@pytest.mark.parametrize("gender,terms,expected,removed,guard", [
    ("male", ["1boy", "Male_Focus", "bad anatomy", "1GIRL", "nsfw", "bad anatomy"],
     ["1girl", "bad anatomy", "nsfw", "bad anatomy"], ["1boy", "Male_Focus"], "1girl"),
    ("female", ["1girl", "female focus", "1boy", "bad anatomy"],
     ["1boy", "bad anatomy"], ["1girl", "female focus"], "1boy"),
    ("unspecified", ["1boy", "nsfw"], ["1boy", "nsfw"], [], None),
])
def test_shared_negative_rules_preserve_legacy_order_and_do_not_mutate(
        gender, terms, expected, removed, guard):
    original = list(terms)
    assert prepare_gender_negative_terms(terms, gender) == (expected, removed, guard)
    assert terms == original


@pytest.mark.parametrize("gender", [None, "auto", "", 1])
def test_shared_negative_rejects_invalid_gender(gender):
    with pytest.raises(ValueError):
        prepare_gender_negative_terms(["nsfw"], gender)


def test_legacy_unspecified_approved_gender_is_unchanged():
    prepared, record = prepare_design_reference_request(
        Request(negative_prompt="1girl, nsfw"), ("jacket",), (Tokenizer(),),
        character_tags=("1boy", "blue hair"), character_gender="unspecified")
    assert prepared.prompt.startswith("1boy, blue hair, ")
    assert prepared.negative_prompt == "1girl, nsfw"
    assert record["approved_character_tags"] == ("1boy", "blue hair")
    assert record["gender_negative_guard"] is None


def test_empty_appearance_does_not_invent_character_features(tmp_path, store):
    source = tmp_path / "character.png"
    save_character_gender(source, "male", store)
    condition = prepare_onepass_gender(source, (), "nsfw", settings=store)
    assert condition.prefix_tags == ("1boy", "male focus", "masculine silhouette")
    assert condition.appearance_tags == ()
    assert condition.negative_prompt == "1girl, nsfw"


def test_character_tags_require_a_sequence(tmp_path, store):
    source = tmp_path / "character.png"
    save_character_gender(source, "female", store)
    with pytest.raises(TypeError, match="태그 목록"):
        prepare_onepass_gender(source, "blue hair", "nsfw", settings=store)
