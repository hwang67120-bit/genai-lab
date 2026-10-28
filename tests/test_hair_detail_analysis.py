from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from genai_lab.clothing_reference import (
    ClothingDesignAnalysisResult,
    ClothingDesignTagCandidate,
)
from genai_lab.hair_detail_analysis import (
    HairDetailAnalysisSettings,
    analyze_hair_details,
    hair_analysis_log_detail,
    hair_detail_group,
    hair_length_geometry,
    hair_observation_masks,
    hair_prompt_delivery_report,
    resolve_hair_detail_settings,
)


def image_mask(values):
    return Image.fromarray(values.astype(np.uint8) * 255)


def base_result():
    return ClothingDesignAnalysisResult(
        model_id="test", execution_provider="CPUExecutionProvider",
        input_width=64, input_height=64, model_input_size=448,
        score_threshold=.35, total_label_count=3, general_label_count=3,
        excluded_rating_label_count=0, excluded_character_label_count=0,
        tag_candidates=(ClothingDesignTagCandidate("blue_hair", "blue hair", .8),),
        elapsed_seconds=.1,
    )


def source_masks():
    hair = np.zeros((64, 64), dtype=bool)
    hair[4:24, 14:50] = True
    hair[20:52, 10:22] = True
    hair[20:52, 42:54] = True
    face = np.zeros_like(hair)
    face[20:44, 22:42] = True
    return hair, face


def test_face_anchored_views_keep_whole_front_and_both_sides():
    hair, face = source_masks()
    with image_mask(hair) as hair_image, image_mask(face) as face_image:
        views, reason = hair_observation_masks(
            hair_image, face_image, HairDetailAnalysisSettings())
    assert reason is None
    assert set(views) == {"whole", "front", "left_side", "right_side", "rear_silhouette_candidate"}
    assert np.array_equal(views["whole"], hair)
    assert all(np.all(selected <= hair) for selected in views.values())


def test_multiview_reuses_session_and_returns_texture_paths(tmp_path, monkeypatch):
    hair, face = source_masks()
    scores = (
        (("blue_hair", .8), ("blunt_bangs", .2), ("sidelocks", .1)),
        (("blue_hair", .7), ("blunt_bangs", .91), ("sidelocks", .2)),
        (("blue_hair", .6), ("blunt_bangs", .1), ("sidelocks", .88)),
        (("blue_hair", .6), ("blunt_bangs", .1), ("sidelocks", .86)),
        (("blue_hair", .6), ("blunt_bangs", .1), ("sidelocks", .20)),
    )
    class Session:
        def __init__(self):
            self.calls = 0
        def analyze(self, image):
            value = scores[self.calls]
            self.calls += 1
            return SimpleNamespace(raw_general_scores=value, tag_candidates=())
    session = Session()
    monkeypatch.setattr(
        "genai_lab.hair_detail_analysis.tempfile.mkdtemp",
        lambda **kwargs: str(tmp_path))
    with Image.new("RGB", (64, 64), "blue") as source:
        with image_mask(hair) as hair_image, image_mask(face) as face_image:
            result = analyze_hair_details(
                session, source, hair_image, face_image, base_result(),
                HairDetailAnalysisSettings())
    assert session.calls == 5
    assert [tag.tag_name for tag in result.tag_candidates] == [
        "blue_hair", "blunt_bangs", "sidelocks"]
    report = result.hair_detail_report
    assert report["session_reused"] is True
    assert report["model_load_count"] == 0
    assert report["optional_detail_tags"] == [
        "blunt_bangs", "sidelocks"]
    geometry_evidence = next(
        item for item in report["evidence"]
        if item["status"] == "mask_geometry_candidate")
    assert geometry_evidence["eligible_for_prompt"] is False
    assert all(Path(view["texture_path"]).is_file() for view in report["views"])
    assert all(Path(view["mask_path"]).is_file() for view in report["views"])
    with Image.open(report["views"][1]["texture_path"]) as texture:
        assert texture.mode == "RGBA"
        assert texture.getchannel("A").getbbox() is not None


@pytest.mark.parametrize("tag, expected", [
    ("blunt_bangs", "front"),
    ("hair_between_eyes", "front"),
    ("sidelocks", "side"),
    ("multicolored_hair", "color"),
    ("medium_hair", "length"),
    ("wavy_hair", "texture"),
    ("high_ponytail", "arrangement"),
    ("jacket", "unresolved"),
])
def test_hair_detail_routing_is_bounded(tag, expected):
    assert hair_detail_group(tag) == expected


def test_unknown_settings_fail_instead_of_becoming_hidden_rules():
    with pytest.raises(ValueError, match="알 수 없는"):
        resolve_hair_detail_settings({
            "reference_analysis": {"hair_detail_analysis": {"guess_bangs": True}}})


def test_length_geometry_uses_face_relative_extent_instead_of_fixed_pixels():
    hair, face = source_masks()
    with image_mask(hair) as hair_image, image_mask(face) as face_image:
        geometry = hair_length_geometry(
            hair_image, face_image, HairDetailAnalysisSettings())
    assert geometry["status"] == "measured"
    assert geometry["suggested_length_band"] == "short hair"
    assert geometry["compatible_length_tags"] == (
        "very short hair", "short hair", "medium hair")
    assert geometry["bottom_extension_face_heights"] == pytest.approx(1 / 3)


def test_local_view_cannot_promote_global_hair_length(tmp_path, monkeypatch):
    hair, face = source_masks()
    scores = (
        (("long_hair", .90), ("short_hair", .10)),
        (("long_hair", .99), ("short_hair", .10)),
        (("long_hair", .99), ("short_hair", .10)),
        (("long_hair", .99), ("short_hair", .10)),
        (("long_hair", .99), ("short_hair", .10)),
    )

    class Session:
        def __init__(self):
            self.calls = 0

        def analyze(self, image):
            value = scores[self.calls]
            self.calls += 1
            return SimpleNamespace(raw_general_scores=value, tag_candidates=())

    monkeypatch.setattr(
        "genai_lab.hair_detail_analysis.tempfile.mkdtemp",
        lambda **kwargs: str(tmp_path))
    with Image.new("RGB", (64, 64), "blue") as source:
        with image_mask(hair) as hair_image, image_mask(face) as face_image:
            result = analyze_hair_details(
                Session(), source, hair_image, face_image, base_result(),
                HairDetailAnalysisSettings())
    assert "long_hair" not in [tag.tag_name for tag in result.tag_candidates]
    long_evidence = next(
        item for item in result.hair_detail_report["evidence"]
        if item["tag"] == "long_hair")
    assert long_evidence["evidence_views"] == ("whole",)
    assert long_evidence["status"] == "geometry_conflict"
    assert long_evidence["eligible_for_prompt"] is False


def test_prompt_delivery_separates_analyzed_approved_and_delivered_tags():
    report = {
        "evidence": [
            {"tag": "long_hair", "max_score": .91},
            {"tag": "short_hair", "max_score": .44},
            {"tag": "purple_eyes", "max_score": .8},
        ],
        "optional_detail_tags": ["long_hair"],
    }
    delivery = hair_prompt_delivery_report(
        ("blue_hair", "long_hair", "purple_eyes"),
        "1boy, blue hair, purple eyes, wearing blue jacket",
        report,
    )
    assert delivery["approved_hair_tags"] == ("blue hair", "long hair")
    assert delivery["delivered_hair_tags"] == ("blue hair",)
    assert delivery["missing_approved_hair_tags"] == ("long hair",)
    assert delivery["analyzed_hair_candidates"] == ("long hair", "short hair")
    assert delivery["analyzed_but_not_delivered"] == ("long hair", "short hair")
    assert delivery["optional_hair_candidates"] == ("long hair",)


def test_analysis_log_lists_evidence_and_marks_it_as_not_delivered():
    detail = hair_analysis_log_detail({
        "length_geometry": {
            "status": "measured",
            "bottom_extension_face_heights": .25,
            "compatible_length_tags": ("short hair", "medium hair"),
        },
        "evidence": [{
            "tag": "long_hair",
            "group": "length",
            "max_score": .8461,
            "status": "geometry_conflict",
            "evidence_views": ("whole",),
            "eligible_for_prompt": False,
        }],
        "optional_detail_tags": [],
    })
    assert "long hair" in detail
    assert "geometry_conflict" in detail
    assert "0.846" in detail
    assert "승인 화면 추가 후보=[]" in detail
    assert "현재 단계 프롬프트 전달=아님" in detail
