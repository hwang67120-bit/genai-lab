from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
import pytest

from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, analyze_clothing_design
from genai_lab.clothing_reference import ClothingDesignAnalysisResult, ClothingDesignTagCandidate
from genai_lab.garment_detail_analysis import (
    analyze_garment_details, detail_view_boxes, detail_group, measured_palette,
)


def result(scores):
    candidates = tuple(ClothingDesignTagCandidate(name, name.replace("_", " "), score)
                       for name, score in scores if score >= .35)
    return ClothingDesignAnalysisResult("mock", "CPU", 100, 180, 448, .35,
        len(scores), len(scores), 0, 0, candidates, .01, tuple(scores))


class Session:
    instances = []
    responses = [
        [("jacket", .9), ("skirt", .8), ("gold_buttons", .15), ("1girl", .99)],
        [("jacket", .3), ("gold_buttons", .85), ("pants", .9), ("blue_hair", .9)],
        [("gold_buttons", .7), ("lace", .6)],
    ]

    def __init__(self, settings):
        self.calls = []
        self.closed = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def analyze(self, image):
        data = self.responses[len(self.calls)]
        self.calls.append(image.copy())
        return result(data)


@pytest.fixture(autouse=True)
def release_sessions():
    Session.instances = []
    yield
    for session in Session.instances:
        for image in session.calls:
            image.close()


def test_actual_entrypoint_reuses_session_and_retains_uncertainty(monkeypatch):
    import genai_lab.clothing_analysis as module
    monkeypatch.setattr(module, "WdTagSession", Session)
    with Image.new("RGBA", (100, 180), "blue") as image:
        before = image.tobytes()
        output = analyze_clothing_design(SimpleNamespace(extracted_image=image),
                                         ClothingDesignAnalysisSettings())
        assert image.tobytes() == before
    assert len(Session.instances) == 1
    assert len(Session.instances[0].calls) == 3
    assert Session.instances[0].closed
    report = output.garment_detail_report
    assert report["layer_order"] == report["absence_status"] == "unresolved"
    assert not report["gender_inference"]
    assert not report["semantic_accuracy_verified"]
    assert report["optional_detail_tags"] == ["gold_buttons", "lace"]
    assert "pants" not in [t.tag_name for t in output.tag_candidates]
    pants = next(e for e in report["evidence"] if e["tag"] == "pants")
    assert pants["location"] == "unresolved"
    assert pants["whole_score"] == 0
    assert not any(e["tag"] in ("1girl", "blue_hair") for e in report["evidence"])
    assert output.raw_general_scores == tuple(Session.responses[0])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("views", [0, 6, True, 2.5])
def test_view_limits(views):
    with Image.new("RGBA", (128, 200), "red") as image, pytest.raises(ValueError):
        detail_view_boxes(image, views)


def test_alpha_coordinates_are_original_coordinates():
    with Image.new("RGBA", (240, 300)) as image:
        image.paste((10, 20, 30, 255), (50, 40, 150, 240))
        boxes = detail_view_boxes(image)
        assert boxes[0] == (50, 40, 150, 240)
        assert len(boxes) == 3
        for l, t, r, b in boxes:
            assert 50 <= l < r <= 150
            assert 40 <= t < b <= 240


def test_tiny_image_uses_no_unhelpful_upscale_views():
    with Image.new("RGB", (20, 40), "blue") as image:
        assert detail_view_boxes(image) == [(0, 0, 20, 40)]


def test_empty_alpha_rejected_before_model_load():
    with Image.new("RGBA", (100, 180)) as image, pytest.raises(ValueError):
        analyze_garment_details(image, ClothingDesignAnalysisSettings(), Session)
    assert not Session.instances


def test_disabled_local_views_preserves_whole_only():
    with Image.new("RGB", (100, 180), "blue") as image:
        output = analyze_garment_details(image, ClothingDesignAnalysisSettings(
            detail_maximum_views=1), Session)
    assert output.garment_detail_report["view_count"] == 1
    assert not output.garment_detail_report["optional_detail_tags"]


def test_palette_ignores_hidden_rgb_and_keeps_two_colors():
    with Image.new("RGBA", (10, 10), (0, 255, 0, 0)) as image:
        image.paste((255, 0, 0, 255), (0, 0, 5, 5))
        image.paste((0, 0, 255, 255), (0, 5, 5, 10))
        record = measured_palette(image)
    assert {tuple(c["rgb"]) for c in record["colors"]} == {(255, 0, 0), (0, 0, 255)}
    assert all(c["sample_fraction"] == .5 for c in record["colors"])
    assert record["semantic_color_assignment"] == "unresolved"


@pytest.mark.parametrize("tag, group", [
    ("suit", "style_context"), ("formal", "style_context"), ("uniform", "style_context"),
    ("pants", "component_candidate"), ("skirt", "component_candidate"),
    ("gold_buttons", "construction"), ("long_sleeves", "shape"),
    ("striped", "surface"), ("1boy", "excluded"), ("novel_detail", "unresolved"),
])
def test_taxonomy_never_infers_gender_or_absence(tag, group):
    assert detail_group(tag) == group


def test_cancellation_releases_session_before_next_inference():
    def check():
        if Session.instances and Session.instances[0].calls:
            raise InterruptedError("cancel")
    with Image.new("RGB", (100, 180), "blue") as image, pytest.raises(InterruptedError):
        analyze_garment_details(image, ClothingDesignAnalysisSettings(), Session, check_running=check)
    assert len(Session.instances[0].calls) == 1
    assert Session.instances[0].closed


def test_timeout_after_first_call_closes_session(monkeypatch):
    import genai_lab.garment_detail_analysis as module
    monkeypatch.setattr(module, "perf_counter",
        lambda: 200. if Session.instances and Session.instances[0].calls else 0.)
    with Image.new("RGB", (100, 180), "blue") as image, pytest.raises(TimeoutError):
        analyze_garment_details(image, ClothingDesignAnalysisSettings(), Session)
    assert len(Session.instances[0].calls) == 1
    assert Session.instances[0].closed


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout(seconds):
    with Image.new("RGB", (100, 180)) as image, pytest.raises(ValueError):
        analyze_garment_details(image, replace(ClothingDesignAnalysisSettings(),
            detail_timeout_seconds=seconds), Session)


def test_gui_approval_splits_core_from_optional(monkeypatch):
    import gui_main
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    with Image.new("RGB", (100, 180), "blue") as image:
        analyzed = analyze_garment_details(image, ClothingDesignAnalysisSettings(), Session)
    monkeypatch.setattr(gui_main, "create_white_background_clothing_preview",
                        lambda candidate: Image.new("RGB", (64, 64), "white"))
    dialog = gui_main.ClothingDesignAnalysisReviewDialog(None, analyzed)
    boxes = dict(dialog.tag_checkboxes)
    assert "pants" not in boxes and "1girl" not in boxes
    assert boxes["gold_buttons"].isChecked()
    boxes["lace"].setChecked(False)
    dialog.approve_selected_tags()
    assert dialog.approved_tag_names == ("jacket", "skirt")
    assert dialog.approved_detail_tag_names == ("gold_buttons",)
    dialog.close()
