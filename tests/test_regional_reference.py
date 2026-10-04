import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from genai_lab.regional_reference import DetectedPart, prepare_extra_references, crop_reference
from genai_lab.visual_reference import VisualInputs, configure_visual_condition, prepare_visual_inputs
from genai_lab.reference_regions import ReferenceRegions
from genai_lab.scene_generation import reference_scales


def mask(box):
    image = Image.new("L", (64, 112))
    image.paste(255, box)
    return image


@pytest.fixture
def source_inputs():
    value = VisualInputs(Image.new("RGB", (64, 112), "white"),
        Image.new("RGB", (32, 32), "blue"), Image.new("RGB", (32, 32), "green"),
        mask((20, 0, 40, 20)), mask((20, 20, 40, 100)))
    yield value
    value.close()


def test_extra_part_exact_pixels_masks_and_scale_reach_encoder(source_inputs):
    source = source_inputs.source
    source.paste("blue", (0, 40, 10, 80))
    part = DetectedPart("tail", mask((0, 40, 10, 80)), mask((0, 40, 10, 80)), .4)
    refs = prepare_extra_references(source, [part],
        [source_inputs.identity_mask, source_inputs.garment_mask])
    source_inputs.extra_references = refs
    assert refs[0].rgb.size[0] == refs[0].rgb.size[1]
    assert set(map(tuple, np.asarray(refs[0].rgb).reshape(-1, 3))) == {(0, 0, 255), (255, 255, 255)}
    seen = {}
    pipe = SimpleNamespace(_execution_device="cpu",
        set_ip_adapter_scale=lambda value: seen.update(scales=value),
        prepare_ip_adapter_image_embeds=lambda **kw: seen.update(images=kw["ip_adapter_image"]) or ["encoded"])
    condition = configure_visual_condition(pipe, source_inputs)
    assert seen["images"] == [[source_inputs.identity, refs[0].rgb, source_inputs.garment]]
    assert seen["scales"] == pytest.approx((.7 + .4 + .45) / 3)
    assert condition == {"ip_adapter_image_embeds": ["encoded"]}
    assert "cross_attention_kwargs" not in condition
    assert reference_scales(source_inputs, .7, 0) == [.7, 0, .4]
    part.close()


@pytest.mark.parametrize("failure", ["overlap", "empty", "unknown", "scale", "size", "unlocated"])
def test_invalid_extra_parts_cannot_silently_fall_back(source_inputs, failure):
    part = DetectedPart("tail", mask((0, 40, 10, 80)), mask((0, 40, 10, 80)), .4)
    if failure == "overlap":
        part.target_region.paste(255, (20, 0, 40, 20))
    elif failure == "empty":
        part.source_mask.paste(0, (0, 0, 64, 112))
    elif failure == "unknown":
        part.name = "../tail"
    elif failure == "scale":
        part.scale = float("nan")
    elif failure == "size":
        part.target_region.close()
        part.target_region = Image.new("L", (8, 8), 255)
    elif failure == "unlocated":
        part.target_region.close()
        part.target_region = None
    try:
        with pytest.raises(ValueError):
            prepare_extra_references(source_inputs.source, [part],
                [source_inputs.identity_mask, source_inputs.garment_mask])
    finally:
        part.close()


def test_optional_analyzer_results_reach_approval_inputs_and_are_closed(source_inputs, tmp_path, monkeypatch):
    value = source_inputs
    def regions(*args, **kwargs):
        return ReferenceRegions({
            "identity": value.identity_mask.copy(), "hair": value.identity_mask.copy(),
            "face": Image.new("L", value.source.size),
            "garment": value.garment_mask.copy(), "foreground": Image.new("L", value.source.size, 255),
        }, {})
    monkeypatch.setattr("genai_lab.reference_regions.analyze_reference_regions", regions)
    calls = []
    class Analyzer:
        def analyze(self, source, output_size, *, cancelled, deadline):
            assert output_size == source.size and callable(cancelled)
            calls.append("analyze")
            return [DetectedPart("tail", mask((0, 40, 10, 80)), mask((0, 40, 10, 80)), .4)]
        def close(self):
            calls.append("close")
    config = {"reference_analysis": {"part_analyzer": Analyzer()}}
    result = prepare_visual_inputs(value.source, value.garment, config, tmp_path)
    try:
        assert calls == ["analyze", "close"]
        assert result.extra_references[0].name == "tail"
        assert set(result.color_evidence) == {"hair", "tail"}
        assert result.analysis_record["additional_parts_status"] == "measured"
    finally:
        result.close()


def test_unimplemented_garment_detail_mode_stops_before_analysis(source_inputs, monkeypatch, tmp_path):
    monkeypatch.setattr("genai_lab.reference_regions.analyze_reference_regions",
                        lambda *a, **k: pytest.fail("미구현 모드에서 분석을 시작하면 안 됩니다."))
    with pytest.raises(ValueError, match="아직 연결되지"):
        prepare_visual_inputs(source_inputs.source, source_inputs.garment,
            {"reference_analysis": {"garment_reference_mode": "detail_references"}}, tmp_path)


def test_input_review_shows_palette_without_auto_approval(source_inputs):
    from PySide6.QtWidgets import QApplication, QLabel
    from genai_lab.part_color_analysis import analyze_part_colors
    from genai_lab.visual_reference_review import VisualInputReview
    app = QApplication.instance() or QApplication([])
    source_inputs.color_evidence = {"hair": analyze_part_colors(source_inputs.source, source_inputs.identity_mask)}
    source_inputs.analysis_record = {"additional_parts_status": "analyzer_not_connected"}
    dialog = VisualInputReview(source_inputs)
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert "자동 검출기 미연결" in text
        assert "색 분포" in text and "패턴 판단 보류" in text
        assert dialog.result() == 0
    finally:
        dialog.close()


def test_default_detector_preview_reaches_ui_without_unverified_condition(source_inputs, tmp_path, monkeypatch):
    value = source_inputs
    monkeypatch.setattr("genai_lab.reference_regions.analyze_reference_regions",
        lambda *a, **k: ReferenceRegions({
            "identity": value.identity_mask.copy(), "hair": value.identity_mask.copy(),
            "face": Image.new("L", value.source.size),
            "garment": value.garment_mask.copy(), "foreground": Image.new("L", value.source.size, 255)}, {}))
    calls = []
    crop_path = tmp_path / "tail.png"
    value.source.save(crop_path)
    class Detector:
        preview_only = True
        report = {"parts": {"tail": {"status": "detected", "target_status": "unresolved",
                                    "crop_path": str(crop_path)}}}
        def analyze(self, source, output_size, **kwargs):
            return [DetectedPart("tail", mask((0, 40, 10, 80)), None, .35)]
        def close(self):
            calls.append("closed")
    monkeypatch.setattr("genai_lab.extra_parts_analysis.create_extra_parts_analyzer", lambda config, **kwargs: Detector())
    result = prepare_visual_inputs(value.source, value.garment, {}, tmp_path)
    try:
        assert calls == ["closed"]
        assert result.extra_references == ()
        assert "tail" in result.color_evidence
        assert result.analysis_record["additional_parts_status"] == "detected_not_conditioned"
        from PySide6.QtWidgets import QApplication, QLabel, QScrollArea
        from genai_lab.visual_reference_review import VisualInputReview
        app = QApplication.instance() or QApplication([])
        dialog = VisualInputReview(result)
        try:
            text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
            assert "귀·꼬리 참조 미전달" in text
            assert dialog.findChildren(QScrollArea)
            assert dialog.result() == 0
        finally:
            dialog.close()
    finally:
        result.close()


def test_source_region_experiment_encodes_four_rgb_references(source_inputs, tmp_path, monkeypatch):
    value = source_inputs
    value.source.paste("blue", (0, 40, 10, 80))
    value.source.paste("purple", (20, 0, 25, 5))
    monkeypatch.setattr("genai_lab.reference_regions.analyze_reference_regions",
        lambda *a, **k: ReferenceRegions({
            "identity": value.identity_mask.copy(), "hair": value.identity_mask.copy(),
            "face": Image.new("L", value.source.size),
            "garment": value.garment_mask.copy(), "foreground": Image.new("L", value.source.size, 255)}, {}))
    class Detector:
        preview_only = True
        report = {"parts": {
            "animal_ears": {
                "status": "detected",
                "accepted_masks": 1,
                "accepted_boxes": [[20, 0, 25, 5]],
                "detection_scores": [0.9],
            },
            "tail": {"status": "detected", "accepted_masks": 1},
        }}
        def analyze(self, source, output_size, **kwargs):
            return [
                DetectedPart(
                    "animal_ears",
                    mask((20, 0, 25, 5)),
                    None,
                    .35,
                ),
                DetectedPart("tail", mask((0, 40, 10, 80)), None, .35),
            ]
        def close(self):
            pass
    monkeypatch.setattr("genai_lab.extra_parts_analysis.create_extra_parts_analyzer", lambda *a, **k: Detector())
    config = {"reference_analysis": {"part_detection": {"conditioning_mode": "source_regions_experiment"}}}
    result = prepare_visual_inputs(value.source, value.garment, config, tmp_path)
    try:
        assert [p.name for p in result.extra_references] == ["animal_ears", "tail"]
        assert result.analysis_record["additional_parts_status"] == "source_regions_experiment"
        assert result.analysis_record["part_detection"]["mode"] == "source_regions_experiment"
        seen = {}
        pipe = SimpleNamespace(_execution_device="cpu",
            set_ip_adapter_scale=lambda scales: seen.update(scales=scales),
            prepare_ip_adapter_image_embeds=lambda **kw: seen.update(images=kw["ip_adapter_image"]) or ["encoded"])
        from genai_lab.reference_experiment_approval import approve_reference_experiment
        approve_reference_experiment(result)
        condition = configure_visual_condition(pipe, result)
        assert seen["scales"] == pytest.approx(.4625)
        assert len(seen["images"][0]) == 4
        assert seen["images"][0][2] is result.extra_references[1].rgb
        assert seen["images"][0][3] is result.garment
        assert (0, 0, 255) in set(map(tuple, np.asarray(seen["images"][0][2]).reshape(-1, 3)))
        assert condition == {"ip_adapter_image_embeds": ["encoded"]}
        assert not np.any(np.asarray(result.identity_mask) & np.asarray(result.extra_references[0].region))
    finally:
        result.close()


def test_partial_overlap_filter_to_encoder_and_review(source_inputs, tmp_path, monkeypatch):
    from genai_lab.extra_parts_analysis import EarTailAnalyzer
    from PySide6.QtWidgets import QApplication, QLabel
    from genai_lab.visual_reference_review import VisualInputReview
    value = source_inputs
    value.source.paste("blue", (0, 40, 10, 80))
    value.source.paste("purple", (20, 0, 25, 5))
    # 꼬리 30%가 의상 마스크에 잘못 포함된 경우를 재현한다.
    value.garment_mask.paste(255, (0, 40, 3, 80))
    monkeypatch.setattr("genai_lab.reference_regions.analyze_reference_regions",
        lambda *a, **k: ReferenceRegions({
            "identity": value.identity_mask.copy(), "hair": value.identity_mask.copy(),
            "face": Image.new("L", value.source.size),
            "garment": value.garment_mask.copy(), "foreground": Image.new("L", value.source.size, 255)}, {}))

    class Backend:
        closed = False
        def detect(self, image, query, threshold):
            if query == "human ears. person ears.":
                return np.empty((0, 4)), np.empty(0)
            self.ears = query == "animal ears."
            boxes = [[20, 0, 25, 5], [20, 0, 40, 20]] if self.ears else [
                [1, 1, 2, 2], [0, 40, 10, 80], [21, 30, 24, 60]]
            return np.array(boxes), np.full(len(boxes), .9)
        def segment(self, image, boxes):
            values = np.zeros((len(boxes), 1, image.height, image.width), dtype=bool)
            for i, (left, top, right, bottom) in enumerate(boxes.astype(int)):
                values[i, 0, top:bottom, left:right] = True
            return values, np.array([[.9], [.9]]) if self.ears else np.array([[.2], [.9], [.9]])
        def close(self):
            self.closed = True

    backend = Backend()
    monkeypatch.setattr("genai_lab.extra_parts_analysis.create_extra_parts_analyzer",
        lambda config, **kwargs: EarTailAnalyzer(backend, debug_dir=tmp_path, **kwargs))
    config = {
        "clothing_reference_generation": {
            "approved_character_tags": ("raccoon tail",),
        },
        "reference_analysis": {
            "part_detection": {
                "conditioning_mode": "source_regions_experiment",
            },
        },
    }
    result = prepare_visual_inputs(value.source, value.garment, config, tmp_path)
    try:
        assert backend.closed
        assert [p.name for p in result.extra_references] == [
            "animal_ears", "tail"]
        saved = json.loads((tmp_path / "parts.json").read_text(encoding="utf-8"))
        decisions = saved["parts"]["tail"]["source_filter"]
        assert [d["candidate_index"] for d in decisions] == [1, 2]
        assert [d["accepted"] for d in decisions] == [True, False]
        assert decisions[0]["garment_overlap"] == pytest.approx(.3)
        assert saved["parts"]["animal_ears"]["source_filter"][1][
            "reason"] == "broad_hair_candidate"
        assert (tmp_path / "source_garment_mask.png").exists()
        # 저품질 및 제외 후보까지 저장해서 원래 번호로 재현할 수 있어야 한다.
        for i in range(3):
            assert (tmp_path / f"tail_candidate_{i}_mask.png").exists()
        with Image.open(tmp_path / "tail_mask.png") as accepted:
            assert np.count_nonzero(accepted) == 400
        seen = {}
        pipe = SimpleNamespace(_execution_device="cpu",
            set_ip_adapter_scale=lambda scales: seen.update(scales=scales),
            prepare_ip_adapter_image_embeds=lambda **kw: seen.update(images=kw["ip_adapter_image"]) or ["encoded"])
        from genai_lab.reference_experiment_approval import approve_reference_experiment
        approve_reference_experiment(result)
        condition = configure_visual_condition(pipe, result)
        assert seen["images"][0][2] is result.extra_references[1].rgb
        assert seen["images"][0][3] is result.garment
        assert (0, 0, 255) in set(map(tuple, np.asarray(seen["images"][0][2]).reshape(-1, 3)))
        assert condition == {"ip_adapter_image_embeds": ["encoded"]}
        app = QApplication.instance() or QApplication([])
        dialog = VisualInputReview(result)
        try:
            assert "부분 겹침 후보 유지" in "\n".join(label.text() for label in dialog.findChildren(QLabel))
            assert dialog.result() == 0
        finally:
            dialog.close()
    finally:
        result.close()


def test_source_region_experiment_continues_without_optional_parts(
        source_inputs, tmp_path, monkeypatch):
    value = source_inputs
    monkeypatch.setattr(
        "genai_lab.reference_regions.analyze_reference_regions",
        lambda *a, **k: ReferenceRegions({
            "identity": value.identity_mask.copy(),
            "hair": value.identity_mask.copy(),
            "face": Image.new("L", value.source.size),
            "garment": value.garment_mask.copy(),
            "foreground": Image.new("L", value.source.size, 255),
        }, {}),
    )

    class Analyzer:
        preview_only = True
        report = {
            "status": "uncertain",
            "parts": {
                "human_ears": {
                    "boxes": [[0, 0, 40, 80]],
                    "accepted_masks": 0,
                    "automatic_conditioning": False,
                },
                "tail": {
                    "boxes": [],
                    "accepted_masks": 0,
                    "automatic_conditioning": False,
                },
            },
        }

        def analyze(self, source, output_size, **kwargs):
            return []

        def close(self):
            pass

    config = {
        "reference_analysis": {
            "color_analysis_enabled": False,
            "part_analyzer": Analyzer(),
            "part_detection": {
                "conditioning_mode": "source_regions_experiment",
            },
        },
    }
    result = prepare_visual_inputs(
        value.source,
        value.garment,
        config,
        tmp_path,
    )
    try:
        assert result.extra_references == ()
        assert result.analysis_record["additional_parts_status"] == (
            "unresolved_skipped"
        )
        skipped = result.analysis_record["part_detection"][
            "no_confirmed_optional_parts"
        ]
        assert skipped["automatic_conditioning"] is False
        assert skipped["continue_with_identity_and_garment"] is True
    finally:
        result.close()
