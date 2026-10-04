"""실험 정책과 입력 계약의 경계 검사. 실제 모델 품질 시험이 아니다."""
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from genai_lab.extra_parts_analysis import AdditionalPartsAnalyzer, validate_part_queries
from genai_lab.part_candidate_policy import measure_candidate_overlap
from genai_lab.regional_reference import DetectedPart, valid_part_name
from genai_lab.source_region_experiment import assign_source_regions, filter_part_instances
from genai_lab.reference_experiment_approval import (
    approve_reference_experiment, require_reference_experiment_approval,
)
from genai_lab.scene_reference import PartReference
from genai_lab.visual_reference import VisualInputs, configure_visual_condition


def mask(box):
    result = Image.new("L", (32, 48))
    result.paste(255, box)
    return result


@pytest.fixture
def inputs():
    value = VisualInputs(
        Image.new("RGB", (32, 48), "white"),
        Image.new("RGB", (16, 16), "blue"),
        Image.new("RGB", (16, 16), "green"),
        mask((8, 0, 24, 12)), mask((8, 12, 24, 44)),
        extra_references=(PartReference("antennae", Image.new("RGB", (8, 8), "purple"),
                                       mask((1, 0, 4, 6)), .35),),
        analysis_record={"part_region_experiment": {
            "region_policy": "source_roi_priority_experiment",
            "requires_input_review": True,
            "identity_overlap_removed": 0, "garment_overlap_removed": 0,
        }},
    )
    yield value
    value.close()


def test_measurement_does_not_make_semantic_or_selection_decisions():
    with mask((1, 1, 7, 7)) as garment, mask((8, 1, 10, 3)) as hair:
        rows = measure_candidate_overlap(
            (np.asarray(garment) == 255)[None], {"garment": garment, "hair": hair})
        assert rows[0]["garment_overlap"] == 1
        assert "accepted" not in rows[0]
        assert "semantic_status" not in rows[0]
        selected, proposals = filter_part_instances(
            "wings", (np.asarray(garment) == 255)[None], {"garment": garment, "hair": hair})
        assert len(selected) == 0
        assert proposals[0]["semantic_status"] == "unresolved"
        assert proposals[0]["selection_status"] == "not_proposed"
        assert proposals[0]["decision_source"] == "experimental_overlap_v2"


@pytest.mark.parametrize("name", ["wings", "horns", "antennae", "fin_2"])
def test_source_region_assignment_supports_dynamic_names(name):
    with mask((8, 0, 24, 12)) as identity, mask((8, 12, 24, 44)) as clothes:
        part = DetectedPart(name, mask((1, 20, 6, 30)), None, .35)
        new_identity, new_clothes, report = assign_source_regions([part], identity, clothes)
        try:
            assert report["conditioned_parts"] == [name]
            assert report["requires_input_review"]
            assert not report["semantic_accuracy_verified"]
        finally:
            part.close()
            new_identity.close()
            new_clothes.close()


@pytest.mark.parametrize("name", ["../tail", "identity", "garment", "hair", "", "a/b", "Wings", "con", "nul", "com1"])
def test_path_and_reserved_name_contract_remains(name):
    assert not valid_part_name(name)


@pytest.mark.parametrize("specs", [
    [], [{"name": "wings", "query": ""}],
    [{"name": "garment", "query": "coat"}],
    [{"name": "wings", "query": "wings"}, {"name": "wings", "query": "wings"}],
    [{"name": "wings", "query": "wings", "hair_context": "false"}],
])
def test_invalid_queries_fail_before_loading_models(specs):
    with pytest.raises(ValueError):
        validate_part_queries(specs)


def test_dynamic_analyzer_missing_detection_is_not_absence(tmp_path):
    calls = []
    class Backend:
        def detect(self, image, query, threshold):
            calls.append(query)
            return np.empty((0, 4)), np.empty(0)
        def close(self):
            pass
    analyzer = AdditionalPartsAnalyzer(Backend(), debug_dir=tmp_path, part_queries=[
        {"name": "antennae", "query": "antennae."},
    ])
    with Image.new("RGB", (32, 48)) as image:
        assert analyzer.analyze(image, image.size, cancelled=lambda: False,
                                deadline=perf_counter()+30) == []
    assert calls == ["antennae."]
    assert set(analyzer.report["parts"]) == {"antennae"}
    entry = analyzer.report["parts"]["antennae"]
    assert entry["presence"] == "not_observed"
    assert entry["status"] == "uncertain"
    assert entry["automatic_conditioning"] is False
    assert entry["outcome_reason"] == "no_candidate_above_threshold"
    analyzer.close()


def test_experiment_cannot_reach_encoder_without_approval(inputs):
    # 승인 검사보다 먼저 모델 속성을 읽으면 AttributeError로 실패한다.
    with pytest.raises(ValueError, match="먼저 승인"):
        configure_visual_condition(SimpleNamespace(), inputs)
    approve_reference_experiment(inputs)
    require_reference_experiment_approval(inputs)
    assert inputs.analysis_record["experimental_input_approval"]["semantic_accuracy_verified"] is False


@pytest.mark.parametrize("change", ["rgb", "mask", "scale", "policy"])
def test_approved_inputs_cannot_change_silently(inputs, change):
    approve_reference_experiment(inputs)
    if change == "rgb":
        inputs.garment.putpixel((0, 0), (1, 2, 3))
    elif change == "mask":
        inputs.garment_mask.putpixel((8, 12), 0)
    elif change == "scale":
        inputs.extra_references[0].scale = .5
    else:
        inputs.analysis_record["part_region_experiment"]["region_policy"] = "changed"
    with pytest.raises(ValueError, match="승인 이후"):
        require_reference_experiment_approval(inputs)


def test_ui_approval_records_input_permission_not_semantic_truth(inputs):
    from PySide6.QtWidgets import QApplication, QLabel
    from genai_lab.visual_reference_review import VisualInputReview
    app = QApplication.instance() or QApplication([])
    dialog = VisualInputReview(inputs)
    try:
        assert "의상 영향 0픽셀 제외" in "\n".join(x.text() for x in dialog.findChildren(QLabel))
        with pytest.raises(ValueError):
            require_reference_experiment_approval(inputs)
        dialog.accept()
        require_reference_experiment_approval(inputs)
        dialog.reject()
        with pytest.raises(ValueError):
            require_reference_experiment_approval(inputs)
    finally:
        dialog.close()


def test_reviewed_dynamic_reference_reaches_encoder(inputs):
    approve_reference_experiment(inputs)
    seen = {}
    pipe = SimpleNamespace(
        _execution_device="cpu",
        set_ip_adapter_scale=lambda value: seen.update(scales=value),
        prepare_ip_adapter_image_embeds=lambda **kw: seen.update(images=kw["ip_adapter_image"]) or ["encoded"],
    )
    condition = configure_visual_condition(pipe, inputs)
    assert len(seen["images"][0]) == 3
    assert seen["images"][0][1] is inputs.extra_references[0].rgb
    assert seen["images"][0][-1] is inputs.garment
    assert condition == {"ip_adapter_image_embeds": ["encoded"]}
