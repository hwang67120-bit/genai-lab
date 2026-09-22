import json
from time import perf_counter
import numpy as np
import pytest
from PIL import Image
from genai_lab.extra_parts_analysis import (
    EarTailAnalyzer,
    create_extra_parts_analyzer,
    deduplicate_ear_candidate_indices,
    filter_tail_garment_overlaps,
)


class Backend:
    def __init__(self):
        self.closed = False
    def detect(self, image, query, threshold):
        return np.array([[1, 2, 5, 8]]), np.array([.9])
    def segment(self, image, boxes):
        masks = np.zeros((1, 1, image.height, image.width), dtype=bool)
        masks[0, 0, 2:8, 1:5] = True
        return masks, np.array([[.9]])
    def close(self):
        self.closed = True


def test_detection_reports_masks_without_inventing_target_or_species(tmp_path):
    backend = Backend()
    analyzer = EarTailAnalyzer(backend, debug_dir=tmp_path)
    with Image.new("RGB", (16, 24), "blue") as image:
        parts = analyzer.analyze(image, image.size, cancelled=lambda: False, deadline=perf_counter()+30)
    assert [p.name for p in parts] == [
        "human_ears", "animal_ears", "tail"]
    assert all(p.target_region is None for p in parts)
    assert analyzer.preview_only
    saved = json.loads((tmp_path / "parts.json").read_text())
    assert saved["parts"]["human_ears"]["status"] == "detected"
    assert saved["parts"]["animal_ears"]["status"] == "detected"
    assert saved["parts"]["human_ears"]["accepted_boxes"] == [[1, 2, 5, 8]]
    assert saved["part_overlap"]["human_ears:animal_ears"][
        "smaller_part_ratio"] == 1.0
    assert saved["parts"]["tail"]["target_status"] == "unresolved"
    assert (tmp_path / "tail_reference.png").exists()
    for part in parts:
        assert part.source_mask.mode == "L"
        assert set(np.unique(part.source_mask)) == {0, 255}
        part.close()
    analyzer.close()
    assert backend.closed


@pytest.mark.parametrize("cause", ["empty", "low_quality", "bad_shape", "nan", "too_many"])
def test_uncertain_and_invalid_masks_not_silently_accepted(tmp_path, cause):
    backend = Backend()
    if cause == "empty":
        backend.detect = lambda *a: (np.empty((0, 4)), np.empty(0))
    if cause == "low_quality":
        backend.segment = lambda *a: (np.ones((1, 1, 24, 16)), np.array([[.2]]))
    if cause == "bad_shape":
        backend.segment = lambda *a: (np.ones((1, 24, 16)), np.array([[.9]]))
    if cause == "nan":
        backend.detect = lambda *a: (np.array([[1, 2, np.nan, 8]]), np.array([.9]))
    if cause == "too_many":
        backend.detect = lambda *a: (np.tile([1, 2, 5, 8], (9, 1)), np.ones(9))
    analyzer = EarTailAnalyzer(backend, debug_dir=tmp_path)
    with Image.new("RGB", (16, 24)) as image:
        if cause in ("empty", "low_quality"):
            assert analyzer.analyze(image, image.size, cancelled=lambda: False, deadline=perf_counter()+30) == []
            assert analyzer.report["status"] == "uncertain"
        else:
            with pytest.raises(ValueError):
                analyzer.analyze(image, image.size, cancelled=lambda: False, deadline=perf_counter()+30)
    analyzer.close()


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_or_timeout_before_model(tmp_path, cancel):
    class Forbidden(Backend):
        def detect(self, *args):
            pytest.fail("중단 이후 모델을 실행하면 안 됩니다")
    analyzer = EarTailAnalyzer(Forbidden(), debug_dir=tmp_path)
    with Image.new("RGB", (16, 24)) as image:
        with pytest.raises(InterruptedError if cancel else TimeoutError):
            analyzer.analyze(image, image.size, cancelled=lambda: cancel, deadline=perf_counter()-1)
    analyzer.close()


def test_factory_disabled_does_not_load_models():
    assert create_extra_parts_analyzer({}) is None


def test_all_filtered_candidates_keep_diagnostics(tmp_path):
    with Image.new("L", (16, 24), 255) as garment, Image.new("L", (16, 24)) as hair:
        analyzer = EarTailAnalyzer(Backend(), debug_dir=tmp_path, source_masks={"garment": garment, "hair": hair})
        with Image.new("RGB", (16, 24)) as image:
            assert analyzer.analyze(image, image.size, cancelled=lambda: False, deadline=perf_counter()+30) == []
        saved = json.loads((tmp_path/"parts.json").read_text(encoding="utf-8"))
        for name in ("human_ears", "animal_ears", "tail"):
            assert saved["parts"][name]["source_filter"][0]["reason"] == "garment_dominated"
            assert (tmp_path/f"{name}_candidate_0_mask.png").exists()
        assert saved["status"] == "uncertain"
        analyzer.close()


def test_ear_candidate_deduplication_drops_two_ear_envelope_and_duplicate():
    boxes = np.array([
        [10, 10, 90, 40],
        [10, 10, 35, 40],
        [65, 10, 90, 40],
        [11, 11, 36, 41],
    ], dtype=float)
    kept, removed = deduplicate_ear_candidate_indices(
        boxes, np.arange(4), np.array([.70, .80, .85, .60]))
    assert kept.tolist() == [1, 2]
    assert {item["candidate_index"]: item["reason"] for item in removed} == {
        0: "enclosing_multiple_ears",
        3: "duplicate_of:1",
    }



def test_tail_preflight_removes_sleeve_like_garment_overlap():
    selected, decisions, rejected = filter_tail_garment_overlaps(
        np.array([0, 1, 2]),
        [
            {"candidate_index": 0, "accepted": True,
             "garment_overlap": .89, "area_pixels": 100},
            {"candidate_index": 1, "accepted": True,
             "garment_overlap": .85, "area_pixels": 120},
            {"candidate_index": 2, "accepted": True,
             "garment_overlap": 0.0, "area_pixels": 300},
        ],
        .75,
    )
    assert selected.tolist() == [2]
    assert rejected == [0, 1]
    assert decisions[0]["reason"] == "tail_garment_overlap_exceeded"
    assert decisions[1]["selection_status"] == "not_proposed"
    assert decisions[2]["accepted"] is True


def test_accessory_observation_is_report_only_and_does_not_change_parts(tmp_path):
    backend = Backend()
    analyzer = EarTailAnalyzer(
        backend,
        debug_dir=tmp_path,
        accessory_observation_settings={
            "enabled": True,
            "tile_grid_size": 1,
            "save_view_images": False,
            "query_groups": [
                {"name": "hair_accessory", "query": "hair ornament."},
            ],
        },
    )
    with Image.new("RGB", (16, 24)) as image:
        parts = analyzer.analyze(
            image,
            image.size,
            cancelled=lambda: False,
            deadline=perf_counter() + 30,
        )
    try:
        assert [part.name for part in parts] == [
            "human_ears",
            "animal_ears",
            "tail",
        ]
        observation = analyzer.report["accessory_observations"]
        assert observation["mode"] == "observe_only"
        assert observation["sam2_invoked"] is False
        assert observation["masks_created"] is False
        assert observation["automatic_conditioning_applied"] is False
    finally:
        for part in parts:
            part.close()
        analyzer.close()


def test_oversized_small_part_is_unresolved_and_not_conditioned(tmp_path):
    class OversizedBackend(Backend):
        def segment(self, image, boxes):
            masks = np.ones((1, 1, image.height, image.width), dtype=bool)
            return masks, np.array([[.99]])

    foreground = Image.new("L", (16, 24), 255)
    garment = Image.new("L", (16, 24), 0)
    hair = Image.new("L", (16, 24), 0)
    analyzer = EarTailAnalyzer(
        OversizedBackend(),
        debug_dir=tmp_path,
        source_masks={
            "foreground": foreground,
            "garment": garment,
            "hair": hair,
        },
        part_queries=[{
            "name": "human_ears",
            "query": "human ears.",
            "hair_context": False,
        }],
        maximum_foreground_area_ratios={"human_ears": .12},
    )
    try:
        with Image.new("RGB", (16, 24)) as image:
            parts = analyzer.analyze(
                image,
                image.size,
                cancelled=lambda: False,
                deadline=perf_counter() + 30,
            )
        assert parts == []
        entry = analyzer.report["parts"]["human_ears"]
        assert entry["status"] == "uncertain"
        assert entry["outcome_reason"] == (
            "small_part_area_upper_bound_exceeded"
        )
        assert entry["automatic_conditioning"] is False
        assert entry["accepted_masks"] == 0
        assert entry["area_validity"]["valid"] is False
        assert entry["area_validity"]["denominator_source"] == (
            "foreground_mask"
        )
    finally:
        foreground.close()
        garment.close()
        hair.close()
        analyzer.close()


def test_output_hair_is_not_forced_through_small_part_area_limit(tmp_path):
    analyzer = EarTailAnalyzer(
        Backend(),
        debug_dir=tmp_path,
        part_queries=[{
            "name": "output_hair",
            "query": "hair.",
            "hair_context": False,
        }],
    )
    try:
        with Image.new("RGB", (16, 24)) as image:
            parts = analyzer.analyze(
                image,
                image.size,
                cancelled=lambda: False,
                deadline=perf_counter() + 30,
            )
        assert [part.name for part in parts] == ["output_hair"]
        validity = analyzer.report["parts"]["output_hair"]["area_validity"]
        assert validity["status"] == "not_applicable"
        assert validity["reason"] == "not_small_part_class"
        assert validity["maximum_foreground_area_ratio"] is None
        assert validity["valid"] is True
    finally:
        for part in parts:
            part.close()
        analyzer.close()
