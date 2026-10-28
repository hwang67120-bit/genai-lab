import numpy as np
import pytest
from PIL import Image

from genai_lab.accessory_analysis import (
    AccessoryObservationPolicy,
    AccessoryView,
    analyze_accessory_observations,
    build_accessory_views,
    estimate_model_input_size,
    map_view_box_to_original,
)


class ObservationBackend:
    def __init__(self, *, conflict=False, empty=False):
        self.conflict = conflict
        self.empty = empty
        self.detect_calls = []
        self.segment_calls = 0

    def detect(self, image, query, threshold):
        self.detect_calls.append((image.size, query, threshold))
        if self.empty:
            return np.empty((0, 4)), np.empty(0)
        active = query.startswith("hair ornament")
        if self.conflict:
            active = active or query.startswith("animal ears")
        if not active:
            return np.empty((0, 4)), np.empty(0)
        boxes = {
            (100, 200): [40, 20, 50, 30],
            (60, 90): [20, 10, 30, 20],
        }
        if image.size not in boxes:
            return np.empty((0, 4)), np.empty(0)
        return np.asarray([boxes[image.size]], dtype=float), np.asarray([.8])

    def segment(self, *args):
        self.segment_calls += 1
        pytest.fail("관찰 전용 장신구 분석은 SAM2를 호출하면 안 됩니다.")


def policy(*, conflict=False):
    queries = [
        {"name": "hair_accessory", "query": "hair ornament."},
    ]
    if conflict:
        queries.insert(0, {"name": "animal_ears", "query": "animal ears."})
    return AccessoryObservationPolicy.from_mapping({
        "enabled": True,
        "tile_grid_size": 1,
        "save_view_images": False,
        "query_groups": queries,
    })


def test_views_and_coordinate_mapping_remain_in_original_space():
    active = AccessoryObservationPolicy(tile_grid_size=2)
    views = build_accessory_views((100, 200), (20, 10, 80, 100), active)
    assert [view.name for view in views] == [
        "full_image",
        "head_crop",
        "head_tile_r0_c0",
        "head_tile_r0_c1",
        "head_tile_r1_c0",
        "head_tile_r1_c1",
    ]
    head = views[1]
    assert map_view_box_to_original(
        (20, 10, 30, 20),
        head,
        (100, 200),
    ) == (40.0, 20.0, 50.0, 30.0)


def test_estimated_resize_records_tall_image_resolution_loss():
    active = AccessoryObservationPolicy()
    assert estimate_model_input_size((936, 2048), active) == (609, 1333)
    assert estimate_model_input_size((350, 350), active) == (800, 800)


def test_same_label_in_full_and_head_view_is_localized_without_sam2(tmp_path):
    backend = ObservationBackend()
    with Image.new("RGB", (100, 200)) as source:
        report = analyze_accessory_observations(
            backend,
            source,
            head_box=(20, 10, 80, 100),
            box_threshold=.35,
            policy=policy(),
            check=lambda: None,
            debug_directory=tmp_path,
        )
    assert report["status"] == "localized"
    assert report["clusters"][0]["source_views"] == [
        "full_image",
        "head_crop",
    ]
    assert report["clusters"][0]["mask_eligible"] is True
    assert report["masks_created"] is False
    assert report["sam2_invoked"] is False
    assert report["automatic_conditioning_applied"] is False
    assert backend.segment_calls == 0
    full_candidate = report["clusters"][0]["candidates"][0]
    assert full_candidate["original_size_px"] == [10.0, 10.0]
    assert full_candidate["estimated_model_object_size_px"] == pytest.approx(
        [66.6, 66.65]
    )


def test_cross_label_same_location_is_conflict_and_never_selects_mask():
    backend = ObservationBackend(conflict=True)
    with Image.new("RGB", (100, 200)) as source:
        report = analyze_accessory_observations(
            backend,
            source,
            head_box=(20, 10, 80, 100),
            box_threshold=.35,
            policy=policy(conflict=True),
            check=lambda: None,
        )
    assert report["status"] == "class_conflict"
    assert report["review_required"] is True
    assert set(report["clusters"][0]["labels"]) == {
        "animal_ears",
        "hair_accessory",
    }
    assert report["clusters"][0]["mask_eligible"] is False
    assert report["automatic_conditioning_applied"] is False


def test_no_detection_is_not_promoted_to_absence():
    backend = ObservationBackend(empty=True)
    with Image.new("RGB", (100, 200)) as source:
        report = analyze_accessory_observations(
            backend,
            source,
            head_box=(20, 10, 80, 100),
            box_threshold=.35,
            policy=policy(),
            check=lambda: None,
        )
    assert report["status"] == "not_detected"
    assert report["presence"] == "not_observed"
    assert report["not_detected_is_absence"] is False
    assert report["automatic_conditioning_applied"] is False


def test_policy_rejects_automatic_mode_and_unknown_settings():
    with pytest.raises(ValueError, match="observe_only"):
        AccessoryObservationPolicy(mode="automatic")
    with pytest.raises(ValueError, match="등록되지 않은"):
        AccessoryObservationPolicy.from_mapping({
            "enabled": True,
            "hard_gate": True,
        })


class BatchedObservationBackend(ObservationBackend):
    def __init__(self):
        super().__init__()
        self.labeled_calls = []

    def detect_labeled(self, image, query_groups, threshold):
        self.labeled_calls.append((image.size, tuple(query_groups), threshold))
        boxes = {
            (100, 200): [40, 20, 50, 30],
            (60, 90): [20, 10, 30, 20],
        }
        return (
            np.asarray([boxes[image.size]], dtype=float),
            np.asarray([.8]),
            ("hair_accessory",),
        )


def test_batched_backend_runs_one_detector_pass_per_view():
    backend = BatchedObservationBackend()
    with Image.new("RGB", (100, 200)) as source:
        report = analyze_accessory_observations(
            backend,
            source,
            head_box=(20, 10, 80, 100),
            box_threshold=.35,
            policy=policy(),
            check=lambda: None,
        )
    assert report["status"] == "localized"
    assert len(backend.labeled_calls) == 2
    assert backend.detect_calls == []