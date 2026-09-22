import numpy as np
from PIL import Image, ImageDraw

from genai_lab.reference_hair_observation import ReferenceHairObserver


def _mask(box=None, size=(100, 100)):
    image = Image.new("L", size, 0)
    if box is not None:
        ImageDraw.Draw(image).rectangle(box, fill=255)
    return image


def _close(masks):
    for image in masks.values():
        image.close()


class FakeBackend:
    def __init__(self, *, qualities=(0.95, 0.90), no_boxes=False):
        self.qualities = np.asarray(qualities, dtype=float)
        self.no_boxes = no_boxes
        self.closed = False

    def detect(self, image, query, threshold):
        if self.no_boxes:
            return np.empty((0, 4)), np.empty((0,))
        return (
            np.asarray([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=float),
            np.asarray([0.40, 0.90], dtype=float),
        )

    def segment(self, image, boxes):
        masks = []
        qualities = []
        for box in boxes:
            mask = np.zeros((image.height, image.width), dtype=bool)
            left, top, right, bottom = (int(value) for value in box)
            mask[top:bottom, left:right] = True
            masks.append(mask)
            original_index = 1 if left == 30 else 0
            qualities.append(self.qualities[original_index])
        return np.asarray(masks), np.asarray(qualities)

    def close(self):
        self.closed = True


def _config(**overrides):
    values = {
        "enabled": True,
        "box_threshold": 0.25,
        "mask_threshold": 0.80,
        "minimum_area_ratio": 0.001,
        "maximum_area_ratio": 0.20,
        "maximum_candidates": 8,
    }
    values.update(overrides)
    return {"reference_hair_observation": values}


def test_selects_highest_detector_score_as_measurement_only_roi():
    backend = FakeBackend()
    observer = ReferenceHairObserver(_config(), backend=backend)
    masks = {
        "hair": _mask((0, 0, 4, 4)),
        "face": _mask((30, 30, 34, 34)),
        "identity": _mask((45, 45, 49, 49)),
    }
    try:
        with Image.new("RGB", (100, 100), "white") as image:
            report = observer.observe(image, masks)
        assert report["status"] == "observed"
        assert report["semantic_mask"] is False
        assert report["automatic_generation_condition"] is False
        assert report["measurement_mask_available"] is True
        assert report["selected"]["detection_score"] == 0.90
        assert report["selected"]["mask_pixels"] == 100
        assert report["selected"]["face_overlap_ratio"] == 0.25
        assert np.count_nonzero(np.asarray(masks["hair"])) == 100
        assert np.count_nonzero(np.asarray(masks["identity"])) == 125
    finally:
        _close(masks)
        observer.close()
    assert backend.closed is True


def test_no_valid_candidate_invalidates_parser_hair_for_a6():
    backend = FakeBackend(qualities=(0.20, 0.30))
    observer = ReferenceHairObserver(_config(), backend=backend)
    masks = {
        "hair": _mask((0, 0, 49, 49)),
        "face": _mask((30, 30, 34, 34)),
        "identity": _mask((0, 0, 49, 49)),
    }
    identity_before = np.asarray(masks["identity"]).copy()
    try:
        with Image.new("RGB", (100, 100), "white") as image:
            report = observer.observe(image, masks)
        assert report["status"] == "unresolved"
        assert report["reason"] == "no_candidate_passed_measurement_bounds"
        assert report["masks_modified"] is True
        assert report["measurement_mask_available"] is False
        assert np.count_nonzero(np.asarray(masks["hair"])) == 0
        assert np.array_equal(np.asarray(masks["identity"]), identity_before)
    finally:
        _close(masks)
        observer.close()


def test_no_detection_invalidates_parser_hair_for_a6():
    backend = FakeBackend(no_boxes=True)
    observer = ReferenceHairObserver(_config(), backend=backend)
    masks = {
        "hair": _mask((0, 0, 49, 49)),
        "face": _mask(),
        "identity": _mask((0, 0, 49, 49)),
    }
    try:
        with Image.new("RGB", (100, 100), "white") as image:
            report = observer.observe(image, masks)
        assert report["status"] == "unresolved"
        assert report["candidate_count"] == 0
        assert np.count_nonzero(np.asarray(masks["hair"])) == 0
    finally:
        _close(masks)
        observer.close()


def test_disabled_observer_keeps_parser_mask_unchanged():
    backend = FakeBackend()
    observer = ReferenceHairObserver(_config(enabled=False), backend=backend)
    masks = {
        "hair": _mask((0, 0, 9, 9)),
        "face": _mask(),
        "identity": _mask((0, 0, 9, 9)),
    }
    try:
        with Image.new("RGB", (100, 100), "white") as image:
            report = observer.observe(image, masks)
        assert report["status"] == "disabled"
        assert report["masks_modified"] is False
        assert np.count_nonzero(np.asarray(masks["hair"])) == 100
    finally:
        _close(masks)
        observer.close()
