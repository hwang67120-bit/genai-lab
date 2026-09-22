import numpy as np
from PIL import Image, ImageDraw

from genai_lab.reference_face_observation import recover_reference_face_region


def _mask(box=None, size=(100, 100)):
    image = Image.new("L", size, 0)
    if box is not None:
        ImageDraw.Draw(image).rectangle(box, fill=255)
    return image


def _close(masks):
    for image in masks.values():
        image.close()


def test_measurable_parser_face_does_not_call_fallback(monkeypatch):
    masks = {
        "face": _mask((40, 20, 59, 39)),
        "identity": _mask((30, 10, 69, 49)),
    }
    monkeypatch.setattr(
        "genai_lab.reference_face_observation._detect_face_boxes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fallback must not run")
        ),
    )
    try:
        with Image.new("RGB", (100, 100)) as image:
            report = recover_reference_face_region(
                image,
                masks,
                {"reference_face_observation": {
                    "enabled": True,
                    "parser_minimum_area_ratio": 0.02,
                }},
            )
        assert report["status"] == "parser_mask_accepted"
        assert report["masks_modified"] is False
    finally:
        _close(masks)


def test_empty_parser_face_recovers_bounded_roi_and_identity(monkeypatch):
    masks = {
        "face": _mask(),
        "identity": _mask((35, 5, 64, 24)),
    }
    monkeypatch.setattr(
        "genai_lab.reference_face_observation._detect_face_boxes",
        lambda *args, **kwargs: [{
            "box": (40.2, 25.1, 60.1, 45.8),
            "confidence": 0.91,
        }],
    )
    try:
        with Image.new("RGB", (100, 100)) as image:
            report = recover_reference_face_region(
                image,
                masks,
                {"reference_face_observation": {
                    "enabled": True,
                    "parser_minimum_area_ratio": 0.002,
                    "minimum_area_ratio": 0.001,
                    "maximum_area_ratio": 0.12,
                }},
            )
        assert report["status"] == "recovered"
        assert report["semantic_mask"] is False
        assert report["box"] == [40, 25, 61, 46]
        assert np.count_nonzero(np.asarray(masks["face"])) == 21 * 21
        assert np.count_nonzero(np.asarray(masks["identity"])) > 30 * 20
    finally:
        _close(masks)


def test_oversized_face_box_is_unresolved_without_mutation(monkeypatch):
    masks = {"face": _mask(), "identity": _mask((35, 5, 64, 24))}
    before = np.asarray(masks["identity"]).copy()
    monkeypatch.setattr(
        "genai_lab.reference_face_observation._detect_face_boxes",
        lambda *args, **kwargs: [{
            "box": (0, 0, 100, 100),
            "confidence": 0.99,
        }],
    )
    try:
        with Image.new("RGB", (100, 100)) as image:
            report = recover_reference_face_region(
                image,
                masks,
                {"reference_face_observation": {
                    "enabled": True,
                    "maximum_area_ratio": 0.12,
                }},
            )
        assert report["status"] == "unresolved"
        assert report["reason"] == "face_box_area_out_of_bounds"
        assert report["masks_modified"] is False
        assert np.array_equal(np.asarray(masks["identity"]), before)
    finally:
        _close(masks)


def test_sparse_parser_face_with_oversized_bbox_uses_bounded_fallback(monkeypatch):
    face = _mask()
    ImageDraw.Draw(face).rectangle((0, 0, 9, 9), fill=255)
    ImageDraw.Draw(face).rectangle((90, 90, 99, 99), fill=255)
    masks = {
        "face": face,
        "identity": _mask((0, 0, 9, 9)),
    }
    monkeypatch.setattr(
        "genai_lab.reference_face_observation._detect_face_boxes",
        lambda *args, **kwargs: [{
            "box": (40, 20, 60, 40),
            "confidence": 0.92,
        }],
    )
    try:
        with Image.new("RGB", (100, 100)) as image:
            report = recover_reference_face_region(
                image,
                masks,
                {"reference_face_observation": {
                    "enabled": True,
                    "parser_minimum_area_ratio": 0.01,
                    "minimum_area_ratio": 0.001,
                    "maximum_area_ratio": 0.12,
                }},
            )
        assert report["status"] == "recovered"
        assert report["reason"] == "parser_face_outside_measurement_bounds"
        assert report["parser_face_area_ratio"] == 0.02
        assert report["parser_face_bbox_area_ratio"] == 1.0
        assert report["box"] == [40, 20, 60, 40]
    finally:
        _close(masks)


def test_generation_face_roi_is_removed_from_garment_influence(monkeypatch):
    masks = {
        "face": _mask(),
        "identity": _mask((0, 0, 9, 9)),
        "garment": _mask((0, 0, 99, 99)),
    }
    monkeypatch.setattr(
        "genai_lab.reference_face_observation._detect_face_boxes",
        lambda *args, **kwargs: [{
            "box": (40, 20, 60, 40),
            "confidence": 0.92,
        }],
    )
    try:
        with Image.new("RGB", (100, 100)) as image:
            report = recover_reference_face_region(
                image,
                masks,
                {"reference_face_observation": {
                    "enabled": True,
                    "minimum_area_ratio": 0.001,
                    "maximum_area_ratio": 0.12,
                }},
                exclude_from_garment=True,
            )
        face = np.asarray(masks["face"]) >= 128
        garment = np.asarray(masks["garment"]) >= 128
        assert report["garment_exclusion_applied"] is True
        assert report["garment_overlap_removed_pixels"] == 400
        assert not np.logical_and(face, garment).any()
    finally:
        _close(masks)
