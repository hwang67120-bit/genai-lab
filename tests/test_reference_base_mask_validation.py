from PIL import Image, ImageDraw

from genai_lab.reference_base_mask_validation import (
    validate_reference_base_masks,
)


def _mask(box=None, size=(100, 100)):
    image = Image.new("L", size)
    if box is not None:
        ImageDraw.Draw(image).rectangle(box, fill=255)
    return image


def _masks(box=(20, 20, 79, 89), size=(100, 100)):
    return {
        "face": _mask((40, 20, 59, 39), size),
        "hair": _mask((35, 15, 64, 44), size),
        "identity": _mask(box, size),
        "foreground": _mask(box, size),
        "garment": _mask((30, 45, 69, 75), size),
    }


def _close(masks):
    for image in masks.values():
        image.close()


def test_matching_reference_and_base_masks_are_diagnostic_consistent():
    source = _masks()
    base = {name: image.copy() for name, image in source.items()}
    try:
        report = validate_reference_base_masks(source, base)
        assert report["status"] == "consistent"
        assert report["blocking"] is False
        assert report["generation_masks_modified"] is False
        assert report["source_masks_reused_for_generation"] is False
        assert report["parts"]["identity"]["metrics"]["iou"] == 1.0
    finally:
        _close(source)
        _close(base)


def test_shifted_base_mask_blocks_automatic_progression():
    source = _masks()
    base = _masks(box=(60, 5, 99, 60))
    try:
        report = validate_reference_base_masks(source, base)
        assert report["status"] == "review_required"
        assert report["parts"]["identity"]["status"] == "review"
        assert "low_canvas_iou" in report["parts"]["identity"]["reasons"]
        assert report["blocking"] is True
        assert report["progression"] == "blocked"
        assert any(
            reason.startswith("identity:")
            for reason in report["blocking_reasons"]
        )
    finally:
        _close(source)
        _close(base)


def test_empty_mask_is_not_measurable_instead_of_low_similarity():
    source = _masks()
    base = _masks()
    base["hair"].close()
    base["hair"] = _mask()
    try:
        report = validate_reference_base_masks(source, base)
        assert report["status"] == "not_measurable"
        assert report["parts"]["hair"]["status"] == "not_measurable"
        assert report["parts"]["hair"]["reasons"] == [
            "base_mask_below_measurement_minimum"
        ]
        assert "iou" not in report["parts"]["hair"]["metrics"]
        assert report["blocking"] is True
    finally:
        _close(source)
        _close(base)


def test_source_resize_is_measurement_only_and_is_reported():
    source = _masks(size=(50, 50), box=(10, 10, 39, 44))
    base = _masks()
    try:
        report = validate_reference_base_masks(source, base)
        assert report["parts"]["identity"]["alignment"] == (
            "normalized_canvas_resize"
        )
        assert report["coordinate_policy"].startswith(
            "source masks are resized only for measurement")
    finally:
        _close(source)
        _close(base)



def test_nonempty_tiny_mask_is_not_measurable_and_has_no_iou():
    source = _masks()
    base = _masks()
    base["face"].close()
    base["face"] = _mask((50, 30, 50, 30))
    try:
        report = validate_reference_base_masks(source, base)
        face = report["parts"]["face"]
        assert face["status"] == "not_measurable"
        assert face["metrics"]["base_pixels"] == 1
        assert "iou" not in face["metrics"]
        assert face["blocks_generation"] is True
    finally:
        _close(source)
        _close(base)


def test_review_can_be_diagnostic_only_when_explicitly_configured():
    source = _masks()
    base = _masks(box=(60, 5, 99, 60))
    try:
        report = validate_reference_base_masks(
            source,
            base,
            settings={"block_on_review": False},
        )
        assert report["status"] == "review_required"
        assert report["blocking"] is False
    finally:
        _close(source)
        _close(base)
