import numpy as np
from PIL import Image, ImageDraw

from genai_lab.hair_inpaint_guard import (
    evaluate_hair_promotion,
    guarded_hair_composite,
    resolve_hair_boundary_guard,
    resolve_hair_promotion_gate,
)


def mask(box, size=(32, 32)):
    result = Image.new("L", size)
    ImageDraw.Draw(result).rectangle(box, fill=255)
    return result


def permissive_boundary(**overrides):
    values = {
        "boundary_width_pixels": 3,
        "feather_radius_pixels": 2,
        "maximum_raw_outside_changed_ratio": 1.0,
        "seam_mean_difference_max": 255,
        "seam_p95_difference_max": 255,
        "seam_changed_ratio_max": 1.0,
    }
    values.update(overrides)
    return resolve_hair_boundary_guard(values)


def test_guarded_composite_restores_protected_and_outside_pixels_exactly():
    base = Image.new("RGB", (32, 32), "blue")
    proposed = Image.new("RGB", (32, 32), "red")
    repair = mask((6, 4, 25, 25))
    protected = mask((12, 10, 19, 20))
    corrected, report = guarded_hair_composite(
        base, proposed, repair, protected, permissive_boundary())
    try:
        assert report["status"] == "passed"
        assert report["final_protected_changed_pixels"] == 0
        assert report["final_outside_changed_pixels"] == 0
        assert corrected.getpixel((15, 15)) == base.getpixel((15, 15))
        assert corrected.getpixel((0, 0)) == base.getpixel((0, 0))
        assert corrected.getpixel((8, 8)) != base.getpixel((8, 8))
    finally:
        corrected.close()
        base.close()
        proposed.close()
        repair.close()
        protected.close()


def test_guarded_composite_rejects_seam_above_threshold():
    base = Image.new("RGB", (32, 32), "black")
    proposed = Image.new("RGB", (32, 32), "white")
    repair = mask((8, 8, 23, 23))
    protected = Image.new("L", (32, 32))
    corrected, report = guarded_hair_composite(
        base, proposed, repair, protected,
        permissive_boundary(
            seam_mean_difference_max=1,
            seam_p95_difference_max=1,
            seam_changed_ratio_max=0.01,
        ))
    try:
        assert corrected is None
        assert report["status"] == "rejected"
        assert report["reason"] == "seam_threshold_exceeded"
    finally:
        base.close()
        proposed.close()
        repair.close()
        protected.close()


def test_hair_promotion_uses_same_coordinate_geometry_and_source_color():
    source = Image.new("RGB", (32, 32), (70, 110, 210))
    before = source.copy()
    corrected = source.copy()
    source_hair = mask((6, 3, 25, 20))
    before_hair = source_hair.copy()
    after_hair = source_hair.copy()
    before_face = mask((11, 10, 20, 21))
    after_face = before_face.copy()
    settings = resolve_hair_promotion_gate({
        "minimum_color_similarity": .8,
        "minimum_shape_iou": .65,
        "maximum_length_change_ratio": .2,
        "maximum_area_change_ratio": .25,
    })
    report = evaluate_hair_promotion(
        source, source_hair, before, before_hair, before_face,
        corrected, after_hair, after_face, settings)
    try:
        assert report["passed"] is True
        assert report["metrics"]["shape_iou"] == 1.0
        assert report["metrics"]["length_change_ratio"] == 0.0
    finally:
        source.close()
        before.close()
        corrected.close()
        source_hair.close()
        before_hair.close()
        after_hair.close()
        before_face.close()
        after_face.close()


def test_hair_promotion_rejects_large_shape_and_length_change():
    source = Image.new("RGB", (32, 32), (70, 110, 210))
    before = source.copy()
    corrected = source.copy()
    source_hair = mask((6, 3, 25, 14))
    before_hair = source_hair.copy()
    after_hair = mask((6, 3, 25, 28))
    before_face = mask((11, 10, 20, 21))
    after_face = before_face.copy()
    report = evaluate_hair_promotion(
        source, source_hair, before, before_hair, before_face,
        corrected, after_hair, after_face,
        resolve_hair_promotion_gate({}))
    try:
        assert report["passed"] is False
        assert report["status"] == "rejected"
        assert report["checks"]["length"] is False
    finally:
        source.close()
        before.close()
        corrected.close()
        source_hair.close()
        before_hair.close()
        after_hair.close()
        before_face.close()
        after_face.close()
