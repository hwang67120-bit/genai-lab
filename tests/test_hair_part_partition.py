import numpy as np
from PIL import Image

from genai_lab.hair_part_partition import (
    partition_hair_regions,
    select_hair_correction_scope,
)
from genai_lab.hair_structure import resolve_hair_structure


def image_mask(values):
    return Image.fromarray(values.astype(np.uint8) * 255)


def masks():
    hair = np.zeros((64, 64), dtype=bool)
    hair[4:24, 14:50] = True
    hair[20:52, 10:22] = True
    hair[20:52, 42:54] = True
    face = np.zeros_like(hair)
    face[20:44, 22:42] = True
    return hair, face


def test_partition_is_disjoint_and_covers_every_hair_pixel():
    hair, face = masks()
    with image_mask(hair) as hair_image, image_mask(face) as face_image:
        regions, report = partition_hair_regions(hair_image, face_image)

    assert report["status"] == "PARTITIONED"
    assert report["disjoint"] is True
    assert report["complete_coverage"] is True
    names = [name for name in regions if name != "whole"]
    combined = np.zeros_like(hair)
    for name in names:
        assert not np.any(combined & regions[name])
        combined |= regions[name]
    assert np.array_equal(combined, hair)


def test_confirmed_front_selects_only_front_region():
    hair, face = masks()
    contract = resolve_hair_structure(("blunt_bangs",))
    with image_mask(hair) as hair_image, image_mask(face) as face_image:
        regions, _ = partition_hair_regions(hair_image, face_image)
    selected, report = select_hair_correction_scope(contract, regions)
    try:
        assert report["mode"] == "confirmed_local_parts"
        assert report["selected_regions"] == ["front"]
        assert np.array_equal(np.asarray(selected) >= 128, regions["front"])
        assert report["unknown_regions_excluded"] is True
    finally:
        selected.close()


def test_confirmed_side_selects_disjoint_left_and_right_regions():
    hair, face = masks()
    contract = resolve_hair_structure(("sidelocks",))
    with image_mask(hair) as hair_image, image_mask(face) as face_image:
        regions, _ = partition_hair_regions(hair_image, face_image)
    selected, report = select_hair_correction_scope(contract, regions)
    try:
        expected = regions["left_side"] | regions["right_side"]
        assert report["selected_regions"] == ["left_side", "right_side"]
        assert np.array_equal(np.asarray(selected) >= 128, expected)
    finally:
        selected.close()


def test_color_uses_whole_hair_but_unlocalized_ahoge_is_deferred():
    hair, face = masks()
    with image_mask(hair) as hair_image, image_mask(face) as face_image:
        regions, _ = partition_hair_regions(hair_image, face_image)

    color, color_report = select_hair_correction_scope(
        resolve_hair_structure(("blue_hair",)), regions)
    try:
        assert color_report["mode"] == "confirmed_global_hair_property"
        assert np.array_equal(np.asarray(color) >= 128, hair)
    finally:
        color.close()

    ahoge, ahoge_report = select_hair_correction_scope(
        resolve_hair_structure(("ahoge",)), regions)
    assert ahoge is None
    assert ahoge_report["reason"] == "pass1_structure_requires_base_regeneration"
    assert ahoge_report["deferred_to_base_retry"] == ["structure"]


def test_length_mismatch_is_not_rewritten_by_pass2():
    hair, face = masks()
    with image_mask(hair) as hair_image, image_mask(face) as face_image:
        regions, _ = partition_hair_regions(hair_image, face_image)

    selected, report = select_hair_correction_scope(
        resolve_hair_structure(("long_hair",)), regions)

    assert selected is None
    assert report["reason"] == "pass1_structure_requires_base_regeneration"
    assert report["deferred_to_base_retry"] == ["length"]


def test_unknown_contract_does_not_edit_any_region():
    hair, face = masks()
    with image_mask(hair) as hair_image, image_mask(face) as face_image:
        regions, _ = partition_hair_regions(hair_image, face_image)
    selected, report = select_hair_correction_scope(
        resolve_hair_structure(()), regions)
    assert selected is None
    assert report["reason"] == "no_confirmed_hair_parts"
