import numpy as np
from PIL import Image

from genai_lab.hair_mask_refinement import (
    HairMaskRefinementSettings,
    refine_hair_mask,
)


def mask(values):
    return Image.fromarray(values.astype(np.uint8) * 255)


def settings(**overrides):
    values = {
        "enabled": True,
        "minimum_hair_pixels": 20,
        "minimum_retained_ratio": 0.5,
        "maximum_missing_ratio": 1.0,
        "protection_padding_pixels": 0,
        "hole_inspection_radius": 1,
        "retry_close_radii": (0,),
    }
    values.update(overrides)
    return HairMaskRefinementSettings(**values)


def test_face_and_animal_ears_are_removed_before_hair_is_returned():
    head = np.ones((20, 20), dtype=bool)
    raw = np.zeros_like(head)
    raw[:12, :] = True
    face = np.zeros_like(head)
    face[5:11, 7:13] = True
    ears = np.zeros_like(head)
    ears[1:4, 2:5] = True
    images = [mask(value) for value in (raw, head, face, ears)]
    try:
        result = refine_hair_mask(*images, settings())
        assert result.status == "repaired_and_accepted"
        assert result.mask is not None
        selected = np.asarray(result.mask) >= 128
        assert not np.any(selected & face)
        assert not np.any(selected & ears)
        assert result.attempts[0]["raw_face_overlap"] > 0
        assert result.attempts[0]["raw_ear_overlap"] > 0
        result.mask.close()
    finally:
        for image in images:
            image.close()


def test_hair_accessory_is_removed_before_hair_is_returned():
    head = np.ones((20, 20), dtype=bool)
    raw = np.zeros_like(head)
    raw[2:16, 3:17] = True
    face = np.zeros_like(head)
    ears = np.zeros_like(head)
    accessory = np.zeros_like(head)
    accessory[3:7, 13:17] = True
    images = [mask(value) for value in (raw, head, face, ears, accessory)]
    try:
        result = refine_hair_mask(
            *images[:4], settings(), hair_accessory_mask=images[4])
        assert result.status == "repaired_and_accepted"
        assert result.mask is not None
        selected = np.asarray(result.mask) >= 128
        assert not np.any(selected & accessory)
        assert result.attempts[0]["raw_accessory_overlap"] > 0
        assert result.attempts[0]["accessory_overlap_after"] == 0
        result.mask.close()
    finally:
        for image in images:
            image.close()


def test_human_and_animal_ears_are_both_removed_from_hair():
    head = np.ones((20, 20), dtype=bool)
    raw = np.zeros_like(head)
    raw[1:16, 2:18] = True
    face = np.zeros_like(head)
    human = np.zeros_like(head)
    human[7:10, 2:5] = True
    animal = np.zeros_like(head)
    animal[1:4, 13:17] = True
    images = [mask(value) for value in (raw, head, face, animal, human)]
    try:
        result = refine_hair_mask(
            *images[:4], settings(), human_ears_mask=images[4])
        assert result.mask is not None
        selected = np.asarray(result.mask) >= 128
        assert not np.any(selected & human)
        assert not np.any(selected & animal)
        attempt = result.attempts[-1]
        assert attempt["raw_human_ear_overlap"] > 0
        assert attempt["raw_animal_ear_overlap"] > 0
        assert attempt["human_ear_overlap_after"] == 0
        assert attempt["animal_ear_overlap_after"] == 0
        result.mask.close()
    finally:
        for image in images:
            image.close()


def test_hole_is_retried_and_closed_before_acceptance():
    head = np.ones((20, 20), dtype=bool)
    raw = np.zeros_like(head)
    raw[4:16, 4:16] = True
    raw[9:11, 9:11] = False
    empty = np.zeros_like(head)
    images = [mask(value) for value in (raw, head, empty)]
    try:
        result = refine_hair_mask(
            images[0], images[1], images[2], None,
            settings(
                minimum_hair_pixels=100,
                maximum_missing_ratio=0.0,
                retry_close_radii=(0, 1, 2),
            ),
        )
        assert result.mask is not None
        assert result.status == "repaired_and_accepted"
        assert len(result.attempts) >= 2
        assert result.attempts[0]["accepted"] is False
        assert result.attempts[-1]["accepted"] is True
        assert np.all((np.asarray(result.mask) >= 128)[9:11, 9:11])
        result.mask.close()
    finally:
        for image in images:
            image.close()


def test_failed_retries_return_structured_unresolved_result():
    raw = np.zeros((12, 12), dtype=bool)
    raw[1, 1] = True
    head = np.ones_like(raw)
    face = np.zeros_like(raw)
    images = [mask(value) for value in (raw, head, face)]
    try:
        result = refine_hair_mask(
            images[0], images[1], images[2], None,
            settings(minimum_hair_pixels=10, retry_close_radii=(0, 1)),
        )
        assert result.status == "unresolved"
        assert result.reason == "hair_mask_refinement_exhausted"
        assert result.mask is None
        assert len(result.attempts) == 2
        assert result.record()["mask_returned"] is False
    finally:
        for image in images:
            image.close()
