import numpy as np
import pytest

from genai_lab.part_spatial_diagnostics import (
    PartSpatialDiagnosticPolicy,
    diagnose_part_spatial_consistency,
)


def _mask(size, box):
    width, height = size
    values = np.zeros((height, width), dtype=bool)
    left, top, right, bottom = box
    values[top:bottom, left:right] = True
    return values


def test_head_and_tail_spatial_diagnostics_are_observe_only():
    size = (20, 20)
    report, rois = diagnose_part_spatial_consistency(
        size=size,
        part_masks={
            "animal_ears": _mask(size, (2, 15, 6, 19)),
            "hair_accessory": _mask(size, (8, 1, 12, 5)),
            "tail": _mask(size, (3, 17, 8, 20)),
        },
    )

    assert report["head_roi"]["source"] == "normalized_image_fallback"
    assert report["head_checks"]["animal_ears"]["status"] == (
        "outside_head_roi"
    )
    assert report["head_checks"]["hair_accessory"]["status"] == (
        "inside_head_roi"
    )
    assert report["tail_connection_check"]["status"] == (
        "outside_body_anchor"
    )
    assert report["review_required"] is True
    assert report["predictions_modified"] is False
    assert report["masks_modified"] is False
    assert report["automatic_conditioning_applied"] is False
    assert rois["head_roi"].shape == (20, 20)
    assert rois["tail_body_anchor_roi"].shape == (20, 20)


def test_same_mask_cross_class_conflict_uses_containment_and_iou():
    size = (20, 20)
    shared = _mask(size, (3, 2, 9, 8))
    report, _ = diagnose_part_spatial_consistency(
        size=size,
        part_masks={
            "animal_ears": shared,
            "hair_accessory": shared.copy(),
        },
    )

    conflicts = report["same_mask_cross_class_conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["parts"] == ["animal_ears", "hair_accessory"]
    assert conflicts[0]["status"] == "same_mask_cross_class_conflict"
    assert conflicts[0]["smaller_part_overlap_ratio"] == 1.0
    assert conflicts[0]["mask_iou"] == 1.0


def test_context_masks_define_head_and_body_anchor_rois():
    size = (20, 20)
    hair = _mask(size, (12, 11, 17, 16))
    garment = _mask(size, (4, 5, 15, 17))
    report, _ = diagnose_part_spatial_consistency(
        size=size,
        part_masks={
            "animal_ears": _mask(size, (12, 11, 17, 15)),
            "tail": _mask(size, (12, 10, 18, 16)),
        },
        context_masks={"hair": hair, "garment": garment},
    )

    assert report["head_roi"]["source"] == "hair_context_bounds"
    assert report["head_checks"]["animal_ears"]["status"] == (
        "inside_head_roi"
    )
    assert report["tail_body_anchor_roi"]["source"] == (
        "garment_context_lower_torso"
    )
    assert report["tail_connection_check"]["status"] == (
        "body_anchor_contact"
    )


def test_policy_rejects_invalid_ranges_and_unknown_keys():
    with pytest.raises(ValueError, match="좌우"):
        PartSpatialDiagnosticPolicy(
            tail_fallback_left_ratio=0.9,
            tail_fallback_right_ratio=0.1,
        )
    with pytest.raises(ValueError, match="등록되지 않은"):
        PartSpatialDiagnosticPolicy.from_mapping({"unknown": 0.5})
