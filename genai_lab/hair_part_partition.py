"""Disjoint, face-anchored hair regions for output-coordinate correction.

These masks are geometric edit scopes, not semantic proof. Every hair pixel is
assigned to exactly one region, and unconfirmed regions are never added to a
local correction target.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PIL import Image

from genai_lab.reference_regions import read_binary_mask


LOCAL_PART_REGIONS = {
    "front": ("front",),
    "side": ("left_side", "right_side"),
}
WHOLE_HAIR_PARTS = frozenset(("texture", "color"))
BASE_RETRY_PARTS = frozenset(("length", "arrangement", "structure"))


def partition_hair_regions(
    hair_mask,
    face_mask,
    *,
    minimum_region_pixels: int = 128,
) -> tuple[dict[str, np.ndarray], dict]:
    """Partition hair into non-overlapping face-relative edit regions."""

    if type(minimum_region_pixels) is not int or minimum_region_pixels < 1:
        raise ValueError("헤어 부분 최소 픽셀 수는 1 이상의 정수여야 합니다.")
    size = hair_mask.size
    hair = read_binary_mask(hair_mask, size)
    face = read_binary_mask(face_mask, size)
    if not hair.any():
        return {}, {
            "status": "UNRESOLVED",
            "reason": "empty_hair_mask",
            "disjoint": True,
            "complete_coverage": True,
        }
    rows, columns = np.nonzero(face)
    if not len(rows):
        return {"whole": hair}, {
            "status": "UNRESOLVED",
            "reason": "missing_face_anchor",
            "disjoint": True,
            "complete_coverage": True,
            "regions": {"whole": int(hair.sum())},
        }

    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    width, height = max(1, right - left), max(1, bottom - top)
    center_x = (left + right) / 2
    yy, xx = np.indices(hair.shape)

    front_window = (
        (xx >= center_x - width * .5)
        & (xx < center_x + width * .5)
        & (yy <= top + height * .55)
    )
    side_bottom = bottom + height * .35
    left_window = (
        (xx < left + width * .35)
        & (yy >= top)
        & (yy < side_bottom)
    )
    right_window = (
        (xx >= right - width * .35)
        & (yy >= top)
        & (yy < side_bottom)
    )
    rear_window = yy >= top + height * .75

    remaining = hair.copy()
    raw = {}
    for name, window in (
        ("front", front_window),
        ("left_side", left_window),
        ("right_side", right_window),
        ("rear_silhouette_candidate", rear_window),
    ):
        selected = remaining & window
        raw[name] = selected
        remaining &= ~selected

    regions = {}
    unresolved = remaining.copy()
    for name, selected in raw.items():
        if int(selected.sum()) >= minimum_region_pixels:
            regions[name] = selected
        else:
            unresolved |= selected
    if unresolved.any():
        regions["unresolved"] = unresolved
    regions["whole"] = hair

    names = [name for name in regions if name not in {"whole", "unresolved"}]
    overlap_pixels = 0
    assigned = np.zeros_like(hair)
    for name in names:
        overlap_pixels += int(np.count_nonzero(assigned & regions[name]))
        assigned |= regions[name]
    if "unresolved" in regions:
        overlap_pixels += int(np.count_nonzero(assigned & regions["unresolved"]))
        assigned |= regions["unresolved"]
    complete = np.array_equal(assigned, hair)
    return regions, {
        "status": "PARTITIONED" if complete and overlap_pixels == 0 else "UNRESOLVED",
        "reason": None if complete and overlap_pixels == 0 else "partition_invariant_failed",
        "disjoint": overlap_pixels == 0,
        "complete_coverage": complete,
        "overlap_pixels": overlap_pixels,
        "regions": {
            name: int(mask.sum()) for name, mask in regions.items()
        },
        "semantic_identity_confirmed": False,
    }


def select_hair_correction_scope(
    contract: Mapping | None,
    regions: Mapping[str, np.ndarray],
    *,
    legacy_whole_if_missing: bool = False,
) -> tuple[Image.Image | None, dict]:
    """Select only approved local regions; unknown parts remain untouched."""

    if not isinstance(contract, Mapping) or not contract.get("confirmed_parts"):
        if legacy_whole_if_missing and "whole" in regions:
            selected = regions["whole"]
            return Image.fromarray(selected.astype(np.uint8) * 255), {
                "status": "SELECTED",
                "mode": "legacy_whole",
                "requested_parts": [],
                "selected_regions": ["whole"],
                "pixel_count": int(selected.sum()),
            }
        return None, {
            "status": "UNRESOLVED",
            "reason": "no_confirmed_hair_parts",
            "requested_parts": [],
            "selected_regions": [],
        }

    confirmed = tuple(dict.fromkeys(contract.get("confirmed_parts", ())))
    local_regions = []
    for part in confirmed:
        for region in LOCAL_PART_REGIONS.get(part, ()):
            if region in regions:
                local_regions.append(region)
    local_regions = list(dict.fromkeys(local_regions))

    deferred_to_base_retry = [
        part for part in confirmed if part in BASE_RETRY_PARTS
    ]
    if local_regions:
        selected = np.zeros_like(regions["whole"])
        for name in local_regions:
            selected |= regions[name]
        mode = "confirmed_local_parts"
        selected_regions = local_regions
    elif any(part in WHOLE_HAIR_PARTS for part in confirmed):
        selected = regions.get("whole")
        mode = "confirmed_global_hair_property"
        selected_regions = ["whole"] if selected is not None else []
    else:
        return None, {
            "status": "UNRESOLVED",
            "reason": (
                "pass1_structure_requires_base_regeneration"
                if deferred_to_base_retry
                else "confirmed_parts_not_safely_localizable"
            ),
            "requested_parts": list(confirmed),
            "unlocalized_parts": deferred_to_base_retry,
            "deferred_to_base_retry": deferred_to_base_retry,
            "selected_regions": [],
        }

    if selected is None or not selected.any():
        return None, {
            "status": "UNRESOLVED",
            "reason": "selected_hair_scope_empty",
            "requested_parts": list(confirmed),
            "selected_regions": selected_regions,
        }
    return Image.fromarray(selected.astype(np.uint8) * 255), {
        "status": "SELECTED",
        "mode": mode,
        "requested_parts": list(confirmed),
        "selected_regions": selected_regions,
        "pixel_count": int(selected.sum()),
        "unknown_regions_excluded": True,
        "deferred_to_base_retry": deferred_to_base_retry,
    }
