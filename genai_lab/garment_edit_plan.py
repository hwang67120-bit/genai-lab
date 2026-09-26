"""Output-coordinate garment coverage and Hard/Soft edit planning.

Reference-image pixels are never projected into a generated Base.  Reference
tags provide semantics; all masks handled here already use Base coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from genai_lab.output_coordinate_control import create_dual_masks
from genai_lab.reference_features import tag_scope


@dataclass
class TargetGarmentCoverage:
    mask: Image.Image
    growth_envelope: Image.Image
    record: dict[str, Any]
    pre_union_mask: Image.Image | None = None

    def close(self) -> None:
        self.mask.close()
        self.growth_envelope.close()
        if self.pre_union_mask is not None:
            self.pre_union_mask.close()


@dataclass
class GarmentEditPlan:
    images: dict[str, Image.Image]
    metrics: dict[str, Any]
    record: dict[str, Any]

    @property
    def hard_edit_domain(self) -> Image.Image:
        return self.images["hard_edit_domain"]

    @property
    def soft_guidance(self) -> Image.Image:
        return self.images["soft_guidance"]

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, image in self.images.items():
            image.save(directory / f"{name}.png")
        manifest = directory / "garment_edit_plan.json"
        manifest.write_text(
            json.dumps(self.record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def close(self) -> None:
        for image in self.images.values():
            image.close()


def _binary(mask: Image.Image, size: tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(mask, Image.Image) or mask.size != size:
        raise ValueError(f"{name} mask does not use generated Base coordinates")
    with mask.convert("L") as gray:
        values = np.asarray(gray, dtype=np.uint8).copy()
    if not np.isin(values, (0, 255)).all():
        raise ValueError(f"{name} mask must be binary 0/255")
    return values == 255


def _optional_binary(
    mask: Image.Image | None,
    size: tuple[int, int],
    name: str,
) -> np.ndarray:
    if mask is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    return _binary(mask, size, name)


def _mask(values: np.ndarray) -> Image.Image:
    return Image.fromarray(values.astype(np.uint8) * 255, mode="L")


def _garment_tags(tags: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in tags:
        if not isinstance(value, str):
            continue
        tag = value.strip().lower().replace("_", " ")
        if tag and tag_scope(tag) == "garment":
            normalized.append(tag)
    return tuple(dict.fromkeys(normalized))


def project_target_garment_coverage(
    source_garment_mask: Image.Image,
    foreground_mask: Image.Image,
    approved_tags: Iterable[str],
    *,
    growth_pixels: int = 12,
) -> TargetGarmentCoverage:
    """Project approved garment semantics into generated Base coordinates."""
    if type(growth_pixels) is not int or not 0 <= growth_pixels <= 64:
        raise ValueError("garment growth pixels must be an integer from 0 to 64")
    size = source_garment_mask.size
    source = _binary(source_garment_mask, size, "source_garment")
    foreground = _binary(foreground_mask, size, "foreground")
    tags = _garment_tags(approved_tags)
    target = np.zeros_like(source)
    growth = np.zeros_like(source)
    if not foreground.any():
        raise ValueError("generated Base foreground mask is empty")
    if not source.any():
        raise ValueError("generated Base source garment mask is empty")

    ys, xs = np.nonzero(foreground)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    height = max(1, bottom - top)
    width = max(1, right - left)
    garment_y, garment_x = np.nonzero(source)
    garment_left = int(garment_x.min())
    garment_right = int(garment_x.max()) + 1

    def row(ratio: float) -> int:
        return min(bottom, max(top, top + round(height * ratio)))

    top_terms = {
        "shirt", "blouse", "jacket", "blazer", "coat", "sweater",
        "cardigan", "hoodie", "vest", "top", "uniform",
    }
    bottom_terms = {
        "skirt", "mini skirt", "microskirt", "pants", "trousers",
        "shorts", "leggings", "tights", "pantyhose", "stockings",
        "thighhighs",
    }
    dress_terms = {"dress", "gown", "bodysuit", "leotard", "swimsuit"}
    shoe_terms = {"footwear", "shoes", "boots", "heels", "loafers", "sneakers"}
    tag_set = set(tags)
    known_terms = top_terms | bottom_terms | dress_terms | shoe_terms | {
        "long sleeves", "sleeves past wrists", "sleeves past fingers",
    }
    semantic_terms = {
        term for tag in tag_set for term in known_terms
        if tag == term or tag.endswith(" " + term)
    }
    has_top = bool(semantic_terms & top_terms)
    has_bottom = bool(semantic_terms & bottom_terms)
    has_dress = bool(semantic_terms & dress_terms)
    has_shoes = bool(semantic_terms & shoe_terms)
    long_sleeves = bool(semantic_terms & {
        "long sleeves", "sleeves past wrists", "sleeves past fingers",
    })
    categories = tuple(
        name for name, present in (
            ("top", has_top), ("bottom", has_bottom),
            ("dress", has_dress), ("shoes", has_shoes),
        ) if present
    )
    if not categories:
        return TargetGarmentCoverage(
            _mask(target),
            _mask(growth),
            {
                "version": "target_garment_coverage_v1",
                "status": "UNRESOLVED",
                "method": "generated_base_foreground_bands_v1",
                "approved_garment_tags": list(tags),
                "categories": [],
                "review_reasons": ["garment_category_unresolved"],
                "coordinate_source": "generated_base",
            },
        )

    if has_top or has_dress:
        y1, y2 = row(0.16), row(0.58 if has_top and not has_dress else 0.90)
        band = np.zeros_like(source)
        band[y1:y2, :] = True
        if long_sleeves or has_dress:
            target |= foreground & band
        else:
            pad = max(2, round((garment_right - garment_left) * 0.15))
            x1 = max(left, garment_left - pad)
            x2 = min(right, garment_right + pad)
            torso = np.zeros_like(source)
            torso[y1:y2, x1:x2] = True
            target |= foreground & torso

    if has_bottom or has_dress:
        y1 = row(0.50)
        if semantic_terms & {"mini skirt", "microskirt", "shorts"}:
            y2 = row(0.70)
        elif semantic_terms & {"skirt"}:
            y2 = row(0.78)
        else:
            y2 = row(0.97)
        band = np.zeros_like(source)
        band[y1:y2, :] = True
        target |= foreground & band
        if growth_pixels and (
            has_dress or bool(semantic_terms & {"skirt", "mini skirt", "microskirt", "coat"})
        ):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (growth_pixels * 2 + 1, growth_pixels * 2 + 1),
            )
            expanded = cv2.dilate(
                (target & band).astype(np.uint8), kernel, iterations=1
            ).astype(bool)
            growth |= expanded & band & ~foreground
            target |= growth

    if has_shoes:
        band = np.zeros_like(source)
        band[row(0.84):bottom, :] = True
        target |= foreground & band

    # Observe the target before the existing source union, without changing it.
    pre_union_mask = _mask(target)
    target |= source
    return TargetGarmentCoverage(
        _mask(target),
        _mask(growth),
        {
            "version": "target_garment_coverage_v1",
            "status": "REVIEW",
            "method": "generated_base_foreground_bands_v1",
            "approved_garment_tags": list(tags),
            "categories": list(categories),
            "long_sleeves": long_sleeves,
            "growth_pixels": growth_pixels,
            "target_pixels": int(np.count_nonzero(target)),
            "growth_pixels_count": int(np.count_nonzero(growth)),
            "foreground_bbox_xywh": [left, top, width, height],
            "review_reasons": ["foreground_band_projection_requires_review"],
            "coordinate_source": "generated_base",
        },
        pre_union_mask=pre_union_mask,
    )


def build_garment_edit_plan(
    generated_base: Image.Image,
    source_garment_removal: Image.Image,
    target_garment_coverage: Image.Image,
    hard_protection: Image.Image,
    foreground: Image.Image,
    *,
    garment_growth_envelope: Image.Image | None = None,
    soft_boundary_protection: Image.Image | None = None,
    conditional_protection: Image.Image | None = None,
    conditional_release: Image.Image | None = None,
    user_add: Image.Image | None = None,
    user_exclude: Image.Image | None = None,
    feather_radius: int = 10,
    soft_boundary_strength: float = 0.5,
) -> GarmentEditPlan:
    """Build auditable Hard/Soft masks in generated Base coordinates."""
    if not 0.0 <= soft_boundary_strength <= 1.0:
        raise ValueError("soft boundary strength must be between 0 and 1")
    size = generated_base.size
    source = _binary(source_garment_removal, size, "source_garment_removal")
    target = _binary(target_garment_coverage, size, "target_garment_coverage")
    hard_keep = _binary(hard_protection, size, "hard_protection")
    fg = _binary(foreground, size, "foreground")
    growth = _optional_binary(
        garment_growth_envelope, size, "garment_growth_envelope")
    conditional = _optional_binary(
        conditional_protection, size, "conditional_protection")
    release = _optional_binary(
        conditional_release, size, "conditional_release")
    add = _optional_binary(user_add, size, "user_add")
    exclude = _optional_binary(user_exclude, size, "user_exclude")

    requested = source | target | add
    editable_domain = fg | growth
    requested_in_domain = requested & editable_domain & ~exclude
    effective_protection = hard_keep | (conditional & ~release)
    conflict = requested_in_domain & effective_protection
    hard_edit = requested_in_domain & ~effective_protection
    if not hard_edit.any():
        raise ValueError("garment edit domain is empty after protection")

    hard_image, soft_image = create_dual_masks(
        _mask(hard_edit), feather_radius=feather_radius
    )
    try:
        soft = np.asarray(soft_image, dtype=np.float32).copy()
        if soft_boundary_protection is not None:
            if soft_boundary_protection.size != size:
                raise ValueError(
                    "soft_boundary_protection mask does not use generated Base coordinates"
                )
            with soft_boundary_protection.convert("L") as gray:
                boundary = np.asarray(gray, dtype=np.float32).copy() / 255.0
            soft *= 1.0 - boundary * soft_boundary_strength
        soft *= hard_edit.astype(np.float32)
        soft_uint8 = np.clip(np.rint(soft), 0, 255).astype(np.uint8)
        if not soft_uint8.any():
            raise ValueError("soft guidance is empty after boundary protection")
        effective_radius = int(soft_image.info.get(
            "effective_feather_radius", feather_radius))
    finally:
        soft_image.close()

    outside_soft = (soft_uint8 > 0) & ~hard_edit
    target_missing = target & ~hard_edit
    target_pixels = int(np.count_nonzero(target))
    missing_pixels = int(np.count_nonzero(target_missing))
    conflict_pixels = int(np.count_nonzero(conflict))
    metrics: dict[str, Any] = {
        "source_removal_pixels": int(np.count_nonzero(source)),
        "target_coverage_pixels": target_pixels,
        "requested_pixels": int(np.count_nonzero(requested)),
        "hard_edit_pixels": int(np.count_nonzero(hard_edit)),
        "hard_protection_pixels": int(np.count_nonzero(hard_keep)),
        "conditional_protection_pixels": int(np.count_nonzero(conditional)),
        "protection_conflict_pixels": conflict_pixels,
        "target_coverage_missing_pixels": missing_pixels,
        "target_coverage_missing_ratio": (
            missing_pixels / target_pixels if target_pixels else 0.0
        ),
        "soft_nonzero_pixels": int(np.count_nonzero(soft_uint8)),
        "soft_outside_hard_pixels": int(np.count_nonzero(outside_soft)),
        "soft_outside_hard_ratio": 0.0,
        "soft_min": int(soft_uint8.min()),
        "soft_max": int(soft_uint8.max()),
        "soft_mean": float(soft_uint8[hard_edit].mean()),
        "requested_feather_radius": feather_radius,
        "effective_feather_radius": effective_radius,
    }
    review_reasons = []
    if conflict_pixels:
        review_reasons.append("edit_protection_overlap_removed")
    if missing_pixels:
        review_reasons.append("target_coverage_reduced_by_domain_or_protection")
    status = "REVIEW" if review_reasons else "PASS"

    with generated_base.convert("RGB") as rgb:
        overlay = np.asarray(rgb, dtype=np.uint8).copy()
    for selected, color in (
        (hard_edit, (255, 60, 60)),
        (effective_protection, (50, 100, 255)),
        (conditional & ~release, (255, 220, 0)),
        (conflict, (180, 70, 220)),
    ):
        overlay[selected] = np.rint(
            overlay[selected] * 0.4 + np.asarray(color) * 0.6
        ).astype(np.uint8)

    images = {
        "source_garment_removal": _mask(source),
        "target_garment_coverage": _mask(target),
        "garment_growth_envelope": _mask(growth),
        "requested_edit": _mask(requested),
        "hard_edit_domain": hard_image,
        "hard_protection": _mask(hard_keep),
        "conditional_protection": _mask(conditional),
        "effective_protection": _mask(effective_protection),
        "mask_conflict": _mask(conflict),
        "soft_guidance": Image.fromarray(soft_uint8, mode="L"),
        "overlay": Image.fromarray(overlay, mode="RGB"),
    }
    record = {
        "version": "garment_edit_plan_v1",
        "status": status,
        "coordinate_source": "generated_base",
        "source_reference_pixel_coordinates_reused": False,
        "hard_mask_role": "maximum_authorized_edit_domain",
        "soft_mask_role": "inward_feathered_condition_strength",
        "review_reasons": review_reasons,
        "metrics": metrics,
    }
    return GarmentEditPlan(images=images, metrics=metrics, record=record)




