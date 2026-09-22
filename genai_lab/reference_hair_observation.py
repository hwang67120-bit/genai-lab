"""Measurement-only head/hair silhouette observation.

GroundingDINO and SAM2 recover a comparable silhouette in source and generated
coordinates. The result is excluded from generation conditions because it can
include forehead or face pixels and is not a semantic hair mask.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


VERSION = "reference_hair_observation_v1"


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("reference_hair_observation", {})
    if not isinstance(raw, Mapping):
        raw = {}
    values = {
        "enabled": raw.get("enabled", False),
        "detector_model_id": str(
            raw.get("detector_model_id", "IDEA-Research/grounding-dino-tiny")
        ),
        "sam_model_id": str(raw.get("sam_model_id", "facebook/sam2.1-hiera-tiny")),
        "cache_dir": Path(raw.get("cache_dir", "D:/genai-cache/huggingface")),
        "query": str(raw.get("query", "hair. hairstyle.")),
        "box_threshold": float(raw.get("box_threshold", 0.25)),
        "mask_threshold": float(raw.get("mask_threshold", 0.80)),
        "minimum_area_ratio": float(raw.get("minimum_area_ratio", 0.001)),
        "maximum_area_ratio": float(raw.get("maximum_area_ratio", 0.20)),
        "maximum_candidates": int(raw.get("maximum_candidates", 8)),
    }
    if not isinstance(values["enabled"], bool):
        raise ValueError("reference_hair_observation enabled must be bool")
    numbers = (
        values["box_threshold"],
        values["mask_threshold"],
        values["minimum_area_ratio"],
        values["maximum_area_ratio"],
    )
    if (
        not all(math.isfinite(value) for value in numbers)
        or not 0 <= values["box_threshold"] <= 1
        or not 0 <= values["mask_threshold"] <= 1
        or not 0 < values["minimum_area_ratio"] < values["maximum_area_ratio"] <= 1
        or not 1 <= values["maximum_candidates"] <= 32
        or not values["detector_model_id"]
        or not values["sam_model_id"]
        or not values["query"].strip()
    ):
        raise ValueError("invalid reference_hair_observation settings")
    return values


def _as_masks(raw_masks: Any, image_size: tuple[int, int]) -> np.ndarray:
    masks = np.asarray(raw_masks)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    expected = (image_size[1], image_size[0])
    if masks.ndim != 3 or tuple(masks.shape[1:]) != expected:
        raise ValueError("hair observation mask coordinates differ from image")
    return masks.astype(bool, copy=False)


class ReferenceHairObserver:
    """Own one cached-model backend for a generation batch."""

    def __init__(self, config: Mapping[str, Any], backend=None):
        self.settings = _settings(config)
        self.backend = backend
        if self.settings["enabled"] and self.backend is None:
            from genai_lab.extra_parts_analysis import TransformersPartsBackend

            self.backend = TransformersPartsBackend(
                self.settings["detector_model_id"],
                self.settings["sam_model_id"],
                str(self.settings["cache_dir"]),
            )

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
        self.backend = None

    @staticmethod
    def _invalidate_measurement_hair(
        masks: Mapping[str, Image.Image],
        image_size: tuple[int, int],
        report: dict[str, Any],
    ) -> dict[str, Any]:
        current = masks.get("hair")
        if isinstance(current, Image.Image):
            masks["hair"] = Image.new("L", image_size, 0)
            current.close()
            report["masks_modified"] = True
            report["measurement_mask_available"] = False
        return report

    def observe(
        self,
        image: Image.Image,
        masks: Mapping[str, Image.Image],
    ) -> dict[str, Any]:
        base = {
            "version": VERSION,
            "usage": "measurement_only",
            "semantic_mask": False,
            "automatic_generation_condition": False,
            "coordinate_space": "current_image",
            "method": "GroundingDINO query plus SAM2 mask",
            "query": self.settings["query"],
            "masks_modified": False,
        }
        hair = masks.get("hair")
        face = masks.get("face")
        if not isinstance(hair, Image.Image):
            return {**base, "status": "unresolved", "reason": "hair_mask_unavailable"}
        identity = masks.get("identity")
        if hair.size != image.size or any(
            isinstance(mask, Image.Image) and mask.size != image.size
            for mask in (face, identity)
        ):
            raise ValueError("reference hair observation coordinates differ from image")
        if not self.settings["enabled"]:
            return {**base, "status": "disabled", "reason": "observation_disabled"}

        boxes, scores = self.backend.detect(
            image,
            self.settings["query"],
            self.settings["box_threshold"],
        )
        boxes = np.asarray(boxes, dtype=float)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if boxes.size == 0:
            return self._invalidate_measurement_hair(
                masks,
                image.size,
                {
                    **base,
                    "status": "unresolved",
                    "reason": "hair_not_detected",
                    "candidate_count": 0,
                },
            )
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(scores):
            raise ValueError("invalid hair observation detector output")

        limit = min(len(boxes), self.settings["maximum_candidates"])
        order = np.argsort(-scores, kind="stable")[:limit]
        selected_boxes = boxes[order]
        raw_masks, raw_quality = self.backend.segment(image, selected_boxes)
        candidates = _as_masks(raw_masks, image.size)
        quality = np.asarray(raw_quality, dtype=float).reshape(-1)
        if len(candidates) != len(selected_boxes) or len(quality) != len(selected_boxes):
            raise ValueError("invalid hair observation segmenter output")

        canvas_pixels = image.width * image.height
        face_array = (
            np.asarray(face.convert("L")) >= 128
            if isinstance(face, Image.Image)
            else None
        )
        diagnostics = []
        accepted = []
        for rank, (original_index, mask, mask_score) in enumerate(
            zip(order, candidates, quality)
        ):
            pixels = int(mask.sum())
            ratio = pixels / canvas_pixels
            face_overlap = (
                float(np.logical_and(mask, face_array).sum() / pixels)
                if pixels and face_array is not None
                else None
            )
            valid = (
                math.isfinite(float(scores[original_index]))
                and math.isfinite(float(mask_score))
                and float(mask_score) >= self.settings["mask_threshold"]
                and self.settings["minimum_area_ratio"]
                <= ratio
                <= self.settings["maximum_area_ratio"]
            )
            detail = {
                "rank": rank,
                "detector_index": int(original_index),
                "box": [float(value) for value in boxes[original_index]],
                "detection_score": float(scores[original_index]),
                "mask_quality_score": float(mask_score),
                "mask_pixels": pixels,
                "canvas_area_ratio": float(ratio),
                "face_overlap_ratio": face_overlap,
                "valid": bool(valid),
            }
            diagnostics.append(detail)
            if valid:
                accepted.append((detail, mask))

        if not accepted:
            return self._invalidate_measurement_hair(
                masks,
                image.size,
                {
                    **base,
                    "status": "unresolved",
                    "reason": "no_candidate_passed_measurement_bounds",
                    "candidate_count": len(diagnostics),
                    "candidates": diagnostics,
                    "minimum_area_ratio": self.settings["minimum_area_ratio"],
                    "maximum_area_ratio": self.settings["maximum_area_ratio"],
                    "minimum_mask_quality": self.settings["mask_threshold"],
                },
            )

        selected_detail, selected_mask = accepted[0]
        replacement = Image.fromarray(selected_mask.astype(np.uint8) * 255)
        previous = masks["hair"]
        masks["hair"] = replacement
        previous.close()
        identity_union_applied = False
        if isinstance(identity, Image.Image):
            union = np.logical_or(
                np.asarray(identity.convert("L")) >= 128,
                selected_mask,
            )
            masks["identity"] = Image.fromarray(union.astype(np.uint8) * 255)
            identity.close()
            identity_union_applied = True
        return {
            **base,
            "status": "observed",
            "reason": "bounded_measurement_candidate_selected",
            "masks_modified": True,
            "identity_union_applied": identity_union_applied,
            "measurement_mask_available": True,
            "candidate_count": len(diagnostics),
            "selected": selected_detail,
            "candidates": diagnostics,
            "minimum_area_ratio": self.settings["minimum_area_ratio"],
            "maximum_area_ratio": self.settings["maximum_area_ratio"],
            "minimum_mask_quality": self.settings["mask_threshold"],
        }
