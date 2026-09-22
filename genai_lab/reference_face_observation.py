"""Bounded face ROI recovery for reference analysis.

The fallback is used only when the SCHP face mask is too small to measure.
It records a detector bounding-box ROI, not a semantic face segmentation.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


VERSION = "reference_face_observation_v1"


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("reference_face_observation", {})
    if not isinstance(raw, Mapping):
        raw = {}
    detail = config.get("detail_correction", {})
    body = config.get("character_body_comparison", {})
    values = {
        "enabled": bool(raw.get("enabled", False)),
        "detector_repository": str(
            raw.get("detector_repository", detail.get("detector_repository", "Bingsu/adetailer"))
        ),
        "model": str(raw.get("model", detail.get("face_model", "face_yolov8s.pt"))),
        "cache_dir": Path(raw.get("cache_dir", body.get("cache_dir", "D:/genai-cache/huggingface"))),
        "minimum_confidence": float(
            raw.get("minimum_confidence", detail.get("minimum_confidence", 0.30))
        ),
        "minimum_area_ratio": float(raw.get("minimum_area_ratio", 0.0005)),
        "maximum_area_ratio": float(
            raw.get("maximum_area_ratio", detail.get("maximum_mask_area_ratio", 0.12))
        ),
        "parser_minimum_area_ratio": float(
            raw.get("parser_minimum_area_ratio", 0.002)
        ),
    }
    ratios = (
        values["minimum_confidence"],
        values["minimum_area_ratio"],
        values["maximum_area_ratio"],
        values["parser_minimum_area_ratio"],
    )
    if (
        not all(math.isfinite(value) for value in ratios)
        or not 0 <= values["minimum_confidence"] <= 1
        or not 0 < values["minimum_area_ratio"] < values["maximum_area_ratio"] <= 1
        or not 0 <= values["parser_minimum_area_ratio"] <= 1
        or not values["detector_repository"]
        or not values["model"]
    ):
        raise ValueError("invalid reference_face_observation settings")
    return values


@lru_cache(maxsize=2)
def _load_model(repository: str, filename: str, cache_dir: str):
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    path = hf_hub_download(
        repo_id=repository,
        filename=filename,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    return YOLO(path)


def _detect_face_boxes(
    image: Image.Image,
    *,
    repository: str,
    model: str,
    cache_dir: Path,
    minimum_confidence: float,
) -> list[dict[str, Any]]:
    detector = _load_model(repository, model, str(cache_dir))
    prediction = detector.predict(
        source=np.asarray(image.convert("RGB")),
        conf=minimum_confidence,
        device="cpu",
        verbose=False,
    )[0]
    boxes = getattr(prediction, "boxes", None)
    if boxes is None:
        return []
    result = [
        {
            "box": tuple(float(value) for value in coordinates),
            "confidence": float(confidence),
        }
        for coordinates, confidence in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        )
    ]
    result.sort(key=lambda item: item["confidence"], reverse=True)
    return result


def recover_reference_face_region(
    image: Image.Image,
    masks: Mapping[str, Image.Image],
    config: Mapping[str, Any],
    *,
    exclude_from_garment: bool = False,
) -> dict[str, Any]:
    """Recover an observable face ROI and union it into identity.

    The supplied mapping must be mutable in practice. No mask is changed when
    the parser face is already measurable or the detector evidence is invalid.
    """
    settings = _settings(config)
    face = masks.get("face")
    identity = masks.get("identity")
    if not isinstance(face, Image.Image) or not isinstance(identity, Image.Image):
        return {
            "version": VERSION,
            "status": "unresolved",
            "reason": "face_or_identity_mask_unavailable",
            "masks_modified": False,
            "automatic_generation_condition": False,
        }
    if face.size != image.size or identity.size != image.size:
        raise ValueError("reference face observation coordinates differ from image")

    canvas_pixels = image.width * image.height
    parser_pixels = int(np.count_nonzero(np.asarray(face.convert("L")) >= 128))
    parser_ratio = parser_pixels / canvas_pixels
    parser_bbox = face.convert("L").getbbox()
    parser_bbox_pixels = (
        (parser_bbox[2] - parser_bbox[0]) * (parser_bbox[3] - parser_bbox[1])
        if parser_bbox is not None else 0
    )
    parser_bbox_ratio = parser_bbox_pixels / canvas_pixels
    base_record = {
        "version": VERSION,
        "parser_face_pixels": parser_pixels,
        "parser_face_area_ratio": parser_ratio,
        "parser_face_bbox": list(parser_bbox) if parser_bbox is not None else None,
        "parser_face_bbox_area_ratio": parser_bbox_ratio,
        "parser_minimum_area_ratio": settings["parser_minimum_area_ratio"],
        "parser_maximum_area_ratio": settings["maximum_area_ratio"],
        "semantic_mask": False,
        "coordinate_space": "current_image",
    }
    parser_measurable = (
        settings["parser_minimum_area_ratio"]
        <= parser_ratio
        <= settings["maximum_area_ratio"]
        and parser_bbox_ratio <= settings["maximum_area_ratio"]
    )
    if parser_measurable:
        return {
            **base_record,
            "status": "parser_mask_accepted",
            "reason": "parser_face_measurable",
            "masks_modified": False,
            "automatic_generation_condition": False,
        }
    if not settings["enabled"]:
        return {
            **base_record,
            "status": "unresolved",
            "reason": "fallback_disabled",
            "masks_modified": False,
            "automatic_generation_condition": False,
        }

    boxes = _detect_face_boxes(
        image,
        repository=settings["detector_repository"],
        model=settings["model"],
        cache_dir=settings["cache_dir"],
        minimum_confidence=settings["minimum_confidence"],
    )
    if not boxes:
        return {
            **base_record,
            "status": "unresolved",
            "reason": "face_not_detected",
            "masks_modified": False,
            "automatic_generation_condition": False,
        }

    selected = boxes[0]
    left, top, right, bottom = selected["box"]
    left = max(0, min(image.width - 1, int(math.floor(left))))
    top = max(0, min(image.height - 1, int(math.floor(top))))
    right = max(left + 1, min(image.width, int(math.ceil(right))))
    bottom = max(top + 1, min(image.height, int(math.ceil(bottom))))
    box_pixels = (right - left) * (bottom - top)
    area_ratio = box_pixels / canvas_pixels
    evidence = {
        **base_record,
        "detector": "Bingsu/adetailer face_yolov8s bounding box",
        "confidence": selected["confidence"],
        "box": [left, top, right, bottom],
        "box_pixels": box_pixels,
        "box_area_ratio": area_ratio,
        "minimum_area_ratio": settings["minimum_area_ratio"],
        "maximum_area_ratio": settings["maximum_area_ratio"],
        "candidate_count": len(boxes),
    }
    if not settings["minimum_area_ratio"] <= area_ratio <= settings["maximum_area_ratio"]:
        return {
            **evidence,
            "status": "unresolved",
            "reason": "face_box_area_out_of_bounds",
            "masks_modified": False,
            "automatic_generation_condition": False,
        }

    recovered = Image.new("L", image.size, 0)
    ImageDraw.Draw(recovered).rectangle(
        (left, top, right - 1, bottom - 1),
        fill=255,
    )
    current_face = masks["face"]
    current_identity = masks["identity"]
    recovered_array = np.asarray(recovered) >= 128
    union = np.logical_or(
        np.asarray(current_identity.convert("L")) >= 128,
        recovered_array,
    )
    garment_overlap_removed_pixels = 0
    garment = masks.get("garment")
    if exclude_from_garment and isinstance(garment, Image.Image):
        garment_array = np.asarray(garment.convert("L")) >= 128
        garment_overlap_removed_pixels = int(
            np.logical_and(garment_array, recovered_array).sum()
        )
        masks["garment"] = Image.fromarray(
            np.logical_and(garment_array, ~recovered_array).astype(np.uint8)
            * 255
        )
        garment.close()
    masks["face"] = recovered
    masks["identity"] = Image.fromarray(union.astype(np.uint8) * 255)
    current_face.close()
    current_identity.close()
    return {
        **evidence,
        "status": "recovered",
        "reason": (
            "parser_face_below_measurement_minimum"
            if parser_ratio < settings["parser_minimum_area_ratio"]
            else "parser_face_outside_measurement_bounds"
        ),
        "masks_modified": True,
        "identity_union_applied": True,
        "garment_exclusion_applied": bool(exclude_from_garment),
        "garment_overlap_removed_pixels": garment_overlap_removed_pixels,
        "automatic_generation_condition": True,
        "usage": "identity_reference_and_measurement_roi",
    }
