"""Lossless trace from source observations to the executed prompt.

This is diagnostic evidence, not a claim that inferred tags or colors are true.
Source observations are never replaced by approved or compiled values.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

SNAPSHOT_VERSION = "reference_condition_snapshot_v1"


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return json.loads(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "record"):
        return _json_value(value.record())
    raise TypeError(f"JSON으로 기록할 수 없는 조건 형식: {type(value).__name__}")


def canonical_json(value):
    return json.dumps(
        _json_value(value), sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")


def build_reference_condition_snapshot(inputs, section, prompt_record=None):
    """Keep observations, user constraints, approvals and compiler output apart."""
    from genai_lab.part_color_descriptions import description_records

    descriptions = tuple(getattr(inputs, "part_color_descriptions", ()))
    approved = tuple(section.get("approved_part_color_descriptions", ()))
    analysis = _json_value(getattr(inputs, "analysis_record", None) or {})
    detection_parts = (
        analysis.get("part_detection", {}).get("parts", {})
        if isinstance(analysis, dict) else {})
    color_by_part = {
        item["part_name"]: item for item in description_records(descriptions)}
    payload = {
        "version": SNAPSHOT_VERSION,
        "observations": {
            "face": {
                "source_region": analysis.get("face") if isinstance(analysis, dict) else None,
            },
            "eyes": _json_value(section.get("eye_color_report")),
            "hair": {
                "mask_refinement": analysis.get("hair_mask_refinement")
                if isinstance(analysis, dict) else None,
                "visual_condition": analysis.get("hair_visual_condition")
                if isinstance(analysis, dict) else None,
                "detail_analysis": _json_value(section.get("hair_detail_report")),
            },
            "human_ears": {
                "detection": detection_parts.get("human_ears"),
                "color": color_by_part.get("human_ears"),
                "contract": analysis.get("ear_contract", {}).get("human")
                if isinstance(analysis, dict) else None,
            },
            "animal_ears": {
                "detection": detection_parts.get(
                    "animal_ears", detection_parts.get("ears")),
                "color": color_by_part.get(
                    "animal_ears", color_by_part.get("ears")),
                "contract": analysis.get("ear_contract", {}).get("animal")
                if isinstance(analysis, dict) else None,
            },
            "tail": {
                "detection": detection_parts.get("tail"),
                "color": color_by_part.get("tail"),
            },
            "garment": _json_value(section.get("garment_detail_report")),
            "source_analysis_full": analysis,
        },
        "user_constraints": {
            "gender": section.get("character_gender", "unspecified"),
        },
        "approved_conditions": {
            "character_tags": list(section.get("approved_character_tags", ())),
            "garment_tags": list(section.get("approved_tags", ())),
            "garment_detail_tags": list(section.get("approved_detail_tags", ())),
            "part_colors": list(description_records(approved)),
        },
        "compiled_prompt": _json_value(
            prompt_record if prompt_record is not None
            else section.get("approved_prompt_record")),
        "claims": {
            "source_observations_are_ground_truth": False,
            "generated_image_matches_conditions": False,
        },
        "ear_contract": analysis.get("ear_contract")
        if isinstance(analysis, dict) else None,
    }
    encoded = canonical_json(payload)
    return dict(payload, fingerprint=hashlib.sha256(encoded).hexdigest())


def write_reference_condition_snapshot(path, inputs, section, prompt_record=None):
    snapshot = build_reference_condition_snapshot(inputs, section, prompt_record)
    Path(path).write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return snapshot
