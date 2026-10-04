"""Offline validation for animal-ear, hair-accessory, and tail analysis."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

from .part_spatial_diagnostics import (
    PartSpatialDiagnosticPolicy,
    diagnose_part_spatial_consistency,
)

VALIDATION_VERSION = "reference_part_validation_v3"
PARTS = ("animal_ears", "hair_accessory", "tail")
CROSS_REVIEW_PARTS = ("animal_ears", "hair_accessory")
LABEL_STATUSES = {"present", "absent", "ambiguous"}
PREDICTION_STATUSES = {"detected", "no_detection", "uncertain"}
FORBIDDEN_DATASET_DIRECTORIES = {
    "outputs",
    "training_candidates",
    "approved",
    "refinement-pending",
}


class ReferencePartValidationError(ValueError):
    """Validation data or analyzer output violates the offline contract."""


@dataclass(frozen=True)
class PartLabel:
    status: str
    mask_path: Path | None


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    image_path: Path
    labels: dict[str, PartLabel]
    context_masks: dict[str, Path]
    protection_masks: dict[str, Path]
    confounders: tuple[str, ...]


def _safe_dataset_path(
    dataset_root: Path,
    value: Any,
    *,
    required: bool,
) -> Path | None:
    if value in (None, ""):
        if required:
            raise ReferencePartValidationError("필수 검증 파일 경로가 없습니다.")
        return None
    if not isinstance(value, str):
        raise ReferencePartValidationError("검증 파일 경로는 문자열이어야 합니다.")
    path = (dataset_root / value).resolve()
    root = dataset_root.resolve()
    if not path.is_relative_to(root):
        raise ReferencePartValidationError(
            "검증 자료 경로는 manifest 폴더 밖을 가리킬 수 없습니다."
        )
    if not path.is_file():
        raise ReferencePartValidationError(f"검증 파일이 없습니다: {path}")
    return path


def _read_manifest_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(text)
        records = value.get("cases") if isinstance(value, Mapping) else value
    if not isinstance(records, list) or not records:
        raise ReferencePartValidationError("검증 manifest에 사례가 없습니다.")
    if not all(isinstance(record, dict) for record in records):
        raise ReferencePartValidationError("검증 사례는 JSON 객체여야 합니다.")
    return records


def load_validation_manifest(path: Path | str) -> tuple[ValidationCase, ...]:
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise ReferencePartValidationError(
            f"검증 manifest가 없습니다: {manifest}"
        )
    dataset_root = manifest.parent
    if any(
        component.lower() in FORBIDDEN_DATASET_DIRECTORIES
        for component in dataset_root.parts
    ):
        raise ReferencePartValidationError(
            "검증 자료는 outputs·학습 후보·승인 결과 폴더와 분리해야 합니다."
        )
    cases: list[ValidationCase] = []
    identifiers: set[str] = set()
    for record in _read_manifest_records(manifest):
        case_id = record.get("id")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id in identifiers
        ):
            raise ReferencePartValidationError(
                "검증 사례 id가 비었거나 중복되었습니다."
            )
        identifiers.add(case_id)
        image_path = _safe_dataset_path(
            dataset_root,
            record.get("image"),
            required=True,
        )
        raw_labels = record.get("parts")
        if not isinstance(raw_labels, Mapping) or set(raw_labels) != set(PARTS):
            raise ReferencePartValidationError(
                "각 사례는 animal_ears, hair_accessory, tail 정답을 모두 가져야 합니다."
            )
        labels: dict[str, PartLabel] = {}
        for part in PARTS:
            raw = raw_labels[part]
            if not isinstance(raw, Mapping):
                raise ReferencePartValidationError(
                    f"{case_id}:{part} 정답 형식이 올바르지 않습니다."
                )
            status = raw.get("status")
            if status not in LABEL_STATUSES:
                raise ReferencePartValidationError(
                    f"{case_id}:{part} 정답 상태가 올바르지 않습니다."
                )
            mask = _safe_dataset_path(
                dataset_root,
                raw.get("mask"),
                required=status == "present",
            )
            if status == "absent" and mask is not None:
                raise ReferencePartValidationError(
                    f"{case_id}:{part} absent 정답에는 마스크를 지정할 수 없습니다."
                )
            labels[part] = PartLabel(status, mask)

        def mask_map(key: str) -> dict[str, Path]:
            raw = record.get(key, {})
            if raw is None:
                return {}
            if not isinstance(raw, Mapping):
                raise ReferencePartValidationError(
                    f"{case_id}:{key} 형식이 올바르지 않습니다."
                )
            return {
                str(name): _safe_dataset_path(
                    dataset_root,
                    value,
                    required=True,
                )
                for name, value in raw.items()
            }

        context = mask_map("context_masks")
        if set(context) not in (set(), {"garment", "hair"}):
            raise ReferencePartValidationError(
                f"{case_id}: context_masks는 garment와 hair를 함께 제공해야 합니다."
            )
        protection = mask_map("protection_masks")
        confounders = record.get("confounders", ())
        if not isinstance(confounders, (list, tuple)) or not all(
            isinstance(item, str) for item in confounders
        ):
            raise ReferencePartValidationError(
                f"{case_id}: confounders는 문자열 배열이어야 합니다."
            )
        cases.append(
            ValidationCase(
                case_id=case_id,
                image_path=image_path,
                labels=labels,
                context_masks=context,
                protection_masks=protection,
                confounders=tuple(confounders),
            )
        )
    return tuple(cases)


def _binary_mask(path: Path, size: tuple[int, int], name: str) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.size != size:
            raise ReferencePartValidationError(
                f"{name} 마스크 좌표가 원본 이미지와 다릅니다."
            )
        values = np.asarray(opened.convert("L"), dtype=np.uint8)
    binary = values >= 128
    if not binary.any():
        raise ReferencePartValidationError(f"{name} 마스크가 비었습니다.")
    return binary


def _open_context_masks(
    paths: Mapping[str, Path],
    size: tuple[int, int],
) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    try:
        for name, path in paths.items():
            binary = _binary_mask(path, size, name)
            images[name] = Image.fromarray(binary.astype(np.uint8) * 255)
        return images
    except BaseException:
        for image in images.values():
            image.close()
        raise


def normalize_prediction(
    part: str,
    analyzer_report: Mapping[str, Any],
) -> dict[str, Any]:
    entry = analyzer_report.get("parts", {}).get(part, {})
    if not isinstance(entry, Mapping):
        return {
            "status": "uncertain",
            "reason": "part_report_missing",
            "raw_status": None,
            "detection_scores": [],
            "mask_quality_scores": [],
            "accepted_masks": 0,
            "accepted_boxes": [],
            "candidate_mask_paths": [],
            "mask_path": None,
            "analyzer_would_condition": False,
            "automatic_conditioning": False,
            "validation_only": True,
        }
    accepted = int(entry.get("accepted_masks", 0) or 0)
    raw_status = entry.get("status")
    mask_path = entry.get("mask_path")
    if raw_status == "detected" and accepted > 0 and mask_path:
        status = "detected"
        reason = str(entry.get("outcome_reason", "accepted_mask_available"))
    elif not entry.get("boxes"):
        status = "no_detection"
        reason = str(
            entry.get("outcome_reason", "no_candidate_above_threshold")
        )
    else:
        status = "uncertain"
        reason = str(
            entry.get(
                "outcome_reason",
                "all_candidates_rejected_or_low_mask_quality",
            )
        )
    return {
        "status": status,
        "reason": reason,
        "raw_status": raw_status,
        "detection_scores": list(entry.get("detection_scores", ())),
        "mask_quality_scores": list(entry.get("mask_quality_scores", ())),
        "accepted_masks": accepted,
        "accepted_boxes": list(entry.get("accepted_boxes", ())),
        "candidate_mask_paths": [
            str(path) for path in entry.get("candidate_mask_paths", ())
        ],
        "mask_path": str(mask_path) if mask_path else None,
        "analyzer_would_condition": bool(
            entry.get("automatic_conditioning", status == "detected")
        ),
        "automatic_conditioning": False,
        "validation_only": True,
    }


def _mask_metrics(
    predicted: np.ndarray,
    expected: np.ndarray,
) -> tuple[float, float]:
    intersection = int(np.count_nonzero(predicted & expected))
    union = int(np.count_nonzero(predicted | expected))
    total = int(np.count_nonzero(predicted)) + int(np.count_nonzero(expected))
    return (
        float(intersection / union) if union else 1.0,
        float((2 * intersection) / total) if total else 1.0,
    )


def _save_overlay(
    source: Image.Image,
    expected: np.ndarray | None,
    predicted: np.ndarray | None,
    path: Path,
) -> None:
    base = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    overlay = base.astype(np.float32)
    if expected is not None:
        overlay[expected] = overlay[expected] * 0.55 + np.array(
            [0, 255, 0], dtype=np.float32
        ) * 0.45
    if predicted is not None:
        overlay[predicted] = overlay[predicted] * 0.55 + np.array(
            [255, 0, 0], dtype=np.float32
        ) * 0.45
    if expected is not None and predicted is not None:
        overlap = expected & predicted
        overlay[overlap] = overlay[overlap] * 0.35 + np.array(
            [255, 220, 0], dtype=np.float32
        ) * 0.65
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def _load_candidate_union(
    paths: list[str],
    size: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    if not paths:
        return None
    union = np.zeros((size[1], size[0]), dtype=bool)
    for index, raw_path in enumerate(paths):
        union |= _binary_mask(
            Path(raw_path),
            size,
            f"{name}:candidate:{index}",
        )
    return union if union.any() else None


def _save_binary_diagnostic_mask(
    mask: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _cross_review_ear_accessory(
    *,
    case_id: str,
    size: tuple[int, int],
    predictions: Mapping[str, Mapping[str, Any]],
    predicted_masks: Mapping[str, np.ndarray],
    output_root: Path,
) -> dict[str, Any]:
    ear = predicted_masks.get("animal_ears")
    accessory = predicted_masks.get("hair_accessory")
    empty = np.zeros((size[1], size[0]), dtype=bool)
    ear_values = ear if ear is not None else empty
    accessory_values = accessory if accessory is not None else empty
    shared = ear_values & accessory_values
    ear_pixels = int(np.count_nonzero(ear_values))
    accessory_pixels = int(np.count_nonzero(accessory_values))
    shared_pixels = int(np.count_nonzero(shared))
    smaller = min(ear_pixels, accessory_pixels)
    overlap_ratio = float(shared_pixels / smaller) if smaller else None
    statuses = {
        name: str(predictions.get(name, {}).get("status", "uncertain"))
        for name in CROSS_REVIEW_PARTS
    }
    both_masks_available = ear_pixels > 0 and accessory_pixels > 0
    if both_masks_available and shared_pixels:
        status = "overlap_ambiguous"
    elif both_masks_available:
        status = "separate_candidates"
    elif "uncertain" in statuses.values():
        status = "unresolved"
    elif "detected" in statuses.values():
        status = "single_candidate"
    else:
        status = "not_observed"
    overlap_path = (
        output_root
        / "cross-review"
        / f"{case_id}-animal_ears-hair_accessory-overlap.png"
    )
    _save_binary_diagnostic_mask(shared, overlap_path)
    return {
        "version": "animal_ear_hair_accessory_cross_review_v1",
        "status": status,
        "parts": statuses,
        "diagnostic_mask_sources": {
            name: predictions.get(name, {}).get(
                "diagnostic_candidate_source",
                "none",
            )
            for name in CROSS_REVIEW_PARTS
        },
        "animal_ears_pixels": ear_pixels,
        "hair_accessory_pixels": accessory_pixels,
        "overlap_pixels": shared_pixels,
        "smaller_part_overlap_ratio": overlap_ratio,
        "overlap_mask_path": str(overlap_path),
        "winner_selected": False,
        "masks_modified": False,
        "automatic_conditioning_applied": False,
        "generation_policy_changed": False,
        "review_required": status in {
            "overlap_ambiguous",
            "unresolved",
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _initial_metrics() -> dict[str, Any]:
    return {
        "present": 0,
        "absent": 0,
        "ambiguous": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "uncertain_present": 0,
        "uncertain_absent": 0,
        "detected": 0,
        "no_detection": 0,
        "uncertain": 0,
        "automated_non_ambiguous": 0,
        "uncertain_non_ambiguous": 0,
        "iou_values": [],
        "dice_values": [],
        "protection_overlap_ratios": [],
    }


def _finalize_metrics(values: dict[str, Any]) -> dict[str, Any]:
    non_ambiguous = values["present"] + values["absent"]
    result = {
        key: value
        for key, value in values.items()
        if not key.endswith("_values")
    }
    result.update({
        "detected_precision": _ratio(
            values["true_positive"],
            values["true_positive"] + values["false_positive"],
        ),
        "strict_present_recall": _ratio(
            values["true_positive"],
            values["present"],
        ),
        "false_positive_rate": _ratio(
            values["false_positive"],
            values["absent"],
        ),
        "uncertain_rate": _ratio(
            values["uncertain_present"] + values["uncertain_absent"],
            non_ambiguous,
        ),
        "diagnostic_coverage": _ratio(
            values["automated_non_ambiguous"],
            non_ambiguous,
        ),
        "automation_coverage": _ratio(
            values["automated_non_ambiguous"],
            non_ambiguous,
        ),
        "automatic_conditioning_coverage": 0.0,
        "mean_mask_iou": (
            float(np.mean(values["iou_values"]))
            if values["iou_values"] else None
        ),
        "mean_mask_dice": (
            float(np.mean(values["dice_values"]))
            if values["dice_values"] else None
        ),
        "maximum_protection_overlap_ratio": (
            float(max(values["protection_overlap_ratios"]))
            if values["protection_overlap_ratios"] else None
        ),
    })
    return result


def run_reference_part_validation(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    analyzer_factory: Callable[..., Any],
    timeout_seconds: float = 120.0,
    configuration: Mapping[str, Any] | None = None,
    diagnostic_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not 1 <= float(timeout_seconds) <= 3600:
        raise ReferencePartValidationError(
            "사례별 분석 제한 시간은 1~3600초여야 합니다."
        )
    cases = load_validation_manifest(manifest_path)
    manifest = Path(manifest_path).resolve()
    output = Path(output_directory).resolve()
    if output.is_relative_to(manifest.parent):
        raise ReferencePartValidationError(
            "검증 보고서 출력은 고정 검증 자료 폴더와 분리해야 합니다."
        )
    output.mkdir(parents=True, exist_ok=True)
    active_diagnostic_policy = PartSpatialDiagnosticPolicy.from_mapping(
        diagnostic_policy
    )
    case_root = output / "cases"
    overlay_root = output / "overlays"
    rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    accessory_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    metrics = {part: _initial_metrics() for part in PARTS}
    spatial_metrics = {
        "head_checks": {
            "inside_head_roi": 0,
            "outside_head_roi": 0,
            "not_observed": 0,
        },
        "tail_connection_checks": {
            "body_anchor_contact": 0,
            "outside_body_anchor": 0,
            "not_observed": 0,
        },
        "cross_class_pairs_checked": 0,
        "same_mask_cross_class_conflicts": 0,
        "review_required_cases": 0,
    }
    accessory_metrics = {
        "not_run": 0,
        "not_detected": 0,
        "present_unlocalized": 0,
        "localized": 0,
        "class_conflict": 0,
        "failed": 0,
        "candidate_count": 0,
        "cluster_count": 0,
        "review_required_cases": 0,
        "sam2_invocations": 0,
    }
    cross_metrics = {
        "overlap_ambiguous": 0,
        "separate_candidates": 0,
        "single_candidate": 0,
        "unresolved": 0,
        "not_observed": 0,
        "overlap_ratios": [],
    }
    case_errors: list[dict[str, str]] = []

    for case in cases:
        debug_directory = case_root / case.case_id
        debug_directory.mkdir(parents=True, exist_ok=True)
        with Image.open(case.image_path) as opened:
            source = opened.convert("RGB")
        context_images: dict[str, Image.Image] = {}
        parts = []
        try:
            context_images = _open_context_masks(
                case.context_masks,
                source.size,
            )
            analyzer = analyzer_factory(
                debug_dir=debug_directory,
                source_masks=context_images or None,
            )
            try:
                parts = analyzer.analyze(
                    source,
                    source.size,
                    cancelled=lambda: False,
                    deadline=perf_counter() + float(timeout_seconds),
                )
                report = json.loads(json.dumps(analyzer.report))
            finally:
                for detected in parts:
                    detected.close()

            protection = {
                name: _binary_mask(path, source.size, name)
                for name, path in case.protection_masks.items()
            }
            case_prediction = {
                "id": case.case_id,
                "image": str(case.image_path),
                "confounders": list(case.confounders),
                "parts": {},
                "analyzer_report_path": str(debug_directory / "parts.json"),
            }
            observation = report.get("accessory_observations")
            if not isinstance(observation, Mapping):
                observation = {
                    "version": "accessory_observation_v1",
                    "mode": "observe_only",
                    "status": "not_run",
                    "presence": "not_observed",
                    "candidate_count": 0,
                    "clusters": [],
                    "review_required": False,
                    "not_detected_is_absence": False,
                    "predictions_modified": False,
                    "masks_created": False,
                    "masks_modified": False,
                    "sam2_invoked": False,
                    "automatic_conditioning_applied": False,
                    "generation_policy_changed": False,
                }
            observation = json.loads(json.dumps(observation))
            observation_status = str(observation.get("status", "failed"))
            if observation_status not in accessory_metrics:
                observation_status = "failed"
            accessory_metrics[observation_status] += 1
            candidate_count = int(observation.get("candidate_count", 0))
            cluster_count = len(observation.get("clusters", ()))
            accessory_metrics["candidate_count"] += candidate_count
            accessory_metrics["cluster_count"] += cluster_count
            if observation.get("review_required", False):
                accessory_metrics["review_required_cases"] += 1
            if observation.get("sam2_invoked", False):
                accessory_metrics["sam2_invocations"] += 1
            case_prediction["accessory_observations"] = observation
            accessory_rows.append({
                "case_id": case.case_id,
                "status": observation_status,
                "presence": observation.get("presence", "not_observed"),
                "candidate_count": candidate_count,
                "cluster_count": cluster_count,
                "review_required": bool(
                    observation.get("review_required", False)
                ),
                "sam2_invoked": bool(
                    observation.get("sam2_invoked", False)
                ),
                "automatic_conditioning_applied": bool(
                    observation.get(
                        "automatic_conditioning_applied",
                        False,
                    )
                ),
            })
            predicted_masks: dict[str, np.ndarray] = {}
            diagnostic_masks: dict[str, np.ndarray] = {}
            for part in PARTS:
                label = case.labels[part]
                prediction = normalize_prediction(part, report)
                predicted_mask = None
                if prediction["mask_path"]:
                    predicted_mask = _binary_mask(
                        Path(prediction["mask_path"]),
                        source.size,
                        f"{case.case_id}:{part}:prediction",
                    )
                    predicted_masks[part] = predicted_mask
                diagnostic_mask = predicted_mask
                diagnostic_source = (
                    "accepted_mask" if predicted_mask is not None else "none"
                )
                if diagnostic_mask is None:
                    diagnostic_mask = _load_candidate_union(
                        prediction["candidate_mask_paths"],
                        source.size,
                        f"{case.case_id}:{part}",
                    )
                    if diagnostic_mask is not None:
                        diagnostic_source = "raw_candidate_union"
                if diagnostic_mask is not None:
                    diagnostic_masks[part] = diagnostic_mask
                    diagnostic_path = (
                        output
                        / "diagnostic-candidates"
                        / f"{case.case_id}-{part}.png"
                    )
                    _save_binary_diagnostic_mask(
                        diagnostic_mask,
                        diagnostic_path,
                    )
                    prediction["diagnostic_candidate_mask_path"] = str(
                        diagnostic_path
                    )
                else:
                    prediction["diagnostic_candidate_mask_path"] = None
                prediction["diagnostic_candidate_source"] = (
                    diagnostic_source
                )
                expected_mask = (
                    _binary_mask(
                        label.mask_path,
                        source.size,
                        f"{case.case_id}:{part}:ground_truth",
                    )
                    if label.mask_path is not None
                    else None
                )
                part_metrics = metrics[part]
                part_metrics[label.status] += 1
                part_metrics[prediction["status"]] += 1
                if label.status != "ambiguous":
                    if prediction["status"] == "uncertain":
                        part_metrics["uncertain_non_ambiguous"] += 1
                    else:
                        part_metrics["automated_non_ambiguous"] += 1
                if label.status == "present":
                    if prediction["status"] == "detected":
                        part_metrics["true_positive"] += 1
                    elif prediction["status"] == "no_detection":
                        part_metrics["false_negative"] += 1
                    else:
                        part_metrics["uncertain_present"] += 1
                elif label.status == "absent":
                    if prediction["status"] == "detected":
                        part_metrics["false_positive"] += 1
                    elif prediction["status"] == "no_detection":
                        part_metrics["true_negative"] += 1
                    else:
                        part_metrics["uncertain_absent"] += 1

                iou = dice = None
                if (
                    label.status == "present"
                    and prediction["status"] == "detected"
                    and expected_mask is not None
                    and predicted_mask is not None
                ):
                    iou, dice = _mask_metrics(
                        predicted_mask,
                        expected_mask,
                    )
                    part_metrics["iou_values"].append(iou)
                    part_metrics["dice_values"].append(dice)

                overlaps: dict[str, float] = {}
                if predicted_mask is not None:
                    predicted_pixels = int(np.count_nonzero(predicted_mask))
                    for name, protected in protection.items():
                        shared = int(np.count_nonzero(
                            predicted_mask & protected
                        ))
                        ratio = (
                            float(shared / predicted_pixels)
                            if predicted_pixels else 0.0
                        )
                        overlaps[name] = ratio
                        part_metrics["protection_overlap_ratios"].append(
                            ratio
                        )

                prediction.update({
                    "ground_truth_status": label.status,
                    "mask_iou": iou,
                    "mask_dice": dice,
                    "protection_overlap": overlaps,
                })
                case_prediction["parts"][part] = prediction
                rows.append({
                    "case_id": case.case_id,
                    "part": part,
                    "ground_truth": label.status,
                    "prediction": prediction["status"],
                    "reason": prediction["reason"],
                    "accepted_masks": prediction["accepted_masks"],
                    "mask_iou": iou,
                    "mask_dice": dice,
                    "confounders": "|".join(case.confounders),
                })
                _save_overlay(
                    source,
                    expected_mask,
                    predicted_mask,
                    overlay_root / f"{case.case_id}-{part}.png",
                )
            context_arrays = {
                name: np.asarray(image.convert("L"), dtype=np.uint8) >= 128
                for name, image in context_images.items()
            }
            spatial_report, roi_masks = diagnose_part_spatial_consistency(
                size=source.size,
                part_masks=diagnostic_masks,
                context_masks=context_arrays,
                policy=active_diagnostic_policy,
            )
            for roi_name, roi_mask in roi_masks.items():
                roi_path = (
                    output
                    / "spatial-diagnostics"
                    / f"{case.case_id}-{roi_name}.png"
                )
                _save_binary_diagnostic_mask(roi_mask, roi_path)
                spatial_report[f"{roi_name}_mask_path"] = str(roi_path)
            case_prediction["spatial_diagnostics"] = spatial_report

            for part_name, check in spatial_report["head_checks"].items():
                spatial_metrics["head_checks"][check["status"]] += 1
                spatial_rows.append({
                    "case_id": case.case_id,
                    "part": part_name,
                    "check": "head_roi",
                    "status": check["status"],
                    "roi_source": spatial_report["head_roi"]["source"],
                    "candidate_overlap_ratio": check[
                        "candidate_overlap_ratio"
                    ],
                    "review_required": check["review_required"],
                })
            tail_check = spatial_report["tail_connection_check"]
            spatial_metrics["tail_connection_checks"][
                tail_check["status"]
            ] += 1
            spatial_rows.append({
                "case_id": case.case_id,
                "part": "tail",
                "check": "body_connection",
                "status": tail_check["status"],
                "roi_source": spatial_report[
                    "tail_body_anchor_roi"
                ]["source"],
                "candidate_overlap_ratio": tail_check[
                    "candidate_overlap_ratio"
                ],
                "review_required": tail_check["review_required"],
            })
            pairs = spatial_report["cross_class_pairs"]
            conflicts = spatial_report[
                "same_mask_cross_class_conflicts"
            ]
            spatial_metrics["cross_class_pairs_checked"] += len(pairs)
            spatial_metrics[
                "same_mask_cross_class_conflicts"
            ] += len(conflicts)
            if spatial_report["review_required"]:
                spatial_metrics["review_required_cases"] += 1
            for pair in pairs:
                spatial_rows.append({
                    "case_id": case.case_id,
                    "part": ":".join(pair["parts"]),
                    "check": "cross_class_mask",
                    "status": pair["status"],
                    "roi_source": "",
                    "candidate_overlap_ratio": pair[
                        "smaller_part_overlap_ratio"
                    ],
                    "review_required": pair["review_required"],
                })

            cross_review = _cross_review_ear_accessory(
                case_id=case.case_id,
                size=source.size,
                predictions=case_prediction["parts"],
                predicted_masks=diagnostic_masks,
                output_root=output,
            )
            ear_accessory_pair = next(
                (
                    item
                    for item in spatial_report["cross_class_pairs"]
                    if set(item["parts"])
                    == {"animal_ears", "hair_accessory"}
                ),
                None,
            )
            cross_review["same_mask_conflict"] = bool(
                ear_accessory_pair
                and ear_accessory_pair["same_mask_conflict"]
            )
            cross_review["mask_iou"] = (
                ear_accessory_pair["mask_iou"]
                if ear_accessory_pair else None
            )
            cross_review["conflict_status"] = (
                ear_accessory_pair["status"]
                if ear_accessory_pair else "not_comparable"
            )
            cross_review["review_required"] = bool(
                cross_review["review_required"]
                or cross_review["same_mask_conflict"]
            )
            case_prediction["animal_ear_hair_accessory_cross_review"] = (
                cross_review
            )
            cross_metrics[cross_review["status"]] += 1
            ratio = cross_review["smaller_part_overlap_ratio"]
            if ratio is not None:
                cross_metrics["overlap_ratios"].append(ratio)
            cross_rows.append({
                "case_id": case.case_id,
                "status": cross_review["status"],
                "animal_ears_status": cross_review["parts"]["animal_ears"],
                "hair_accessory_status": cross_review["parts"][
                    "hair_accessory"
                ],
                "overlap_pixels": cross_review["overlap_pixels"],
                "smaller_part_overlap_ratio": ratio,
                "mask_iou": cross_review["mask_iou"],
                "same_mask_conflict": cross_review[
                    "same_mask_conflict"
                ],
                "conflict_status": cross_review["conflict_status"],
                "winner_selected": False,
                "masks_modified": False,
                "automatic_conditioning_applied": False,
            })
            predictions.append(case_prediction)
        except BaseException as error:
            case_errors.append({
                "id": case.case_id,
                "error_type": type(error).__name__,
                "message": str(error),
            })
        finally:
            for image in context_images.values():
                image.close()
            source.close()

    prediction_path = output / "predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in predictions
        ),
        encoding="utf-8",
    )
    with (output / "cases.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "part",
                "ground_truth",
                "prediction",
                "reason",
                "accepted_masks",
                "mask_iou",
                "mask_dice",
                "confounders",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    with (output / "spatial-diagnostics.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "part",
                "check",
                "status",
                "roi_source",
                "candidate_overlap_ratio",
                "review_required",
            ),
        )
        writer.writeheader()
        writer.writerows(spatial_rows)

    with (output / "cross-review.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "status",
                "animal_ears_status",
                "hair_accessory_status",
                "overlap_pixels",
                "smaller_part_overlap_ratio",
                "mask_iou",
                "same_mask_conflict",
                "conflict_status",
                "winner_selected",
                "masks_modified",
                "automatic_conditioning_applied",
            ),
        )
        writer.writeheader()
        writer.writerows(cross_rows)

    with (output / "accessory-observations.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "status",
                "presence",
                "candidate_count",
                "cluster_count",
                "review_required",
                "sam2_invoked",
                "automatic_conditioning_applied",
            ),
        )
        writer.writeheader()
        writer.writerows(accessory_rows)

    finalized_cross_metrics = {
        key: value
        for key, value in cross_metrics.items()
        if key != "overlap_ratios"
    }
    finalized_cross_metrics["mean_smaller_part_overlap_ratio"] = (
        float(np.mean(cross_metrics["overlap_ratios"]))
        if cross_metrics["overlap_ratios"] else None
    )

    result = {
        "version": VALIDATION_VERSION,
        "status": (
            "completed" if not case_errors else "completed_with_errors"
        ),
        "created_at": datetime.now().astimezone().isoformat(),
        "manifest": str(Path(manifest_path).resolve()),
        "case_count": len(cases),
        "completed_case_count": len(predictions),
        "failed_case_count": len(case_errors),
        "case_errors": case_errors,
        "parts": {
            part: _finalize_metrics(values)
            for part, values in metrics.items()
        },
        "animal_ear_hair_accessory_cross_review": (
            finalized_cross_metrics
        ),
        "spatial_diagnostics": {
            "version": "part_spatial_diagnostics_v1",
            "mode": "observe_only",
            "policy": active_diagnostic_policy.snapshot(),
            **spatial_metrics,
            "predictions_modified": False,
            "masks_modified": False,
            "automatic_conditioning_applied": False,
            "generation_policy_changed": False,
        },
        "accessory_observations": {
            "version": "accessory_observation_v1",
            "mode": "observe_only",
            **accessory_metrics,
            "not_detected_is_absence": False,
            "predictions_modified": False,
            "masks_created": False,
            "masks_modified": False,
            "automatic_conditioning_applied": False,
            "generation_policy_changed": False,
        },
        "generation_invoked": False,
        "automatic_threshold_approval": False,
        "semantic_accuracy_verified": False,
        "notes": [
            "uncertain is reported and never removed from coverage metrics",
            "no_detection is not automatically promoted to not_present",
            "detected masks remain diagnostic and never enable conditioning",
            "ear-accessory overlap never selects a winner or edits either mask",
            "spatial diagnostics record head ROI, body connection, and cross-class conflicts only",
            "multi-scale accessory observations never create or select a mask",
            "this report does not authorize generation conditioning",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "config-snapshot.json").write_text(
        json.dumps(
            dict(configuration or {}),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return result

