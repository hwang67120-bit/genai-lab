import json
from pathlib import Path

from PIL import Image
import pytest

from genai_lab.reference_analysis_validation import (
    ReferencePartValidationError,
    load_validation_manifest,
    normalize_prediction,
    run_reference_part_validation,
)


def _save_image(path: Path, mode: str = "RGB", box=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, (16, 16), "white" if mode == "RGB" else 0)
    if box is not None:
        image.paste(255, box)
    image.save(path)
    image.close()


def _write_manifest(root: Path, records: list[dict]) -> Path:
    path = root / "labels.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    return path


class FakeAnalyzer:
    def __init__(self, debug_dir: Path, plans: dict):
        self.debug_dir = Path(debug_dir)
        self.plans = plans
        self.report = {}

    def analyze(self, source, output_size, *, cancelled, deadline):
        assert output_size == source.size
        assert not cancelled()
        plan = self.plans[self.debug_dir.name]
        parts = {}
        for name in ("animal_ears", "hair_accessory", "tail"):
            status = plan[name]
            if status == "detected":
                mask_path = self.debug_dir / f"{name}_mask.png"
                _save_image(mask_path, "L", (2, 2, 8, 8))
                parts[name] = {
                    "status": "detected",
                    "boxes": [[2, 2, 8, 8]],
                    "detection_scores": [0.9],
                    "mask_quality_scores": [0.95],
                    "accepted_masks": 1,
                    "accepted_boxes": [[2, 2, 8, 8]],
                    "mask_path": str(mask_path),
                    "outcome_reason": "accepted_mask_available",
                    "automatic_conditioning": True,
                }
            elif status == "no_detection":
                parts[name] = {
                    "status": "uncertain",
                    "boxes": [],
                    "detection_scores": [],
                    "accepted_masks": 0,
                    "outcome_reason": "no_candidate_above_threshold",
                }
            else:
                candidate_path = (
                    self.debug_dir / f"{name}_candidate_0_mask.png"
                )
                _save_image(candidate_path, "L", (2, 2, 8, 8))
                parts[name] = {
                    "status": "uncertain",
                    "boxes": [[2, 2, 8, 8]],
                    "detection_scores": [0.51],
                    "mask_quality_scores": [0.2],
                    "accepted_masks": 0,
                    "candidate_mask_paths": [str(candidate_path)],
                    "outcome_reason": (
                        "all_candidates_rejected_or_low_mask_quality"
                    ),
                }
        self.report = {"status": "detected", "parts": parts}
        (self.debug_dir / "parts.json").write_text(
            json.dumps(self.report),
            encoding="utf-8",
        )
        return []


def test_validation_reports_detection_absence_and_ambiguous_separately(
    tmp_path,
):
    dataset = tmp_path / "validation"
    _save_image(dataset / "images" / "present.png")
    _save_image(dataset / "masks" / "ears.png", "L", (2, 2, 8, 8))
    _save_image(dataset / "images" / "ambiguous.png")
    manifest = _write_manifest(dataset, [
        {
            "id": "present",
            "image": "images/present.png",
            "parts": {
                "animal_ears": {
                    "status": "present",
                    "mask": "masks/ears.png",
                },
                "hair_accessory": {
                    "status": "present",
                    "mask": "masks/ears.png",
                },
                "tail": {"status": "absent"},
            },
        },
        {
            "id": "ambiguous",
            "image": "images/ambiguous.png",
            "parts": {
                "animal_ears": {"status": "ambiguous"},
                "hair_accessory": {"status": "ambiguous"},
                "tail": {"status": "ambiguous"},
            },
        },
    ])
    plans = {
        "present": {
            "animal_ears": "detected",
            "hair_accessory": "detected",
            "tail": "no_detection",
        },
        "ambiguous": {
            "animal_ears": "detected",
            "hair_accessory": "uncertain",
            "tail": "uncertain",
        },
    }

    result = run_reference_part_validation(
        manifest,
        tmp_path / "report",
        analyzer_factory=lambda debug_dir, source_masks: FakeAnalyzer(
            debug_dir,
            plans,
        ),
        configuration={"test": True},
    )

    ears = result["parts"]["animal_ears"]
    accessory = result["parts"]["hair_accessory"]
    tail = result["parts"]["tail"]
    assert ears["true_positive"] == 1
    assert ears["detected_precision"] == 1.0
    assert ears["strict_present_recall"] == 1.0
    assert ears["automation_coverage"] == 1.0
    assert ears["mean_mask_iou"] == 1.0
    assert ears["mean_mask_dice"] == 1.0
    assert accessory["true_positive"] == 1
    assert accessory["mean_mask_iou"] == 1.0
    assert tail["true_negative"] == 1
    assert tail["uncertain"] == 1
    assert tail["automation_coverage"] == 1.0
    assert result["generation_invoked"] is False
    assert result["automatic_threshold_approval"] is False
    for name in (
        "metrics.json",
        "predictions.jsonl",
        "cases.csv",
        "cross-review.csv",
        "spatial-diagnostics.csv",
        "accessory-observations.csv",
        "config-snapshot.json",
    ):
        assert (tmp_path / "report" / name).is_file()
    assert (
        tmp_path / "report" / "overlays" / "present-animal_ears.png"
    ).is_file()
    assert (
        tmp_path / "report" / "overlays" / "present-hair_accessory.png"
    ).is_file()
    overlap_path = (
        tmp_path
        / "report"
        / "cross-review"
        / "present-animal_ears-hair_accessory-overlap.png"
    )
    assert overlap_path.is_file()
    predictions = [
        json.loads(line)
        for line in (
            tmp_path / "report" / "predictions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    present = predictions[0]
    assert present["parts"]["animal_ears"]["automatic_conditioning"] is False
    assert present["parts"]["animal_ears"]["analyzer_would_condition"] is True
    cross = present["animal_ear_hair_accessory_cross_review"]
    assert cross["status"] == "overlap_ambiguous"
    assert cross["winner_selected"] is False
    assert cross["masks_modified"] is False
    assert cross["automatic_conditioning_applied"] is False
    assert cross["same_mask_conflict"] is True
    assert cross["conflict_status"] == (
        "same_mask_cross_class_conflict"
    )
    spatial = present["spatial_diagnostics"]
    assert spatial["mode"] == "observe_only"
    assert spatial["head_checks"]["animal_ears"]["status"] == (
        "inside_head_roi"
    )
    assert spatial["predictions_modified"] is False
    assert spatial["masks_modified"] is False
    assert spatial["automatic_conditioning_applied"] is False
    assert result["spatial_diagnostics"][
        "same_mask_cross_class_conflicts"
    ] >= 1
    assert result["animal_ear_hair_accessory_cross_review"][
        "overlap_ambiguous"
    ] >= 1


def test_uncertain_is_not_automatic_conditioning_and_reduces_coverage(
    tmp_path,
):
    dataset = tmp_path / "validation"
    _save_image(dataset / "image.png")
    _save_image(dataset / "ears.png", "L", (2, 2, 8, 8))
    manifest = _write_manifest(dataset, [{
        "id": "uncertain",
        "image": "image.png",
        "parts": {
            "animal_ears": {"status": "present", "mask": "ears.png"},
            "hair_accessory": {"status": "absent"},
            "tail": {"status": "absent"},
        },
    }])
    plans = {
        "uncertain": {
            "animal_ears": "uncertain",
            "hair_accessory": "no_detection",
            "tail": "no_detection",
        },
    }
    result = run_reference_part_validation(
        manifest,
        tmp_path / "report",
        analyzer_factory=lambda debug_dir, source_masks: FakeAnalyzer(
            debug_dir,
            plans,
        ),
    )
    ears = result["parts"]["animal_ears"]
    assert ears["uncertain_present"] == 1
    assert ears["automation_coverage"] == 0.0
    prediction = json.loads(
        (tmp_path / "report" / "predictions.jsonl").read_text(
            encoding="utf-8"
        )
    )["parts"]["animal_ears"]
    assert prediction["status"] == "uncertain"
    assert prediction["automatic_conditioning"] is False
    assert prediction["diagnostic_candidate_source"] == "raw_candidate_union"
    assert Path(prediction["diagnostic_candidate_mask_path"]).is_file()


def test_no_detection_is_observation_not_absence():
    prediction = normalize_prediction(
        "animal_ears",
        {"parts": {"animal_ears": {
            "status": "uncertain",
            "presence": "not_observed",
            "boxes": [],
            "accepted_masks": 0,
        }}},
    )
    assert prediction["status"] == "no_detection"
    assert prediction["automatic_conditioning"] is False


def test_manifest_rejects_generation_or_training_directories(tmp_path):
    forbidden = tmp_path / "outputs" / "reference-validation"
    _save_image(forbidden / "image.png")
    manifest = _write_manifest(forbidden, [{
        "id": "bad",
        "image": "image.png",
        "parts": {
            "animal_ears": {"status": "absent"},
            "hair_accessory": {"status": "absent"},
            "tail": {"status": "absent"},
        },
    }])
    with pytest.raises(ReferencePartValidationError, match="분리"):
        load_validation_manifest(manifest)


def test_manifest_rejects_files_outside_dataset_root(tmp_path):
    dataset = tmp_path / "validation"
    _save_image(tmp_path / "outside.png")
    manifest = _write_manifest(dataset, [{
        "id": "escape",
        "image": "../outside.png",
        "parts": {
            "animal_ears": {"status": "absent"},
            "hair_accessory": {"status": "absent"},
            "tail": {"status": "absent"},
        },
    }])
    with pytest.raises(ReferencePartValidationError, match="밖"):
        load_validation_manifest(manifest)


def test_report_output_cannot_pollute_validation_dataset(tmp_path):
    dataset = tmp_path / "validation"
    _save_image(dataset / "image.png")
    manifest = _write_manifest(dataset, [{
        "id": "clean",
        "image": "image.png",
        "parts": {
            "animal_ears": {"status": "absent"},
            "hair_accessory": {"status": "absent"},
            "tail": {"status": "absent"},
        },
    }])
    with pytest.raises(ReferencePartValidationError, match="보고서 출력"):
        run_reference_part_validation(
            manifest,
            dataset / "reports",
            analyzer_factory=lambda **kwargs: None,
        )
