"""Diagnostic-only person counting for generated images."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PersonCountDiagnosticSettings:
    enabled: bool = False
    expected_count: int = 1
    detector_model_id: str = "IDEA-Research/grounding-dino-tiny"
    cache_dir: str = "D:/genai-cache/huggingface"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    minimum_area_ratio: float = 0.02
    duplicate_iou: float = 0.65
    blocking: bool = False

    def __post_init__(self) -> None:
        if type(self.expected_count) is not int or self.expected_count < 1:
            raise ValueError("person count expected_count는 1 이상이어야 합니다.")
        for name in (
            "box_threshold", "text_threshold", "minimum_area_ratio", "duplicate_iou"
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 < float(value) <= 1
            ):
                raise ValueError(f"person count {name}은 0 초과 1 이하여야 합니다.")
        if type(self.blocking) is not bool:
            raise ValueError("person count blocking은 boolean이어야 합니다.")

    def record(self) -> dict[str, Any]:
        return {
            "version": "person_count_diagnostic_v1",
            **self.__dict__,
            "policy": "diagnostic_only" if not self.blocking else "blocking",
        }


def resolve_person_count_diagnostic(
    config: Mapping[str, Any],
) -> PersonCountDiagnosticSettings:
    raw = config.get("person_count_diagnostic", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("person_count_diagnostic는 항목 묶음이어야 합니다.")
    allowed = set(PersonCountDiagnosticSettings.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"알 수 없는 인물 수 진단 설정: {sorted(unknown)}")
    return PersonCountDiagnosticSettings(**dict(raw))


def _iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _deduplicate(
    boxes: list[tuple[float, ...]],
    scores: list[float],
    maximum_iou: float,
) -> list[int]:
    selected: list[int] = []
    for index in sorted(range(len(boxes)), key=lambda item: scores[item], reverse=True):
        if all(_iou(boxes[index], boxes[kept]) < maximum_iou for kept in selected):
            selected.append(index)
    return selected


def _default_detector(
    image: Image.Image,
    settings: PersonCountDiagnosticSettings,
) -> tuple[list[tuple[float, ...]], list[float]]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_path = snapshot_download(
        settings.detector_model_id,
        cache_dir=settings.cache_dir,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path, local_files_only=True
    ).to("cpu").eval()
    try:
        with torch.inference_mode():
            inputs = processor(images=image, text="person.", return_tensors="pt")
            outputs = model(**inputs)
            post = processor.post_process_grounded_object_detection
            threshold_name = (
                "box_threshold"
                if "box_threshold" in inspect.signature(post).parameters
                else "threshold"
            )
            detected = post(
                outputs,
                inputs.input_ids,
                text_threshold=settings.text_threshold,
                target_sizes=[(image.height, image.width)],
                **{threshold_name: settings.box_threshold},
            )[0]
        boxes = [tuple(float(value) for value in box) for box in detected["boxes"]]
        scores = [float(value) for value in detected["scores"]]
        return boxes, scores
    finally:
        del model, processor


def evaluate_person_count(
    image: Image.Image,
    config: Mapping[str, Any],
    *,
    detector: Callable[
        [Image.Image, PersonCountDiagnosticSettings],
        tuple[list[tuple[float, ...]], list[float]],
    ] | None = None,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    settings = resolve_person_count_diagnostic(config)
    base = settings.record()
    if not settings.enabled:
        return {**base, "status": "DISABLED", "action": "none"}
    detector = detector or _default_detector
    try:
        boxes, scores = detector(image.convert("RGB"), settings)
        if len(boxes) != len(scores):
            raise ValueError("인물 검출 상자와 점수 개수가 다릅니다.")
        canvas_area = image.width * image.height
        candidates: list[tuple[float, ...]] = []
        candidate_scores: list[float] = []
        for box, score in zip(boxes, scores):
            if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / canvas_area
            if area_ratio >= settings.minimum_area_ratio:
                candidates.append((x1, y1, x2, y2))
                candidate_scores.append(float(score))
        kept = _deduplicate(candidates, candidate_scores, settings.duplicate_iou)
        accepted_boxes = [candidates[index] for index in kept]
        accepted_scores = [candidate_scores[index] for index in kept]
        detected_count = len(accepted_boxes)
        matches = detected_count == settings.expected_count
        record: dict[str, Any] = {
            **base,
            "status": "PASS" if matches else "REVIEW",
            "action": "diagnostic",
            "detected_count": detected_count,
            "matches_expected": matches,
            "boxes_xyxy": [list(box) for box in accepted_boxes],
            "scores": accepted_scores,
            "raw_detection_count": len(boxes),
            "return_blocked": bool(settings.blocking and not matches),
        }
        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            overlay = image.convert("RGB").copy()
            draw = ImageDraw.Draw(overlay)
            for index, (box, score) in enumerate(
                zip(accepted_boxes, accepted_scores), start=1
            ):
                draw.rectangle(box, outline=(255, 70, 70), width=3)
                draw.text((box[0] + 3, box[1] + 3), f"person {index} {score:.2f}", fill=(255, 70, 70))
            overlay_path = output_directory / "person-count-overlay.png"
            overlay.save(overlay_path)
            overlay.close()
            record["overlay_path"] = str(overlay_path)
        return record
    except Exception as error:
        return {
            **base,
            "status": "UNAVAILABLE",
            "action": "diagnostic",
            "detected_count": None,
            "matches_expected": None,
            "return_blocked": False,
            "error": f"{type(error).__name__}: {error}",
        }

