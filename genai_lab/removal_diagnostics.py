"""의상 제거 전처리의 수치 증거. 이미지/판정/모델 설정은 변경하지 않는다."""

from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from datetime import datetime
from functools import wraps
import inspect
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from genai_lab.image_digest import calculate_image_pixel_sha256

DIAGNOSTIC_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "debug_removal"
_active = ContextVar("removal_diagnostic_run", default=None)
_logger = logging.getLogger("genai_lab.removal_diagnostics")
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(handler)


def image_evidence(image):
    evidence = {
        "size": list(image.size), "mode": image.mode,
        "pixel_sha256": calculate_image_pixel_sha256(image, image.mode),
    }
    if image.mode != "L":
        return evidence
    mask = np.asarray(image) >= 128
    evidence["selected_pixels"] = int(mask.sum())
    # 내부 0 영역에는 원래의 틈도 포함된다. 의상 잔재라는 뜻이 아니다.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~mask).astype(np.uint8), connectivity=8)
    del labels
    holes = []
    height, width = mask.shape
    for x, y, w, h, area in stats[1:count]:
        if x > 0 and y > 0 and x + w < width and y + h < height:
            holes.append({"pixels": int(area), "bbox_xywh": [int(v) for v in (x, y, w, h)]})
    evidence["enclosed_zero_components"] = len(holes)
    evidence["enclosed_zero_pixels"] = sum(hole["pixels"] for hole in holes)
    evidence["largest_enclosed_zero_regions"] = sorted(
        holes, key=lambda hole: hole["pixels"], reverse=True)[:8]
    evidence["zero_region_meaning"] = "geometric_only_not_confirmed_clothing_residue"
    return evidence


def summarize(value, depth=0):
    if isinstance(value, Image.Image):
        return image_evidence(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if depth >= 6:
        return {"type": type(value).__name__}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: summarize(getattr(value, f.name), depth + 1) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): summarize(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [summarize(v, depth + 1) for v in value]
    return {"type": type(value).__name__}


def brief_numbers(value, prefix=""):
    items = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict):
                items.extend(brief_numbers(child, name))
            elif isinstance(child, (int, float)) and (
                "pixel" in key or "components" in key
            ):
                items.append(f"{name}={child}")
    return items


def emit(run, stage, event, **payload):
    """로그 저장 실패는 원래 연산의 성공/실패를 바꾸지 않는다."""
    record = {
        "time": datetime.now().astimezone().isoformat(),
        "run_id": run["id"], "source_rgb_sha256": run.get("source"),
        "stage": stage, "event": event, **payload,
    }
    try:
        path = run["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception as error:
        _logger.warning("[의상 제거 진단] 로그 저장 실패: %s", error)
    _logger.info("[의상 제거][%s][%s] %s elapsed=%s log=%s",
                 run["id"], stage, event, payload.get("elapsed_seconds", "-"), run["path"])
    numbers = brief_numbers(payload)
    if numbers:
        _logger.info("[의상 제거][%s][%s] %s", run["id"], stage, " | ".join(numbers[:12]))
    if event in ("failed", "diagnostic_error"):
        _logger.error("[의상 제거][%s][%s] %s: %s",
                      run["id"], stage, payload.get("error_type", event), payload.get("error"))


def record_removal_event(stage, event, **payload):
    run = _active.get()
    if run is not None:
        emit(run, stage, event, **payload)


def trace_removal(stage):
    """같은 전처리 호출의 하위 단계를 실행 ID로 묶고 예외도 기록한다."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def traced(*args, **kwargs):
            run = _active.get()
            token = None
            if run is None:
                identifier = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:12]
                run = {"id": identifier, "path": DIAGNOSTIC_ROOT / (identifier + ".jsonl")}
                token = _active.set(run)
            started = perf_counter()
            try:
                try:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    source = next((bound.arguments[k] for k in (
                        "character_image", "source_image", "source"
                    ) if k in bound.arguments), None)
                    if isinstance(source, Image.Image):
                        run["source"] = calculate_image_pixel_sha256(source, "RGB")
                    emit(run, stage, "start", inputs=summarize(bound.arguments),
                         coverage_meaning="selected_mask_coverage_not_visual_clothing_removal")
                except Exception as error:
                    emit(run, stage, "diagnostic_error", error=str(error))
                try:
                    result = function(*args, **kwargs)
                except Exception as error:
                    emit(run, stage, "failed", error_type=type(error).__name__,
                         error=str(error), elapsed_seconds=round(perf_counter() - started, 4))
                    raise
                try:
                    emit(run, stage, "completed", outputs=summarize(result),
                         elapsed_seconds=round(perf_counter() - started, 4))
                except Exception as error:
                    emit(run, stage, "diagnostic_error", error=str(error))
                return result
            finally:
                if token is not None:
                    _active.reset(token)
        return traced
    return decorate
