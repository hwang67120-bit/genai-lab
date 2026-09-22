"""Observe-only multi-scale diagnostics for small head accessories."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from PIL import Image

VERSION = "accessory_observation_v1"
DEFAULT_QUERIES = (
    ("animal_ears", "animal ears."),
    ("costume_ears", "costume ears. animal ear headband."),
    ("hair_accessory", "hair ornament. hairpin. hair clip. hair tie. hair ribbon."),
    ("headphones", "headphones. headset."),
    ("hat", "hat. cap. headwear."),
)


@dataclass(frozen=True)
class AccessoryObservationPolicy:
    mode: str = "observe_only"
    tile_grid_size: int = 2
    tile_overlap_ratio: float = .25
    model_short_edge: int = 800
    model_max_long_edge: int = 1333
    minimum_cluster_iou: float = .10
    maximum_normalized_center_distance: float = .35
    minimum_localized_views: int = 2
    maximum_candidates_per_view_label: int = 8
    save_view_images: bool = True
    query_groups: tuple[tuple[str, str], ...] = DEFAULT_QUERIES

    def __post_init__(self):
        if self.mode != "observe_only":
            raise ValueError("장신구 확대 분석은 observe_only만 지원합니다.")
        if not 1 <= self.tile_grid_size <= 3:
            raise ValueError("장신구 확대 타일 격자는 1~3이어야 합니다.")
        if not 0 <= self.tile_overlap_ratio < .75:
            raise ValueError("장신구 확대 타일 중첩 비율이 올바르지 않습니다.")
        if not 64 <= self.model_short_edge <= 4096:
            raise ValueError("장신구 분석 모델 짧은 변 설정이 올바르지 않습니다.")
        if not self.model_short_edge <= self.model_max_long_edge <= 8192:
            raise ValueError("장신구 분석 모델 긴 변 설정이 올바르지 않습니다.")
        if not 0 <= self.minimum_cluster_iou <= 1:
            raise ValueError("장신구 후보 군집 IoU가 올바르지 않습니다.")
        if not 0 <= self.maximum_normalized_center_distance <= 2:
            raise ValueError("장신구 후보 중심 거리 설정이 올바르지 않습니다.")
        if not 1 <= self.minimum_localized_views <= 10:
            raise ValueError("장신구 위치 합의 관측 수가 올바르지 않습니다.")
        if not 1 <= self.maximum_candidates_per_view_label <= 32:
            raise ValueError("장신구 관측 후보 상한이 올바르지 않습니다.")
        names = set()
        for item in self.query_groups:
            if (not isinstance(item, tuple) or len(item) != 2
                    or not all(isinstance(value, str) for value in item)):
                raise ValueError("장신구 관측 질의 형식이 올바르지 않습니다.")
            name, query = item
            if not name or name in names or not query.strip():
                raise ValueError("장신구 관측 질의 이름이 비었거나 중복되었습니다.")
            names.add(name)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        raw = dict(value or {})
        if not raw.pop("enabled", False):
            return None
        raw.pop("analysis_scope", None)
        configured = raw.pop("query_groups", None)
        if configured is not None:
            if not isinstance(configured, (list, tuple)):
                raise ValueError("장신구 관측 질의는 배열이어야 합니다.")
            queries = []
            for item in configured:
                if not isinstance(item, Mapping) or set(item) != {"name", "query"}:
                    raise ValueError("장신구 관측 질의 설정 형식이 올바르지 않습니다.")
                queries.append((str(item["name"]), str(item["query"])))
            raw["query_groups"] = tuple(queries)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("등록되지 않은 장신구 관측 설정입니다: " + ", ".join(sorted(unknown)))
        return cls(**raw)

    def snapshot(self):
        result = asdict(self)
        result["query_groups"] = [
            {"name": name, "query": query} for name, query in self.query_groups
        ]
        return result


@dataclass(frozen=True)
class AccessoryView:
    name: str
    original_box: tuple[int, int, int, int]


def _box(values, size):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("장신구 관측 좌표가 올바르지 않습니다.")
    width, height = size
    result = (
        max(0, min(width, round(values[0]))),
        max(0, min(height, round(values[1]))),
        max(0, min(width, round(values[2]))),
        max(0, min(height, round(values[3]))),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("장신구 관측 영역이 비었습니다.")
    return result


def _windows(length, count, overlap):
    if count == 1:
        return ((0, length),)
    window = min(length, ceil(length / (1 + (count - 1) * (1 - overlap))))
    maximum = max(0, length - window)
    starts = [round(maximum * index / (count - 1)) for index in range(count)]
    return tuple((start, min(length, start + window)) for start in starts)


def build_accessory_views(size, head_box, policy):
    width, height = size
    left, top, right, bottom = _box(head_box, size)
    views = [AccessoryView("full_image", (0, 0, width, height))]
    if (left, top, right, bottom) != (0, 0, width, height):
        views.append(AccessoryView("head_crop", (left, top, right, bottom)))
    if policy.tile_grid_size > 1:
        xs = _windows(right-left, policy.tile_grid_size, policy.tile_overlap_ratio)
        ys = _windows(bottom-top, policy.tile_grid_size, policy.tile_overlap_ratio)
        for row, (y1, y2) in enumerate(ys):
            for column, (x1, x2) in enumerate(xs):
                views.append(AccessoryView(
                    f"head_tile_r{row}_c{column}",
                    (left+x1, top+y1, left+x2, top+y2),
                ))
    unique, seen = [], set()
    for view in views:
        if view.original_box not in seen:
            seen.add(view.original_box)
            unique.append(view)
    return tuple(unique)


def estimate_model_input_size(size, policy):
    width, height = size
    scale = min(
        policy.model_short_edge / min(width, height),
        policy.model_max_long_edge / max(width, height),
    )
    return max(1, round(width*scale)), max(1, round(height*scale))


def map_view_box_to_original(values, view, original_size):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("장신구 후보 좌표가 올바르지 않습니다.")
    x1, y1, x2, y2 = values
    left, top, right, bottom = view.original_box
    x1, x2 = np.clip((x1, x2), 0, right-left)
    y1, y2 = np.clip((y1, y2), 0, bottom-top)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("장신구 후보 좌표가 비었습니다.")
    width, height = original_size
    return (
        min(width, left+float(x1)), min(height, top+float(y1)),
        min(width, left+float(x2)), min(height, top+float(y2)),
    )


def _iou(first, second):
    intersection = max(0., min(first[2], second[2])-max(first[0], second[0])) * max(
        0., min(first[3], second[3])-max(first[1], second[1]))
    union = ((first[2]-first[0])*(first[3]-first[1])
             + (second[2]-second[0])*(second[3]-second[1]) - intersection)
    return intersection/union if union else 0.


def _center_distance(first, second):
    scale = max(1., first[2]-first[0], first[3]-first[1],
                second[2]-second[0], second[3]-second[1])
    first_center = ((first[0]+first[2])/2, (first[1]+first[3])/2)
    second_center = ((second[0]+second[2])/2, (second[1]+second[3])/2)
    return float(np.hypot(first_center[0]-second_center[0],
                          first_center[1]-second_center[1]) / scale)


def _cluster(candidates, policy):
    groups = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        group = next((items for items in groups if any(
            _iou(candidate["box_in_original"], item["box_in_original"])
            >= policy.minimum_cluster_iou
            or _center_distance(candidate["box_in_original"], item["box_in_original"])
            <= policy.maximum_normalized_center_distance
            for item in items)), None)
        (groups.append([candidate]) if group is None else group.append(candidate))
    reports = []
    for index, items in enumerate(groups):
        labels = sorted({item["label"] for item in items})
        views = sorted({item["source_view"] for item in items})
        status = ("class_conflict" if len(labels) > 1
                  else "localized" if len(views) >= policy.minimum_localized_views
                  else "present_unlocalized")
        reports.append({
            "cluster_index": index, "status": status, "labels": labels,
            "best_score_by_label": {
                label: max(item["score"] for item in items if item["label"] == label)
                for label in labels},
            "source_views": views, "independent_view_count": len(views),
            "candidate_count": len(items),
            "union_box_in_original": [
                min(item["box_in_original"][0] for item in items),
                min(item["box_in_original"][1] for item in items),
                max(item["box_in_original"][2] for item in items),
                max(item["box_in_original"][3] for item in items)],
            "review_required": status != "localized",
            "mask_eligible": status == "localized", "candidates": items,
        })
    return reports


def _detect_view(backend, crop, policy, box_threshold, check):
    """Return labeled boxes, using one model pass when the backend supports it."""
    query_lookup = dict(policy.query_groups)
    detect_labeled = getattr(backend, "detect_labeled", None)
    if callable(detect_labeled):
        boxes, scores, labels = detect_labeled(
            crop,
            policy.query_groups,
            box_threshold,
        )
        check()
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels = tuple(str(label) for label in labels)
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("장신구 관측 후보 좌표·점수·라벨 수가 다릅니다.")
        counts = {}
        result = []
        for box, score, label in zip(boxes, scores, labels):
            counts[label] = counts.get(label, 0) + 1
            if counts[label] > policy.maximum_candidates_per_view_label:
                raise ValueError(f"{label}: 장신구 관측 후보 수가 상한을 넘었습니다.")
            result.append((
                label,
                query_lookup.get(label, label),
                box,
                float(score),
            ))
        return result
    result = []
    for label, query in policy.query_groups:
        boxes, scores = backend.detect(crop, query, box_threshold)
        check()
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(boxes) != len(scores):
            raise ValueError("장신구 관측 후보 좌표와 점수 수가 다릅니다.")
        if len(boxes) > policy.maximum_candidates_per_view_label:
            raise ValueError(f"{label}: 장신구 관측 후보 수가 상한을 넘었습니다.")
        result.extend(
            (label, query, box, float(score))
            for box, score in zip(boxes, scores)
        )
    return result


def analyze_accessory_observations(
    backend, source: Image.Image, *, head_box, box_threshold, policy,
    check, debug_directory: Path | str | None = None,
):
    """Collect boxes only; never call SAM2 or alter the analyzer result."""
    if not np.isfinite(box_threshold) or not 0 <= box_threshold <= 1:
        raise ValueError("장신구 관측 검출 임계값이 올바르지 않습니다.")
    views = build_accessory_views(source.size, head_box, policy)
    view_directory = Path(debug_directory)/"accessory-views" if debug_directory else None
    if view_directory and policy.save_view_images:
        view_directory.mkdir(parents=True, exist_ok=True)
    candidates, view_reports = [], []
    for view in views:
        check()
        with source.crop(view.original_box).convert("RGB") as crop:
            estimated = estimate_model_input_size(crop.size, policy)
            saved_path = None
            if view_directory and policy.save_view_images:
                saved_path = view_directory/f"{view.name}.png"
                crop.save(saved_path)
            count = 0
            scale_x, scale_y = estimated[0]/crop.width, estimated[1]/crop.height
            for label, query, candidate_box, score in _detect_view(
                backend,
                crop,
                policy,
                box_threshold,
                check,
            ):
                if not np.isfinite(candidate_box).all() or not np.isfinite(score):
                    raise ValueError("장신구 관측 후보에 유한하지 않은 값이 있습니다.")
                if float(score) < box_threshold:
                    continue
                original = map_view_box_to_original(candidate_box, view, source.size)
                candidates.append({
                    "label": label, "query": query, "source_view": view.name,
                    "view_original_box": list(view.original_box),
                    "box_in_view": [float(value) for value in candidate_box],
                    "box_in_original": list(original),
                    "original_size_px": [original[2]-original[0], original[3]-original[1]],
                    "estimated_model_input_size": list(estimated),
                    "estimated_model_object_size_px": [
                        float(candidate_box[2]-candidate_box[0])*scale_x,
                        float(candidate_box[3]-candidate_box[1])*scale_y],
                    "score": float(score),
                })
                count += 1
            view_reports.append({
                "name": view.name, "box_in_original": list(view.original_box),
                "view_size": list(crop.size),
                "estimated_model_input_size": list(estimated),
                "candidate_count": count,
                "image_path": str(saved_path) if saved_path else None,
            })
    clusters = _cluster(candidates, policy)
    statuses = {item["status"] for item in clusters}
    if not clusters:
        status, presence = "not_detected", "not_observed"
    elif "class_conflict" in statuses:
        status, presence = "class_conflict", "uncertain"
    elif "localized" in statuses:
        status, presence = "localized", "model_candidate"
    else:
        status, presence = "present_unlocalized", "uncertain"
    return {
        "version": VERSION, "mode": "observe_only", "status": status,
        "presence": presence, "head_box_in_original": list(_box(head_box, source.size)),
        "views": view_reports, "clusters": clusters,
        "candidate_count": len(candidates),
        "review_required": status not in {"localized", "not_detected"},
        "not_detected_is_absence": False,
        "predictions_modified": False, "masks_created": False,
        "masks_modified": False, "sam2_invoked": False,
        "automatic_conditioning_applied": False,
        "generation_policy_changed": False, "policy": policy.snapshot(),
    }
