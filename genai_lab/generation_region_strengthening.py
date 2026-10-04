"""작은 부위의 생성용 IP-Adapter 영역만 축소 손실에 맞춰 보강한다."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from genai_lab.reference_regions import read_binary_mask


_ROOT_KEYS = {"enabled", "parts"}
_PART_KEYS = {
    "strong_threshold",
    "middle_queries",
    "minimum_middle_cells",
    "coarse_queries",
    "minimum_coarse_cells",
    "maximum_area_growth",
    "maximum_radius_ratio",
}


def _strong_cell_count(selected: np.ndarray, queries: int, threshold: float) -> int:
    """설치된 Diffusers와 같은 IP-Adapter mask 축소 함수로 남는 셀을 센다."""
    import torch
    from diffusers.image_processor import IPAdapterMaskProcessor

    tensor = torch.from_numpy(selected.astype(np.float32, copy=False)).unsqueeze(0)
    downsampled = IPAdapterMaskProcessor.downsample(
        tensor, batch_size=1, num_queries=queries, value_embed_dim=1
    )
    return int(torch.count_nonzero(downsampled[:, :, 0] > threshold).item())


def _validate_policy(settings: dict, part_name: str) -> dict | None:
    if not isinstance(settings, dict) or set(settings) - _ROOT_KEYS:
        raise ValueError("생성용 부위 영역 보강 설정 형식 오류")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("생성용 부위 영역 보강 enabled는 bool이어야 합니다.")
    parts = settings.get("parts", {})
    if not isinstance(parts, dict) or any(not isinstance(name, str) for name in parts):
        raise ValueError("생성용 부위 영역 보강 parts 설정 오류")
    if not enabled or part_name not in parts:
        return None
    policy = parts[part_name]
    if not isinstance(policy, dict) or set(policy) != _PART_KEYS:
        raise ValueError(f"{part_name}: 생성용 영역 보강 정책 키 오류")

    threshold = policy["strong_threshold"]
    growth = policy["maximum_area_growth"]
    radius_ratio = policy["maximum_radius_ratio"]
    finite_values = (threshold, growth, radius_ratio)
    if (not all(type(value) in (int, float) and math.isfinite(value)
                for value in finite_values)
            or not 0 < threshold <= 1
            or growth <= 1
            or not 0 < radius_ratio <= .10):
        raise ValueError(f"{part_name}: 생성용 영역 보강 수치 오류")
    for key in ("middle_queries", "minimum_middle_cells",
                "coarse_queries", "minimum_coarse_cells"):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f"{part_name}: 생성용 영역 보강 셀 기준 오류")
    if policy["middle_queries"] <= policy["coarse_queries"]:
        raise ValueError(f"{part_name}: attention query 단계 순서 오류")
    return policy


def strengthen_generation_region(
    part_name: str,
    source_mask: Image.Image,
    settings: dict | None,
    blocked_mask: Image.Image | None = None,
) -> tuple[Image.Image, dict]:
    """원본 검출 마스크는 변경하지 않고 생성에 사용할 복사본만 보강한다."""
    if settings is None:
        settings = {}
    selected = read_binary_mask(source_mask, source_mask.size)
    blocked = (
        np.zeros_like(selected)
        if blocked_mask is None
        else read_binary_mask(blocked_mask, source_mask.size)
    )
    if np.any(selected & blocked):
        raise ValueError(
            f"{part_name}: 원본 부위 마스크가 보호할 반환 마스크와 겹칩니다.")
    source_pixels = int(selected.sum())
    policy = _validate_policy(settings, part_name)
    if policy is None:
        return source_mask.copy(), {
            "version": "attention_survival_dilation_v2",
            "part": part_name,
            "status": "disabled_or_not_configured",
            "changed": False,
            "source_pixels": source_pixels,
            "generation_pixels": source_pixels,
            "source_mask_preserved": True,
            "reference_rgb_changed": False,
            "scale_changed": False,
            "blocked_mask_applied": blocked_mask is not None,
            "blocked_pixels_removed": 0,
        }

    threshold = float(policy["strong_threshold"])
    requirements = (
        (int(policy["middle_queries"]), int(policy["minimum_middle_cells"])),
        (int(policy["coarse_queries"]), int(policy["minimum_coarse_cells"])),
    )
    maximum_growth = float(policy["maximum_area_growth"])
    maximum_radius = max(1, int(round(min(source_mask.size) *
                                      float(policy["maximum_radius_ratio"]))))

    def measure(candidate: np.ndarray) -> dict:
        return {
            str(queries): {
                "minimum_cells": minimum,
                "strong_cells": _strong_cell_count(candidate, queries, threshold),
            }
            for queries, minimum in requirements
        }

    def satisfied(values: dict) -> bool:
        return all(item["strong_cells"] >= item["minimum_cells"]
                   for item in values.values())

    source_stats = measure(selected)
    if satisfied(source_stats):
        return source_mask.copy(), {
            "version": "attention_survival_dilation_v2",
            "part": part_name,
            "status": "already_sufficient",
            "changed": False,
            "radius_pixels": 0,
            "source_pixels": source_pixels,
            "generation_pixels": source_pixels,
            "area_growth": 1.0,
            "strong_threshold": threshold,
            "attention_cells": source_stats,
            "maximum_area_growth": maximum_growth,
            "maximum_radius_pixels": maximum_radius,
            "source_mask_preserved": True,
            "reference_rgb_changed": False,
            "scale_changed": False,
            "blocked_mask_applied": blocked_mask is not None,
            "blocked_pixels_removed": 0,
        }

    source_u8 = selected.astype(np.uint8)
    last_stats = source_stats
    last_growth = 1.0
    last_blocked_pixels_removed = 0
    for radius in range(1, maximum_radius + 1):
        diameter = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        dilated = cv2.dilate(
            source_u8, kernel, iterations=1).astype(bool)
        candidate = dilated & ~blocked
        blocked_pixels_removed = int((dilated & blocked).sum())
        last_blocked_pixels_removed = blocked_pixels_removed
        generation_pixels = int(candidate.sum())
        growth = generation_pixels / source_pixels
        if growth > maximum_growth:
            break
        stats = measure(candidate)
        last_stats, last_growth = stats, growth
        if satisfied(stats):
            return Image.fromarray(candidate.astype(np.uint8) * 255), {
                "version": "attention_survival_dilation_v2",
                "part": part_name,
                "status": "strengthened",
                "changed": True,
                "radius_pixels": radius,
                "kernel": "ellipse",
                "source_pixels": source_pixels,
                "generation_pixels": generation_pixels,
                "area_growth": growth,
                "strong_threshold": threshold,
                "attention_cells": stats,
                "source_attention_cells": source_stats,
                "maximum_area_growth": maximum_growth,
                "maximum_radius_pixels": maximum_radius,
                "source_mask_preserved": True,
                "reference_rgb_changed": False,
                "scale_changed": False,
                "position_policy": "source_roi_unchanged",
                "blocked_mask_applied": blocked_mask is not None,
                "blocked_pixels_removed": blocked_pixels_removed,
            }
    if blocked_mask is not None:
        return source_mask.copy(), {
            "version": "attention_survival_dilation_v2",
            "part": part_name,
            "status": "blocked_fallback_source",
            "changed": False,
            "radius_pixels": 0,
            "source_pixels": source_pixels,
            "generation_pixels": source_pixels,
            "area_growth": 1.0,
            "strong_threshold": threshold,
            "attention_cells": source_stats,
            "last_constrained_attention_cells": last_stats,
            "maximum_area_growth": maximum_growth,
            "maximum_radius_pixels": maximum_radius,
            "source_mask_preserved": True,
            "reference_rgb_changed": False,
            "scale_changed": False,
            "position_policy": "source_roi_unchanged",
            "blocked_mask_applied": True,
            "blocked_pixels_removed": last_blocked_pixels_removed,
            "fallback_reason": "protected_return_mask_has_priority",
        }
    raise ValueError(
        f"{part_name}: 생성용 영역을 안전 한도 안에서 보강하지 못했습니다. "
        f"최대 면적 증가={maximum_growth:g}, 최대 반경={maximum_radius}px, "
        f"마지막 면적 증가={last_growth:.3f}, 축소 셀={last_stats}"
    )
