"""Mechanical generated-output checks; never a semantic quality judge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np
import torch
from PIL import Image


class GeneratedOutputIntegrityError(RuntimeError):
    def __init__(self, message: str, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


@dataclass(frozen=True)
class GeneratedImageIntegritySettings:
    enabled: bool = False
    retry_count: int = 0
    retry_final_latent_failure: bool = True
    retry_decoded_rgb_failure: bool = False
    continue_after_corruption: bool = True
    preserve_corrupted_image: bool = True
    high_jump_difference: float = 80.0
    maximum_adjacent_difference: float = 12.0
    maximum_high_jump_ratio: float = 0.03
    maximum_laplacian_variance: float = 1500.0
    maximum_extreme_pixel_ratio: float = 0.05
    minimum_failed_checks: int = 3
    reject_adjacent_and_extreme_pair: bool = True

    def __post_init__(self) -> None:
        if self.retry_count < 0 or self.high_jump_difference <= 0:
            raise ValueError("생성 이미지 무결성 설정값이 올바르지 않습니다.")
        ratios = (self.maximum_high_jump_ratio, self.maximum_extreme_pixel_ratio)
        if any(value < 0 or value > 1 for value in ratios):
            raise ValueError("생성 이미지 무결성 비율은 0~1이어야 합니다.")
        if self.maximum_adjacent_difference < 0 or self.maximum_laplacian_variance < 0:
            raise ValueError("생성 이미지 무결성 임계값은 0 이상이어야 합니다.")
        if self.minimum_failed_checks not in range(1, 5):
            raise ValueError("minimum_failed_checks는 1~4이어야 합니다.")

    def record(self) -> dict[str, Any]:
        return {"version": "generated_image_integrity_v2", **self.__dict__,
                "semantic_quality_judgement": False}


def resolve_generated_image_integrity(config: Mapping[str, Any]) -> GeneratedImageIntegritySettings:
    raw = config.get("generated_image_integrity", {})
    if not isinstance(raw, Mapping):
        raw = {}
    names = GeneratedImageIntegritySettings.__dataclass_fields__
    values = {name: raw[name] for name in names if name in raw}
    return GeneratedImageIntegritySettings(**values)


def _latent_report(latents: Any, *, retryable: bool) -> dict[str, Any]:
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
        return {"status": "invalid", "reason": "final_latents_not_4d_tensor",
                "failure_stage": "final_latent", "retryable": retryable,
                "type": type(latents).__name__}
    finite = bool(torch.isfinite(latents).all().item())
    report = {"status": "passed" if finite else "invalid",
              "reason": None if finite else "final_latents_non_finite",
              "failure_stage": "final_latent", "retryable": retryable and not finite,
              "shape": list(latents.shape), "dtype": str(latents.dtype), "finite": finite}
    if finite:
        values = latents.detach().float()
        horizontal = torch.diff(values, dim=-1).abs()
        vertical = torch.diff(values, dim=-2).abs()
        report.update(minimum=float(values.min()), maximum=float(values.max()),
                      mean=float(values.mean()), standard_deviation=float(values.std()),
                      spatial_metrics={
                          "horizontal_difference_mean": float(horizontal.mean()),
                          "vertical_difference_mean": float(vertical.mean()),
                          "horizontal_difference_max": float(horizontal.max()),
                          "vertical_difference_max": float(vertical.max()),
                      })
    return report


def wrap_final_latent_audit(callback: Callable | None, total_steps: int,
                            audit_record: MutableMapping[str, Any],
                            policy_record: Mapping[str, Any]) -> Callable:
    def audited(pipe: Any, step_index: int, timestep: Any,
                callback_kwargs: MutableMapping[str, Any]):
        result = callback_kwargs
        if callback is not None:
            returned = callback(pipe, step_index, timestep, callback_kwargs)
            if returned is not None:
                result = returned
        if step_index == total_steps - 1:
            report = _latent_report(
                result.get("latents"),
                retryable=bool(policy_record.get("retry_final_latent_failure", True)),
            )
            audit_record.clear()
            audit_record.update(report)
            if report["status"] != "passed":
                raise GeneratedOutputIntegrityError("최종 denoising latent 무결성 검사 실패", report)
        return result

    schedule = getattr(callback, "reference_step_schedule_record", None)
    if schedule is not None:
        audited.reference_step_schedule_record = schedule
    audited.generated_image_integrity_record = dict(policy_record)
    return audited


def analyze_generated_image_integrity(image: Image.Image,
                                      settings: GeneratedImageIntegritySettings) -> dict[str, Any]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    horizontal, vertical = np.abs(np.diff(rgb, axis=1)), np.abs(np.diff(rgb, axis=0))
    adjacent = float((horizontal.mean() + vertical.mean()) / 2.0)
    gray = rgb.mean(axis=2)
    gray_h, gray_v = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    jumps = np.count_nonzero(gray_h >= settings.high_jump_difference)
    jumps += np.count_nonzero(gray_v >= settings.high_jump_difference)
    jump_ratio = float(jumps / max(gray_h.size + gray_v.size, 1))
    center = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * center
    lap_variance = float(np.var(lap, dtype=np.float64)) if lap.size else 0.0
    extreme_ratio = float(np.mean((rgb <= 1.0) | (rgb >= 254.0)))
    failed = []
    if adjacent > settings.maximum_adjacent_difference: failed.append("adjacent_difference")
    if jump_ratio > settings.maximum_high_jump_ratio: failed.append("high_jump_ratio")
    if lap_variance > settings.maximum_laplacian_variance: failed.append("laplacian_variance")
    if extreme_ratio > settings.maximum_extreme_pixel_ratio: failed.append("extreme_pixel_ratio")
    critical_pair = (
        settings.reject_adjacent_and_extreme_pair
        and "adjacent_difference" in failed
        and "extreme_pixel_ratio" in failed
    )
    corrupted = len(failed) >= settings.minimum_failed_checks or critical_pair
    return {"version": "generated_image_integrity_report_v2",
            "status": "corrupted" if corrupted else "passed",
            "failure_stage": "decoded_rgb",
            "retryable": corrupted and settings.retry_decoded_rgb_failure,
            "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
            "metrics": {"adjacent_difference": adjacent, "high_jump_ratio": jump_ratio,
                        "laplacian_variance": lap_variance, "extreme_pixel_ratio": extreme_ratio},
            "failed_checks": failed, "failed_check_count": len(failed),
            "minimum_failed_checks": settings.minimum_failed_checks,
            "critical_pair_triggered": critical_pair,
            "semantic_quality_judgement": False}
