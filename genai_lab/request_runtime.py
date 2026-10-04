"""Reset mutable diffusion state at product-request boundaries."""
from __future__ import annotations

from typing import Any


def reset_request_runtime(pipeline: Any, *, boundary: str) -> dict[str, Any]:
    """Clear request-owned state without unloading shared model weights."""
    if boundary not in {"request_start", "request_end", "request_failed"}:
        raise ValueError(f"지원하지 않는 요청 경계입니다: {boundary}")
    report: dict[str, Any] = {
        "version": "request_runtime_reset_v1",
        "boundary": boundary,
        "pipeline_present": pipeline is not None,
        "actions": [],
        "errors": [],
    }
    if pipeline is None:
        report["status"] = "not_required"
        return report

    if hasattr(pipeline, "maybe_free_model_hooks"):
        try:
            pipeline.maybe_free_model_hooks()
            report["actions"].append("model_hooks_released")
        except Exception as error:  # cleanup remains diagnostic
            report["errors"].append(
                f"model_hooks:{type(error).__name__}:{error}"
            )
    if hasattr(pipeline, "_interrupt"):
        try:
            pipeline._interrupt = False
            report["actions"].append("interrupt_cleared")
        except Exception as error:
            report["errors"].append(
                f"interrupt:{type(error).__name__}:{error}"
            )
    if hasattr(pipeline, "set_ip_adapter_scale"):
        try:
            pipeline.set_ip_adapter_scale(0.0)
            report["actions"].append("ip_adapter_scale_zeroed")
        except Exception as error:
            report["errors"].append(
                f"ip_adapter_scale:{type(error).__name__}:{error}"
            )

    cached_name = "_genai_lab_part_inpaint_pipeline"
    cached = getattr(pipeline, cached_name, None)
    if cached is not None:
        try:
            if hasattr(cached, "maybe_free_model_hooks"):
                cached.maybe_free_model_hooks()
            delattr(pipeline, cached_name)
            report["actions"].append("cached_inpaint_pipeline_released")
        except Exception as error:
            report["errors"].append(
                f"cached_inpaint:{type(error).__name__}:{error}"
            )

    report["status"] = "completed_with_warnings" if report["errors"] else "completed"
    return report
