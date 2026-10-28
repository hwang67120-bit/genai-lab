"""승인된 1차 latent를 같은 SDXL 구성요소로 저노이즈 정밀화한다."""

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class LatentRefinementSettings:
    enabled: bool
    inference_steps: int
    strength: float
    guidance_scale: float
    latent_scale_factor: float

    def record(self) -> dict[str, Any]:
        return {
            "version": "sdxl_latent_img2img_refinement_v1",
            "enabled": self.enabled,
            "inference_steps": self.inference_steps,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "latent_scale_factor": self.latent_scale_factor,
            "validation": "tensor_shape_channels_and_finite_values",
            "semantic_output_verification": False,
            "pipeline_reuse": "AutoPipelineForImage2Image.from_pipe",
        }


def resolve_latent_refinement(config: dict[str, Any], request) -> LatentRefinementSettings:
    """설정을 엄격히 검사한다. 생략 시 기존 단일 패스를 유지한다."""
    raw = config.get("reference_analysis", {}).get("latent_refinement", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("latent 정밀화 설정은 객체여야 합니다.")

    enabled = bool(raw.get("enabled", False))
    steps = raw.get("inference_steps", 10)
    strength = float(raw.get("strength", 0.25))
    guidance = float(raw.get("guidance_scale", request.guidance_scale))
    scale = float(raw.get("latent_scale_factor", 1.0))

    if type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError("latent 정밀화 반복 횟수는 1~50이어야 합니다.")
    if not math.isfinite(strength) or not 0 < strength <= 0.5:
        raise ValueError("latent 정밀화 strength는 0 초과 0.5 이하여야 합니다.")
    if not math.isfinite(guidance) or guidance <= 0:
        raise ValueError("latent 정밀화 guidance_scale은 양의 유한 값이어야 합니다.")
    if not math.isfinite(scale) or not 1.0 <= scale <= 1.5:
        raise ValueError("latent 확대 배율은 1.0~1.5여야 합니다.")
    return LatentRefinementSettings(enabled, steps, strength, guidance, scale)


def prepare_refinement_latents(latents, request, settings, *, vae_scale_factor=8):
    """1차 출력이 외부 RGB가 아닌 정상 latent인지 검사하고 선택적으로 확대한다."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
        raise ValueError("1차 생성 결과가 4차원 latent 텐서가 아닙니다.")
    if latents.shape[0] != 1 or latents.shape[1] != 4 or latents.numel() == 0:
        raise ValueError("1차 latent의 배치 또는 채널 형식이 잘못됐습니다.")
    if not torch.isfinite(latents).all().item():
        raise ValueError("1차 latent에 유한하지 않은 값이 있습니다.")
    expected = (request.height // int(vae_scale_factor), request.width // int(vae_scale_factor))
    if tuple(latents.shape[-2:]) != expected:
        raise ValueError(
            f"1차 latent 크기가 요청과 다릅니다: {tuple(latents.shape[-2:])} != {expected}"
        )
    if settings.latent_scale_factor == 1.0:
        return latents
    return functional.interpolate(
        latents,
        scale_factor=settings.latent_scale_factor,
        mode="bicubic",
        align_corners=False,
    )


def refinement_pipeline_from(pipeline):
    """체크포인트를 다시 올리지 않고 같은 모듈을 공유하는 Img2Img 래퍼를 만든다."""
    cached = getattr(pipeline, "_genai_lab_latent_refinement_pipeline", None)
    if cached is not None:
        return cached
    from diffusers import AutoPipelineForImage2Image

    refined = AutoPipelineForImage2Image.from_pipe(pipeline)
    pipeline._genai_lab_latent_refinement_pipeline = refined
    return refined

