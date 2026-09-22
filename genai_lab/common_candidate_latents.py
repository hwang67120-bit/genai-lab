"""한 후보 배치가 공유하는 초기 노이즈와 제한된 후보 변형을 만든다."""

from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True)
class CommonCandidateLatentSettings:
    enabled: bool
    correlation: float
    retry_correlation: float
    retry_start_candidate_index: int | None
    variation_seed_offset: int
    configured_enabled: bool = False
    disabled_reason: str | None = None

    def record(self):
        return {
            "version": "common_candidate_latents_v3",
            "enabled": self.enabled,
            "configured_enabled": self.configured_enabled,
            "disabled_reason": self.disabled_reason,
            "correlation": self.correlation,
            "retry_correlation": self.retry_correlation,
            "retry_start_candidate_index": self.retry_start_candidate_index,
            "variation_seed_offset": self.variation_seed_offset,
            "seed_rule": "base_seed + variation_seed_offset + candidate_index",
            "variance_policy": "rho*base + sqrt(1-rho^2)*variation",
            "generator_device": "cpu",
        }


def resolve_common_candidate_latents(config):
    raw = config.get("common_candidate_latents", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("공통 후보 latent 설정은 객체여야 합니다.")
    configured_enabled = bool(raw.get("enabled", False))
    generation_mode = (config.get("generation", {}) or {}).get("mode")
    disabled_reason = None
    enabled = configured_enabled
    if configured_enabled and generation_mode == "image_to_image":
        enabled = False
        disabled_reason = "image_to_image_must_encode_the_approved_source_image"
    correlation = float(raw.get("correlation", .96))
    retry_correlation = float(raw.get("retry_correlation", correlation))
    candidate_pipeline = config.get("candidate_pipeline", {}) or {}
    retry_start_candidate_index = candidate_pipeline.get("maximum_attempts")
    offset = raw.get("variation_seed_offset", 10000)
    if (not math.isfinite(correlation) or not 0 < correlation < 1
            or not math.isfinite(retry_correlation)
            or not 0 < retry_correlation < 1):
        raise ValueError("공통 후보 latent 상관계수는 0 초과 1 미만이어야 합니다.")
    if (retry_start_candidate_index is not None
            and (type(retry_start_candidate_index) is not int
                 or retry_start_candidate_index < 1)):
        raise ValueError("추가 후보 latent 시작 인덱스를 결정할 수 없습니다.")
    if type(offset) is not int or not 1 <= offset < 2**32:
        raise ValueError("후보 변형 시드 오프셋은 1 이상 2^32 미만 정수여야 합니다.")
    return CommonCandidateLatentSettings(
        enabled, correlation, retry_correlation,
        retry_start_candidate_index, offset,
        configured_enabled, disabled_reason)


def latent_shape(pipeline, request):
    scale = int(getattr(pipeline, "vae_scale_factor", 8))
    unet_config = getattr(getattr(pipeline, "unet", None), "config", None)
    channels = int(getattr(unet_config, "in_channels", 0))
    if scale < 1 or channels < 1 or request.width % scale or request.height % scale:
        raise ValueError("공통 후보 latent 크기를 계산할 수 없습니다.")
    return (1, channels, request.height // scale, request.width // scale)


def create_common_base_latents(pipeline, request, settings):
    import torch
    shape = latent_shape(pipeline, request)
    dtype = pipeline.unet.dtype
    if not getattr(dtype, "is_floating_point", False):
        raise ValueError("공통 후보 latent는 부동소수점 dtype이어야 합니다.")
    generator = torch.Generator(device="cpu").manual_seed(request.seed)
    return torch.randn(shape, generator=generator, dtype=dtype, device="cpu")


def correlated_candidate_latents(base_latents, base_seed, candidate_index, settings, device):
    import torch
    if not settings.enabled:
        raise ValueError("비활성 공통 후보 latent 정책을 실행할 수 없습니다.")
    if type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("후보 latent 인덱스는 0 이상의 정수여야 합니다.")
    if not isinstance(base_latents, torch.Tensor) or base_latents.device.type != "cpu":
        raise ValueError("공통 기준 latent는 CPU tensor여야 합니다.")
    variation_seed = base_seed + settings.variation_seed_offset + candidate_index
    generator = torch.Generator(device="cpu").manual_seed(variation_seed)
    variation = torch.randn(
        tuple(base_latents.shape), generator=generator, dtype=base_latents.dtype, device="cpu")
    retry_candidate = (
        settings.retry_start_candidate_index is not None
        and candidate_index >= settings.retry_start_candidate_index
    )
    rho = (
        settings.retry_correlation if retry_candidate
        else settings.correlation
    )
    candidate = rho * base_latents + math.sqrt(1.0 - rho**2) * variation
    moved = candidate.to(device=device, dtype=base_latents.dtype)
    digest = hashlib.sha256(candidate.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    return moved, {
        "version": "correlated_candidate_latent_v2",
        "base_seed": base_seed,
        "candidate_index": candidate_index,
        "variation_seed": variation_seed,
        "correlation": rho,
        "latent_phase": "retry" if retry_candidate else "initial",
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
        "sha256": digest,
    }


def validate_candidate_latent(arguments, request, policy, candidate_record):
    import torch
    latent = arguments.get("latents")
    if not policy.get("enabled"):
        if latent is not None:
            raise ValueError("승인되지 않은 공통 후보 latent가 연결됐습니다.")
        return
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4 or not torch.isfinite(latent).all().item():
        raise ValueError("공통 후보 latent tensor가 유효하지 않습니다.")
    if not isinstance(candidate_record, dict):
        raise ValueError("후보 latent 생성 기록이 없습니다.")
    expected_index = request.candidate_number - 1
    expected_base_seed = (request.seed - expected_index) % (2**32)
    expected_seed = policy["variation_seed_offset"] + expected_base_seed + expected_index
    retry_start = policy.get("retry_start_candidate_index")
    retry_candidate = retry_start is not None and expected_index >= retry_start
    expected_correlation = (
        policy["retry_correlation"] if retry_candidate
        else policy["correlation"]
    )
    expected_phase = "retry" if retry_candidate else "initial"
    if (candidate_record.get("candidate_index") != expected_index
            or candidate_record.get("base_seed") != expected_base_seed
            or candidate_record.get("variation_seed") != expected_seed
            or candidate_record.get("correlation") != expected_correlation
            or candidate_record.get("latent_phase") != expected_phase
            or candidate_record.get("shape") != list(latent.shape)
            or candidate_record.get("dtype") != str(latent.dtype)):
        raise ValueError("승인된 공통 후보 latent 규칙과 실제 후보 기록이 다릅니다.")
    cpu = latent.detach().to("cpu").contiguous()
    digest = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    if digest != candidate_record.get("sha256"):
        raise ValueError("후보 latent tensor가 생성 기록 이후 변경됐습니다.")
