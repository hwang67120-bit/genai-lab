"""승인된 IP-Adapter 참조 강도를 확산 단계별로 전환한다."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReferenceScalePhase:
    end_ratio: float
    end_step: int
    scales: tuple[float, ...]
    scale_factors: tuple[tuple[str, float], ...]


class ReferenceScaleSchedule:
    """한 후보 안에서만 사용하는 상태 있는 Diffusers step-end callback."""

    def __init__(self, names, phases, record):
        self.names = tuple(names)
        self.phases = tuple(phases)
        self.reference_step_schedule_record = record
        self._phase_index = 0

    @property
    def initial_scales(self):
        return list(self.phases[0].scales)

    @property
    def initial_scale(self):
        from genai_lab.reference_order import unmasked_adapter_scale
        return unmasked_adapter_scale(self.phases[0].scales)

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        del timestep
        completed_steps = int(step_index) + 1
        next_index = self._phase_index
        while (next_index < len(self.phases) - 1
               and completed_steps >= self.phases[next_index].end_step):
            next_index += 1
        if next_index != self._phase_index:
            self._phase_index = next_index
            from genai_lab.reference_order import unmasked_adapter_scale
            pipeline.set_ip_adapter_scale(
                unmasked_adapter_scale(self.phases[next_index].scales)
            )
        return callback_kwargs


def _settings(config):
    value = config.get("reference_analysis", {}).get("reference_step_schedule", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("단계별 참조 강도 설정은 객체여야 합니다.")
    return value


def effective_denoising_steps(total_steps, generation_mode, strength=None):
    """Diffusers가 실제로 호출하는 denoising callback 횟수를 계산한다."""
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("전체 추론 스텝 수가 잘못됐습니다.")
    if generation_mode != "image_to_image":
        return total_steps

    try:
        resolved_strength = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_to_image strength가 잘못됐습니다.") from exc
    if not math.isfinite(resolved_strength) or not 0 < resolved_strength <= 1:
        raise ValueError("image_to_image strength는 0보다 크고 1 이하여야 합니다.")
    return max(1, min(int(total_steps * resolved_strength), total_steps))


def build_reference_scale_schedule(config, entries, total_steps):
    """활성 참조 순서와 승인 강도를 기준으로 단계별 스케줄을 만든다."""
    entries = tuple(entries)
    names = tuple(entry.name for entry in entries)
    if not names or len(set(names)) != len(names):
        raise ValueError("단계별 참조 강도에 고유한 참조 순서가 필요합니다.")
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("단계별 참조 강도의 전체 스텝 수가 잘못됐습니다.")

    base_scales = {entry.name: float(entry.scale) for entry in entries}
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for value in base_scales.values()):
        raise ValueError("승인된 기본 참조 강도가 0~1 범위를 벗어났습니다.")

    settings = _settings(config)
    if not settings.get("enabled", False):
        return None, {
            "version": "ip_adapter_step_scale_v1",
            "enabled": False,
            "reference_order": list(names),
            "total_steps": total_steps,
            "base_scales": base_scales,
        }

    raw_phases = settings.get("phases")
    if not isinstance(raw_phases, list) or len(raw_phases) < 2:
        raise ValueError("단계별 참조 강도에는 두 개 이상의 phases가 필요합니다.")

    phases = []
    phase_records = []
    previous_ratio = 0.0
    previous_end_step = 0
    for raw in raw_phases:
        if not isinstance(raw, dict):
            raise ValueError("단계별 참조 강도의 phase 형식이 잘못됐습니다.")
        ratio = float(raw.get("end_ratio", float("nan")))
        factors = raw.get("scale_factors")
        if (not math.isfinite(ratio) or not previous_ratio < ratio <= 1
                or not isinstance(factors, dict)):
            raise ValueError("단계별 참조 강도의 비율 또는 배율이 잘못됐습니다.")
        missing = set(names) - set(factors)
        if missing:
            raise ValueError(f"단계별 참조 배율 누락: {sorted(missing)}")

        resolved = []
        used_factors = []
        for name in names:
            factor = float(factors[name])
            scale = base_scales[name] * factor
            if (not math.isfinite(factor) or factor < 0
                    or not math.isfinite(scale) or not 0 <= scale <= 1):
                raise ValueError(f"{name} 단계별 참조 강도가 0~1 범위를 벗어났습니다.")
            used_factors.append((name, factor))
            resolved.append(scale)

        end_step = max(1, math.ceil(total_steps * ratio))
        if end_step <= previous_end_step:
            raise ValueError("현재 스텝 수에서 실행되지 않는 참조 단계가 있습니다.")
        phase = ReferenceScalePhase(ratio, end_step, tuple(resolved), tuple(used_factors))
        phases.append(phase)
        phase_records.append({
            "end_ratio": ratio,
            "end_step": end_step,
            "scale_factors": dict(used_factors),
            "resolved_scales": dict(zip(names, resolved)),
        })
        previous_ratio = ratio
        previous_end_step = end_step

    if not math.isclose(previous_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("마지막 참조 단계의 end_ratio는 1.0이어야 합니다.")

    record = {
        "version": "ip_adapter_step_scale_v1",
        "enabled": True,
        "reference_order": list(names),
        "total_steps": total_steps,
        "base_scales": base_scales,
        "phases": phase_records,
    }
    return ReferenceScaleSchedule(names, phases, record), record
