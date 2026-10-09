"""성별 표시와 독립적인 요청별 체형 값을 관리한다."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping


VERSION = "request_body_morphology_v1"
SAMPLING_MODE = "sha256_per_component_uniform_0_05_0_95"
COMPONENT_NAMES = (
    "shoulder_width",
    "pelvis_width",
    "waist_hip_ratio",
    "chest_volume",
    "muscle_mass",
    "body_fat",
    "jaw_angularity",
    "height_ratio",
    "limb_thickness",
)


@dataclass(frozen=True)
class BodyMorphologyVector:
    """한 생성 요청에서 한 번 추출한 시각적 체형 조건을 변경 없이 유지한다."""

    source_seed: int
    shoulder_width: float
    pelvis_width: float
    waist_hip_ratio: float
    chest_volume: float
    muscle_mass: float
    body_fat: float
    jaw_angularity: float
    height_ratio: float
    limb_thickness: float

    def __post_init__(self) -> None:
        if type(self.source_seed) is not int or not 0 <= self.source_seed < 2**32:
            raise ValueError("체형 벡터 기준 시드는 0 이상 2**32 미만 정수여야 합니다.")
        for name in COMPONENT_NAMES:
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"체형 벡터 {name}은 0~1 유한한 수여야 합니다.")

    def values(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in COMPONENT_NAMES}

    def record(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "sampling_mode": SAMPLING_MODE,
            "source_seed": self.source_seed,
            "values": self.values(),
            "gender_gate_input": False,
            "model_conditioning_applied": False,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "BodyMorphologyVector":
        if record.get("version") != VERSION:
            raise ValueError("지원하지 않는 체형 벡터 버전입니다.")
        values = record.get("values")
        if not isinstance(values, Mapping) or set(values) != set(COMPONENT_NAMES):
            raise ValueError("체형 벡터 구성요소가 일치하지 않습니다.")
        return cls(
            source_seed=int(record["source_seed"]),
            **{name: float(values[name]) for name in COMPONENT_NAMES},
        )


def _component_value(seed: int, name: str) -> float:
    payload = f"{VERSION}:{seed}:{name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return round(0.05 + 0.90 * unit, 6)


def freeze_body_morphology(seed: int) -> BodyMorphologyVector:
    """Python 전역 난수 상태를 바꾸지 않고 재현 가능한 벡터를 만든다."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("체형 벡터 기준 시드는 0 이상 2**32 미만 정수여야 합니다.")
    return BodyMorphologyVector(
        source_seed=seed,
        **{name: _component_value(seed, name) for name in COMPONENT_NAMES},
    )


def request_body_morphology(request: Any) -> BodyMorphologyVector:
    """요청에 확정된 벡터를 반환한다. 구형 객체에만 새로 계산한다."""
    vector = getattr(request, "body_morphology", None)
    if vector is None:
        return freeze_body_morphology(int(request.seed))
    if not isinstance(vector, BodyMorphologyVector):
        raise ValueError("요청 체형 벡터 타입이 올바르지 않습니다.")
    return vector
