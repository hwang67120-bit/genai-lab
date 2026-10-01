"""One-pass input defaults, separate from the active product configuration.

Threshold source: outputs/pose-identity-20260927/skeleton-check/rule-values.json.
Normalization source: outputs/e2e-20261001/analyze.py.
C6 source: outputs/face-crop-c5-20260928/methods_c5.md and methods_c6.md.
No model paths, imports, downloads, or global environment changes on import.
"""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class PoseNormalizationSettings:
    width: int = 736
    height: int = 1232
    top_margin: float = 0.20
    bottom_margin: float = 0.10
    side_margin: float = 0.15
    background: int = 128
    # Same resource ceiling as PoseReferenceSettings; bounds the padded canvas.
    maximum_canvas_pixels: int = 40_000_000

    def __post_init__(self):
        if any(type(v) is not int or v < 256 or v % 8 for v in (self.width, self.height)):
            raise ValueError('제어 크기는 256 이상, 8의 배수여야 합니다.')
        if not 0 <= self.background <= 255 or self.maximum_canvas_pixels < 1:
            raise ValueError('정규화 배경/최대 크기 설정 오류')
        for v in (self.top_margin, self.bottom_margin, self.side_margin):
            if not math.isfinite(v) or v < 0:
                raise ValueError('정규화 여백은 유한한 음이 아닌 비율이어야 합니다.')


@dataclass(frozen=True)
class PoseCheckSettings:
    confidence: float = 0.30
    upper_arm: float = 1.181502639381011
    forearm: float = 0.9363656085038689
    thigh: float = 1.564899848145202
    shank: float = 1.3923638730204642
    bilateral_length_ratio: float = 2.0
    collapse_ratio: float = 0.10
    torso_width_ratio: float = 0.10
    overlap_ratio: float = 0.15
    guessed_confidence_upper: float = 0.50
    guessed_count: int = 2
    tag_threshold: float = 0.35

    def __post_init__(self):
        for name, v in vars(self).items():
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f'검사 문턱값 오류: {name}')
        if not self.confidence < self.guessed_confidence_upper <= 1:
            raise ValueError('추측 관절 구간 오류')
        if self.tag_threshold > 1 or type(self.guessed_count) is not int:
            raise ValueError('태그 문턱값/추측 관절 개수 오류')


@dataclass(frozen=True)
class InputPolicies:
    # Measurements do not decide handling; changing these never changes a mask.
    rules: tuple[tuple[str, str], ...] = (
        ('K1', 'reject'), ('K2', 'reject'), ('K3', 'warn'), ('K4', 'warn'),
        ('K5', 'warn'), ('K6', 'warn'), ('K7', 'warn'),
        ('GUESS', 'warn'), ('FACE_DIR', 'warn'),
    )
    high_angle: str = 'warn'  # Decision pending: warn / out_of_scope.

    def __post_init__(self):
        names = [name for name, _ in self.rules]
        expected = {f'K{i}' for i in range(1, 8)} | {'GUESS', 'FACE_DIR'}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError('입력 검사 정책 키가 누락되거나 중복됐습니다.')
        if any(v not in ('reject', 'warn') for _, v in self.rules):
            raise ValueError('검사 정책은 reject/warn이어야 합니다.')
        if self.high_angle not in ('warn', 'out_of_scope'):
            raise ValueError('하이앵글 정책 오류')


@dataclass(frozen=True)
class CharacterInputSettings:
    slim_ratio: float = 0.48
    slim_policy: str = 'ps'  # Decision pending; consumed by stage 3, not here.
    head_model: str = 'head_detect_v2.0_s'
    head_confidence: float = 0.4
    head_iou: float = 0.7
    head_margin: float = 0.10
    arm_band_ratio: float = 0.35
    chin_margin: float = 0.03
    crop_margin: float = 0.05
    face_confidence: float = 0.30
    foreground_threshold: int = 128
    arm_zone_side: float = 0.15
    arm_zone_top: float = 0.25
    arm_zone_bottom: float = 0.15

    def __post_init__(self):
        if self.slim_policy not in ('ps', 'ns', 'off'):
            raise ValueError('마른 체형 정책 오류')
        if not self.head_model:
            raise ValueError('머리 검출 모델이 필요합니다.')
        for name, value in vars(self).items():
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f'캐릭터 입력 설정 오류: {name}')
        if not 0 < self.face_confidence <= 1 or type(self.foreground_threshold) is not int or not 1 <= self.foreground_threshold <= 255:
            raise ValueError('얼굴 신뢰도/전경 문턱값 오류')
        if not 0 < self.slim_ratio or not 0 < self.head_confidence <= 1 or not 0 < self.head_iou <= 1:
            raise ValueError('체형 비율/머리 검출 문턱값 오류')


@dataclass(frozen=True)
class OnePassInputSettings:
    normalization: PoseNormalizationSettings = field(default_factory=PoseNormalizationSettings)
    checks: PoseCheckSettings = field(default_factory=PoseCheckSettings)
    policies: InputPolicies = field(default_factory=InputPolicies)
    character: CharacterInputSettings = field(default_factory=CharacterInputSettings)
