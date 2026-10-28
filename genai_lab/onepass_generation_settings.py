"""사용자가 명시적으로 요청한 1회 생성의 로컬 설정이다."""
from dataclasses import dataclass, field
import math
from pathlib import Path

from genai_lab.onepass_prompt_settings import OnePassPromptSettings


def scheduler_config():
    """확정한 exposure-v2 A.scheduler_config를 사용하며 시험 결과 폴더를 읽지 않는다."""
    return {
        "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
        "beta_schedule": "scaled_linear", "trained_betas": None,
        "prediction_type": "epsilon", "timestep_spacing": "leading",
        "steps_offset": 1, "rescale_betas_zero_snr": False,
        "_class_name": "EulerAncestralDiscreteScheduler",
        "_diffusers_version": "0.26.3", "interpolation_type": "linear",
        "sample_max_value": 1.0, "set_alpha_to_one": False,
        "skip_prk_steps": True,
    }


@dataclass(frozen=True)
class OnePassGenerationSettings:
    # 이미 고정한 토크나이저·모델 자료를 공유하며 별도 루트를 만들지 않는다.
    model_root: Path = field(default_factory=lambda: OnePassPromptSettings().tokenizer_root)
    adapter_root: Path = Path("D:/genai-cache/models/t2i-adapter-openpose-sdxl-1.0")
    ip_root: Path = Path(
        "D:/genai-cache/huggingface/models--h94--IP-Adapter/snapshots/"
        "018e402774aeeddd60609b4ecdb7e298259dc729"
    )
    ip_subfolder: str = "sdxl_models"
    ip_weight_name: str = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    image_encoder_subfolder: str = "models/image_encoder"
    width: int = 736
    height: int = 1232
    steps: int = 28
    guidance_scale: float = 5.5
    adapter_scale: float = 1.2
    adapter_steps: int = 11
    ip_start: int = 11
    ip_scale: float = 0.9
    max_reserved_gib: float = 6.5

    def __post_init__(self):
        if any(type(v) is not int for v in (
            self.width, self.height, self.steps, self.adapter_steps, self.ip_start
        )):
            raise ValueError("크기와 단계는 정수여야 합니다.")
        if (min(self.width, self.height) <= 0 or self.width % 8 or self.height % 8
                or not 0 < self.adapter_steps <= self.steps
                or not 0 < self.ip_start < self.steps):
            raise ValueError("잘못된 이미지 크기 또는 단계 일정입니다.")
        if not all(math.isfinite(v) and v > 0 for v in (
            self.guidance_scale, self.adapter_scale, self.ip_scale, self.max_reserved_gib
        )):
            raise ValueError("생성 강도와 메모리 한도는 양의 유한값이어야 합니다.")

    @property
    def adapter_factor(self):
        # 설치된 Diffusers가 int(steps * factor)를 쓰므로 같은 정수 구간 안쪽 값을 사용한다.
        factor = (self.adapter_steps + 0.25) / self.steps
        if int(self.steps * factor) != self.adapter_steps:
            raise ValueError("어댑터 단계 수를 정확히 표현할 수 없습니다.")
        return min(factor, 1.0)

    def model_record(self):
        def revision(path):
            path = Path(path)
            return path.name if path.parent.name == "snapshots" else None
        return {
            "base": {"id": "cagliostrolab/animagine-xl-3.1",
                     "path": str(self.model_root), "revision": revision(self.model_root)},
            "adapter": {"id": "TencentARC/t2i-adapter-openpose-sdxl-1.0",
                        "path": str(self.adapter_root), "revision": revision(self.adapter_root)},
            "ip_adapter": {"id": "h94/IP-Adapter", "path": str(self.ip_root),
                           "weight": self.ip_subfolder + "/" + self.ip_weight_name,
                           "revision": revision(self.ip_root)},
            "image_encoder": {"path": str(self.ip_root / self.image_encoder_subfolder),
                              "revision": revision(self.ip_root)},
        }
