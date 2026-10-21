"""생성 가속 선택지(D-141, 시험용 — 제품 기본값 아님). 단계 수를 줄이도록 학습된 LoRA와 그에 맞는 스케줄러·CFG.

lightning8: lightx2v/Qwen-Image-Edit-2511-Lightning 8단계 bf16 LoRA(Apache-2.0, rev 107d238d, 2026-10-21 사용자 승인으로 받음).
설정은 제작자 예제(ModelTC/Qwen-Image-Lightning generate_with_diffusers.py)를 따른다: 8단계, true CFG 1.0(부정 지시문 계산 없음),
FlowMatchEuler shift 3(base/max_shift = ln 3, shift_terminal 없음).
주의: CFG 1.0이면 부정 지시문이 쓰이지 않는다 → D-139(없는 부위를 부정 지시문으로)의 효과가 사라질 수 있다.
"""
import math
from pathlib import Path

LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
    "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
    "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True, "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
ACCELERATORS = {
    "lightning8": dict(
        repo="lightx2v/Qwen-Image-Edit-2511-Lightning", revision="107d238d68b8387e32843d0b4a2e24ceb0d3ad86",
        folder="Qwen-Image-Edit-2511-Lightning", weight_name="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        size=849608296, sha256="a9e81a58a78f260f67b337a6f615e8fa4cd3bc79847c77b7d61a581b789b1ba8",
        steps=8, true_cfg_scale=1.0, scheduler=LIGHTNING_SCHEDULER),
}


def accelerator(name):
    """이름 → 명세. 빈 이름은 None(가속 없음, 검증된 40단계/CFG 4.0)."""
    if not name:
        return None
    if name not in ACCELERATORS:
        raise ValueError("알 수 없는 가속 선택지")
    return ACCELERATORS[name]


def lora_path(name, model_root):
    """가속 LoRA 파일 경로: 기준 모델 폴더와 같은 models 폴더 아래."""
    spec = accelerator(name)
    return Path(model_root).parent / spec["folder"] / spec["weight_name"]


def check_file(name, model_root):
    """크기만 빠르게 확인(SHA는 작업 프로세스가 불러오기 직전에 한 번 대조)."""
    path = lora_path(name, model_root)
    spec = accelerator(name)
    if not path.is_file() or path.stat().st_size != spec["size"]:
        raise ValueError(f"가속 LoRA 파일이 없거나 크기가 다릅니다: {path}")
    return path
