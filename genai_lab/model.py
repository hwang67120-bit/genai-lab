"""모델 불러오기와 GPU 배치만 담당한다."""

from pathlib import Path
from typing import Any


def prepare_pipeline(config: dict[str, Any]):
    """기준 모델과 선택적인 참조 그림 장치를 GPU에 준비한다."""
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    model = config["model"]
    style = config["style"]
    cache_dir = Path(model["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"모델 준비 중: {model['id']}")
    family = model["family"]
    pipeline_class = (
        StableDiffusionXLPipeline if family == "sdxl" else StableDiffusionPipeline
    )
    try:
        pipeline = pipeline_class.from_pretrained(
            model["id"],
            torch_dtype=torch.float16,
            cache_dir=str(cache_dir),
            use_safetensors=True,
        )
    except OSError as error:
        raise RuntimeError(
            "모델을 내려받거나 캐시에서 불러오지 못했습니다. "
            "인터넷 연결과 D:\\genai-cache의 남은 공간을 확인하세요.\n"
            f"원인: {error}"
        ) from error

    if style["enabled"]:
        print("참조 그림 특징 전달 장치(IP-Adapter) 연결 중")
        pipeline.load_ip_adapter(
            style["adapter_repository"],
            subfolder=style["adapter_subfolder"],
            weight_name=style["adapter_weight"],
            cache_dir=str(cache_dir),
        )
        pipeline.set_ip_adapter_scale(float(style["scale"]))

    if family == "sdxl":
        print("GPU 메모리 절약: 사용 중인 모델 부분만 GPU로 이동")
        pipeline.enable_model_cpu_offload()
        return pipeline
    return pipeline.to("cuda")

