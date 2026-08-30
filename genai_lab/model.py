"""모델 불러오기와 GPU 배치만 담당한다."""

from pathlib import Path
from typing import Any


def prepare_pipeline(
    config: dict[str, Any],
    pose_control_enabled: bool = False,
):
    """기준 모델, 참조 그림 장치와 선택적인 자세 제어 모델을 준비한다."""
    import torch
    from diffusers import (
        AutoPipelineForImage2Image,
        ControlNetModel,
        StableDiffusionPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )

    model = config["model"]
    generation = config["generation"]
    style = config["style"]
    cache_dir = Path(model["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    family = model["family"]
    pose_control = config.get("pose_control", {})
    if pose_control_enabled:
        if family != "sdxl":
            raise RuntimeError(
                "현재 자세 제어는 SDXL 베이스 모델에서만 사용할 수 있습니다."
            )
        if generation.get("mode", "text_to_image") != "image_to_image":
            raise RuntimeError(
                "현재 자세 제어는 원본 유지 이미지 수정 방식에서만 사용할 수 있습니다."
            )
        if not pose_control.get("enabled", False):
            raise RuntimeError(
                "자세 승인은 완료됐지만 설정 'pose_control.enabled'가 꺼져 있습니다."
            )

    print(f"모델 준비 중: {model['id']}")
    try:
        if pose_control_enabled:
            controlnet_model_id = str(pose_control["model_id"])
            print(f"자세 제어 모델 준비 중: {controlnet_model_id}")
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                torch_dtype=torch.float16,
                cache_dir=str(cache_dir),
                use_safetensors=True,
            )
            pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                model["id"],
                controlnet=controlnet,
                torch_dtype=torch.float16,
                cache_dir=str(cache_dir),
                use_safetensors=True,
            )
        else:
            pipeline_class = (
                StableDiffusionXLPipeline
                if family == "sdxl"
                else StableDiffusionPipeline
            )
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

    if (
        generation.get("mode", "text_to_image") == "image_to_image"
        and not pose_control_enabled
    ):
        print("원본 유지: 기존 모델을 이미지 수정 방식으로 전환")
        pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)

    pipeline._genai_lab_pose_control_enabled = pose_control_enabled

    if family == "sdxl":
        print("GPU 메모리 절약: 사용 중인 모델 부분만 GPU로 이동")
        pipeline.enable_model_cpu_offload()
        return pipeline
    return pipeline.to("cuda")

