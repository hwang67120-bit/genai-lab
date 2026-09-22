"""SDXL Inpaint + IP-Adapter Plus 의상 생성 실행기."""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

from PIL import Image
try:
    from .generation_inputs import resolve_inference_size, prepare_prompt_for_clip
except ImportError:
    from generation_inputs import resolve_inference_size, prepare_prompt_for_clip

try:
    from .inpaint_runtime_probe import RuntimeProbe
except ImportError:
    from inpaint_runtime_probe import RuntimeProbe


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D garment inpaint runner")
    parser.add_argument(
        "--operation",
        choices=("garment_inpaint", "body_restoration"),
        default="garment_inpaint",
    )
    for name in (
        "base-model-id", "adapter-repository",
        "adapter-subfolder", "adapter-weight",
        "adapter-image-encoder-subfolder", "cache-dir", "initial-image",
        "mask-image", "garment-image", "output-image", "prompt",
        "prompt-record-file",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--model-variant", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--inference-width", type=int, default=None)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--ip-adapter-scale", type=float, required=True)
    parser.add_argument("--padding-mask-crop", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--body-pose-control-image")
    parser.add_argument("--body-pose-controlnet-model-id")
    parser.add_argument("--body-pose-conditioning-scale", type=float)
    parser.add_argument("--body-pose-guidance-start", type=float)
    parser.add_argument("--body-pose-guidance-end", type=float)
    return parser.parse_args()




def resize_inference_inputs(initial, mask, pose, size):
    """동일 캔버스에서 RGB/자세는 연속색, 하드 마스크는 최근접으로 변환한다."""
    if mask.size != initial.size or (pose is not None and pose.size != initial.size):
        raise ValueError("초기 이미지·마스크·자세 크기가 다릅니다.")
    return (
        initial.resize(size, Image.Resampling.LANCZOS),
        mask.resize(size, Image.Resampling.NEAREST),
        pose.resize(size, Image.Resampling.BILINEAR) if pose is not None else None,
    )


def load_image_copy(path: str, mode: str) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert(mode).copy()








def validate_ip_adapter_dimensions(pipeline, adapter_weight: str) -> dict[str, int]:
    """Plus 어댑터 투영층과 CLIP hidden size를 추론 전에 대조한다."""
    image_encoder = getattr(pipeline, "image_encoder", None)
    encoder_projection = getattr(
        getattr(pipeline, "unet", None),
        "encoder_hid_proj",
        None,
    )
    projection_layers = getattr(
        encoder_projection,
        "image_projection_layers",
        None,
    )
    if image_encoder is None or not projection_layers:
        raise RuntimeError(
            "IP-Adapter 이미지 인코더 또는 투영층을 확인할 수 없습니다."
        )
    projection_input = getattr(projection_layers[0], "proj_in", None)
    if projection_input is None or not hasattr(projection_input, "in_features"):
        raise RuntimeError("IP-Adapter Plus 입력 차원을 확인할 수 없습니다.")
    actual = int(image_encoder.config.hidden_size)
    expected = int(projection_input.in_features)
    if actual != expected:
        raise RuntimeError(
            "IP-Adapter 이미지 인코더 차원이 맞지 않습니다: "
            f"인코더={actual}, 어댑터={expected}, 가중치={adapter_weight}"
        )
    return {
        "image_encoder_hidden_size": actual,
        "adapter_projection_input_size": expected,
    }


def emit_progress(
    progress_path: Path,
    *,
    phase: str,
    started_at: float,
    phase_started_at: float,
    current_step: int | None = None,
    configured_steps: int | None = None,
    message: str = "",
) -> None:
    """부모 프로세스가 읽을 수 있는 JSONL 진행 이벤트를 즉시 기록한다."""
    now = perf_counter()
    event = {
        "phase": phase,
        "phase_elapsed_seconds": round(now - phase_started_at, 3),
        "total_elapsed_seconds": round(now - started_at, 3),
        "current_step": current_step,
        "configured_steps": configured_steps,
        "message": message,
    }
    with progress_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()


def main() -> None:
    arguments = parse_arguments()
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLInpaintPipeline,
    )

    started_at = perf_counter()
    progress_path = Path(arguments.progress_file)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    emit_progress(
        progress_path,
        phase="runner_started",
        started_at=started_at,
        phase_started_at=started_at,
        message="started",
    )
    dtype = torch.float16 if arguments.dtype == "float16" else torch.bfloat16
    expected_size = (arguments.width, arguments.height)
    inference_size = resolve_inference_size(expected_size, arguments.inference_width)
    initial = load_image_copy(arguments.initial_image, "RGB")
    mask = load_image_copy(arguments.mask_image, "L")
    garment = load_image_copy(arguments.garment_image, "RGB")
    body_pose = None
    body_pose_arguments = (
        arguments.body_pose_control_image,
        arguments.body_pose_controlnet_model_id,
        arguments.body_pose_conditioning_scale,
        arguments.body_pose_guidance_start,
        arguments.body_pose_guidance_end,
    )
    if arguments.operation == "body_restoration":
        if any(value is None for value in body_pose_arguments):
            raise RuntimeError(
                "body_restoration에는 DWPose ControlNet 인자 5개가 모두 필요합니다."
            )
        body_pose = load_image_copy(arguments.body_pose_control_image, "RGB")
    elif any(value is not None for value in body_pose_arguments):
        raise RuntimeError(
            "garment_inpaint에는 신체 복원용 DWPose 인자를 전달할 수 없습니다."
        )
    pipeline = None
    controlnet = None
    generated = None
    result = None
    probe = RuntimeProbe(progress_path.with_name("runtime_trace.jsonl"), torch)
    try:
        probe.wrap(ControlNetModel, "from_pretrained", "controlnet.load")
        probe.wrap(StableDiffusionXLControlNetInpaintPipeline, "from_pretrained", "controlnet_inpaint.load")
        probe.wrap(StableDiffusionXLInpaintPipeline, "from_pretrained", "inpaint.load")
        size_inputs = [("initial", initial), ("mask", mask)]
        if body_pose is not None:
            size_inputs.append(("body_pose", body_pose))
        for name, image in size_inputs:
            if image.size != expected_size:
                raise RuntimeError(
                    f"{name} image size {image.size} != {expected_size}"
                )
        if inference_size != expected_size:
            resized = resize_inference_inputs(initial, mask, body_pose, inference_size)
            for old_image in (initial, mask, body_pose):
                if old_image is not None:
                    old_image.close()
            initial, mask, body_pose = resized
            if not mask.getbbox():
                raise ValueError("축소 후 제거 마스크가 비었습니다. 더 큰 해상도를 선택하세요.")
        phase_started_at = perf_counter()
        emit_progress(
            progress_path,
            phase="pipeline_loading",
            started_at=started_at,
            phase_started_at=phase_started_at,
            message="started",
        )
        if arguments.operation == "body_restoration":
            controlnet = ControlNetModel.from_pretrained(
                arguments.body_pose_controlnet_model_id,
                torch_dtype=dtype,
                cache_dir=arguments.cache_dir,
                use_safetensors=True,
            )
            pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                arguments.base_model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                variant=arguments.model_variant,
                cache_dir=arguments.cache_dir,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                arguments.base_model_id,
                torch_dtype=dtype,
                variant=arguments.model_variant,
                cache_dir=arguments.cache_dir,
                use_safetensors=True,
            )
        emit_progress(
            progress_path,
            phase="pipeline_loading",
            started_at=started_at,
            phase_started_at=phase_started_at,
            message="completed",
        )
        phase_started_at = perf_counter()
        emit_progress(
            progress_path,
            phase="ip_adapter_loading",
            started_at=started_at,
            phase_started_at=phase_started_at,
            message="started",
        )
        adapter_dimensions = None
        if arguments.operation == "garment_inpaint":
            pipeline.load_ip_adapter(
                arguments.adapter_repository,
                subfolder=arguments.adapter_subfolder,
                weight_name=arguments.adapter_weight,
                image_encoder_folder=arguments.adapter_image_encoder_subfolder,
                cache_dir=arguments.cache_dir,
            )
            adapter_dimensions = validate_ip_adapter_dimensions(
                pipeline,
                arguments.adapter_weight,
            )
            pipeline.set_ip_adapter_scale(arguments.ip_adapter_scale)
        emit_progress(
            progress_path,
            phase="ip_adapter_loading",
            started_at=started_at,
            phase_started_at=phase_started_at,
            message=(
                "completed"
                if arguments.operation == "garment_inpaint"
                else "skipped_for_body_restoration"
            ),
        )
        probe.wrap(pipeline, "enable_model_cpu_offload", "pipeline.install_cpu_offload")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_tiling()
        probe.attach_pipeline(pipeline)
        tokenizers = (pipeline.tokenizer, pipeline.tokenizer_2)
        effective_prompt, prompt_record = prepare_prompt_for_clip(
            arguments.prompt,
            tokenizers,
        )
        effective_negative_prompt, negative_prompt_record = (
            prepare_prompt_for_clip(arguments.negative_prompt, tokenizers)
        )
        prompt_record_path = Path(arguments.prompt_record_file)
        prompt_record_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_record_path.write_text(
            json.dumps(
                {
                    "operation": arguments.operation,
                    "adapter_dimensions": adapter_dimensions,
                    "body_pose_controlnet": (
                        {
                            "model_id": arguments.body_pose_controlnet_model_id,
                            "conditioning_scale": (
                                arguments.body_pose_conditioning_scale
                            ),
                            "guidance_start": arguments.body_pose_guidance_start,
                            "guidance_end": arguments.body_pose_guidance_end,
                        }
                        if body_pose is not None
                        else None
                    ),
                    "inference_size": inference_size,
                    "output_size": expected_size,
                    "prompt": prompt_record,
                    "negative_prompt": negative_prompt_record,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        generator = torch.Generator(device="cpu").manual_seed(arguments.seed)
        diffusion_started_at = perf_counter()
        callback_count = 0
        emit_progress(
            progress_path,
            phase="diffusion_running",
            started_at=started_at,
            phase_started_at=diffusion_started_at,
            current_step=0,
            configured_steps=arguments.inference_steps,
            message="started",
        )

        def report_diffusion_step(
            pipe,
            step_index: int,
            timestep,
            callback_kwargs: dict,
        ) -> dict:
            del pipe, step_index, timestep
            nonlocal callback_count
            callback_count += 1
            emit_progress(
                progress_path,
                phase="diffusion_running",
                started_at=started_at,
                phase_started_at=diffusion_started_at,
                current_step=callback_count,
                configured_steps=arguments.inference_steps,
                message="step_completed",
            )
            return callback_kwargs

        generation_arguments = dict(
            prompt=effective_prompt,
            negative_prompt=effective_negative_prompt,
            image=initial,
            mask_image=mask,
            width=inference_size[0],
            height=inference_size[1],
            strength=arguments.strength,
            num_inference_steps=arguments.inference_steps,
            guidance_scale=arguments.guidance_scale,
            padding_mask_crop=(arguments.padding_mask_crop if inference_size == expected_size else None),
            generator=generator,
            callback_on_step_end=report_diffusion_step,
        )
        if arguments.operation == "garment_inpaint":
            generation_arguments["ip_adapter_image"] = garment
        else:
            generation_arguments.update({
                "control_image": body_pose,
                "controlnet_conditioning_scale": (
                    arguments.body_pose_conditioning_scale
                ),
                "control_guidance_start": arguments.body_pose_guidance_start,
                "control_guidance_end": arguments.body_pose_guidance_end,
            })
        with probe.measure("pipeline.generate"):
            generated = pipeline(**generation_arguments).images[0]
        emit_progress(
            progress_path,
            phase="diffusion_running",
            started_at=started_at,
            phase_started_at=diffusion_started_at,
            current_step=callback_count,
            configured_steps=arguments.inference_steps,
            message="completed",
        )
        phase_started_at = perf_counter()
        emit_progress(
            progress_path,
            phase="output_saving",
            started_at=started_at,
            phase_started_at=phase_started_at,
            message="started",
        )
        result = generated.convert("RGB")
        if result.size != inference_size:
            raise RuntimeError(f"생성 결과 크기 {result.size} != 연산 크기 {inference_size}")
        if result.size != expected_size:
            restored_size = result.resize(expected_size, Image.Resampling.LANCZOS)
            result.close()
            result = restored_size
        output_path = Path(arguments.output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        emit_progress(
            progress_path,
            phase="output_saving",
            started_at=started_at,
            phase_started_at=phase_started_at,
            message="completed",
        )
        emit_progress(
            progress_path,
            phase="completed",
            started_at=started_at,
            phase_started_at=started_at,
            current_step=callback_count,
            configured_steps=arguments.inference_steps,
            message="completed",
        )
    finally:
        probe.detach()
        probe.emit("start", "cleanup")
        for image in (initial, mask, garment, body_pose):
            if image is None:
                continue
            image.close()
        if result is not None:
            result.close()
        if generated is not None:
            generated.close()
        if pipeline is not None:
            del pipeline
        if controlnet is not None:
            del controlnet
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        probe.emit("completed", "cleanup")
        probe.close()


if __name__ == "__main__":
    main()
