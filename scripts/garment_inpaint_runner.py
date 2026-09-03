"""SDXL Inpaint + IP-Adapter Plus 의상 생성 실행기."""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

from PIL import Image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D garment inpaint runner")
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
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--ip-adapter-scale", type=float, required=True)
    parser.add_argument("--padding-mask-crop", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--progress-file", required=True)
    return parser.parse_args()


def load_image_copy(path: str, mode: str) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert(mode).copy()


def _token_count(tokenizer, text: str) -> int:
    token_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def _tokenizer_limit(tokenizer) -> int:
    configured = int(getattr(tokenizer, "model_max_length", 77))
    return configured if 2 <= configured < 100_000 else 77


def prepare_prompt_for_clip(
    text: str,
    tokenizers: tuple,
) -> tuple[str, dict[str, object]]:
    """쉼표 단위 의미를 보존하며 모든 CLIP 토크나이저 한도에 맞춘다."""
    usable_tokenizers = tuple(item for item in tokenizers if item is not None)
    if not usable_tokenizers:
        raise RuntimeError("SDXL CLIP 토크나이저를 찾을 수 없습니다.")
    segments = tuple(part.strip() for part in text.split(",") if part.strip())
    retained: list[str] = []
    for segment in segments:
        candidate = ", ".join((*retained, segment))
        if all(
            _token_count(tokenizer, candidate) <= _tokenizer_limit(tokenizer)
            for tokenizer in usable_tokenizers
        ):
            retained.append(segment)
        else:
            break
    effective = ", ".join(retained)
    if text.strip() and not effective:
        raise RuntimeError(
            "프롬프트 첫 구문이 CLIP 최대 토큰 길이를 초과합니다."
        )
    limits = tuple(_tokenizer_limit(item) for item in usable_tokenizers)
    original_counts = tuple(_token_count(item, text) for item in usable_tokenizers)
    effective_counts = tuple(
        _token_count(item, effective) for item in usable_tokenizers
    )
    return effective, {
        "original": text,
        "effective": effective,
        "truncated": effective != text.strip(),
        "tokenizer_limits": limits,
        "original_token_counts": original_counts,
        "effective_token_counts": effective_counts,
        "retained_segment_count": len(retained),
        "source_segment_count": len(segments),
    }


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
    from diffusers import StableDiffusionXLInpaintPipeline

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
    initial = load_image_copy(arguments.initial_image, "RGB")
    mask = load_image_copy(arguments.mask_image, "L")
    garment = load_image_copy(arguments.garment_image, "RGB")
    pipeline = None
    generated = None
    result = None
    try:
        for name, image in (("initial", initial), ("mask", mask)):
            if image.size != expected_size:
                raise RuntimeError(
                    f"{name} image size {image.size} != {expected_size}"
                )
        phase_started_at = perf_counter()
        emit_progress(
            progress_path,
            phase="pipeline_loading",
            started_at=started_at,
            phase_started_at=phase_started_at,
            message="started",
        )
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
            message="completed",
        )
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_tiling()
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
                    "adapter_dimensions": adapter_dimensions,
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

        generated = pipeline(
            prompt=effective_prompt,
            negative_prompt=effective_negative_prompt,
            image=initial,
            mask_image=mask,
            ip_adapter_image=garment,
            width=arguments.width,
            height=arguments.height,
            strength=arguments.strength,
            num_inference_steps=arguments.inference_steps,
            guidance_scale=arguments.guidance_scale,
            padding_mask_crop=arguments.padding_mask_crop,
            generator=generator,
            callback_on_step_end=report_diffusion_step,
        ).images[0]
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
        for image in (initial, mask, garment):
            image.close()
        if result is not None:
            result.close()
        if generated is not None:
            generated.close()
        if pipeline is not None:
            del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
