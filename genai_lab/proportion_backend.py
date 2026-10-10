"""검증한 자세·스케치 실행기다. 기준 생성은 기존 단일 어댑터로 유지한다."""
import time
from PIL import Image
from genai_lab.onepass_generation import (
    DiffusersOnePassBackend, OnePassGenerationError, check_pipeline_support,
    encode_prompt_plan, pipeline_callback_kwargs)
from genai_lab.onepass_generation_settings import scheduler_config
from genai_lab.proportion_inputs import checked_image, validate_models, require
from genai_lab.generation_cleanup import GenerationCleanupError, cleanup_steps, release_cuda_cache


class StepGuard:
    def __init__(self, observation, settings, inputs, cuda):
        self.observation, self.settings, self.inputs, self.cuda = observation, settings, inputs, cuda

    def __call__(self, module, args, kwargs, output):
        self.observation(module, args, kwargs, output)
        calls = self.observation.calls
        index, call = len(calls) - 1, calls[-1]
        scale = self.inputs.ip_early if index < self.settings.ip_start else self.settings.ip_scale
        if (index >= self.settings.steps or call["ip_scales"] != [scale]
                or call["adapter_applied"] != (index < self.settings.adapter_steps)):
            raise OnePassGenerationError(f"비율 생성 적용 일정 불일치: {index}")
        if self.cuda.max_memory_reserved() / 2**30 > self.settings.max_reserved_gib:
            raise OnePassGenerationError("비율 생성 reserved 6.5 GiB 초과")


class GuardedBase:
    """변경하지 않은 1단계 실행기의 단계 정보를 읽기 전용으로 검사한다."""
    def __init__(self, backend, settings):
        self.backend, self.settings = backend, settings
        self.pipe, self.callback_mode = backend.pipe, backend.callback_mode
        self.start_memory = getattr(backend, "start_memory", None)

    def generate(self, inputs, images, seed, observation, cancelled):
        guard = StepGuard(observation, self.settings, inputs, self.backend.torch.cuda)
        return self.backend.generate(inputs, images, seed, guard, cancelled)

    def close(self):
        self.pipe = None  # 래퍼가 파이프라인을 붙잡지 않게 먼저 끊는다.
        self.backend.close()


class ProportionBackend(DiffusersOnePassBackend):
    def __init__(self, settings, options, maps):
        self.models = validate_models(settings, options)
        self.settings, self.maps = settings, maps
        self.pipe, self.adapter_handles = None, []
        self.forward_counts = [0, 0]
        import torch
        import diffusers
        from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, MultiAdapter, EulerAncestralDiscreteScheduler
        from transformers import CLIPVisionModelWithProjection
        self.torch = torch
        self.callback_mode = check_pipeline_support(StableDiffusionXLAdapterPipeline)
        self.scheduler_type = EulerAncestralDiscreteScheduler
        self.versions = {"torch": torch.__version__, "diffusers": diffusers.__version__}
        require(torch.cuda.is_available(), "CUDA 장치가 없습니다.")
        self.start_memory = release_cuda_cache(torch)
        try:
            pose = T2IAdapter.from_pretrained(str(settings.adapter_root), torch_dtype=torch.float16,
                                              use_safetensors=True, local_files_only=True)
            sketch = T2IAdapter.from_pretrained(str(options.sketch_root), torch_dtype=torch.float16,
                                                variant="fp16", use_safetensors=True, local_files_only=True)
            adapter = MultiAdapter([pose, sketch])
            encoder = CLIPVisionModelWithProjection.from_pretrained(
                str(settings.ip_root / settings.image_encoder_subfolder), torch_dtype=torch.float16,
                use_safetensors=True, local_files_only=True)
            self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                str(settings.model_root), adapter=adapter, image_encoder=encoder,
                torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
            self.pipe.load_ip_adapter(str(settings.ip_root), subfolder=settings.ip_subfolder,
                                     weight_name=settings.ip_weight_name, image_encoder_folder=None, local_files_only=True)
            self.pipe.set_ip_adapter_scale(0.)
            self.pipe.model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->adapter->unet->vae"
            self.pipe.enable_model_cpu_offload()
            for index, model in enumerate(self.pipe.adapter.adapters):
                def count(_model, _args, _output, index=index):
                    self.forward_counts[index] += 1
                self.adapter_handles.append(model.register_forward_hook(count))
        except BaseException:
            self.close()
            raise

    def record_metadata(self, seed):
        return {"models": {**self.settings.model_record(), "adapters": [self.models["pose"], self.models["sketch"]]},
                "adapter_input": "pose_and_head_sketch", "adapter_strengths": [1.2, .5],
                "shared_adapter_steps": list(range(11)), "sketch": self.maps[seed],
                "proportion_stage": "second_pass",
                "outline_source": self.maps[seed].get("outline_source", "base"),
                "original_sha256": self.maps[seed].get("original_sha256")}

    def generate(self, inputs, images, seed, observation, cancelled):
        torch, pipe, settings = self.torch, self.pipe, self.settings
        entry = self.maps[seed]
        pixels = checked_image(entry["path"], entry["sha256"], (settings.width, settings.height), "RGB")
        contour = Image.fromarray(pixels)
        pipe.scheduler = self.scheduler_type.from_config(scheduler_config())
        pipe.set_ip_adapter_scale(inputs.ip_early)
        guard = StepGuard(observation, settings, inputs, torch.cuda)
        handle = pipe.unet.register_forward_hook(guard, with_kwargs=True)
        self.forward_counts[:] = [0, 0]
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            embeds = encode_prompt_plan(inputs.prompt, (pipe.text_encoder, pipe.text_encoder_2),
                                         pipe._execution_device, pipe.text_encoder_2.dtype)
            result = pipe(**embeds, image=[images[0], contour], ip_adapter_image=images[1],
                          width=settings.width, height=settings.height, num_inference_steps=settings.steps,
                          guidance_scale=settings.guidance_scale, adapter_conditioning_scale=[1.2, .5],
                          adapter_conditioning_factor=settings.adapter_factor,
                          generator=torch.Generator(device="cuda").manual_seed(seed),
                          **pipeline_callback_kwargs(pipe, settings, cancelled, self.callback_mode))
            torch.cuda.synchronize()
            require(len(result.images) == 1, "출력은 한 장이어야 합니다.")
            # 횟수가 틀려도 원본을 반환한다. 호출부가 저장한 뒤 기록하고 거부한다.
            return result.images[0], {
                "generation_seconds": time.perf_counter() - started,
                "timing_scope": "prompt_encoding_and_pipeline", "adapter_forward_counts": list(self.forward_counts),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "versions": self.versions, "scheduler_actual_config": dict(pipe.scheduler.config)}
        finally:
            handle.remove()
            contour.close()

    def close(self):
        handles, self.adapter_handles = getattr(self, "adapter_handles", []), []
        errors = cleanup_steps([(f"어댑터 훅 {index}", handle.remove)
                                for index, handle in enumerate(handles)])
        del handles
        errors.extend(cleanup_steps((("비율 모델 정리", super().close),)))
        if errors:
            raise GenerationCleanupError("비율 모델 정리 실패: " + "; ".join(errors))
