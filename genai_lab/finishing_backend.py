"""Owned SDXL img2img pipeline, optional verified face reference, no downloads or fallback."""
import inspect
from contextlib import nullcontext
import time
from genai_lab.onepass_generation import (DiffusersOnePassBackend, OnePassCancelled,
    OnePassGenerationError, encode_prompt_plan, validate_local_models)
from genai_lab.onepass_generation_settings import scheduler_config
from genai_lab.finishing_reference import IP_SCALE, observe_face_reference
from genai_lab.finishing_memory import prepare_face_cache, unet_attention
from genai_lab.finishing_offload import configure_offload


class FinishingBackend:
    def __init__(self, settings, *, face_reference=False):
        validate_local_models(settings)
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
        self.torch, self.settings, self.pipe = torch, settings, None
        self.scheduler_type = EulerAncestralDiscreteScheduler
        self.embeds = None
        self.face_reference_enabled = face_reference
        self.face_reference = None
        self.face_cache = None
        self.face_preparation = None
        self.offload = None
        self.offload_record = None
        try:
            if not torch.cuda.is_available():
                raise OnePassGenerationError("CUDA 장치가 없습니다.")
            extra = {}
            if face_reference:
                from transformers import CLIPVisionModelWithProjection
                extra["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
                    str(settings.ip_root / settings.image_encoder_subfolder),
                    torch_dtype=torch.float16, local_files_only=True)
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(str(settings.model_root),
                torch_dtype=torch.float16, use_safetensors=True, local_files_only=True, **extra)
            if face_reference:
                self.pipe.load_ip_adapter(str(settings.ip_root), subfolder=settings.ip_subfolder,
                    weight_name=settings.ip_weight_name, image_encoder_folder=None, local_files_only=True)
                self.pipe.set_ip_adapter_scale(IP_SCALE)
                self.offload_record = {}
                self.offload = configure_offload(self.pipe, torch, self.offload_record)
            else:
                self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_tiling()
        except BaseException as error:
            if self.offload_record is not None:
                error.finishing_offload = self.offload_record
            self.close()
            raise

    def set_face_reference(self, reference):
        if not self.face_reference_enabled:
            raise OnePassGenerationError("얼굴 참조가 꺼진 마무리 파이프라인입니다.")
        with reference.open_image():
            pass
        self.face_reference = reference

    def guard(self, cancelled):
        if cancelled():
            raise OnePassCancelled("마무리를 취소했습니다. 생성 원본과 완료된 중간 결과는 보관됩니다.")
        if self.torch.cuda.max_memory_reserved() / 2**30 > 6.5:
            raise OnePassGenerationError("마무리 reserved 6.5 GiB 초과")

    def prepare(self, prompt, cancelled):
        torch, pipe = self.torch, self.pipe
        torch.cuda.reset_peak_memory_stats()
        self.guard(cancelled)
        encoders = (pipe.text_encoder.to("cuda"), pipe.text_encoder_2.to("cuda"))
        try:
            self.embeds = encode_prompt_plan(prompt, encoders, "cuda", torch.float16)
            self.guard(cancelled)
        finally:
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        if self.face_reference_enabled:
            self.prepare_face_embeddings(cancelled)

    def prepare_face_embeddings(self, cancelled):
        if self.face_cache is not None:
            self.face_cache.verify(self.face_reference)
            return
        if self.face_reference is None:
            raise OnePassGenerationError("확인된 마무리 얼굴 참조가 없습니다.")
        self.face_preparation = {}
        self.face_cache = prepare_face_cache(self.pipe, self.face_reference, self.torch,
            lambda:self.guard(cancelled), self.face_preparation)

    def refine(self, image, seed, cancelled):
        torch, pipe = self.torch, self.pipe
        pipe.scheduler = self.scheduler_type.from_config(scheduler_config())
        torch.cuda.reset_peak_memory_stats()
        self.guard(cancelled)
        calls = [0]
        def after_unet(*_):
            calls[0] += 1
            self.guard(cancelled)
            if calls[0] > 9:
                raise OnePassGenerationError("마무리 img2img 호출 수 초과")
        hook = pipe.unet.register_forward_hook(after_unet)
        parameters = inspect.signature(pipe.__call__).parameters
        callback = {}
        if "callback_on_step_end" in parameters:
            def step(_pipe, _index, _timestep, values):
                self.guard(cancelled)
                return values
            callback["callback_on_step_end"] = step
        elif "callback" in parameters:
            callback.update(callback=lambda *_:self.guard(cancelled), callback_steps=1)
        else:
            hook.remove()
            raise OnePassGenerationError("img2img 취소·메모리 확인 callback을 지원하지 않습니다.")
        started = time.perf_counter()
        face_cache, face_hook = None, None
        ip_calls = []
        self.last_refine = {}
        offload = getattr(self,"offload",None)
        metrics = {}
        if offload is not None:
            offload.begin_stage()
        try:
            extra = {}
            if getattr(self, "face_reference_enabled", False):
                if self.face_reference is None:
                    raise OnePassGenerationError("확인된 마무리 얼굴 참조가 없습니다.")
                face_cache = self.face_cache
                if face_cache is None:
                    raise OnePassGenerationError("얼굴 임베딩 사전 계산이 필요합니다.")
                face_cache.verify(self.face_reference)
                pipe.set_ip_adapter_scale(IP_SCALE)
                def observe(module, args, kwargs):
                    observe_face_reference(module,args,kwargs,ip_calls)
                    if len(ip_calls) == 1:
                        face_cache.verify_delivered(kwargs["added_cond_kwargs"]["image_embeds"])
                face_hook = pipe.unet.register_forward_pre_hook(observe, with_kwargs=True)
                extra["ip_adapter_image_embeds"] = list(face_cache.tensors)
            with unet_attention(pipe.unet) if face_cache is not None else nullcontext():
                result = pipe(**self.embeds, image=image, strength=.35, num_inference_steps=28,
                    guidance_scale=5.5, generator=torch.Generator(device="cuda").manual_seed(seed), **callback, **extra)
            torch.cuda.synchronize()
            self.guard(cancelled)
            if calls[0] != 9:
                raise OnePassGenerationError("마무리 img2img 호출 수 불일치")
            if face_cache is not None and len(ip_calls) != calls[0]:
                raise OnePassGenerationError("마무리 IP 적용 호출 수 불일치")
            if len(result.images) != 1:
                raise OnePassGenerationError("마무리 결과는 한 장이어야 합니다.")
            if offload is not None:
                offload.verify_stage(calls[0])
            metrics = {"seconds":time.perf_counter()-started, "unet_calls":calls[0],
                "max_reserved_bytes":torch.cuda.max_memory_reserved(),
                "max_allocated_bytes":torch.cuda.max_memory_allocated(), "ip_adapter":"loaded" if face_cache is not None else "not_loaded",
                "ip_scale":IP_SCALE if face_cache is not None else 0.0,
                "ip_applied_calls":len(ip_calls), "ip_observations":ip_calls,
                "scheduler_actual_config":dict(pipe.scheduler.config),
                **({"ip_embeddings":face_cache.record["embeddings"],
                    "attention_policy":face_cache.record["attention"]} if face_cache is not None else {}),
                **({"offload":self.offload_record,"ip_device_observations":dict(offload.observations)} if offload else {})}
            return result.images[0], metrics
        finally:
            self.last_refine = {"unet_calls":calls[0], "ip_applied_calls":len(ip_calls),
                "ip_observations":ip_calls, "max_reserved_bytes":torch.cuda.max_memory_reserved(),
                "max_allocated_bytes":torch.cuda.max_memory_allocated(), "seconds":time.perf_counter()-started}
            if offload is not None:
                self.last_refine.update(offload=self.offload_record,ip_device_observations=dict(offload.observations))
            if face_hook is not None: face_hook.remove()
            hook.remove()
            if offload is not None:
                release_started = time.perf_counter()
                try:
                    offload.release()
                except BaseException as error:
                    self.last_refine["offload_release_error"] = str(error)
                    raise
                finally:
                    release_metrics = {"offload_release_seconds":time.perf_counter()-release_started,
                                       "seconds_including_release":time.perf_counter()-started}
                    self.last_refine.update(release_metrics)
                    metrics.update(release_metrics)

    def close(self):
        offload = getattr(self,"offload",None)
        if offload is not None:
            offload.close()
            self.offload = None
        cache = getattr(self, "face_cache", None)
        if cache is not None:
            cache.close()
            self.face_cache = None
        self.embeds = None
        DiffusersOnePassBackend.close(self)
