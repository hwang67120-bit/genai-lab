"""선택적 1회 생성의 원본 후보를 만든다. 실행기 생성 전에는 모델을 가져오지 않는다. 4단계 결과는 미검토 후보이며 최종 승인 결과가 아니다. 신·구 단계
종료 콜백은 같은 얼굴 참조 강도표를 사용한다.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import io
import json
from pathlib import Path
import time
from typing import Callable

from PIL import Image

from genai_lab.onepass_generation_settings import OnePassGenerationSettings, scheduler_config
from genai_lab.onepass_prompt import OnePassPrompt, AppearanceOverrides, assemble_onepass_prompt
from genai_lab.onepass_prompt_settings import OnePassPromptSettings
from genai_lab.onepass_pose import InputDecision


class OnePassGenerationError(RuntimeError):
    pass


class OnePassCancelled(OnePassGenerationError):
    pass


@dataclass(frozen=True)
class OnePassInputs:
    prompt: OnePassPrompt
    control_file: Path | None
    face_file: Path
    control_sha256: str | None
    face_sha256: str
    ip_early: float = 0.0
    pose_mode: str = field(kw_only=True)
    input_decision: InputDecision | None = None
    user_choice: str | None = None
    # 하위 수준 N0i 재현에서만 명시적으로 사용한다. 사용자 선택 조립기는 설정하지 않는다.
    diagnostic_ip_override: bool = False

    def __post_init__(self):
        if type(self.ip_early) not in (int, float) or self.ip_early not in (0.0, 0.5):
            raise ValueError("초기 IP scale은 정면 0.0 또는 비정면 0.5여야 합니다.")
        if type(self.diagnostic_ip_override) is not bool:
            raise ValueError("진단 옵션은 bool이어야 합니다.")
        if self.diagnostic_ip_override and (self.user_choice is not None or self.input_decision is not None):
            raise ValueError("N0i 진단 옵션은 제품 사용자 선택 경로에 사용할 수 없습니다.")
        if self.pose_mode not in ("with_pose", "without_pose"):
            raise ValueError("명시적인 자세 모드가 필요합니다.")
        if self.pose_mode == "with_pose" and (self.control_file is None or self.control_sha256 is None):
            raise ValueError("with_pose에는 제어 이미지와 SHA-256이 필요합니다.")
        if self.pose_mode == "without_pose":
            if self.control_file is not None or self.control_sha256 is not None:
                raise ValueError("without_pose에는 제어 이미지를 전달하지 않습니다.")
            if self.ip_early != 0.0 and not self.diagnostic_ip_override:
                raise ValueError("without_pose 초기 IP는 0.0입니다. N0i 재현은 명시적 진단 옵션만 허용합니다.")
        if self.user_choice is not None:
            expected_choice = "proceed_" + self.pose_mode
            if (self.user_choice != expected_choice or self.input_decision is None
                    or self.user_choice not in self.input_decision.choices
                    or (self.input_decision.status == "reject" and self.pose_mode == "with_pose")):
                raise ValueError("입력 확인 결과와 사용자 선택이 일치하지 않습니다.")
        elif self.input_decision is not None:
            raise ValueError("입력 확인 결과에 대한 사용자 선택이 필요합니다.")
        if self.face_file is None:
            raise ValueError("얼굴 참조 파일은 모든 모드에서 필요합니다.")
        hashes = (self.face_sha256,) if self.pose_mode == "without_pose" else (self.control_sha256, self.face_sha256)
        for value in hashes:
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("제어·얼굴 입력의 SHA-256이 필요합니다.")


def prepare_onepass_inputs(*, choice, decision, gender, groups, garment_tags, pose_tags,
                           slim, tokenizers, face_file, face_sha256, control_file=None,
                           control_sha256=None, ip_early=0.0,
                           prompt_settings=OnePassPromptSettings(), appearance=AppearanceOverrides()):
    """명시적인 화면 선택으로 요청을 만든다. 기본 선택이나 문구 재작성은 없다. 자세 없음은 빈 자세 태그로 기존 조립기를 쓰며 방향에서 계산한 초기 얼굴 참조를
    이어받지 않는다. 진단용 원문은 OnePassInputs에 넣고 별도로 기록할 수 있다.
    """
    if choice not in decision.choices or choice not in ('proceed_with_pose', 'proceed_without_pose'):
        raise ValueError('생성 가능한 명시적 사용자 선택이 필요합니다.')
    without = choice == 'proceed_without_pose'
    prompt = assemble_onepass_prompt(gender, groups, garment_tags, () if without else pose_tags,
                                    slim=slim, tokenizers=tokenizers, settings=prompt_settings, appearance=appearance)
    return OnePassInputs(prompt, None if without else control_file, face_file,
                         None if without else control_sha256, face_sha256,
                         0.0 if without else ip_early,
                         pose_mode='without_pose' if without else 'with_pose',
                         input_decision=decision, user_choice=choice)


@dataclass(frozen=True)
class OnePassRequest:
    inputs: OnePassInputs
    seeds: tuple[int, ...]
    settings: OnePassGenerationSettings = field(default_factory=OnePassGenerationSettings)

    def __post_init__(self):
        if len(self.seeds) != 4 or len(set(self.seeds)) != 4:
            raise ValueError("요청에는 서로 다른 seed 4개가 필요합니다.")
        for seed in self.seeds:
            validate_seed(seed)


@dataclass(frozen=True)
class OnePassCandidate:
    seed: int
    path: Path
    record_path: Path
    record: dict


@dataclass(frozen=True)
class OnePassBatch:
    directory: Path
    candidates: tuple[OnePassCandidate, ...]
    review_stage: str = "onepass_raw_unreviewed"
    final_return_eligible: bool = False


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed는 0 이상 2**63 미만 정수여야 합니다.")


def write_record(path, record):
    Path(path).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def check_pipeline_support(pipeline_type):
    parameters = inspect.signature(pipeline_type.__call__).parameters
    required = {
        "ip_adapter_image", "adapter_conditioning_factor",
        "adapter_conditioning_scale", "prompt_embeds", "negative_prompt_embeds",
        "pooled_prompt_embeds", "negative_pooled_prompt_embeds",
    }
    missing = sorted(required - parameters.keys())
    if "callback_on_step_end" in parameters:
        callback_mode = "callback_on_step_end"
    elif {"callback", "callback_steps"} <= parameters.keys():
        callback_mode = "legacy_callback"
    else:
        callback_mode = None
        missing.append("callback_on_step_end 또는 callback + callback_steps")
    if missing:
        raise OnePassGenerationError(
            "설치된 AdapterPipeline 호출 계약 미지원: " + ", ".join(missing)
        )
    return callback_mode


def validate_local_models(settings):
    """모델 라이브러리를 가져오기 전에 필요한 로컬 자료를 확인한다."""
    base = Path(settings.model_root)
    adapter = Path(settings.adapter_root)
    ip = Path(settings.ip_root)
    required = [
        base / "model_index.json", adapter / "config.json",
        adapter / "diffusion_pytorch_model.safetensors",
        ip / settings.ip_subfolder / settings.ip_weight_name,
        ip / settings.image_encoder_subfolder / "config.json",
    ]
    for sub in ("unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
        required.append(base / sub)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("로컬 모델 경로 누락 (다운로드하지 않음): " + ", ".join(missing))


def validate_prompt(prompt):
    if len(prompt.encoders) != 2:
        raise ValueError("텍스트 인코더 두 개의 청크 계획이 필요합니다.")
    counts = [len(chunks) for plan in prompt.encoders
              for chunks in (plan.positive, plan.negative)]
    if not counts[0] or len(set(counts)) != 1:
        raise ValueError("긍정·부정과 두 인코더의 청크 수가 같아야 합니다.")
    for plan in prompt.encoders:
        for chunk in plan.positive + plan.negative:
            if len(chunk.token_ids) != 77 or any(
                type(v) is not int or v < 0 for v in chunk.token_ids
            ):
                raise ValueError("청크마다 유효한 토큰 ID 77개가 필요합니다.")


def read_inputs(inputs, settings):
    """디코딩한 바이트 자체의 해시를 계산해 파일 변경으로 생기는 불일치를 막는다."""
    validate_prompt(inputs.prompt)
    images = []
    try:
        for path, expected in (
            (inputs.control_file, inputs.control_sha256), (inputs.face_file, inputs.face_sha256)
        ):
            if path is None:
                # Diffusers 0.38은 적용 비율이 0이어도 이미지 전처리와 어댑터 특징을 계산한다.
                # 검은 대체 입력에는 참조 자세가 없으며 그 특징은
                # UNet에 전달되지 않는다. 존재하지 않는 자세 파일 해시를 만들지 않는다.
                images.append(Image.new("RGB", (settings.width, settings.height), (0, 0, 0)))
                continue
            data = Path(path).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise OnePassGenerationError(f"입력 이미지 SHA-256 불일치: {path}")
            with Image.open(io.BytesIO(data)) as image:
                if image.mode != "RGB":
                    raise OnePassGenerationError(f"전처리 완료 RGB 이미지가 필요합니다: {path}")
                images.append(image.copy())
        if images[0].size != (settings.width, settings.height):
            raise OnePassGenerationError("제어 이미지 크기가 출력과 다릅니다. 여기서 재변환하지 않습니다.")
        return tuple(images)
    except BaseException:
        for image in images:
            image.close()
        raise


def encode_prompt_plan(prompt, encoders, device, dtype):
    """run_int의 encode_pcs 방식으로 인코딩하며 두 번째 인코더의 요약 출력은 첫 조각을 쓴다."""
    import torch

    validate_prompt(prompt)
    if len(encoders) != 2:
        raise ValueError("텍스트 인코더 두 개가 필요합니다.")
    result = {}
    with torch.no_grad():
        for polarity, sequence_key, pooled_key in (
            ("positive", "prompt_embeds", "pooled_prompt_embeds"),
            ("negative", "negative_prompt_embeds", "negative_pooled_prompt_embeds"),
        ):
            sequence, pooled = [], None
            for index in range(len(getattr(prompt.encoders[0], polarity))):
                parts = []
                for encoder_index, encoder in enumerate(encoders):
                    chunk = getattr(prompt.encoders[encoder_index], polarity)[index]
                    ids = torch.tensor([chunk.token_ids], dtype=torch.long, device=device)
                    output = encoder(ids, output_hidden_states=True)
                    parts.append(output.hidden_states[-2])
                    if index == 0 and encoder_index == 1:
                        pooled = output[0]
                sequence.append(torch.cat(parts, dim=-1))
            if pooled is None or pooled.ndim != 2:
                raise OnePassGenerationError("두 번째 인코더의 pooled 출력이 없습니다.")
            result[sequence_key] = torch.cat(sequence, dim=1).to(device=device, dtype=dtype)
            result[pooled_key] = pooled
    return result


def step_end_callback(settings, cancelled):
    def callback(pipeline, index, _timestep, callback_kwargs):
        if cancelled():
            raise OnePassCancelled("사용자가 생성을 취소했습니다.")
        if index == settings.ip_start - 1:
            pipeline.set_ip_adapter_scale(settings.ip_scale)
        return callback_kwargs
    return callback


def pipeline_callback_kwargs(pipeline, settings, cancelled, callback_mode):
    """구형 잠재값을 바꾸지 않고 같은 단계 종료 동작으로 변환한다."""
    callback = step_end_callback(settings, cancelled)
    if callback_mode == "callback_on_step_end":
        return {"callback_on_step_end": callback}
    if callback_mode == "legacy_callback":
        def legacy_callback(step_idx, timestep, _latents):
            callback(pipeline, step_idx, timestep, {})
        return {"callback": legacy_callback, "callback_steps": 1}
    raise OnePassGenerationError(f"지원하지 않는 callback_mode: {callback_mode}")


class UNetObservation:
    """순전파 뒤 읽기 전용 훅이다. 잔차를 주입하거나 scheduler.step을 감싸지 않는다."""

    def __init__(self, cancelled=lambda: False):
        self.calls = []
        self.cancelled = cancelled

    def __call__(self, module, args, kwargs, _output):
        if self.cancelled():
            raise OnePassCancelled("사용자가 생성을 취소했습니다.")
        scales = []
        for processor in module.attn_processors.values():
            if hasattr(processor, "scale"):
                value = processor.scale
                scales.extend(float(v) for v in (value if isinstance(value, (list, tuple)) else [value]))
        # SDXL은 순전파 중 잔차 목록을 바꾸지만 None과 목록의 구분은 남는다.
        residuals = kwargs.get("down_intrablock_additional_residuals")
        self.calls.append({
            "index": len(self.calls), "ip_scales": sorted(set(scales)),
            "adapter_applied": residuals is not None,
        })
        # None을 반환하면 UNet 결과를 바꾸지 않는다.


def schedule_errors(calls, settings, ip_early, pose_mode="with_pose"):
    errors = []
    if len(calls) != settings.steps:
        errors.append(f"UNet 호출 수 {len(calls)} != {settings.steps}")
    for index, call in enumerate(calls):
        expected = ip_early if index < settings.ip_start else settings.ip_scale
        if call["ip_scales"] != [expected]:
            errors.append(f"UNet {index} IP scale {call['ip_scales']} != {[expected]}")
    actual = [i for i, call in enumerate(calls) if call["adapter_applied"]]
    expected_adapter = [] if pose_mode == "without_pose" else list(range(settings.adapter_steps))
    if actual != expected_adapter:
        errors.append(f"실제 어댑터 적용 단계 불일치: {actual}")
    return errors


class DiffusersOnePassBackend:
    """요청마다 파이프라인 하나를 관리하고 지원되는 단계 종료 콜백을 고른다."""

    def __init__(self, settings):
        validate_local_models(settings)
        from diffusers import StableDiffusionXLAdapterPipeline
        # 모델을 하나라도 로드하기 전에 지원 콜백을 선택한다.
        self.callback_mode = check_pipeline_support(StableDiffusionXLAdapterPipeline)
        import torch
        import diffusers
        from diffusers import T2IAdapter, EulerAncestralDiscreteScheduler
        from transformers import CLIPVisionModelWithProjection

        if not torch.cuda.is_available():
            raise OnePassGenerationError("CUDA 장치가 없습니다.")
        self.torch = torch
        self.scheduler_type = EulerAncestralDiscreteScheduler
        self.settings = settings
        self.versions = {"torch": torch.__version__, "diffusers": diffusers.__version__}
        adapter = T2IAdapter.from_pretrained(
            str(settings.adapter_root), torch_dtype=torch.float16,
            use_safetensors=True, local_files_only=True,
        )
        encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(settings.ip_root / settings.image_encoder_subfolder),
            torch_dtype=torch.float16, local_files_only=True,
        )
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            str(settings.model_root), adapter=adapter, image_encoder=encoder,
            torch_dtype=torch.float16, use_safetensors=True, local_files_only=True,
        )
        try:
            self.pipe.load_ip_adapter(
                str(settings.ip_root), subfolder=settings.ip_subfolder,
                weight_name=settings.ip_weight_name, image_encoder_folder=None,
                local_files_only=True,
            )
            self.pipe.set_ip_adapter_scale(0.0)
            # 현재 인스턴스만 변경한다. 다음 UNet 호출에서 어댑터 가중치를 내린다.
            self.pipe.model_cpu_offload_seq = (
                "text_encoder->text_encoder_2->image_encoder->adapter->unet->vae"
            )
            self.pipe.enable_model_cpu_offload()
        except BaseException:
            self.close()
            raise

    def generate(self, inputs, images, seed, observation, cancelled):
        settings, torch, pipe = self.settings, self.torch, self.pipe
        # 이미지마다 새 스케줄러와 시작 강도를 사용한다. 어댑터도 매 호출 새로 실행한다.
        pipe.scheduler = self.scheduler_type.from_config(scheduler_config())
        pipe.set_ip_adapter_scale(inputs.ip_early)
        handle = pipe.unet.register_forward_hook(observation, with_kwargs=True)
        start = None
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            embeds = encode_prompt_plan(
                inputs.prompt, (pipe.text_encoder, pipe.text_encoder_2),
                pipe._execution_device, pipe.text_encoder_2.dtype,
            )
            result = pipe(
                **embeds, image=images[0], ip_adapter_image=images[1],
                width=settings.width, height=settings.height,
                num_inference_steps=settings.steps, guidance_scale=settings.guidance_scale,
                adapter_conditioning_scale=settings.adapter_scale,
                adapter_conditioning_factor=0.0 if inputs.pose_mode == "without_pose" else settings.adapter_factor,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                **pipeline_callback_kwargs(pipe, settings, cancelled, self.callback_mode),
            )
            torch.cuda.synchronize()
            if len(result.images) != 1:
                raise OnePassGenerationError("한 번의 호출에서 이미지 한 장이어야 합니다.")
            return result.images[0], {
                "generation_seconds": time.perf_counter() - start,
                "timing_scope": "prompt_encoding_and_pipeline",
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "versions": self.versions,
                "scheduler_actual_config": dict(pipe.scheduler.config),
            }
        finally:
            handle.remove()

    def close(self):
        import gc
        pipe = getattr(self, "pipe", None)
        self.pipe = None
        if pipe is not None:
            remove = getattr(pipe, "remove_all_hooks", None)
            if callable(remove):
                remove()
        del pipe
        gc.collect()
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()


def generate_onepass_image(
    backend, inputs, seed, directory, *, settings=OnePassGenerationSettings(),
    cancelled=lambda: False, expected_sha256=None,
):
    """한 번만 시도한다. 검사에 실패해도 원본과 관찰 기록은 보존한다."""
    validate_seed(seed)
    if cancelled():
        raise OnePassCancelled("이미지 실행 전에 취소됐습니다.")
    images = read_inputs(inputs, settings)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    observation = UNetObservation(cancelled)
    record = {
        "version": "onepass_generation_v1", "seed": seed, "completed": False,
        "callback_mode": backend.callback_mode,
        "model_cpu_offload_seq": getattr(backend.pipe, "model_cpu_offload_seq", None),
        "gates_executed": False, "review_stage": "onepass_raw_unreviewed",
        "final_return_eligible": False, "settings": asdict(settings),
        "models": settings.model_record(), "scheduler_requested_config": scheduler_config(),
        "prompt": asdict(inputs.prompt),
        **({"skin_tone": inputs.prompt.rules["skin_tone"]} if "skin_tone" in inputs.prompt.rules else {}),
        "inputs": {"control_file": str(inputs.control_file) if inputs.control_file is not None else None, "face_file": str(inputs.face_file),
                   "control_sha256": inputs.control_sha256, "face_sha256": inputs.face_sha256},
        "pose_mode": inputs.pose_mode, "user_choice": inputs.user_choice,
        "input_decision": asdict(inputs.input_decision) if inputs.input_decision is not None else None,
        "diagnostic_ip_override": inputs.diagnostic_ip_override,
        "adapter_input": "black_placeholder_no_pose" if inputs.pose_mode == "without_pose" else "control_file",
        "ip_early": inputs.ip_early,
        "adapter_factor": 0.0 if inputs.pose_mode == "without_pose" else settings.adapter_factor,
        "expected_sha256": expected_sha256, "valid": False,
    }
    record["settings"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in record["settings"].items()
    }
    metadata = getattr(backend, "record_metadata", None)
    if metadata is not None:
        record.update(metadata(seed))
    record_path = directory / "run.json"
    write_record(record_path, record)
    try:
        image, timing = backend.generate(inputs, images, seed, observation, cancelled)
        record.update(timing)
        raw = directory / "raw.png"
        image.save(raw, format="PNG")
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        errors = schedule_errors(observation.calls, settings, inputs.ip_early, inputs.pose_mode)
        reserved_gib = timing["max_memory_reserved_bytes"] / 2**30
        if reserved_gib > settings.max_reserved_gib:
            errors.append(f"reserved {reserved_gib:.6f} GiB > {settings.max_reserved_gib} GiB")
        if expected_sha256 is not None and digest != expected_sha256:
            errors.append("원시 PNG SHA-256 불일치")
        record.update(
            raw_sha256=digest, max_reserved_gib=reserved_gib, completed=True,
            valid=not errors, validation_errors=errors,
        )
        if errors:
            raise OnePassGenerationError("; ".join(errors))
        return OnePassCandidate(seed, raw, record_path, record)
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["unet_calls"] = observation.calls
        record["adapter_applied_calls"] = [
            i for i, call in enumerate(observation.calls) if call["adapter_applied"]
        ]
        write_record(record_path, record)
        for item in images:
            item.close()


def generate_onepass_request(
    request, directory, *, on_image: Callable[[OnePassCandidate], None] = lambda _image: None,
    cancelled=lambda: False, backend_factory=DiffusersOnePassBackend, proportion=None,
):
    """재시도 없이 네 seed를 생성한다. 기능이 꺼지면 원본 후보의 OnePassBatch를 반환한다. proportion.enabled를 명시하면 원본·제품
    참조를 구분한 ProportionBatch를 반환한다. 검토 연결을 승인하기 전에는 화면에서 이 설정을 사용하지 않는다.
    """
    if not isinstance(request, OnePassRequest):
        raise TypeError("명시적 OnePassRequest가 필요합니다.")
    if proportion is not None:
        from genai_lab.proportion_inputs import ProportionOptions
        if not isinstance(proportion, ProportionOptions):
            raise TypeError("명시적 ProportionOptions가 필요합니다.")
        if proportion.enabled:
            from genai_lab.proportion_generation import generate_proportion_batch
            return generate_proportion_batch(
                request.inputs, request.seeds, directory, settings=request.settings, options=proportion,
                cancelled=cancelled, on_image=on_image, base_factory=backend_factory)
    if cancelled():
        raise OnePassCancelled("요청 시작 전에 취소됐습니다.")
    for image in read_inputs(request.inputs, request.settings):
        image.close()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    backend = None
    candidates = []
    manifest = {"seeds": list(request.seeds), "completed_seeds": [],
                "status": "started", "gates_executed": False, "final_return_eligible": False}
    write_record(directory / "request.json", manifest)
    try:
        backend = backend_factory(request.settings)
        for seed in request.seeds:
            candidate = generate_onepass_image(
                backend, request.inputs, seed, directory / f"seed-{seed}",
                settings=request.settings, cancelled=cancelled,
            )
            candidates.append(candidate)
            manifest["completed_seeds"].append(seed)
            write_record(directory / "request.json", manifest)
            on_image(candidate)
        manifest["status"] = "raw_completed"
        return OnePassBatch(directory, tuple(candidates))
    except BaseException as error:
        manifest.update(status="stopped", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            write_record(directory / "request.json", manifest)
