"""별도 Qwen 환경의 진입점이다. torch를 가져오기 전에 패키지를 검사한다. 근거: diffusers v0.38.0
pipeline_qwenimage_edit_plus.py와 승인 B 실행기. CPU 단위 테스트에서 가져와도 안전하며 main()만 추론한다.
"""
import gc
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

from genai_lab.qwen_preservation import PreservationSpec, file_sha, json_sha
from genai_lab.qwen_pose_prompt import PoseEditInstructions, assemble_pose_edit_prompt
from genai_lab.qwen_pose_settings import QwenPoseSettings, check_runtime_versions, output_dimensions, validate_model_files
from genai_lab.qwen_record_io import write_json, publish_progress, write_terminal


class WorkerCancelled(RuntimeError):
    pass


SESSION_KINDS = ("free_generation",)
SESSION_IDLE_SECONDS = 1800          # 다음 작업이 30분 안에 안 오면 끝낸다(부모가 사라진 경우 대비 — stdin 감시와 별개)


def reuse_key(request):
    """세션 재사용 조건(D-180): 요청에서 seed만 뺀 내용(지시문·입력 그림 경로·SHA·설정·메모리 정책·크기)이 같아야 한다."""
    return json_sha({k: v for k, v in request.items() if k != "seed"})


def session_reusable(request):
    """세션은 제품 기본 경로만: 자유 생성, 가속 LoRA·그림체 LoRA 없음(어댑터를 붙인 파이프라인은 넘기지 않는다)."""
    spec = request.get("free_spec") or {}
    return request.get("task_kind") in SESSION_KINDS and not spec.get("accelerator") and not spec.get("style_lora")


class SessionCache:
    """한 작업 프로세스 안에서 다음 요청에 넘기는 것과 넘기지 않는 것(D-180).

    넘김: 불러온 파이프라인(GGUF 변환기·VAE, 블록 오프로드 설정), 문장 임베딩(CPU), 첫 작업의 모델 파일 검사 기록, 처리기 입력 기록.
    넘기지 않음(요청마다 새로): seed와 생성기, 입력 그림(다시 읽고 SHA 확인), model_inputs 폴더, 전처리 감시 함수, 메모리 감시·상한,
    단계 콜백·미리보기·진행 기록, 단계 시간, 스케줄러 상태(파이프라인 호출이 단계를 새로 정함).
    작업이 끝나면 VAE를 CPU로 내리고 GPU 캐시를 비운다. 작업이 실패·취소되면 넘길 것을 모두 버리고 프로세스가 끝난다."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.key = self.pipe = self.tr = self.embeddings = self.models = None
        self.processor_inputs = self.embedding_shapes = self.cond_sha = None
        self.vae_tiling = False
        self.first_job = None
        self.job = 0

    def ready(self):
        return self.pipe is not None

    def models_record(self):
        return dict(self.models, files_rechecked=False, reused_from_job=self.first_job,
                    note="같은 프로세스가 첫 작업에서 검사한 파일로 불러 둔 가중치를 그대로 쓴다 — 파일을 다시 읽지 않으므로 다시 검사하지 않는다")

    def keep(self, request, pipe, tr, embeddings, record, model_inputs):
        if self.first_job is None:
            self.first_job = self.job
            self.models = dict(record["models"])
            self.processor_inputs = [dict(row) for row in record.get("processor_inputs", [])]
            self.embedding_shapes = dict(record["embedding_shapes"])
            self.cond_sha = cond_sha(model_inputs)
        self.key, self.pipe, self.tr, self.embeddings = reuse_key(request), pipe, tr, embeddings
        self.vae_tiling = bool(record.get("vae_tiling"))
        pipe.vae.to("cpu")                # 다음 작업 전까지 다른 GPU 작업(결과 분석의 분할)과 겹치지 않게 GPU에서 내린다


def cond_sha(model_inputs):
    return {p.name: file_sha(p) for p in sorted(Path(model_inputs).glob("vl_picture*.png"))}


def check_session(cache, request):
    """세션 작업이면 앞 작업의 파이프라인·임베딩을 쓸 수 있는지. 조건이 다르면 멈춘다(실행기는 이런 요청을 세션에 보내지 않는다)."""
    if cache is None:
        return False
    if not session_reusable(request):
        raise RuntimeError("세션 대상이 아닌 요청(가속·그림체 LoRA 또는 다른 편집 종류)")
    if not cache.ready():
        return False
    if cache.key != reuse_key(request):
        raise RuntimeError("세션 재사용 조건 불일치 — seed 밖의 내용이 앞 작업과 다릅니다")
    return True


class UntruncatedProcessor:
    """텍스트나 토큰 번호를 바꾸지 않고 실제 처리기 입력을 관찰한다."""
    def __init__(self, processor, record):
        self.processor, self.record = processor, record

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        if kwargs.get("truncation"):
            raise ValueError("Qwen 프롬프트 자동 절단 금지")
        result = self.processor(*args, **kwargs)
        ids = result["input_ids"]
        self.record.setdefault("processor_inputs", []).append({
            "sequence_length": int(ids.shape[-1]), "text": kwargs.get("text"),
            "truncation": False})
        return result


def validate_request(request):
    kind = request.get("task_kind", "pose_edit")
    if kind == "tail_edit":
        from genai_lab.qwen_tail_edit import validate_tail_request
        return validate_tail_request(request)
    if kind == "outfit_edit":
        from genai_lab.qwen_outfit_edit import validate_outfit_request
        return validate_outfit_request(request)
    if kind == "free_generation":          # D-123: 기준 이미지 + 머리 크롭, 편집과 같은 메모리 정책
        from genai_lab.qwen_free_generation import validate_free_request
        return validate_free_request(request)
    if kind == "style_pass":               # D-161 그림체 후처리 한 번(시험 전용): 결과 그림 1장 + 트리거 문장
        from genai_lab.style_pass import validate_style_request
        return validate_style_request(request)
    if kind != "pose_edit":
        raise ValueError("지원하지 않는 편집 종류")
    if request.get("schema_version") != 1 or request.get("images") != 2:
        raise ValueError("B 조건은 기준 이미지와 골격 2장만 지원합니다.")
    if request.get("skeleton_color_order") != "openpose_rgb_before_t2i_reversal":
        raise ValueError("Qwen 반전 전 골격 계약 오류")
    spec = PreservationSpec.from_record(request["spec"])
    spec.verify_image()
    prompt = assemble_pose_edit_prompt(spec, PoseEditInstructions(**request["instructions"]))
    if prompt != request["prompt"]:
        raise ValueError("확인된 보존 명세와 전달 프롬프트가 다릅니다.")
    if file_sha(request["skeleton_path"]) != request["skeleton_sha256"]:
        raise ValueError("골격 SHA 불일치")
    seed = request["seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed 형식 오류")
    return spec


def generate(request, directory, record, cache=None):
    # torch·diffusers·bitsandbytes를 가져오기 전에 검사하도록 의도한 순서다.
    record["versions"] = check_runtime_versions()
    import importlib.metadata
    record["runtime"] = {"python": sys.executable, "packages": {
        name: str(importlib.metadata.distribution(name).locate_file("")) for name in record["versions"]}}
    settings = QwenPoseSettings(**request["settings"])
    spec = validate_request(request)
    reuse = check_session(cache, request)           # 세션(D-180): 앞 작업과 seed만 다른 요청이면 불러 둔 것을 쓴다
    record["models"] = cache.models_record() if reuse else validate_model_files(settings)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen CUDA 장치가 없습니다.")
    images, p1, te, pipe, tr = [], None, None, None, None
    outfit = request.get("task_kind") in ("outfit_edit", "free_generation", "style_pass")
    guard = None
    if outfit:  # D-094: 6.5GiB 할당 상한과 단계별 기록(시험 09~11 구성). 자세·꼬리 경로는 바꾸지 않는다.
        from genai_lab.outfit_edit import MemoryGuard
        guard = MemoryGuard(torch.cuda, request["memory"]["reserved_limit_gib"], directory / "memory.json")
        record["memory_cap_fraction"] = guard.apply_cap()
        torch.cuda.reset_peak_memory_stats()
    if cache is not None:       # 세션: 이 작업 시작 때 이 프로세스가 GPU에 남긴 것(앞 작업 뒤 정리 확인)
        record["session_memory_at_start"] = {"allocated_bytes": int(torch.cuda.memory_allocated()),
                                             "reserved_bytes": int(torch.cuda.memory_reserved())}
    def stage(name):
        if guard is not None:
            guard.check(name)
    def begin(name):
        if guard is not None:
            guard.begin(name)
    pe = pm = ne = nm = None
    original_preprocess = None
    completed = False
    start = time.perf_counter()
    def save(): publish_progress(directory, record)
    def check():
        if (directory / "cancel.request").exists():
            raise WorkerCancelled("사용자 취소")
    try:
        reference = (None if request.get("task_kind") == "style_pass"      # D-161 후처리: 결과 그림 1장만
                     else (spec.crop_path, spec.crop_sha256) if request.get("task_kind") == "tail_edit"
                     else (spec.reference_path, spec.reference_sha256) if request.get("task_kind") == "free_generation"
                     else (spec.garment_path, spec.garment_sha256) if outfit
                     else (request["skeleton_path"], request["skeleton_sha256"]))
        record["task_kind"] = request.get("task_kind", "pose_edit")
        sources = [(spec.image_path, spec.image_sha256)] + ([reference] if reference else [])
        if request.get("task_kind") == "free_generation" and getattr(spec, "skeleton_path", ""):   # D-131 자세 골격(image 3)
            sources.append((spec.skeleton_path, spec.skeleton_sha256))
        for path, expected in sources:
            data = Path(path).read_bytes()
            import hashlib
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("이미지 읽기 중 해시 변경")
            with Image.open(io.BytesIO(data)) as source:
                images.append(source.convert("RGB"))
        width, height = calculate_dimensions(1024 * 1024, images[0].width / images[0].height)
        if [width, height] != request["output_size"] or (width, height) != output_dimensions(*images[0].size):
            raise ValueError("출력 크기 계약 불일치")
        mi = directory / "model_inputs"
        mi.mkdir()
        (mi / "prompt.txt").write_text(request["prompt"]["positive"], encoding="utf-8")
        record.update(output_size=[width, height], input_sizes=[list(i.size) for i in images],
                      gpu_name=torch.cuda.get_device_name(0),
                      total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                      memory_note="PyTorch allocated/reserved only; dedicated/shared sampling unavailable",
                      dedicated_memory=None, shared_memory=None, stage_seconds={}, steps=[])
        # 텍스트 인코더·변환기 단계는 run_identity.py와 같으며 대체 메모리 정책은 없다.
        check()
        if reuse:
            cond = []
            for index, image in enumerate(images, 1):
                cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
                cache.pipe.image_processor.resize(image, ch, cw).save(mi / f"vl_picture{index}_{cw}x{ch}.png")
            if cond_sha(mi) != cache.cond_sha:
                raise RuntimeError("세션 조건 그림이 앞 작업과 다릅니다")
            pe, pm, ne, nm = cache.embeddings
            record["processor_inputs"] = [dict(row) for row in cache.processor_inputs]
            record["embedding_shapes"] = dict(cache.embedding_shapes)
            record["text_encode_reused"] = {"from_job": cache.first_job, "note": "지시문·입력 그림이 같아 첫 작업의 임베딩을 그대로 씀"}
            begin("encoded_and_released")
            stage("encoded_and_released")
        else:
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            begin("text_encoder_load")
            if outfit:  # lm_head(1.02GiB)는 문장 인코딩에 쓰지 않으므로 CPU에 두고 계산하지 않는다(D-090, 시험 10)
                from genai_lab.outfit_edit import text_encoder_device_map
                te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(Path(settings.model_root) / "text_encoder"), torch_dtype=torch.bfloat16,
                    local_files_only=True, device_map=text_encoder_device_map(),
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"],
                        llm_int8_enable_fp32_cpu_offload=True))
                te.lm_head = torch.nn.Identity()
                gc.collect()
            else:
                te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(Path(settings.model_root) / "text_encoder"), torch_dtype=torch.bfloat16,
                    local_files_only=True, device_map={"": 0},
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"]))
            stage("text_encoder_loaded")
            p1 = QwenImageEditPlusPipeline.from_pretrained(settings.model_root, transformer=None, vae=None,
                text_encoder=te, torch_dtype=torch.bfloat16, local_files_only=True)
            p1.processor = UntruncatedProcessor(p1.processor, record)
            cond = []
            for index, image in enumerate(images, 1):
                cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
                c = p1.image_processor.resize(image, ch, cw)
                c.save(mi / f"vl_picture{index}_{cw}x{ch}.png")
                cond.append(c)
            begin("encode")
            with torch.no_grad():
                pe, pm = p1.encode_prompt(prompt=request["prompt"]["positive"], image=cond, device=torch.device("cuda"))
                ne, nm = p1.encode_prompt(prompt=request["prompt"].get("negative", " "), image=cond, device=torch.device("cuda"))
            pe, pm, ne, nm = (v.to("cpu") if v is not None else None for v in (pe, pm, ne, nm))
            record["embedding_shapes"] = {"positive": list(pe.shape), "negative": list(ne.shape)}
            observed = record["processor_inputs"]
            # 설치된 파이프라인은 본문이 아닌 고정 시스템 시작 문구 64개 토큰을 제거한다.
            start_index = p1.prompt_template_encode_start_idx
            if any(int(emb.shape[1]) != row["sequence_length"] - start_index
                   for emb, row in zip((pe, ne), observed)) or len(observed) != 2:
                raise RuntimeError("실제 인코딩 길이가 절단 없는 전처리 길이와 다릅니다.")
            record["stage_seconds"]["text_encode"] = time.perf_counter() - tick
            record["text_encode_memory"] = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                                           "reserved_bytes": torch.cuda.max_memory_reserved()}
            del te, p1
            te = p1 = None
            gc.collect()
            torch.cuda.empty_cache()
            record["text_encoder_released"] = True
            stage("encoded_and_released")
        save()
        check()
        steps, true_cfg = settings.steps, settings.true_cfg_scale       # 가속 LoRA가 있으면 아래에서 바꾼다
        if reuse:
            pipe, tr = cache.pipe, cache.tr
            begin("transformer_load")
            pipe.vae.to("cuda")
            accel_name = ""
            if cache.vae_tiling:
                record["vae_tiling"] = True
            record["pipeline_reused"] = {"from_job": cache.first_job, "note": "불러 둔 변환기·VAE·블록 오프로드를 그대로 씀"}
            stage("transformer_loaded")
        else:
            tick = time.perf_counter()
            begin("transformer_load")
            tr = QwenImageTransformer2DModel.from_single_file(settings.gguf_file,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16,
                config=settings.model_root, subfolder="transformer", local_files_only=True)
            if getattr(tr.config, "zero_cond_t", None) is not True:
                raise RuntimeError("zero_cond_t=True 계약 불일치")
            pipe = QwenImageEditPlusPipeline.from_pretrained(settings.model_root, transformer=tr,
                text_encoder=None, tokenizer=None, processor=None, torch_dtype=torch.bfloat16, local_files_only=True)
            accel_name = getattr(spec, "accelerator", "") if request.get("task_kind") == "free_generation" else ""
            if accel_name:      # D-141 시험용 가속: 오프로드 전에 LoRA를 붙이고 스케줄러·단계·CFG를 제작자 설정으로
                from diffusers import FlowMatchEulerDiscreteScheduler
                from genai_lab.qwen_accelerators import accelerator, lora_path
                accel = accelerator(accel_name)
                path = lora_path(accel_name, settings.model_root)
                if file_sha(str(path)) != accel["sha256"]:
                    raise ValueError("가속 LoRA SHA 불일치")
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(accel["scheduler"])
                pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=accel_name)
                steps, true_cfg = accel["steps"], accel["true_cfg_scale"]
                record["accelerator"] = dict(name=accel_name, repo=accel["repo"], revision=accel["revision"], lora=str(path),
                                             sha256=accel["sha256"], steps=steps, true_cfg_scale=true_cfg,
                                             note="시험용 가속(D-141). CFG 1.0이면 부정 지시문은 쓰이지 않는다")
            style_name = (getattr(spec, "style_lora", "") if request.get("task_kind") == "free_generation"
                          else spec.lora_name if request.get("task_kind") == "style_pass" else "")
            if style_name:      # D-159 그림체 LoRA 맛보기(시험 전용): 오프로드 전에 붙이고 가속 LoRA와 함께면 강도를 나눠 건다
                from genai_lab.style_loras import check_strength, lora_path as style_path, style_lora
                style = style_lora(style_name)
                path = style_path(style_name, settings.model_root)
                if not style["sha256"] or file_sha(str(path)) != style["sha256"]:
                    raise ValueError("그림체 LoRA SHA 불일치(또는 미기록)")
                strength = check_strength(spec.style_strength)
                pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name="style")
                names, weights = ([accel_name], [1.0]) if accel_name else ([], [])
                pipe.set_adapters(names + ["style"], weights + [strength])
                record["style_lora"] = dict(name=style_name, repo=style["repo"], revision=style["revision"], lora=str(path),
                                            sha256=style["sha256"], strength=strength, adapters=names + ["style"],
                                            note="그림체 맛보기(D-159, 시험 전용 — 제품 아님)")
                # D-161 실제 적용 모듈: 'style' 어댑터가 걸린 층과 실제 배율(강도 × alpha/r)을 남긴다
                applied = {name: float(module.scaling["style"]) for name, module in tr.named_modules()
                           if "style" in (getattr(module, "lora_A", None) or {})}
                write_json(directory / "style_modules.json", dict(count=len(applied), modules=applied))
                kinds = {}
                for name in applied:
                    kind_name = name.split(".", 2)[-1] if name.startswith("transformer_blocks.") else name
                    kinds[kind_name] = kinds.get(kind_name, 0) + 1
                record["style_lora"].update(applied_modules=len(applied), applied_kinds=kinds,
                                            applied_scalings=sorted(set(applied.values())), modules_file="style_modules.json")
            pipe.vae.to("cuda")
            if outfit and request["memory"]["vae_tiling"]:  # 타일 없이는 6.5GiB 상한에서 VAE 복원이 실패했다(시험 10 진단 C)
                pipe.vae.enable_tiling()
                record["vae_tiling"] = True
            tr.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                    offload_type="block_level", num_blocks_per_group=1)
            stage("transformer_loaded")
            record["stage_seconds"]["pipeline_load"] = time.perf_counter() - tick
        record["scheduler"] = {"class": type(pipe.scheduler).__name__, "config": dict(pipe.scheduler.config)}
        record["offload"] = settings.offload
        record["zero_cond_t"] = True
        original_preprocess = pipe.image_processor.preprocess
        count = [0]
        def preprocess(image, h, w, *args, **kwargs):
            tensor = original_preprocess(image, h, w, *args, **kwargs)
            count[0] += 1
            Image.fromarray(((tensor[0].float().clamp(-1, 1) + 1) * 127.5).round().byte()
                            .permute(1, 2, 0).cpu().numpy()).save(mi / f"vae_picture{count[0]}_{w}x{h}.png")
            return tensor
        pipe.image_processor.preprocess = preprocess
        previous = [time.perf_counter()]
        try:        # 생성 과정 미리보기(2026-10-22): 단계마다 잠재값을 대략 색으로 바꿔 preview-NNN.png. 결과에는 영향 없음
            from genai_lab.generation_preview import load_factors
            preview = [load_factors()]
            preview_failures = [0]
        except Exception as error:      # noqa: BLE001 — 미리보기는 보조
            preview = [None]
            preview_failures = [0]
            record["preview_error"] = str(error)[:200]
        def callback(pipeline, index, timestep, kwargs):
            check()
            stage(f"step{index}")
            begin("decode" if index + 1 == steps else f"step{index + 1}")
            now = time.perf_counter()
            record["steps"].append({"index": index, "seconds": now - previous[0]})
            previous[0] = now
            if preview[0] is not None:
                try:
                    from genai_lab.generation_preview import preview_rgb, write_preview
                    latents = kwargs["latents"][:1].detach().float().cpu().numpy()
                    write_preview(directory, preview_rgb(latents, height, width, pipeline.vae_scale_factor, preview[0]), index)
                    preview_failures[0] = 0
                except Exception as error:      # noqa: BLE001 — 한 번 실패해도 다음 단계에 다시 쓴다. 연속 5번이면 끈다. 생성은 계속
                    preview_failures[0] += 1
                    record.setdefault("preview_errors", []).append(dict(step=index, error=str(error)[:200]))
                    if preview_failures[0] >= 5:
                        preview[0] = None
                        record["preview_error"] = str(error)[:200]
            save()
            return kwargs
        record["call_arguments"] = {"height": height, "width": width, "true_cfg_scale": true_cfg,
            "guidance_scale": settings.guidance_scale, "num_inference_steps": steps,
            "num_images_per_prompt": 1, "seed": request["seed"], "generator_device": "cpu",
            "positive": request["prompt"]["positive"], "negative": request["prompt"].get("negative", " "), "images": len(images)}
        save()
        if guard is None:
            torch.cuda.reset_peak_memory_stats()
        begin("step0")
        tick = time.perf_counter()
        result = pipe(image=images, prompt_embeds=pe.to("cuda"), prompt_embeds_mask=pm.to("cuda") if pm is not None else None,
            negative_prompt_embeds=ne.to("cuda"), negative_prompt_embeds_mask=nm.to("cuda") if nm is not None else None,
            true_cfg_scale=true_cfg, guidance_scale=settings.guidance_scale,
            num_inference_steps=steps, height=height, width=width,
            generator=torch.Generator("cpu").manual_seed(request["seed"]), callback_on_step_end=callback)
        record["stage_seconds"]["generation"] = time.perf_counter() - tick
        stage("decoded")
        if guard is not None:
            record["memory_stages"] = guard.stages
        record["generation_memory"] = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                                      "reserved_bytes": torch.cuda.max_memory_reserved()}
        if len(result.images) != 1 or len(record["steps"]) != steps or count[0] != len(images):
            raise RuntimeError("이미지/호출/입력 관측 수 불일치")
        if result.images[0].size != (width, height):
            raise RuntimeError("실제 출력 크기 불일치")
        result.images[0].save(directory / "raw.png")
        record["raw_sha256"] = file_sha(directory / "raw.png")
        record["model_inputs"] = {p.name: file_sha(p) for p in sorted(mi.iterdir())}
        record["seconds"] = time.perf_counter() - start
        completed = True
    except torch.OutOfMemoryError as error:
        if guard is not None:
            guard.fail(str(error).splitlines()[0])
            record["memory_stages"] = guard.stages
        raise
    finally:
        for image in images:
            image.close()
        if pipe is not None and original_preprocess is not None:
            pipe.image_processor.preprocess = original_preprocess
        if cache is not None and completed and pipe is not None:
            cache.keep(request, pipe, tr, (pe, pm, ne, nm), record, directory / "model_inputs")   # 다음 작업에 넘김
        else:
            if cache is not None:
                cache.clear()               # 실패·취소: 아무것도 넘기지 않는다(프로세스도 끝난다)
            if pipe is not None:
                remove = getattr(pipe, "remove_all_hooks", None)
                if callable(remove): remove()
        pipe = tr = te = p1 = pe = pm = ne = nm = None
        result = None
        gc.collect()
        torch.cuda.empty_cache()


def run_one(path, cache=None, session=None):
    """요청 하나. 세션이면 cache·session 기록을 받는다. 반환: (종료 코드, 상태)."""
    path = Path(path).resolve()
    directory = path.parent
    if any((directory / name).exists() for name in ("run.json", "raw.png", "run.final.json", "progress")):
        raise SystemExit("기존 실행 결과를 덮어쓰거나 자동 재시도하지 않습니다.")
    record = {"status": "starting", "gates_executed": False, "final_return_eligible": False}
    if session is not None:
        record["session"] = dict(session)
    code = 1
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
        record["request_sha256"] = json_sha(request)
        write_json(directory / "run.json", record)
        if cache is not None:
            cache.job = session["job"]
        if cache is None:
            generate(request, directory, record)
        else:
            generate(request, directory, record, cache)
        record["status"] = "completed"
        code = 0
    except BaseException as error:
        record.update(status="cancelled" if isinstance(error, WorkerCancelled) else "failed",
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        write_terminal(directory, record)
    return code, record["status"]


def watch_parent(stream):
    """실행기가 끝나 표준 입력이 닫히면 바로 끝낸다(남은 세션 프로세스가 GPU를 붙잡지 않게)."""
    import threading

    def wait():
        try:
            while stream.read(1024):
                pass
        finally:
            os._exit(3)
    threading.Thread(target=wait, daemon=True).start()


def serve(control, *, idle_seconds=SESSION_IDLE_SECONDS, watch=True, sleep=time.sleep, run=run_one):
    """세션(D-180): control 폴더의 job-000001.json, job-000002.json…을 차례로 처리한다. close 파일이 생기면 끝낸다.
    한 작업이라도 완료되지 않으면(실패·취소) 넘길 상태를 버리고 바로 끝낸다 — 다음 요청은 새 프로세스가 맡는다."""
    control = Path(control).resolve()
    if watch:
        watch_parent(sys.stdin)
    cache = SessionCache()
    number, idle_since = 1, time.monotonic()
    while True:
        if (control / "close").exists():
            return 0
        job = control / f"job-{number:06d}.json"
        if job.exists():
            data = json.loads(job.read_text(encoding="utf-8"))
            code, status = run(data["request"], cache=cache, session=dict(control=str(control), job=number, pid=os.getpid()))
            # 요청 폴더의 run.final.json·run.json을 다 쓴 뒤에 알린다(실행기가 쓰는 중인 기록을 열지 않게)
            write_json(control / f"job-{number:06d}.done.json", dict(status=status, code=code))
            if status != "completed":
                return code or 1
            number, idle_since = number + 1, time.monotonic()
            continue
        if time.monotonic() - idle_since > idle_seconds:
            return 0
        sleep(.2)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2 and args[0] == "--session":
        return serve(args[1])
    if len(args) != 1:
        raise SystemExit("usage: python -m genai_lab.qwen_pose_worker REQUEST.json | --session CONTROL_DIR")
    return run_one(args[0])[0]


if __name__ == "__main__":
    raise SystemExit(main())
