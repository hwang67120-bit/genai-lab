"""Separate Qwen runtime entry point. Package checks run before torch imports.

Reference: diffusers v0.38.0 pipeline_qwenimage_edit_plus.py; approved B runner.
This module is safe to import for CPU unit tests; main() alone performs inference.
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
from genai_lab.qwen_pose_edit import write_json


class UntruncatedProcessor:
    """Observe real processor inputs without rewriting text or token IDs."""
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


def generate(request, directory, record):
    # This is deliberately before importing torch/diffusers/bitsandbytes.
    record["versions"] = check_runtime_versions()
    import importlib.metadata
    record["runtime"] = {"python": sys.executable, "packages": {
        name: str(importlib.metadata.distribution(name).locate_file("")) for name in record["versions"]}}
    settings = QwenPoseSettings(**request["settings"])
    spec = validate_request(request)
    record["models"] = validate_model_files(settings)
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
    pe = pm = ne = nm = None
    original_preprocess = None
    start = time.perf_counter()
    def save(): write_json(directory / "run.json", record)
    def check():
        if (directory / "cancel.request").exists():
            raise RuntimeError("사용자 취소")
    try:
        for path, expected in ((spec.image_path, spec.image_sha256),
                               (request["skeleton_path"], request["skeleton_sha256"])):
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
        # TE and transformer stages match run_identity.py; no alternative memory policy.
        check()
        tick = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(Path(settings.model_root) / "text_encoder"), torch_dtype=torch.bfloat16,
            local_files_only=True, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"]))
        p1 = QwenImageEditPlusPipeline.from_pretrained(settings.model_root, transformer=None, vae=None,
            text_encoder=te, torch_dtype=torch.bfloat16, local_files_only=True)
        p1.processor = UntruncatedProcessor(p1.processor, record)
        cond = []
        for index, image in enumerate(images, 1):
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
            c = p1.image_processor.resize(image, ch, cw)
            c.save(mi / f"vl_picture{index}_{cw}x{ch}.png")
            cond.append(c)
        with torch.no_grad():
            pe, pm = p1.encode_prompt(prompt=request["prompt"]["positive"], image=cond, device=torch.device("cuda"))
            ne, nm = p1.encode_prompt(prompt=" ", image=cond, device=torch.device("cuda"))
        pe, pm, ne, nm = (v.to("cpu") if v is not None else None for v in (pe, pm, ne, nm))
        record["embedding_shapes"] = {"positive": list(pe.shape), "negative": list(ne.shape)}
        observed = record["processor_inputs"]
        # The installed pipeline removes the fixed system prefix (64 IDs), not body text.
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
        save()
        check()
        tick = time.perf_counter()
        tr = QwenImageTransformer2DModel.from_single_file(settings.gguf_file,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16,
            config=settings.model_root, subfolder="transformer", local_files_only=True)
        if getattr(tr.config, "zero_cond_t", None) is not True:
            raise RuntimeError("zero_cond_t=True 계약 불일치")
        pipe = QwenImageEditPlusPipeline.from_pretrained(settings.model_root, transformer=tr,
            text_encoder=None, tokenizer=None, processor=None, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.vae.to("cuda")
        tr.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                offload_type="block_level", num_blocks_per_group=1)
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
        def callback(pipeline, index, timestep, kwargs):
            check()
            now = time.perf_counter()
            record["steps"].append({"index": index, "seconds": now - previous[0]})
            previous[0] = now
            save()
            return kwargs
        record["call_arguments"] = {"height": height, "width": width, "true_cfg_scale": settings.true_cfg_scale,
            "guidance_scale": settings.guidance_scale, "num_inference_steps": settings.steps,
            "num_images_per_prompt": 1, "seed": request["seed"], "generator_device": "cpu",
            "positive": request["prompt"]["positive"], "negative": " ", "images": 2}
        save()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        result = pipe(image=images, prompt_embeds=pe.to("cuda"), prompt_embeds_mask=pm.to("cuda") if pm is not None else None,
            negative_prompt_embeds=ne.to("cuda"), negative_prompt_embeds_mask=nm.to("cuda") if nm is not None else None,
            true_cfg_scale=settings.true_cfg_scale, guidance_scale=settings.guidance_scale,
            num_inference_steps=settings.steps, height=height, width=width,
            generator=torch.Generator("cpu").manual_seed(request["seed"]), callback_on_step_end=callback)
        record["stage_seconds"]["generation"] = time.perf_counter() - tick
        record["generation_memory"] = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                                      "reserved_bytes": torch.cuda.max_memory_reserved()}
        if len(result.images) != 1 or len(record["steps"]) != settings.steps or count[0] != 2:
            raise RuntimeError("이미지/호출/입력 관측 수 불일치")
        if result.images[0].size != (width, height):
            raise RuntimeError("실제 출력 크기 불일치")
        result.images[0].save(directory / "raw.png")
        record["raw_sha256"] = file_sha(directory / "raw.png")
        record["model_inputs"] = {p.name: file_sha(p) for p in sorted(mi.iterdir())}
        record["seconds"] = time.perf_counter() - start
    finally:
        for image in images:
            image.close()
        if pipe is not None:
            if original_preprocess is not None:
                pipe.image_processor.preprocess = original_preprocess
            remove = getattr(pipe, "remove_all_hooks", None)
            if callable(remove): remove()
        pipe = tr = te = p1 = pe = pm = ne = nm = None
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m genai_lab.qwen_pose_worker REQUEST.json")
    path = Path(args[0]).resolve()
    directory = path.parent
    if (directory / "run.json").exists() or (directory / "raw.png").exists():
        raise SystemExit("기존 실행 결과를 덮어쓰거나 자동 재시도하지 않습니다.")
    record = {"status": "starting", "gates_executed": False, "final_return_eligible": False}
    code = 1
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
        record["request_sha256"] = json_sha(request)
        write_json(directory / "run.json", record)
        generate(request, directory, record)
        record["status"] = "completed"
        code = 0
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        write_json(directory / "run.json", record)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
