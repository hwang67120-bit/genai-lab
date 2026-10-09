"""잠금된 머리 확대 시험: CPU 확인 → 얼굴 보호 측정 → 머리 재생성 → 원본에 합성."""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "outputs/head-redraw-20261010"
sys.path.insert(0, str(ROOT))
from genai_lab.proportion_inputs import sha, json_value, checked_image, validate_models, require
from genai_lab.qwen_record_io import write_json
from genai_lab.onepass_prompt import plan_prompt_chunks
from genai_lab.onepass_prompt_tokenizers import load_onepass_tokenizers
from genai_lab.onepass_generation import encode_prompt_plan
from genai_lab.onepass_generation_settings import scheduler_config
from scripts.head_contour_trial_inputs import restore_request
from scripts.body_outline_source_trial import restore_options

LIMIT = 6.5
PREVIEW_SHA = "c1f723c24a2eddde6197662ba6486b7f94c9e2497ddcec8ab28b9010e0e0e1f2"
REPO = Path("D:/genai-cache/tools/see-through")
CHECKPOINT = Path("D:/genai-cache/huggingface/models--24yearsold--l2d_sam_iter2/snapshots/0b0608310fb7a89e32ecd7397147a249540cf1e7/checkpoint-18000.pt")
TAGS = ("hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth", "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear", "tail", "wings", "objects")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight():
    preview = read(D / "preview/preview.json")
    require(sha(D / "preview/preview.json") == PREVIEW_SHA, "미리보기 잠금 불일치")
    confirmation = read(D / "preview/user-confirmation.json")
    require(confirmation["confirmed"] and confirmation["preview_sha256"] == PREVIEW_SHA, "사용자 머리 확인 불일치")
    run = Path(preview["run"])
    lock = read(run / "generation/preflight.json")
    inputs, settings = restore_request({"inputs": {"BASE": lock["inputs"]}, "settings": lock["settings"]})
    options = restore_options(lock["options"])
    checked_image(options.head.normalized_file, options.head.normalized_sha256, (736, 1232), "RGB")
    checked_image(options.head.mask_file, options.head.mask_sha256, (736, 1232), "L")
    require(sha(inputs.face_file) == inputs.face_sha256, "얼굴 참조 SHA 불일치")
    with Image.open(inputs.face_file) as face:
        require(face.mode == "RGB" and min(face.size) > 0, "얼굴 참조 크기·모드 오류")
    models = validate_models(settings, options)
    tokenizers = load_onepass_tokenizers()
    require(plan_prompt_chunks(inputs.prompt.positive, inputs.prompt.negative, tokenizers) == inputs.prompt.encoders, "원래 문구 토큰 재현 실패")
    terms = ("1boy", "male focus", "masculine silhouette", "purple eyes", "short hair", "blue hair", "multicolored hair", "animal ears", "raccoon ears")
    available = [x.strip() for x in inputs.prompt.positive.split(",")]
    require(all(x in available for x in terms), "잠금 문구에 필요한 머리 태그가 없습니다")
    positive = ", ".join((*terms, "portrait", "close-up", "white background"))
    prompt = replace(inputs.prompt, positive=positive, encoders=plan_prompt_chunks(positive, inputs.prompt.negative, tokenizers))
    manifest = {str(D / "preview/preview.json"): PREVIEW_SHA, str(inputs.face_file): inputs.face_sha256}
    require(len(preview["seeds"]) == 2, "시험 seed는 두 개여야 합니다")
    for seed in preview["seeds"]:
        row = preview["per_seed"][str(seed)]
        checked_image(row["raw"], row["raw_sha256"], (736, 1232), "RGB")
        manifest[row["raw"]] = row["raw_sha256"]
        for name, mode in (("gen_crop", "RGB"), ("mask", "L"), ("sketch", "L")):
            path = D / "preview" / f"{name}_{seed}.png"
            checked_image(path, row[name + "_sha256"], (1024, 1024), mode)
            manifest[str(path)] = row[name + "_sha256"]
    return preview, inputs, settings, options, prompt, models, manifest


def blend_result(raw, redrawn, mask, features, box):
    size = (box[2] - box[0], box[3] - box[1])
    local = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)) > 127
    face = np.asarray(Image.fromarray(features).resize(size, Image.Resampling.NEAREST)) > 127
    support = cv2.dilate(local.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    # 최종 4px 확장에도 눈·코·입이 들어가지 않도록 보호 영역을 다시 제한한다.
    require(not np.any(local & face), "원래 편집 마스크가 얼굴 특징을 덮습니다")
    expanded_overlap = int((support & face).sum())
    support &= ~face
    soft = np.asarray(Image.fromarray(support.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(2))).astype(float) / 255
    soft[~support] = 0
    patch = np.asarray(redrawn.resize(size, Image.Resampling.LANCZOS), dtype=float)
    result = raw.copy()
    old = raw[box[1]:box[3], box[0]:box[2]]
    result[box[1]:box[3], box[0]:box[2]] = np.rint(old * (1 - soft[..., None]) + patch * soft[..., None]).astype(np.uint8)
    allowed = np.zeros(raw.shape[:2], bool)
    allowed[box[1]:box[3], box[0]:box[2]] = support
    require(np.array_equal(raw[~allowed], result[~allowed]), "편집 영역 밖 픽셀 변경")
    require(np.array_equal(old[face], result[box[1]:box[3], box[0]:box[2]][face]), "얼굴 특징 픽셀 변경")
    return result, {"outside_equal": True, "face_equal": True, "expanded_face_overlap_removed": expanded_overlap}


class RestoreSteps:
    def __init__(self, initial, noise, mask, sigmas, guard=lambda: None):
        self.initial, self.noise, self.mask, self.sigmas = initial, noise, mask, sigmas
        self.guard, self.calls = guard, []

    def __call__(self, index, timestep, latents):
        import torch
        require(index == len(self.calls), "단계 콜백 순서 불일치")
        reference = self.initial + self.noise * self.sigmas[index + 1].to(latents.device)
        latents.copy_(reference if index <= 10 else torch.where(self.mask, latents, reference))
        require(torch.equal(latents[~self.mask], reference[~self.mask]), "잠재 공간 보호 실패")
        self.calls.append({"index": index, "sigma_next": float(self.sigmas[index + 1]), "outside_restored": True})
        self.guard()


def check_cpu():
    import torch
    initial = torch.ones((1, 4, 2, 2))
    noise = torch.full_like(initial, 2)
    mask = torch.zeros_like(initial, dtype=torch.bool)
    mask[..., 0, 0] = True
    sigmas = torch.linspace(1, 0, 29)
    restore = RestoreSteps(initial, noise, mask, sigmas)
    for index in range(28):
        value = torch.full_like(initial, 9)
        restore(index, None, value)
        if index <= 10:
            require(torch.equal(value, initial + noise * sigmas[index + 1]), "초기 단계 복원 실패")
        else:
            require((value[mask] == 9).all(), "편집 영역 손상")
    require(torch.equal(value[~mask], initial[~mask]), "마지막 단계 노이즈 제거 실패")
    raw = np.full((40, 40, 3), 40, np.uint8)
    mask = np.zeros((20, 20), np.uint8); mask[4:9, 4:9] = 255
    face = np.zeros_like(mask); face[9:11, 9:11] = 255
    result, checks = blend_result(raw, Image.new("RGB", (20, 20), "white"), mask, face, (10, 10, 30, 30))
    require(checks["expanded_face_overlap_removed"] > 0 and not np.array_equal(result, raw), "얼굴 보호 합성 시험 실패")
    print("CPU 검사 통과: 28단계 복원·마지막 노이즈·편집 영역·최종 얼굴 보호·영역 밖 픽셀", flush=True)


def memory_guard(torch):
    value = torch.cuda.max_memory_reserved() / 2**30
    require(value <= LIMIT, f"reserved 메모리 한도 초과: {value:.3f} GiB")


def prepare_features(output):
    import torch
    # 별도 프로세스가 종료된 뒤 생성 모델을 올려 GPU 모델이 겹치지 않게 한다.
    sys.path.insert(0, str(REPO / "common")); sys.path.insert(0, str(REPO))
    from modules.semanticsam import SemanticSam
    preview = read(D / "preview/preview.json")
    require(sha(CHECKPOINT) == "dda82b916f6c340bc5a6c3a077ff1214f9906906abe3707b1076d55b508f253f", "부위 분할 가중치 불일치")
    model = SemanticSam(class_num=19)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model = model.to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    for seed in preview["seeds"]:
        image = np.asarray(Image.open(D / f"preview/gen_crop_{seed}.png").convert("RGB"))
        with torch.inference_mode():
            parts = model.inference(image)[0]
        parts = parts.cpu().numpy() if hasattr(parts, "cpu") else np.asarray(parts)
        features = np.logical_or.reduce([parts[TAGS.index(name)] > 0 for name in ("eyes", "nose", "mouth")])
        require(features.any(), "얼굴 특징 분할이 비었습니다")
        Image.fromarray(features.astype(np.uint8) * 255).save(output / f"features_{seed}.png")
        memory_guard(torch)
    write_json(output / "features.json", {"checkpoint_sha256": sha(CHECKPOINT), "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})


def memory_snapshot(torch, pipe, output, stage):
    """실제 사용량·예약량과 모델의 GPU 상주량을 단계마다 기록한다."""
    row = {"stage": stage, "seconds": time.perf_counter(),
           "allocated_bytes": torch.cuda.memory_allocated(),
           "reserved_bytes": torch.cuda.memory_reserved(),
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
           "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    row["gpu_parameter_bytes"] = {
        name: sum(p.numel()*p.element_size() for p in getattr(pipe,name).parameters()
                  if p.device.type == "cuda")
        for name in ("text_encoder", "text_encoder_2", "image_encoder", "adapter", "unet", "vae")}
    with (output / "memory.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def generate(prepared, output, state):
    import torch
    from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler
    from transformers import CLIPVisionModelWithProjection
    preview, inputs, settings, options, prompt, models, manifest = prepared
    adapter = T2IAdapter.from_pretrained(str(options.sketch_root), variant="fp16", torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    encoder = CLIPVisionModelWithProjection.from_pretrained(str(settings.ip_root / settings.image_encoder_subfolder), torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(str(settings.model_root), adapter=adapter, image_encoder=encoder, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    pipe.load_ip_adapter(str(settings.ip_root), subfolder=settings.ip_subfolder, weight_name=settings.ip_weight_name, image_encoder_folder=None, local_files_only=True)
    pipe.set_ip_adapter_scale(.9)
    initial = {}
    offload = None
    adapter_hook = None
    try:
        memory_snapshot(torch, pipe, output, "models_loaded_cpu")
        # 인코딩은 fp32로 끝낸 뒤 VAE를 내려 생성 모델과 동시에 상주하지 않게 한다.
        torch.cuda.reset_peak_memory_stats()
        pipe.vae.to(device="cuda", dtype=torch.float32)
        for seed in preview["seeds"]:
            with Image.open(D / f"preview/gen_crop_{seed}.png") as image:
                tensor = pipe.image_processor.preprocess(image.convert("RGB")).to("cuda", torch.float32)
            with torch.inference_mode():
                latent = pipe.vae.encode(tensor).latent_dist.mean * pipe.vae.config.scaling_factor
            initial[seed] = latent.cpu().half()
            del tensor, latent
            memory_guard(torch)
        state["vae_encode_peak_gib"] = torch.cuda.max_memory_reserved() / 2**30
        pipe.vae.to(device="cpu", dtype=torch.float16); torch.cuda.empty_cache()
        memory_snapshot(torch, pipe, output, "vae_released")
        if state["memory_mode"] == "block":
            from accelerate import cpu_offload_with_hook
            from genai_lab.finishing_offload import configure_offload
            from genai_lab.finishing_memory import attention_policy
            state["offload"] = {}
            offload = configure_offload(pipe, torch, state["offload"])
            _, adapter_hook = cpu_offload_with_hook(pipe.adapter, torch.device("cuda:0"))
            # 스케치 잔차를 계산한 뒤 어댑터 가중치를 내려 UNet과 겹치지 않게 한다.
            offload.observer_hooks.append(pipe.unet.register_forward_pre_hook(lambda *_:adapter_hook.offload()))
            state["attention_policy"] = attention_policy(pipe)
        else:
            pipe.model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->adapter->unet->vae"
            pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        encoders = (pipe.text_encoder.to("cuda"), pipe.text_encoder_2.to("cuda"))
        embeds = encode_prompt_plan(prompt, encoders, "cuda", torch.float16)
        pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu"); torch.cuda.empty_cache()
        memory_snapshot(torch, pipe, output, "text_encoders_released")
        for seed in preview["seeds"]:
            if offload is not None:offload.begin_stage()
            destination = output / f"seed-{seed}"; destination.mkdir()
            for path, digest in manifest.items():require(sha(path) == digest, "실행 중 입력 변경: " + path)
            torch.cuda.reset_peak_memory_stats()
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config())
            pipe.scheduler.set_timesteps(28, device="cuda")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            init = initial[seed].to("cuda")
            noise = torch.randn(init.shape, generator=generator, device="cuda", dtype=init.dtype)
            mask = np.asarray(Image.open(D / f"preview/mask_{seed}.png").convert("L"))
            down = cv2.resize(mask.astype(np.float32) / 255, (128, 128), interpolation=cv2.INTER_AREA) >= .5
            latent_mask = torch.from_numpy(down).to("cuda")[None, None].expand_as(init)
            callback = RestoreSteps(init, noise, latent_mask, pipe.scheduler.sigmas, lambda: memory_guard(torch))
            observations = []
            def observe(module, args, kwargs, result):
                observations.append({"batch": int(args[0].shape[0]), "adapter": kwargs.get("down_intrablock_additional_residuals") is not None})
                memory_snapshot(torch, pipe, output, f"seed-{seed}:unet-{len(observations)}")
                memory_guard(torch)
            counts = []
            handles = [pipe.unet.register_forward_hook(observe, with_kwargs=True), pipe.adapter.register_forward_hook(lambda *args: counts.append(1))]
            started = time.perf_counter()
            record = {"seed": seed, "status": "generating", "prompt": prompt.positive, "negative": prompt.negative, "sketch_strength": .8, "ip_scale": .9, "steps": 28, "actual_start_index": 11, "memory_limit_gib": LIMIT, "memory_mode": state["memory_mode"]}
            write_json(destination / "run.json", record)
            try:
                from genai_lab.finishing_memory import unet_attention
                memory_snapshot(torch, pipe, output, f"seed-{seed}:before_pipeline")
                with unet_attention(pipe.unet) if offload is not None else nullcontext(), Image.open(D / f"preview/sketch_{seed}.png") as sketch, Image.open(inputs.face_file) as face:
                    result = pipe(**embeds, image=sketch.convert("RGB"), ip_adapter_image=face.convert("RGB"), width=1024, height=1024, num_inference_steps=28, guidance_scale=settings.guidance_scale, adapter_conditioning_scale=.8, adapter_conditioning_factor=1., generator=generator, callback=callback, callback_steps=1).images[0]
                if offload is not None:offload.verify_stage(28)
                memory_snapshot(torch, pipe, output, f"seed-{seed}:decoded")
                result.save(destination / "head_1024.png")
                require(len(observations)==28 and len(callback.calls)==28 and len(counts)==1, "모델·복원 호출 수 불일치")
                require(all(x["batch"]==2 and x["adapter"] for x in observations), "CFG 또는 스케치 적용 불일치")
                row = preview["per_seed"][str(seed)]
                raw = np.asarray(Image.open(row["raw"]).convert("RGB"))
                features = np.asarray(Image.open(output / f"features_{seed}.png").convert("L"))
                blended, checks = blend_result(raw, result, mask, features, preview["box"])
                Image.fromarray(blended).save(destination / "raw_redraw.png")
                memory_guard(torch)
                record.update(status="review_pending", seconds=time.perf_counter()-started, checks=checks, unet_calls=observations, callback=callback.calls, adapter_calls=len(counts), max_reserved_gib=torch.cuda.max_memory_reserved()/2**30, head_sha256=sha(destination / "head_1024.png"), result_sha256=sha(destination / "raw_redraw.png"))
                state["completed"].append(seed)
                print(seed, record["status"], round(record["seconds"], 2), record["max_reserved_gib"], flush=True)
            except BaseException as error:
                record.update(status="failed", error=str(error), unet_calls=observations,
                              callback=callback.calls, adapter_calls=len(counts))
                record["failure_memory"] = memory_snapshot(torch, pipe, output, f"seed-{seed}:failed")
                raise
            finally:
                if offload is not None:
                    record["ip_observations"] = dict(offload.observations)
                    offload.release()
                    adapter_hook.offload()
                for handle in handles:handle.remove()
                write_json(destination / "run.json", json_value(record))
                write_json(output / "status.json", json_value(state))
                del init, noise, latent_mask
                torch.cuda.empty_cache()
    finally:
        if offload is not None:offload.close()
        else:pipe.maybe_free_model_hooks()
        if adapter_hook is not None:
            adapter_hook.offload()
            adapter_hook.remove()
        memory_snapshot(torch, pipe, output, "models_released")
        del pipe
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--memory-mode", choices=("model", "block"), default="model")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--output", type=Path, default=D / "R1-gpu-01")
    args = parser.parse_args()
    if args.features:
        prepare_features(args.output); return
    check_cpu()
    prepared = preflight()
    if not args.run:
        print("입력·모델 SHA·기존 프롬프트 재현 통과. GPU 생성 없음", flush=True); return
    args.output.mkdir(exist_ok=False)
    state = {"memory_mode": args.memory_mode, "status": "preparing", "completed": [], "gpu_authorized_by_user": True, "runner_sha256": sha(__file__), "models": prepared[5], "input_sha256": prepared[6], "prompt": prepared[4].positive, "note": "구형 콜백 시험이며 표준 이미지 재처리와 동일하다고 주장하지 않음. 최종 마스크 확장에서 얼굴 특징을 다시 제외함."}
    write_json(args.output / "status.json", json_value(state))
    try:
        subprocess.run([sys.executable, "-B", str(Path(__file__)), "--features", "--output", str(args.output)], check=True)
        state["status"] = "generating"
        generate(prepared, args.output, state)
        state["status"] = "review_pending"
    except BaseException as error:
        state.update(status="failed", error=str(error)); raise
    finally:
        write_json(args.output / "status.json", json_value(state))


if __name__ == "__main__":
    main()
