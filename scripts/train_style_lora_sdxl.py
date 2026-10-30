"""SDXL 스타일 LoRA 학습(D-204, 모드 B 그림체 교체 준비). diffusers + peft만 쓴다(새 설치 없음).

학습 자료: 그림 + 같은 이름의 .txt 캡션(호출어 + 내용만). 메모리를 앱 상한(6.5GiB) 안에 두려고
  1) VAE로 잠재값, 두 문장 인코더로 문장 임베딩을 먼저 계산해 두고 그 모델들은 내린다
  2) 본체 UNet은 fp8로 보관(계산은 fp16), 기울기 체크포인트, LoRA(rank 16, 주의층 q·k·v·out)만 학습, 8비트 AdamW
사용: python train_style_lora_sdxl.py --data <train 폴더> --base <realvis 폴더> --output <새 폴더> [--max-steps 10]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1030)
    p.add_argument("--memory-cap-gib", type=float, default=6.5)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1024)
    # D-207: 얼굴 영역의 오차를 줄여 학습(LoRA가 학습 자료의 공통 얼굴을 덜 배우게). {"boxes": {그림 이름: [x0,y0,x1,y1]}}
    p.add_argument("--face-boxes", type=Path)
    p.add_argument("--face-weight", type=float, default=1.0)
    return p.parse_args()


def loss_weights(args, items):
    """그림마다 잠재값 크기의 오차 가중치(얼굴 상자 안 = face_weight, 밖 = 1). 상자 파일이 없으면 None."""
    if args.face_boxes is None:
        return None
    from PIL import Image
    boxes = json.loads(args.face_boxes.read_text(encoding="utf-8"))["boxes"]
    h, w = args.height // 8, args.width // 8
    weights = []
    for img_path, _ in items:
        m = torch.ones(1, 1, h, w)
        box = boxes.get(img_path.stem)
        if box:
            with Image.open(img_path) as im:
                sx, sy = args.width / im.width, args.height / im.height
            x0, y0, x1, y1 = box
            m[..., int(y0 * sy / 8):int(math.ceil(y1 * sy / 8)), int(x0 * sx / 8):int(math.ceil(x1 * sx / 8))] = args.face_weight
        weights.append(m)
    return weights


def encode_all(args, items, device):
    """잠재값과 문장 임베딩을 한 번만 계산한다. 반환 뒤 VAE·문장 인코더는 메모리에서 내린다."""
    from PIL import Image
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers import AutoencoderKL
    import numpy as np
    # 모델 폴더에는 fp16 파일만 있다. 불러온 뒤 fp32로 올려 인코딩한다(fp16 VAE의 NaN을 피함).
    vae = AutoencoderKL.from_pretrained(str(args.base / "vae"), torch_dtype=torch.float16, variant="fp16",
                                        local_files_only=True).to(device, torch.float32)
    vae.enable_tiling()
    latents = []
    with torch.no_grad():
        for img_path, _ in items:
            im = Image.open(img_path).convert("RGB").resize((args.width, args.height), Image.LANCZOS)
            x = torch.from_numpy(np.asarray(im).astype("float32") / 127.5 - 1).permute(2, 0, 1)[None].to(device)
            latents.append((vae.encode(x).latent_dist.sample() * vae.config.scaling_factor).to("cpu", torch.float16))
    del vae
    torch.cuda.empty_cache()
    tok1 = CLIPTokenizer.from_pretrained(str(args.base / "tokenizer"), local_files_only=True)
    tok2 = CLIPTokenizer.from_pretrained(str(args.base / "tokenizer_2"), local_files_only=True)
    te1 = CLIPTextModel.from_pretrained(str(args.base / "text_encoder"), torch_dtype=torch.float16, variant="fp16", local_files_only=True).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(str(args.base / "text_encoder_2"), torch_dtype=torch.float16, variant="fp16",
                                                      local_files_only=True).to(device)
    embeds, truncated = [], 0
    with torch.no_grad():
        for _, caption in items:
            ids1 = tok1(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
            ids2 = tok2(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
            truncated += int(len(tok1(caption).input_ids) > 77)
            h1 = te1(ids1, output_hidden_states=True).hidden_states[-2]
            out2 = te2(ids2, output_hidden_states=True)
            prompt = torch.cat([h1, out2.hidden_states[-2]], dim=-1)
            embeds.append((prompt.to("cpu"), out2.text_embeds.to("cpu")))
    del te1, te2
    torch.cuda.empty_cache()
    return latents, embeds, truncated


def save_lora(unet, folder):
    from peft.utils import get_peft_model_state_dict
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    folder.mkdir(parents=True, exist_ok=True)
    layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(str(folder), unet_lora_layers=layers, safe_serialization=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    items = [(p, p.with_suffix(".txt").read_text(encoding="utf-8").strip()) for p in sorted(args.data.glob("*.png"))]
    report = dict(images=len(items), args={k: str(v) for k, v in vars(args).items()}, started=time.time())
    t0 = time.time()
    latents, embeds, truncated = encode_all(args, items, device)
    weights = loss_weights(args, items)
    report.update(encode_seconds=round(time.time() - t0, 1), captions_truncated=truncated,
                  face_masked=None if weights is None else dict(weight=args.face_weight, images=sum(int((w < 1).any()) for w in weights)))

    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    import bitsandbytes as bnb
    scheduler = DDPMScheduler.from_pretrained(str(args.base / "scheduler"), local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(str(args.base / "unet"), torch_dtype=torch.float16, variant="fp16", local_files_only=True)
    unet.requires_grad_(False)
    # 합성곱 층은 fp16 그대로: fp8 보관 + 기울기 체크포인트에서 다시 계산할 때 합성곱 무게가 fp8로 남아 오류가 났다.
    unet.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16,
                                  skip_modules_classes=(torch.nn.Conv2d,))
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    unet.enable_gradient_checkpointing()
    unet.to(device)
    unet.train()
    optimizer = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    time_ids = torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]], dtype=torch.float16, device=device)
    report["trainable_parameters"] = sum(p.numel() for p in params)
    torch.cuda.reset_peak_memory_stats()
    losses, t1 = [], time.time()
    order = []
    for step in range(1, args.max_steps + 1):
        if not order:
            order = list(range(len(items)))
            random.shuffle(order)
        i = order.pop()
        x0 = latents[i].to(device)
        noise = torch.randn_like(x0)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
        xt = scheduler.add_noise(x0, noise, t)
        prompt, pooled = (e.to(device) for e in embeds[i])
        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(xt, t, encoder_hidden_states=prompt,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
            if weights is None:
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            else:
                w = weights[i].to(device)
                loss = (w * (pred.float() - noise.float()) ** 2).sum() / (w.sum() * pred.shape[1])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss))
        if step % 10 == 0 or step == args.max_steps:
            report.update(step=step, seconds_per_step=round((time.time() - t1) / step, 2), loss_last10=round(sum(losses[-10:]) / len(losses[-10:]), 4),
                          peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            print(json.dumps({k: report[k] for k in ("step", "seconds_per_step", "loss_last10", "peak_reserved_gib")}), flush=True)
            (args.output / "train-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if step % args.save_every == 0 or step == args.max_steps:
            save_lora(unet, args.output / f"step-{step:05d}")
    report.update(state="completed", finished=time.time(), loss_curve=[round(v, 4) for v in losses])
    (args.output / "train-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
