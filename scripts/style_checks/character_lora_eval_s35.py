"""S35 캐릭터 기억 LoRA 효과(S34 학습 결과): 학습에 쓰지 않은 새 장면 3개 × seed 2로
A 지금 앱(그림체 LoRA만) · B + 자동 고치기 문장 · C<단계> 그림체 + 캐릭터 LoRA(트리거 낱말, 세기 1.0)를 비교. 지킬 태그(첫 진단)를
몇 개 잃었는지와 CCIP 차이. 이 PC, 앱과 같은 설정(4비트·4단계·1024x1360·상한 6.5GiB).
사용: python character_lora_eval_s35.py <앱 작업 폴더(flux)> <캐릭터 LoRA 폴더(checkpoint-*)> <출력 폴더> [단계 ...]
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

from genai_lab.identity_autofix import KINDS_TABLE, fixed_prompt, load_kinds, lost_tags, remaining, to_fix

SCENES = (35001, 35002, 35003)
SEEDS = (42, 43)
TRIGGER = "f1chr character."


def main():
    R, lora_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    steps = sys.argv[4:] or ["400", "600"]
    req = json.loads((R / "request.json").read_text(encoding="utf-8"))
    prepared = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
    import torch
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.scene_composer import compose_scene
    from genai_lab.style_redraw_flux import (GUIDANCE, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, SEEDS as APP_SEEDS, STEPS, STYLE_LORA_WEIGHT,
                                             flux_prompt)
    from style_redraw_worker import MEMORY_CAP_GIB
    settings = ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)
    with WdTagSession(settings) as t:
        tag = lambda p: dict(t.analyze(Image.open(p).convert("RGB")).raw_general_scores)
        original = tag(R / "character.png")
        watch = to_fix(lost_tags(original, [tag(R / f"result-raw-seed{s}.png") for s in APP_SEEDS]), load_kinds(Path(r"D:\Code\genai-lab") / KINDS_TABLE))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(req["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    style = Path(req["flux_style_lora"]); pipe.load_lora_weights(str(style.parent), weight_name=style.name, adapter_name="style")
    for s in steps:
        pipe.load_lora_weights(str(lora_dir / f"checkpoint-{s}"), weight_name="pytorch_lora_weights.safetensors", adapter_name=f"char{s}")
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    src = Image.open(R / "character.png").convert("RGB")
    variants = ["A", "B"] + [f"C{s}" for s in steps]
    records = []
    for sc in SCENES:
        scene = compose_scene(sc)["text"]
        base = flux_prompt(prepared["traits"], False, prepared.get("body_parts"), scene, prepared.get("body_part_kinds"))
        for v in variants:
            if v.startswith("C"):
                pipe.set_adapters(["style", f"char{v[1:]}"], adapter_weights=[STYLE_LORA_WEIGHT, 1.0]); prompt = f"{TRIGGER} {base}"
            else:
                pipe.set_adapters(["style"], adapter_weights=[STYLE_LORA_WEIGHT]); prompt = fixed_prompt(base, watch) if v == "B" else base
            for seed in SEEDS:
                torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                image = pipe(image=[src], prompt=prompt, width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], num_inference_steps=STEPS,
                             guidance_scale=GUIDANCE, max_sequence_length=MAX_SEQUENCE_LENGTH,
                             generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
                f = out / f"sc{sc}-{v}-s{seed}.png"; image.save(f)
                records.append(dict(scene=sc, variant=v, seed=seed, file=f.name, seconds=round(time.time() - t0, 1),
                                    peak_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2)))
                print(records[-1], flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    import huggingface_hub.constants as hub
    hub.HF_HUB_CACHE = str(req["model_cache"])
    from imgutils.metrics import ccip_difference, ccip_extract_feature
    feat0 = ccip_extract_feature(src)
    with WdTagSession(settings) as t:
        for r in records:
            im = Image.open(out / r["file"]).convert("RGB")
            r["lost"] = remaining(original, dict(t.analyze(im).raw_general_scores), watch)
            r["ccip"] = round(float(ccip_difference(feat0, ccip_extract_feature(im))), 3)
    summary = {v: dict(mean_lost=round(sum(len(r["lost"]) for r in records if r["variant"] == v) / (len(SCENES) * len(SEEDS)), 2),
                       furry_kept=sum("furry" not in r["lost"] for r in records if r["variant"] == v),
                       mean_ccip=round(sum(r["ccip"] for r in records if r["variant"] == v) / (len(SCENES) * len(SEEDS)), 3)) for v in variants}
    W, H = 260, 345
    sheet = Image.new("RGB", (W * (len(variants) + 1), (H + 22) * len(SCENES)), "white"); dr = ImageDraw.Draw(sheet)
    for i, sc in enumerate(SCENES):
        y = i * (H + 22)
        sheet.paste(src.resize((W, H)), (0, y + 22)); dr.text((4, y + 5), "original", fill="black")
        for j, v in enumerate(variants):
            r = next(x for x in records if x["scene"] == sc and x["variant"] == v and x["seed"] == SEEDS[0])
            sheet.paste(Image.open(out / r["file"]).convert("RGB").resize((W, H)), ((j + 1) * W, y + 22))
            dr.text(((j + 1) * W + 4, y + 5), f"{v} lost {len(r['lost'])} ccip {r['ccip']}", fill="black")
    sheet.save(out / "charlora-eval.png")
    (out / "charlora-eval.json").write_text(json.dumps(dict(watch=watch, summary=summary, records=records), ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
