"""S52 비율 문장(D-247) 전후: S51 점검의 T009(옷 그대로·옷 바꿈) 작업 폴더의 장면·문장 그대로, 비율 문장만 더해 같은 seed로.
사용: python proportion_check_s52.py <출력 폴더> <이름>=<작업 폴더(flux)> ...
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import json, sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

SEEDS = (42, 43)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cases = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    import torch
    import genai_lab.style_redraw_flux as flux
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from style_redraw_worker import MEMORY_CAP_GIB
    first = json.loads((next(iter(cases.values())) / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(first["model_cache"]), local_files_only=True)) as t:
        originals = {n: dict(t.analyze(Image.open(R / "character.png").convert("RGB")).raw_general_scores) for n, R in cases.items()}
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(first["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    lora = Path(first["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[flux.STYLE_LORA_WEIGHT]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    labels = []
    for name, R in cases.items():
        p = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
        refs = [Image.open(R / "character.png").convert("RGB")] + ([Image.open(R / "garment.png").convert("RGB")] if p.get("with_garment") else [])
        sentence = flux.proportion_sentence(originals[name], p.get("gender") or p.get("guessed_gender"))
        print(name, "sentence:", sentence, flush=True)
        for label, extra in (("before", ""), ("after", sentence)):
            prompt = flux.flux_prompt(p["traits"], bool(p.get("with_garment")), p.get("body_parts"), p.get("scene"), p.get("body_part_kinds"), extra)
            for seed in SEEDS:
                pipe(image=refs, prompt=prompt, width=flux.OUTPUT_SIZE[0], height=flux.OUTPUT_SIZE[1], num_inference_steps=flux.STEPS,
                     guidance_scale=flux.GUIDANCE, max_sequence_length=flux.MAX_SEQUENCE_LENGTH,
                     generator=torch.Generator(device="cpu").manual_seed(seed)).images[0].save(out / f"{name}-{label}-s{seed}.png")
                labels.append((name, label, seed))
    W, H = 260, 345
    sheet = Image.new("RGB", (W * 4, (H + 22) * len(cases)), "white"); dr = ImageDraw.Draw(sheet)
    for i, name in enumerate(cases):
        for j, (label, seed) in enumerate([("before", 42), ("before", 43), ("after", 42), ("after", 43)]):
            sheet.paste(Image.open(out / f"{name}-{label}-s{seed}.png").convert("RGB").resize((W, H)), (j * W, i * (H + 22) + 22))
            dr.text((j * W + 4, i * (H + 22) + 5), f"{name} {label} s{seed}", fill="black")
    sheet.save(out / "proportion-before-after.png")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
