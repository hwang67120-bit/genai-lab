"""S49 털 수인 그림체(사용자 11/2 "그림체는 1판이 내가 제시한 것에 가장 가깝다, 퍼리는 따로 그림체를 적용해야 할 것 같다"): 1판 그림체 LoRA는
사람 캐릭터로 학습해 털 수인 얼굴을 사람 쪽으로 당긴다(F1·N2). 털 수인일 때 LoRA 세기를 낮추면(1.0·0.6·0.3·0) 털·동물 얼굴이 남는지,
그림체는 얼마나 남는지. 앱과 같은 설정(장면 4판), 자동 고치기·얼굴 다듬기는 끔, 같은 장면·seed.
사용: python furry_style_weight_s49.py <출력 폴더> <이름>=<앱 작업 폴더(flux)> ...
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

WEIGHTS = (1.0, 0.6, 0.3, 0.0)
SCENES = (39001, 39004)
SEED = 42
FURRY = ("furry", "furry_female", "furry_male", "body_fur", "animal_nose", "snout")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cases = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    import torch
    import genai_lab.style_redraw_flux as flux
    from genai_lab.scene_composer import compose_scene
    from style_redraw_worker import MEMORY_CAP_GIB
    first = json.loads((next(iter(cases.values())) / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(first["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    lora = Path(first["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    records = []
    for name, R in cases.items():
        prepared = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
        ref = Image.open(R / "character.png").convert("RGB")
        for sc in SCENES:
            prompt = flux.flux_prompt(prepared["traits"], False, prepared.get("body_parts"), compose_scene(sc)["text"],
                                      prepared.get("body_part_kinds"))
            for w in WEIGHTS:
                pipe.set_adapters(["style"], adapter_weights=[w])
                f = out / f"{name}-sc{sc}-w{w}.png"
                pipe(image=[ref], prompt=prompt, width=flux.OUTPUT_SIZE[0], height=flux.OUTPUT_SIZE[1], num_inference_steps=flux.STEPS,
                     guidance_scale=flux.GUIDANCE, max_sequence_length=flux.MAX_SEQUENCE_LENGTH,
                     generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0].save(f)
                records.append(dict(name=name, scene=sc, weight=w, file=f.name)); print(f.name, flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(first["model_cache"]), local_files_only=True)) as t:
        for r in records:
            s = dict(t.analyze(Image.open(out / r["file"]).convert("RGB")).raw_general_scores)
            r["furry_tags"] = {k: round(s.get(k, 0.0), 2) for k in FURRY if s.get(k, 0.0) >= 0.2}
            r["furry"] = round(s.get("furry", 0.0), 2)
    W, H = 260, 345
    rows = [(n, sc) for n in cases for sc in SCENES]
    sheet = Image.new("RGB", (W * (len(WEIGHTS) + 1), (H + 22) * len(rows)), "white"); dr = ImageDraw.Draw(sheet)
    for i, (n, sc) in enumerate(rows):
        y = i * (H + 22)
        sheet.paste(Image.open(cases[n] / "character.png").convert("RGB").resize((W, H)), (0, y + 22)); dr.text((4, y + 5), f"{n} original", fill="black")
        for j, w in enumerate(WEIGHTS):
            r = next(x for x in records if x["name"] == n and x["scene"] == sc and x["weight"] == w)
            sheet.paste(Image.open(out / r["file"]).convert("RGB").resize((W, H)), ((j + 1) * W, y + 22))
            dr.text(((j + 1) * W + 4, y + 5), f"LoRA {w}  furry {r['furry']}", fill="black")
    sheet.save(out / "furry-weights.png")
    (out / "furry-weights.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in records:
        print(r["name"], r["scene"], r["weight"], r["furry_tags"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
