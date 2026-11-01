"""S32 자동 고치기 한 바퀴 시험: 앞 실행(작업 폴더)의 원본·후보 태그를 비교해 사라진 태그를 찾고, AI 태그 분류(tag_kinds)로
몸·입은 것만 남겨 '빠진 특징' 문장을 자동으로 붙여 같은 seed로 다시 그린다. 사람이 고른 항목 없음.
사용: python autofix_round_s32.py <작업 폴더(flux)> <출력 폴더>
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image

PRESENT, GONE = 0.5, 0.35


def main():
    R, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    req = json.loads((R / "request.json").read_text(encoding="utf-8"))
    prompt = json.loads((R / "prepare.json").read_text(encoding="utf-8"))["prompt"]
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.style_redraw_flux import SEEDS
    settings = ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)
    with WdTagSession(settings) as t:
        tag = lambda p: dict(t.analyze(Image.open(p).convert("RGB")).raw_general_scores)
        original = tag(R / "character.png")
        before = {s: tag(R / f"result-raw-seed{s}.png") for s in SEEDS}
    lost = sorted({k for s in SEEDS for k, v in original.items() if v >= PRESENT and before[s].get(k, 0.0) < GONE})
    from genai_lab.tag_kinds import TagKindClassifier
    model = Path(req["flux_model"])
    with TagKindClassifier(model / "text_encoder", model / "tokenizer") as c:
        kinds = c.classify(lost)
    keep = [k for k in lost if kinds[k]["kind"] in ("body", "outfit")]
    fix = ("These features of the character in image 1 are missing and must be shown clearly: "
           + ", ".join(k.replace("_", " ") for k in keep) + ". ") if keep else ""
    fixed_prompt = prompt.replace("Keep the same face,", fix + "Keep the same face,", 1) if fix else prompt
    print("lost", lost); print("keep", keep)
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, STEPS, STYLE_LORA_WEIGHT
    from style_redraw_worker import MEMORY_CAP_GIB
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    lora = Path(req["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[STYLE_LORA_WEIGHT]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    ref = Image.open(R / "character.png").convert("RGB")
    for s in SEEDS:
        pipe(image=[ref], prompt=fixed_prompt, width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], num_inference_steps=STEPS, guidance_scale=GUIDANCE,
             max_sequence_length=MAX_SEQUENCE_LENGTH, generator=torch.Generator(device="cpu").manual_seed(s)).images[0].save(out / f"fixed-seed{s}.png")
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    with WdTagSession(settings) as t:
        after = {s: dict(t.analyze(Image.open(out / f"fixed-seed{s}.png").convert("RGB")).raw_general_scores) for s in SEEDS}
    rows = {s: dict(before={k: round(before[s].get(k, 0.0), 2) for k in keep}, after={k: round(after[s].get(k, 0.0), 2) for k in keep},
                    still_lost=[k for k in keep if after[s].get(k, 0.0) < GONE]) for s in SEEDS}
    (out / "autofix-round.json").write_text(json.dumps(dict(lost=lost, kinds=kinds, keep=keep, fix=fix, prompt=fixed_prompt, rows=rows),
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    for s in SEEDS:
        print(s, rows[s])


if __name__ == "__main__":
    main()
