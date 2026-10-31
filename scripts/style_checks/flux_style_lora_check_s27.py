"""S27 그림체 LoRA 확인(S26 학습 결과): 앱과 같은 FLUX.2-klein-4B 빠른 판(e7b7dc27, NF4, 4단계, 1024x1360, 상한 6.5GiB)에
학습 중간 저장본을 하나씩 붙여, 같은 캐릭터·장면·seed(앱과 같은 42·43·44)로 LoRA 없음과 비교한다. 동일성은 고정 특징표(D-228),
손 개수는 앱과 같은 손 검출(500단계 미리보기에서 손이 사라진 팔이 나와서, 원본보다 적으면 '손 사라짐').
문장은 앱 문장(flux_prompt) 앞에 학습 트리거 낱말을 붙인 것. 이 PC, GPU.
사용: python flux_style_lora_check_s27.py <LoRA 폴더(checkpoint-*/ 들어 있음)> <출력 폴더> [단계 또는 단계@세기 ...]
     (예: 500 500@0.7 — 세기를 낮추면 그림체는 약해지고 원래 특징은 더 남는지 본다, D-235 귀 문제)
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab"); sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image, ImageDraw
from finish_quality_s18 import FOLDERS, SCENE
from flux_style_lora_s26 import TRIGGER


def main():
    lora_dir, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    steps = sys.argv[3:] or sorted((p.name.split("-")[1] for p in lora_dir.glob("checkpoint-*")), key=int)
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, SEEDS, STEPS, flux_prompt
    from genai_lab.identity_anchors import anchors, compare, summary_ko
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from style_redraw_worker import MEMORY_CAP_GIB, count_hands
    req = json.loads((FOLDERS["U002"] / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(req["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    specs = [(s.split("@")[0], float(s.split("@")[1]) if "@" in s else 1.0) for s in steps]
    for s in dict.fromkeys(step for step, _ in specs):
        pipe.load_lora_weights(str(lora_dir / f"checkpoint-{s}"), weight_name="pytorch_lora_weights.safetensors", adapter_name=f"s{s}")
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    weights = {f"s{step}" + (f"w{w:g}" if w != 1.0 else ""): (f"s{step}", w) for step, w in specs}
    variants = ["none"] + list(weights)
    records = []
    for name, folder in FOLDERS.items():
        prepared = json.loads((folder / "prepare.json").read_text(encoding="utf-8"))
        base_prompt = flux_prompt(prepared["traits"], False, prepared.get("body_parts"), SCENE)
        # 시험용 문장 바꾸기(D-235 귀 문제): S27_PROMPT_SUB="옛말=>새말;..." — 앱 문장이 '동물 귀'만 말하고 종류를 말하지 않아서
        for pair in filter(None, os.environ.get("S27_PROMPT_SUB", "").split(";")):
            old, new = pair.split("=>")
            base_prompt = base_prompt.replace(old, new)
        ref = Image.open(folder / "character.png").convert("RGB")
        for v in variants:
            if v == "none":
                pipe.disable_lora(); prompt = base_prompt
            else:
                adapter, weight = weights[v]
                pipe.enable_lora(); pipe.set_adapters([adapter], adapter_weights=[weight]); prompt = f"{TRIGGER} {base_prompt}"
            for seed in SEEDS:
                torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                image = pipe(image=[ref], prompt=prompt, width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], num_inference_steps=STEPS,
                             guidance_scale=GUIDANCE, max_sequence_length=MAX_SEQUENCE_LENGTH,
                             generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
                image.save(out / f"{name}-{v}-s{seed}.png")
                records.append(dict(name=name, variant=v, seed=seed, seconds=round(time.time() - t0, 1),
                                    peak_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2)))
                print(records[-1], flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    for name, folder in FOLDERS.items():
        mine = [r for r in records if r["name"] == name]
        original_hands = json.loads((folder / "prepare.json").read_text(encoding="utf-8")).get("original_hands")
        for r, n in zip(mine, count_hands([Image.open(out / f"{name}-{r['variant']}-s{r['seed']}.png") for r in mine], req)):
            r["hands"], r["original_hands"] = n, original_hands
            r["hands_lost"] = None if n is None or original_hands is None else n < original_hands
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)) as tagger:
        tag = lambda p: dict(tagger.analyze(Image.open(p).convert("RGB")).raw_general_scores)
        for r in records:
            report = compare(anchors(tag(FOLDERS[r["name"]] / "character.png")), tag(out / f"{r['name']}-{r['variant']}-s{r['seed']}.png"))
            r["anchors"] = summary_ko(report)
            r["anchors_kept"] = f"{report['kept']}/{report['kept'] + report['changed'] + report['lost']}"
    W, H = 260, 345
    for name in FOLDERS:          # 줄 = seed, 칸 = LoRA 없음·단계. 칸 위에 손 개수(원본보다 적으면 '손-')
        sheet = Image.new("RGB", (W * len(variants), (H + 22) * len(SEEDS)), "white"); dr = ImageDraw.Draw(sheet)
        for row, seed in enumerate(SEEDS):
            for i, v in enumerate(variants):
                r = next(x for x in records if x["name"] == name and x["variant"] == v and x["seed"] == seed)
                sheet.paste(Image.open(out / f"{name}-{v}-s{seed}.png").convert("RGB").resize((W, H)), (i * W, row * (H + 22) + 22))
                flag = " HAND-" if r.get("hands_lost") else ""
                dr.text((i * W + 4, row * (H + 22) + 5), f"{name} {v} s{seed} hands {r.get('hands')}{flag}", fill="red" if flag else "black")
        sheet.save(out / f"{name}-lora-steps.png")
    (out / "lora-check.json").write_text(json.dumps(dict(lora_dir=str(lora_dir), trigger=TRIGGER, records=records),
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
