"""S34 캐릭터 기억 자동 학습 자료(사용자 11/1 "처음 보는 털 동물 캐릭터를 자동으로 학습한 다음 생성"): 원본 그림 1장 → 학습 자료를 사람 손 없이.
① 원본과 그 변형(좌우 뒤집기, 허벅지 위·상반신 자르기) ② 앱과 같은 FLUX + 그림체 LoRA로 장면 조합 여러 개 × seed 2를 그리되,
첫 진단(앱 후보가 과반에서 잃은 몸·입은 것 태그, identity_autofix)을 '빠진 특징' 문장으로 미리 붙인다 ③ 그린 것마다 지킬 태그를
몇 개 잃었는지(remaining)와 CCIP 차이로 자동 선별(전부 잃은 것 빼고 덜 잃은 순 상위 12장, CCIP 0.178 미만). 설명문 = 트리거 낱말 + 장면 문장(캐릭터는 트리거가 맡게).
사용: python character_lora_data_s34.py <앱 작업 폴더(flux)> <출력 폴더> [장면 수, 기본 10] [트리거 낱말, 기본 'f1chr character.']
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageOps

from genai_lab.identity_autofix import KINDS_TABLE, fixed_prompt, load_kinds, lost_tags, remaining, to_fix

TRIGGER = "f1chr character."
SEEDS = (11, 12)
TOP_K = 12          # 잃은 개수가 적은 순으로 상위 12장(첫 판 '잃은 것 1개 이하'는 20장 중 2장 — 동물 코·몸 털은 그려져도 점수가 낮게 나옴)
CCIP_SAME = 0.178


def main():
    global TRIGGER
    R, out = Path(sys.argv[1]), Path(sys.argv[2]); scenes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    TRIGGER = sys.argv[4] if len(sys.argv) > 4 else TRIGGER
    data = out / "data"; data.mkdir(parents=True, exist_ok=True)
    req = json.loads((R / "request.json").read_text(encoding="utf-8"))
    prepared = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
    import torch
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.scene_composer import compose_scene
    from genai_lab.style_redraw_flux import (GUIDANCE, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, SEEDS as APP_SEEDS, STEPS, STYLE_LORA_WEIGHT,
                                             flux_prompt)
    from style_redraw_worker import MEMORY_CAP_GIB
    settings = ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)
    kinds = load_kinds(Path(r"D:\Code\genai-lab") / KINDS_TABLE)
    with WdTagSession(settings) as t:
        tag = lambda p: dict(t.analyze(Image.open(p).convert("RGB")).raw_general_scores)
        original = tag(R / "character.png")
        app = [tag(R / f"result-raw-seed{s}.png") for s in APP_SEEDS]
    watch = to_fix(lost_tags(original, app), kinds)
    print("watch", watch, flush=True)
    rows = []
    src = Image.open(R / "character.png").convert("RGB")
    w, h = src.size
    variants = {"orig": src, "orig-flip": ImageOps.mirror(src), "orig-thigh": src.crop((0, 0, w, int(h * 0.62))),
                "orig-upper": src.crop((int(w * 0.1), 0, int(w * 0.9), int(h * 0.42)))}
    for name, im in variants.items():
        im.save(data / f"{name}.png")
        rows.append(dict(file=f"{name}.png", source="original", text=f"{TRIGGER} The character standing with one hand on the hip, simple background."))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(req["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    lora = Path(req["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[STYLE_LORA_WEIGHT]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    made = []
    started = time.time()
    for i in range(scenes):
        scene = compose_scene(34000 + i)["text"]
        prompt = fixed_prompt(flux_prompt(prepared["traits"], False, prepared.get("body_parts"), scene, prepared.get("body_part_kinds")), watch)
        for seed in SEEDS:
            path = out / f"gen-{i:02d}-s{seed}.png"
            pipe(image=[src], prompt=prompt, width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                 max_sequence_length=MAX_SEQUENCE_LENGTH, generator=torch.Generator(device="cpu").manual_seed(seed)).images[0].save(path)
            made.append(dict(file=path.name, scene=scene, seed=seed))
            print(path.name, round(time.time() - started), "s", flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    import huggingface_hub.constants as hub
    hub.HF_HUB_CACHE = str(req["model_cache"])
    from imgutils.metrics import ccip_difference, ccip_extract_feature
    base = ccip_extract_feature(src)
    with WdTagSession(settings) as t:
        for m in made:
            scores = dict(t.analyze(Image.open(out / m["file"]).convert("RGB")).raw_general_scores)
            m["lost"] = remaining(original, scores, watch)
            m["ccip"] = round(float(ccip_difference(base, ccip_extract_feature(Image.open(out / m["file"]).convert("RGB")))), 3)
    # 지킬 태그를 전부 잃은 것(사람으로 돌아간 것)·CCIP 다른 캐릭터는 빼고, 덜 잃은 순 → CCIP 순으로 상위 TOP_K
    ok = sorted((m for m in made if len(m["lost"]) < len(watch) and m["ccip"] < CCIP_SAME), key=lambda m: (len(m["lost"]), m["ccip"]))
    chosen = {m["file"] for m in ok[:TOP_K]}
    for m in made:
        m["chosen"] = m["file"] in chosen
        if m["chosen"]:
            Image.open(out / m["file"]).save(data / m["file"])
            rows.append(dict(file=m["file"], source="generated", text=f"{TRIGGER} {m['scene']}", lost=m["lost"], ccip=m["ccip"]))
    with (data / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(dict(file_name=r["file"], text=r["text"]), ensure_ascii=False) + "\n")
    for r in rows:
        r["sha256"] = hashlib.sha256((data / r["file"]).read_bytes()).hexdigest()
    (out / "character-data.json").write_text(json.dumps(dict(source=str(R), trigger=TRIGGER, watch=watch, rule=dict(top_k=TOP_K, ccip=CCIP_SAME, exclude="지킬 태그 전부 잃음"),
                                                             generated=made, data=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    print("DATA", len(rows), "chosen", sum(m["chosen"] for m in made), "of", len(made), flush=True)


if __name__ == "__main__":
    main()
