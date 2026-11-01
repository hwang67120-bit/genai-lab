"""S39 장면 품질 측정(사용자 11/1 "손 숨김·간판 글자 보정, 나중 연동을 생각해 배경 품질도 조금이라도"): 앱과 같은 설정
(FLUX.2-klein-4B + 그림체 LoRA, 앱 문장 flux_prompt + 장면 조합)으로 캐릭터마다 장면 seed 여러 개를 그려 자동으로 센다:
손(손 검출 개수가 원본보다 적음 = 숨김), 글자(원본에 없던 글자 태그가 0.35 이상), CCIP. 배경은 사람 판정용 판.
자동 고치기(D-241)는 끈 상태 — 장면 문장 자체의 품질을 본다. 같은 장면 seed·그림 seed로 고치기 전후를 비교한다.
사용: python scene_quality_s39.py <출력 폴더> <이름>=<앱 작업 폴더(flux)> ...   (환경 변수 S39_SCENES=장면 seed 시작:개수, 기본 39001:12)
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

TEXT_TAGS = ("text", "english_text", "chinese_text", "japanese_text", "korean_text", "sign", "signature", "watermark", "artist_name",
             "speech_bubble", "logo", "character_name", "copyright_name", "web_address", "dated", "twitter_username", "translation_request")
TEXT_MIN = 0.35
SEED = 42


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cases = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    spec = os.environ.get("S39_SCENES", "39001:12")     # 시작:개수 또는 쉼표 목록(간판이 잘 생기는 장소만 고른 seed 등)
    if ":" in spec:
        start, count = (int(x) for x in spec.split(":"))
        scenes = list(range(start, start + count))
    else:
        scenes = [int(x) for x in spec.split(",")]
    import torch
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.scene_composer import VERSION, compose_scene
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, STEPS, STYLE_LORA_WEIGHT, flux_prompt
    from style_redraw_worker import MEMORY_CAP_GIB, count_hands
    first = json.loads((next(iter(cases.values())) / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(first["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    lora = Path(first["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[STYLE_LORA_WEIGHT]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    records = []
    started = time.time()
    for name, R in cases.items():
        prepared = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
        ref = Image.open(R / "character.png").convert("RGB")
        for sc in scenes:
            scene = compose_scene(sc)
            prompt = flux_prompt(prepared["traits"], False, prepared.get("body_parts"), scene["text"], prepared.get("body_part_kinds"))
            f = out / f"{name}-sc{sc}.png"
            pipe(image=[ref], prompt=prompt, width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                 max_sequence_length=MAX_SEQUENCE_LENGTH, generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0].save(f)
            records.append(dict(name=name, scene=sc, file=f.name, place=scene["parts"]["place"], action=scene["parts"]["action"],
                                original_hands=prepared.get("original_hands"), prompt=prompt))
            print(f.name, round(time.time() - started), "s", flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    import huggingface_hub.constants as hub
    hub.HF_HUB_CACHE = str(first["model_cache"])
    from imgutils.metrics import ccip_difference, ccip_extract_feature
    hands = count_hands([Image.open(out / r["file"]).convert("RGB") for r in records], first)
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(first["model_cache"]), local_files_only=True)) as t:
        originals = {n: dict(t.analyze(Image.open(R / "character.png").convert("RGB")).raw_general_scores) for n, R in cases.items()}
        feats = {n: ccip_extract_feature(Image.open(R / "character.png").convert("RGB")) for n, R in cases.items()}
        for r, n in zip(records, hands):
            im = Image.open(out / r["file"]).convert("RGB")
            s = dict(t.analyze(im).raw_general_scores)
            r["hands"] = n
            r["hands_lost"] = None if n is None or r["original_hands"] is None else n < r["original_hands"]
            r["new_text"] = sorted(k for k in TEXT_TAGS if s.get(k, 0.0) >= TEXT_MIN and originals[r["name"]].get(k, 0.0) < TEXT_MIN)
            r["ccip"] = round(float(ccip_difference(feats[r["name"]], ccip_extract_feature(im))), 3)
    summary = dict(images=len(records), hands_lost=sum(bool(r["hands_lost"]) for r in records),
                   text=sum(bool(r["new_text"]) for r in records), mean_ccip=round(sum(r["ccip"] for r in records) / len(records), 3))
    W, H = 240, 319
    for name in cases:
        mine = [r for r in records if r["name"] == name]
        sheet = Image.new("RGB", (W * 6, (H + 22) * ((len(mine) + 5) // 6)), "white"); dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(mine):
            x, y = (k % 6) * W, (k // 6) * (H + 22)
            sheet.paste(Image.open(out / r["file"]).convert("RGB").resize((W, H)), (x, y + 22))
            flags = ("HAND- " if r["hands_lost"] else "") + ("TEXT " if r["new_text"] else "")
            dr.text((x + 4, y + 5), f"{name} {r['scene']} {flags}", fill="red" if flags else "black")
        sheet.save(out / f"{name}-sheet.png")
    (out / "scene-quality.json").write_text(json.dumps(dict(composer_version=VERSION, seed=SEED, summary=summary, records=records),
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), flush=True)
    for r in records:
        if r["hands_lost"] or r["new_text"]:
            print(r["file"], "hands", r["hands"], "/", r["original_hands"], r["new_text"], "|", r["place"], "|", r["action"][:50], flush=True)


if __name__ == "__main__":
    main()
