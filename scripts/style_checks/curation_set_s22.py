"""S22 고르기용 그림 묶음(사용자 10/31 "1단계부터": 남은 클라우드 크레딧으로 '고른 그림으로 다듬는 LoRA'를 시험하기 전 자료).
고정한 그림체(D-230: S4w 문장, 처음부터 1024x1360, FLUX.2-klein-4B e7b7dc27, Apache-2.0)로 직접 짠 문장의 오리지널 성인 캐릭터를 그린다.
학습 자료 규칙(D-203): 'Original adult character'로 시작, 작가·작품 이름 없음. 어려 보이는 옷(교복 등)·노출 옷 없음.
다 그린 뒤 WD 태그 안전 검사(blocked·review 있으면 제외), 그림마다 출처(모델·리비전·seed·문장·SHA) 기록, 번호 붙인 12장 판 5장.
사용: python curation_set_s22.py <출력 폴더> [장 수, 기본 60]
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, hashlib, json, random, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

WHO = ("a woman in her twenties", "a man in his twenties", "a woman in her thirties", "a man in his thirties",
       "a tall woman in her late twenties", "a broad-shouldered man in his late twenties")
SKIN = ("fair", "light", "olive", "tan", "brown", "dark brown")
HAIR = ("short black hair", "long straight chestnut hair", "a high blonde ponytail", "short spiky silver hair", "wavy shoulder-length red hair",
        "a navy blue bob", "long braided brown hair", "short curly black hair", "an undercut with dark green hair on top", "long white hair tied low",
        "pink hair in a loose side braid", "messy ash-grey hair")
EYES = ("brown", "green", "blue", "amber", "grey", "violet", "hazel")
OUTFIT = ("a beige trench coat over a turtleneck sweater and slacks", "a knit cardigan, a blouse and a long pleated skirt",
          "a leather jacket, a plain t-shirt and jeans", "a tailored navy business suit", "an oversized hoodie and cargo pants",
          "a military-style long coat with boots", "a fantasy knight armour with a cloak", "a hooded mage robe with embroidered trim",
          "a denim jacket over a sundress that reaches the ankles", "a chef's white jacket and apron", "a mechanic's jumpsuit with tool belt",
          "a wool coat, scarf and gloves")
ACCESSORY = ("", "", "small earrings", "round glasses", "a hair clip", "a wristwatch", "a scarf")
TAIL = " No lettering, logo, watermark or multiple people."


def describe(rng):
    accessory = rng.choice(ACCESSORY)
    return (f"Original adult character: {rng.choice(WHO)} with {rng.choice(SKIN)} skin, {rng.choice(HAIR)} and {rng.choice(EYES)} eyes, "
            f"wearing {rng.choice(OUTFIT)}" + (f" and {accessory}" if accessory else "") + ".")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.scene_composer import compose_scene
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, STEPS, STYLE
    from genai_lab.style_redraw_pipeline import LEGACY
    from style_redraw_worker import MEMORY_CAP_GIB
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(str(LEGACY["flux_model"]), torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    records = []
    for i in range(1, count + 1):
        rng = random.Random(1000 + i)
        prompt = f"{describe(rng)} {compose_scene(2000 + i)['text']} Rendered as a {STYLE}.{TAIL}"
        seed = 5000 + i
        t0 = time.time()
        image = pipe(prompt=prompt, width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                     max_sequence_length=MAX_SEQUENCE_LENGTH, generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
        path = out / f"C{i:03d}.png"
        image.save(path)
        records.append(dict(id=f"C{i:03d}", file=path.name, seed=seed, prompt=prompt, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            seconds=round(time.time() - t0, 1)))
        print(records[-1]["id"], records[-1]["seconds"], "s", flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.free_generation import safety_observation
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(r"D:\genai-cache\huggingface"), local_files_only=True)) as tagger:
        for r in records:
            safety = safety_observation(dict(tagger.analyze(Image.open(out / r["file"]).convert("RGB")).raw_general_scores))
            r["safety"] = safety
            r["excluded"] = bool(safety["blocked"] or safety["review"])
    usable = [r for r in records if not r["excluded"]]
    W, H = 240, 319
    for n in range(0, len(usable), 12):
        group = usable[n:n + 12]
        sheet = Image.new("RGB", (W * 6, (H + 24) * 2), "white"); dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(group):
            row, col = divmod(k, 6)
            sheet.paste(Image.open(out / r["file"]).convert("RGB").resize((W, H)), (col * W, row * (H + 24) + 24))
            dr.rectangle([col * W, row * (H + 24), col * W + 70, row * (H + 24) + 22], fill="black")
            dr.text((col * W + 6, row * (H + 24) + 5), r["id"], fill="white")
        sheet.save(out / f"pick-sheet-{n // 12 + 1}.png")
    (out / "curation-set.json").write_text(json.dumps(dict(
        model="black-forest-labs/FLUX.2-klein-4B", revision="e7b7dc27f91deacad38e78976d1f2b499d76a294", license="Apache-2.0",
        size=list(OUTPUT_SIZE), steps=STEPS, guidance=GUIDANCE, style="S4w (D-230 고정 그림체)", rule="D-203 직접 짠 문장, 작가·작품 이름 없음",
        records=records, excluded=[r["id"] for r in records if r["excluded"]]), ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE usable", len(usable), "excluded", [r["id"] for r in records if r["excluded"]], flush=True)


if __name__ == "__main__":
    main()
