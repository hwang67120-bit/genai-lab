"""S18 완성도 올리기(사용자 10/31 "(다)로 먼저": 그림체 방향은 FLUX에 맡기고 해상도·마감으로 세이버 참고 3번 쪽 완성도까지).
같은 캐릭터·장면·seed로 A(지금 768x1024) · B(처음부터 1024x1360) · C(A를 참조로 1152x1536 '같은 그림을 더 선명하게' 2단계).
LoRA 없음, 이 PC, GPU, 상한 6.5GiB. 색 맞춤은 하지 않은 원래 그림끼리 비교. 고정 특징표(D-228)로 동일성 근거.
사용: python finish_quality_s18.py <출력 폴더>
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

B = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025\character-style-binding-20261027")
FOLDERS = {"U002": B / "s6b-scene-unseen-20261031/U002-base", "U007": B / "s6c-scene-tail-20261031/U007-base"}
SCENE = "standing in a sunlit greenhouse full of plants, one hand raised to brush the hair back, looking at the viewer"
REFINE = ("Keep everything in image 1 exactly the same: the same character, face, hair, eyes, pose, outfit, colours, background and "
          "composition. Redraw it as a sharper, cleaner, higher-resolution finished illustration with crisp clean line art, smooth "
          "polished shading and fine details in the eyes, hair strands and fabric. No lettering, logo, watermark or multiple people.")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, STEPS, flux_prompt
    from genai_lab.identity_anchors import anchors, compare, summary_ko
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from style_redraw_worker import MEMORY_CAP_GIB
    req = json.loads((FOLDERS["U002"] / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(req["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    gen = lambda: torch.Generator(device="cpu").manual_seed(42)
    records = []
    def run(name, variant, refs, prompt, w, h):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        image = pipe(image=refs, prompt=prompt, width=w, height=h, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                     max_sequence_length=MAX_SEQUENCE_LENGTH, generator=gen()).images[0]
        image.save(out / f"{name}-{variant}.png")
        records.append(dict(name=name, variant=variant, size=[w, h], seconds=round(time.time() - t0, 1),
                            peak_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2)))
        print(records[-1], flush=True)
        return image
    for name, folder in FOLDERS.items():
        prepared = json.loads((folder / "prepare.json").read_text(encoding="utf-8"))
        prompt = flux_prompt(prepared["traits"], False, prepared.get("body_parts"), SCENE)
        ref = Image.open(folder / "character.png").convert("RGB")
        a = run(name, "A", [ref], prompt, 768, 1024)
        run(name, "B", [ref], prompt, 1024, 1360)
        run(name, "C", [a], REFINE, 1152, 1536)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)) as tagger:
        tag = lambda p: dict(tagger.analyze(Image.open(p).convert("RGB")).raw_general_scores)
        for r in records:
            found = anchors(tag(FOLDERS[r["name"]] / "character.png"))
            r["anchors"] = summary_ko(compare(found, tag(out / f"{r['name']}-{r['variant']}.png")))
    W, H = 330, 440
    sheet = Image.new("RGB", (W * 4, (H + 22) * 2), "white"); dr = ImageDraw.Draw(sheet)
    for row, name in enumerate(FOLDERS):
        tiles = [(FOLDERS[name] / "character.png", "original")] + [(out / f"{name}-{v}.png", v) for v in ("A", "B", "C")]
        for i, (p, t) in enumerate(tiles):
            sheet.paste(Image.open(p).convert("RGB").resize((W, H)), (i * W, row * (H + 22) + 22))
            dr.text((i * W + 4, row * (H + 22) + 5), f"{name} {t}", fill="black")
    sheet.save(out / "finish-overview.png")
    for name in FOLDERS:          # 얼굴 확대(세부 비교)
        crops = []
        for v in ("A", "B", "C"):
            im = Image.open(out / f"{name}-{v}.png").convert("RGB"); w, h = im.size
            crops.append(im.crop((int(w * 0.28), int(h * 0.05), int(w * 0.72), int(h * 0.33))).resize((440, 420)))
        strip = Image.new("RGB", (440 * 3, 442), "white"); d2 = ImageDraw.Draw(strip)
        for i, c in enumerate(crops):
            strip.paste(c, (i * 440, 22)); d2.text((i * 440 + 4, 5), f"{name} {'ABC'[i]} face", fill="black")
        strip.save(out / f"{name}-face-crops.png")
    (out / "finish.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
