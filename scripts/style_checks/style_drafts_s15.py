"""S15 그림체 시안(사용자 10/31 "그림체 시안 진행해 줘", 목표 = 세이버 참고 4번 화려한 애니메이션 일러스트 또는 3번 매끈한 반실사).
같은 캐릭터·같은 장면·같은 seed에 그림체 문장만 바꾼다: S4w(지금, 1번 쪽) · S5a(4번 쪽) · S5b(3번 쪽). LoRA 없음. 이 PC, GPU.
결과마다 고정 특징표(D-228)로 동일성 근거를 같이 남긴다.
사용: python style_drafts_s15.py <출력 폴더>
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
import numpy as np
from PIL import Image, ImageDraw

B = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025\character-style-binding-20261027")
FOLDERS = {"U002": B / "s6b-scene-unseen-20261031/U002-base", "U007": B / "s6c-scene-tail-20261031/U007-base"}
SCENE = "standing in a sunlit greenhouse full of plants, one hand raised to brush the hair back, looking at the viewer"
STYLES = {
    "S5a": ("vibrant high-quality anime illustration: crisp clean line art, glossy vivid colours with rich saturated tones, smooth cel "
            "shading with soft gradients, luminous eyes with bright highlights, shiny silky hair with sharp light streaks, dramatic "
            "glowing light with soft bloom, floating light particles, dynamic composition, polished key visual finish"),
    "S5b": ("polished semi-realistic anime illustration: clean thin line art, smooth flawless skin with soft airbrushed shading and a "
            "natural light blush, large expressive anime eyes with clear reflections, sleek glossy hair with smooth strands, soft "
            "lighting with a gentle rim light, clean fabric shading without heavy texture, refined high-resolution finish"),
}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, PARTS, STEPS, STYLE, colour_match, flux_prompt
    from genai_lab.identity_anchors import compare, summary_ko
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from style_redraw_worker import MEMORY_CAP_GIB, person_and_parts
    styles = dict(S4w=STYLE, **STYLES)
    req = json.loads((FOLDERS["U002"] / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(req["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    torch.cuda.reset_peak_memory_stats()
    raws = []
    for name, folder in FOLDERS.items():
        prepared = json.loads((folder / "prepare.json").read_text(encoding="utf-8"))
        base = flux_prompt(prepared["traits"], False, prepared.get("body_parts"), SCENE)
        ref = Image.open(folder / "character.png").convert("RGB")
        for key, text in styles.items():
            prompt = base.replace(STYLE, text)
            t0 = time.time()
            image = pipe(image=[ref], prompt=prompt, width=ref.width, height=ref.height, num_inference_steps=STEPS,
                         guidance_scale=GUIDANCE, max_sequence_length=MAX_SEQUENCE_LENGTH,
                         generator=torch.Generator(device="cpu").manual_seed(42)).images[0]
            image.save(out / f"{name}-{key}-raw.png")
            raws.append((name, key, np.asarray(image), prompt))
            print(name, key, round(time.time() - t0, 1), "s", flush=True)
    peak = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    rows = []
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)) as tagger:
        for name, key, raw, prompt in raws:
            folder = FOLDERS[name]
            prepared = json.loads((folder / "prepare.json").read_text(encoding="utf-8"))
            _, masks = person_and_parts(raw, req)
            fixed, _ = colour_match(raw, np.asarray(Image.open(folder / "character.png").convert("RGB")), masks,
                                    dict(np.load(folder / "original-masks.npz")), parts=PARTS, scene_lighting=True)
            Image.fromarray(fixed).save(out / f"{name}-{key}.png")
            original_anchors = prepared.get("identity_anchors")
            if original_anchors is None:
                from genai_lab.identity_anchors import anchors
                original_anchors = anchors(dict(tagger.analyze(Image.open(folder / "character.png").convert("RGB")).raw_general_scores))
            report = compare(original_anchors, dict(tagger.analyze(Image.fromarray(fixed)).raw_general_scores))
            rows.append(dict(name=name, style=key, anchors=summary_ko(report), prompt=prompt))
    W, H = 330, 440
    sheet = Image.new("RGB", (W * 4, (H + 44) * 2), "white"); dr = ImageDraw.Draw(sheet)
    for r, name in enumerate(FOLDERS):
        tiles = [(FOLDERS[name] / "character.png", "original", "")] + [
            (out / f"{name}-{k}.png", k, next(x["anchors"] for x in rows if x["name"] == name and x["style"] == k)) for k in styles]
        for i, (p, t, a) in enumerate(tiles):
            sheet.paste(Image.open(p).convert("RGB").resize((W, H)), (i * W, r * (H + 44) + 22))
            dr.text((i * W + 4, r * (H + 44) + 5), f"{name} {t}", fill="black")
            dr.text((i * W + 4, r * (H + 44) + 24 + H), a[:52], fill="black")
    sheet.save(out / "style-drafts.png")
    (out / "style-drafts.json").write_text(json.dumps(dict(peak_reserved_gib=peak, scene=SCENE, styles=styles, rows=rows),
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE peak", peak, flush=True)


if __name__ == "__main__":
    main()
