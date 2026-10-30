"""S6 장면 시험(사용자 10/31): LoRA 없이 FLUX.2-klein으로 자세·장면을 바꿔도 같은 캐릭터로 보이는가. 이 PC, GPU.
캐릭터 3명(V002·T017 = 10명 중, U001 = 처음 보는 그림) × 장면 3개, seed 42. 결과 = 원래 생성(raw) + 색 맞춤(제품 후처리와 같음) + 손 개수.
사용: python scene_test_s6.py <출력 폴더>
"""
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
import numpy as np
from PIL import Image

BASE = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025\character-style-binding-20261027")
UNSEEN = Path(r"C:\Users\user\Downloads\참조 이미지\4a46d080207f26a008ef828ac42f0b33.jpg")
SCENES = {
    "walk": "walking along a rainy city street at night with glowing shop signs, mid-stride, one hand in a pocket",
    "bench": "sitting on a wooden park bench on an autumn afternoon, relaxed, hands resting on the knees",
    "library": "standing in a library aisle between tall bookshelves, turning to look back over the shoulder, holding a book",
}


def scene_prompt(traits, scene):
    from genai_lab.style_redraw_flux import STYLE
    described = "; ".join(r["text"].strip() for r in traits if r.get("text", "").strip())
    who = f" ({described})" if described else ""
    return (f"Show the same adult character as in the reference image{who} in a new scene: {scene}. Keep the same face, hairstyle, "
            f"hair colour, eye colour, skin tone, body proportions and outfit exactly as in the reference; do not lighten or darken "
            f"the skin. Full body, the whole character inside the frame, face clearly visible. Render as a new {STYLE}. "
            f"No lettering, logo, watermark or multiple people.")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    from genai_lab.style_redraw_pipeline import run_style_redraw
    from genai_lab.studio_generation import StudioRuntime
    rt = StudioRuntime.from_environment()
    # 처음 보는 캐릭터: 정식 모드 B로 특징 문장·원본 마스크·같은 자세 기준 결과를 만든다
    unseen_dir = out / "U001-base"
    if not (unseen_dir / "outcome.json").is_file():
        t0 = time.time()
        o = run_style_redraw(UNSEEN, unseen_dir, rt)
        print("U001 base", o["state"], o.get("message", ""), round(time.time() - t0), "s", flush=True)
    folders = {"V002": BASE / "s5s-merged-10-20261031" / "V002", "T017": BASE / "s5s-merged-10-20261031" / "T017", "U001": unseen_dir}
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, PARTS, STEPS, colour_match
    from style_redraw_worker import MEMORY_CAP_GIB, count_hands, person_and_parts
    req = json.loads((unseen_dir / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(req["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    torch.cuda.reset_peak_memory_stats()
    raws, records = [], []
    for name, folder in folders.items():
        traits = json.loads((folder / "prepare.json").read_text(encoding="utf-8"))["traits"]
        ref = Image.open(folder / "character.png").convert("RGB")
        for key, scene in SCENES.items():
            prompt = scene_prompt(traits, scene)
            t0 = time.time()
            image = pipe(image=[ref], prompt=prompt, width=ref.width, height=ref.height, num_inference_steps=STEPS,
                         guidance_scale=GUIDANCE, max_sequence_length=MAX_SEQUENCE_LENGTH,
                         generator=torch.Generator(device="cpu").manual_seed(42)).images[0]
            image.save(out / f"{name}-{key}-raw.png")
            raws.append((name, key, np.asarray(image)))
            records.append(dict(name=name, scene=key, prompt=prompt, seconds=round(time.time() - t0, 1)))
            print(name, key, records[-1]["seconds"], "s", flush=True)
    peak = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    hands = count_hands([r for _, _, r in raws], req)
    for (name, key, raw), rec, h in zip(raws, records, hands):
        folder = folders[name]
        original = np.asarray(Image.open(folder / "character.png").convert("RGB"))
        om = dict(np.load(folder / "original-masks.npz"))
        _, masks = person_and_parts(raw, req)
        fixed, shifts = colour_match(raw, original, masks, om, parts=PARTS)
        Image.fromarray(fixed).save(out / f"{name}-{key}.png")
        rec.update(hands=h, colour_shift=shifts)
    original_hands = {n: json.loads((f / "prepare.json").read_text(encoding="utf-8")).get("original_hands") for n, f in folders.items()}
    (out / "scene-test.json").write_text(json.dumps(dict(peak_reserved_gib=peak, original_hands=original_hands, records=records,
                                                         lora=None, seed=42, steps=STEPS), ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE peak", peak, flush=True)


if __name__ == "__main__":
    main()
