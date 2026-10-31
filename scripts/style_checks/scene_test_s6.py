"""S6 장면 시험(사용자 10/31, D-219): LoRA 없이 FLUX.2-klein으로 자세·장면을 바꿔도 같은 캐릭터로 보이는가. 이 PC, GPU.
캐릭터마다 장면 3개, seed 42. 결과 = 원래 생성(raw) + 색 맞춤(제품 후처리와 같음) + 손 개수.
처음 보는 캐릭터는 정식 모드 B 준비 단계(관절·안전·특징 문장·원본 마스크)만 돌리고 같은 자세 후보 생성은 건너뛴다.
사용: python scene_test_s6.py <출력 폴더> 이름=그림경로 ... (이름=기존 모드 B 실행 폴더도 가능: character.png·prepare.json이 있으면)
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")   # 손 검출 등 허브 라이브러리를 불러오기 전에
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
import numpy as np
from PIL import Image

SCENES = {
    "walk": "walking along a rainy city street at night with glowing shop signs, mid-stride, one hand in a pocket",
    "bench": "sitting on a wooden park bench on an autumn afternoon, relaxed, hands resting on the knees",
    "library": "standing in a library aisle between tall bookshelves, turning to look back over the shoulder, holding a book",
}


def scene_prompt(traits, scene, body_parts=None):
    from genai_lab.outfit_edit import body_parts_sentence
    from genai_lab.style_redraw_flux import STYLE
    described = "; ".join(r["text"].strip() for r in traits if r.get("text", "").strip())
    who = f" ({described})" if described else ""
    return (f"Show the same adult character as in the reference image{who} in a new scene: {scene}. Keep the same face, hairstyle, "
            f"hair colour, eye colour, skin tone, body proportions and outfit exactly as in the reference; do not lighten or darken "
            f"the skin. {body_parts_sentence(None if body_parts is None else tuple(body_parts))}Full body, the whole character inside the frame, face clearly visible. Render as a new {STYLE}. "
            f"No lettering, logo, watermark or multiple people.")


def prepared_folder(name, source, out, rt):
    """기존 실행 폴더면 그대로, 그림이면 준비 단계까지만 돌린 폴더. 거절(안전 등)이면 None."""
    from genai_lab.style_redraw_pipeline import default_run, run_style_redraw
    source = Path(source)
    if source.is_dir():
        return source
    folder = out / f"{name}-base"
    if not (folder / "prepare.json").is_file():
        base = default_run(rt)
        run_style_redraw(source, folder, rt, run=lambda stage, d: 1 if stage == "generate" else base(stage, d), adult_confirmed=True)
    prepared = json.loads((folder / "prepare.json").read_text(encoding="utf-8")) if (folder / "prepare.json").is_file() else {}
    print(name, "prepare", prepared.get("state"), prepared.get("code", ""), flush=True)
    return folder if prepared.get("state") == "ready" else None


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    from genai_lab.studio_generation import StudioRuntime
    rt = StudioRuntime.from_environment()
    folders, refused = {}, []
    for arg in sys.argv[2:]:
        name, source = arg.split("=", 1)
        folder = prepared_folder(name, source, out, rt)
        (folders.__setitem__(name, folder) if folder else refused.append(name))
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, PARTS, STEPS, body_part_tags, colour_match, missing_body_parts
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from style_redraw_worker import MEMORY_CAP_GIB, count_hands, person_and_parts
    req = json.loads((next(iter(folders.values())) / "request.json").read_text(encoding="utf-8"))
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
        prepared = json.loads((folder / "prepare.json").read_text(encoding="utf-8"))
        traits, body_parts = prepared["traits"], prepared.get("body_parts")
        ref = Image.open(folder / "character.png").convert("RGB")
        for key, scene in SCENES.items():
            prompt = scene_prompt(traits, scene, body_parts)
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
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)) as tagger:
        tags = [dict(tagger.analyze(Image.fromarray(r)).raw_general_scores) for _, _, r in raws]
    for (name, key, raw), rec, h, scores in zip(raws, records, hands, tags):
        folder = folders[name]
        original = np.asarray(Image.open(folder / "character.png").convert("RGB"))
        om = dict(np.load(folder / "original-masks.npz"))
        _, masks = person_and_parts(raw, req)
        fixed, shifts = colour_match(raw, original, masks, om, parts=PARTS)
        Image.fromarray(fixed).save(out / f"{name}-{key}.png")
        body_parts = json.loads((folder / "prepare.json").read_text(encoding="utf-8")).get("body_parts")
        rec.update(hands=h, colour_shift=shifts, body_parts=body_parts, missing_body_parts=missing_body_parts(body_parts, scores),
                   body_part_tags=body_part_tags(scores))
    original_hands = {n: json.loads((f / "prepare.json").read_text(encoding="utf-8")).get("original_hands") for n, f in folders.items()}
    (out / "scene-test.json").write_text(json.dumps(dict(peak_reserved_gib=peak, original_hands=original_hands, refused=refused,
                                                         folders={k: str(v) for k, v in folders.items()}, records=records,
                                                         lora=None, seed=42, steps=STEPS), ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE peak", peak, "refused", refused, "hands", [r.get("hands") for r in records],
          "missing", {f"{r['name']}-{r['scene']}": r["missing_body_parts"] for r in records if r["missing_body_parts"]}, flush=True)


if __name__ == "__main__":
    main()
