"""S45 그림체 문장 S4w → S5 비교(사용자 11/2 "얼굴은 2D 같아야 하고 머리카락 묘사를 높이고 옷 섬유 묘사를 더 — 아직 진짜 사람을 미형으로
만든 것 같다"). S4w는 'realistic soft skin rendering', 'deep realistic fabric folds'처럼 실사 쪽을 요구한다. 앱과 같은 설정
(그림체 LoRA, 장면 4판, 자동 고치기·얼굴 다듬기는 끔)으로 같은 캐릭터·장면·seed를 두 문장으로 그려 얼굴 확대 판을 만든다.
사용: python style_sentence_s45.py <출력 폴더> <이름>=<앱 작업 폴더(flux)> ...
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

S5 = ("high-quality 2D anime illustration: a clean 2D anime face with large expressive anime eyes, detailed irises with bright "
      "highlights, a small simple nose and small mouth, soft cel shading on the skin with gentle blush and no realistic skin texture; "
      "extremely detailed hair with many fine individual strands, layered locks, flyaway wisps and glossy highlights; detailed clothing "
      "with visible fabric texture, weave, stitching, seams and natural folds; crisp clean line art, soft lighting with gentle bloom, "
      "polished professional finish")
SCENES = (39001, 39004, 40009)
SEED = 42


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cases = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    import torch
    import genai_lab.style_redraw_flux as flux
    from genai_lab.scene_composer import compose_scene
    from style_redraw_worker import MEMORY_CAP_GIB
    styles = {"S4w": flux.STYLE, "S5": S5}
    first = json.loads((next(iter(cases.values())) / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(first["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    lora = Path(first["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[flux.STYLE_LORA_WEIGHT]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    records = []
    for name, R in cases.items():
        prepared = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
        ref = Image.open(R / "character.png").convert("RGB")
        for sc in SCENES:
            for label, style in styles.items():
                flux.STYLE = style
                prompt = flux.flux_prompt(prepared["traits"], False, prepared.get("body_parts"), compose_scene(sc)["text"],
                                          prepared.get("body_part_kinds"))
                f = out / f"{name}-sc{sc}-{label}.png"
                pipe(image=[ref], prompt=prompt, width=flux.OUTPUT_SIZE[0], height=flux.OUTPUT_SIZE[1], num_inference_steps=flux.STEPS,
                     guidance_scale=flux.GUIDANCE, max_sequence_length=flux.MAX_SEQUENCE_LENGTH,
                     generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0].save(f)
                records.append(dict(name=name, scene=sc, style=label, file=f.name)); print(f.name, flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    from genai_lab.face_detail import head_box, square
    from genai_lab.studio_generation import StudioRuntime
    head_cache = StudioRuntime.from_environment().head_cache
    W = 330
    rows = [(n, sc) for n in cases for sc in SCENES]
    sheet = Image.new("RGB", (W * 2, (W + 22) * len(rows)), "white"); dr = ImageDraw.Draw(sheet)
    import numpy as np
    for i, (n, sc) in enumerate(rows):
        for j, label in enumerate(styles):
            im = Image.open(out / f"{n}-sc{sc}-{label}.png").convert("RGB")
            found = head_box(np.asarray(im), head_cache)
            l, t, s = square(found[0], *im.size) if found else (0, 0, min(im.size))
            sheet.paste(im.crop((l, t, l + s, t + s)).resize((W, W)), (j * W, i * (W + 22) + 22))
            dr.text((j * W + 4, i * (W + 22) + 5), f"{n} {sc} {label}", fill="black")
    sheet.save(out / "style-sentence-faces.png")
    (out / "style-sentence.json").write_text(json.dumps(dict(styles=styles, seed=SEED, records=records), ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
