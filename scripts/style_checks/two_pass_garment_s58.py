"""S58 옷 두 단계(D-249 6번 상반신 맨몸 + 옷 그림: 장면 한 번에는 옷을 안 입힘). ① 장면 없이 '원본 자세 그대로 옷만 갈아입히기'(flux_prompt
장면 없는 옷 문장) → ② 그 그림을 참조(그림 1)로 장면 문장(옷 그림 없이). 같은 seed. 서명 글자를 따라 그리지 않게 문장 하나.
사용: python two_pass_garment_s58.py <작업 폴더(flux)> <출력 폴더>
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import json, sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

NO_SIGN = "Do not copy any signature, watermark or text from image 1. "


def main():
    R, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    import torch
    import genai_lab.style_redraw_flux as flux
    from style_redraw_worker import MEMORY_CAP_GIB
    req = json.loads((R / "request.json").read_text(encoding="utf-8")); p = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
    gender = p.get("gender") or p.get("guessed_gender")
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(req["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    lora = Path(req["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[flux.STYLE_LORA_WEIGHT]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    ch = Image.open(R / "character.png").convert("RGB"); gar = Image.open(R / "garment.png").convert("RGB")
    dress = flux.flux_prompt(p["traits"], True, p.get("body_parts"), None, p.get("body_part_kinds"), p.get("framing", flux.FULL_BODY_FRAMING),
                             gender, p.get("garment_words"))
    dress = dress.replace("Keep the same face,", NO_SIGN + "Keep the same face,", 1)
    no_clothes = [dict(r, text="") if r.get("name") == "clothes" else r for r in p["traits"]]
    gen = lambda s: torch.Generator(device="cpu").manual_seed(s)
    for seed in (42, 43):
        dressed = pipe(image=[ch, gar], prompt=dress, width=ch.width, height=ch.height, num_inference_steps=4, guidance_scale=1.0,
                       max_sequence_length=256, generator=gen(seed)).images[0]
        dressed.save(out / f"dressed-s{seed}.png")
        scene = flux.flux_prompt(no_clothes, False, p.get("body_parts"), p["scene"], p.get("body_part_kinds"),
                                 p.get("framing", flux.FULL_BODY_FRAMING), gender)
        scene = scene.replace("Keep the same face,", NO_SIGN + "Keep the same face,", 1)
        pipe(image=[dressed], prompt=scene, width=1024, height=1360, num_inference_steps=4, guidance_scale=1.0, max_sequence_length=256,
             generator=gen(seed)).images[0].save(out / f"scene-s{seed}.png")
        print("seed", seed, flush=True)
    W, H = 300, 398
    o = Image.new("RGB", (W * 6, H + 22), "white"); d = ImageDraw.Draw(o)
    for j, (im, l) in enumerate([(ch, "original"), (gar, "garment"), (Image.open(out / "dressed-s42.png"), "step1 s42"), (Image.open(out / "scene-s42.png"), "step2 s42"),
                                 (Image.open(out / "dressed-s43.png"), "step1 s43"), (Image.open(out / "scene-s43.png"), "step2 s43")]):
        o.paste(im.convert("RGB").resize((W, H)), (j * W, 22)); d.text((j * W + 4, 5), l, fill="black")
    o.save(out / "two-pass.png"); print("DONE", flush=True)


if __name__ == "__main__":
    main()
