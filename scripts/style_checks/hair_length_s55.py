"""S55 머리 길이(D-248, 사용자 11/2 T009 머리카락이 잘못 만들어짐): S54 T009 작업 폴더의 문장 그대로 머리 길이 강조 문장 / 머리 넘기기 동작만 바꿈 / 둘 다, 같은 seed 3개.
사용: python hair_length_s55.py <작업 폴더(flux)> <출력 폴더>
"""
import os, json, sys
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from style_redraw_worker import MEMORY_CAP_GIB
R = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
req = json.loads((R / "request.json").read_text(encoding="utf-8")); p = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
base = p["prompt"]
emph = "Keep the hair length exactly as in image 1: short hair that stays short; do not make the hair longer and do not tie it into a ponytail. "
no_tuck = base.replace("tucking a strand of hair behind one ear with one hand", "standing with one hand resting on the hip")
variants = {"emph": base.replace("Keep the same face,", emph + "Keep the same face,"), "noaction": no_tuck,
            "both": no_tuck.replace("Keep the same face,", emph + "Keep the same face,")}
torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
from diffusers import Flux2KleinPipeline
from diffusers.quantizers import PipelineQuantizationConfig
quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
pipe = Flux2KleinPipeline.from_pretrained(req["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True, local_files_only=True)
lora = Path(req["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
pipe.set_adapters(["style"], adapter_weights=[1.0]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
ref = Image.open(R / "character.png").convert("RGB")
for k, pr in variants.items():
    for s in (42, 43, 44):
        pipe(image=[ref], prompt=pr, width=1024, height=1360, num_inference_steps=4, guidance_scale=1.0, max_sequence_length=256,
             generator=torch.Generator(device="cpu").manual_seed(s)).images[0].save(out / f"{k}-s{s}.png")
pipe = None
from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
keys = ("short_hair", "medium_hair", "long_hair", "ponytail")
with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)) as t:
    for k in variants:
        for s in (42, 43, 44):
            sc = dict(t.analyze(Image.open(out / f"{k}-s{s}.png").convert("RGB")).raw_general_scores)
            print(k, s, {x: round(sc.get(x, 0), 2) for x in keys if sc.get(x, 0) > 0.15}, flush=True)
W, H = 240, 319
o = Image.new("RGB", (W * 3, (H + 22) * 3), "white"); d = ImageDraw.Draw(o)
for i, k in enumerate(variants):
    for j, s in enumerate((42, 43, 44)):
        o.paste(Image.open(out / f"{k}-s{s}.png").convert("RGB").resize((W, H)), (j * W, i * (H + 22) + 22)); d.text((j * W + 4, i * (H + 22) + 5), f"{k} s{s}", fill="black")
o.save(out / "hair-variants.png")
