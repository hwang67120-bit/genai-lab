"""S48 그림체 LoRA 1판(앱의 500단계) vs 2판(S47 부위별 자료) 저장본 비교. 앱과 같은 설정(4비트·4단계·1024x1360, 장면 4판), 같은 캐릭터·장면·seed.
2판은 그림체 문장 S5(2D 얼굴·머리카락·옷감, S45)와 함께. 얼굴 확대(머리 검출 정사각형)와 전신 판.
사용: python style_lora_v2_eval_s48.py <2판 LoRA 폴더(checkpoint-*)> <출력 폴더> <단계,…> <이름>=<앱 작업 폴더(flux)> ...
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab"); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image, ImageDraw

SCENE = 39001
SEED = 42


def main():
    lora_dir, out, steps = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
    cases = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[4:])}
    out.mkdir(parents=True, exist_ok=True)
    import torch
    import genai_lab.style_redraw_flux as flux
    from genai_lab.scene_composer import compose_scene
    from style_sentence_s45 import S5
    from style_redraw_worker import MEMORY_CAP_GIB
    first = json.loads((next(iter(cases.values())) / "request.json").read_text(encoding="utf-8"))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(first["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    v1 = Path(first["flux_style_lora"]); pipe.load_lora_weights(str(v1.parent), weight_name=v1.name, adapter_name="v1")
    for s in steps:
        pipe.load_lora_weights(str(lora_dir / f"checkpoint-{s}"), weight_name="pytorch_lora_weights.safetensors", adapter_name=f"v2s{s}")
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    variants = [("v1", flux.STYLE)] + [(f"v2s{s}", S5) for s in steps]
    scene = compose_scene(SCENE)["text"]
    for name, R in cases.items():
        prepared = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
        ref = Image.open(R / "character.png").convert("RGB")
        for v, style in variants:
            pipe.set_adapters([v], adapter_weights=[1.0])
            saved = flux.STYLE; flux.STYLE = style
            prompt = flux.flux_prompt(prepared["traits"], False, prepared.get("body_parts"), scene, prepared.get("body_part_kinds"))
            flux.STYLE = saved
            pipe(image=[ref], prompt=prompt, width=flux.OUTPUT_SIZE[0], height=flux.OUTPUT_SIZE[1], num_inference_steps=flux.STEPS,
                 guidance_scale=flux.GUIDANCE, max_sequence_length=flux.MAX_SEQUENCE_LENGTH,
                 generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0].save(out / f"{name}-{v}.png")
            print(name, v, flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    from genai_lab.face_detail import head_box, square
    from genai_lab.studio_generation import StudioRuntime
    head_cache = StudioRuntime.from_environment().head_cache
    W = 300
    labels = [v for v, _ in variants]
    faces = Image.new("RGB", (W * len(labels), (W + 22) * len(cases)), "white"); df = ImageDraw.Draw(faces)
    full = Image.new("RGB", (W * len(labels), (int(W * 1.33) + 22) * len(cases)), "white"); dg = ImageDraw.Draw(full)
    for i, name in enumerate(cases):
        for j, v in enumerate(labels):
            im = Image.open(out / f"{name}-{v}.png").convert("RGB")
            found = head_box(np.asarray(im), head_cache)
            l, t, s = square(found[0], *im.size) if found else (0, 0, min(im.size))
            faces.paste(im.crop((l, t, l + s, t + s)).resize((W, W)), (j * W, i * (W + 22) + 22)); df.text((j * W + 4, i * (W + 22) + 5), f"{name} {v}", fill="black")
            full.paste(im.resize((W, int(W * 1.33))), (j * W, i * (int(W * 1.33) + 22) + 22)); dg.text((j * W + 4, i * (int(W * 1.33) + 22) + 5), f"{name} {v}", fill="black")
    faces.save(out / "v1-v2-faces.png"); full.save(out / "v1-v2-full.png")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
