"""S16 그림체 시안 — 참조 없이 문장만(사용자 10/31: 그림체는 아직 학습한 게 없어 편집 경로에서는 문장으로 거의 안 움직인다, D-229).
목표(세이버 참고 4번 화려한 애니메이션 일러스트 / 3번 매끈한 반실사)를 FLUX.2-klein 문장만으로 그릴 수 있는지 보고, 고른 그림체를
FLUX 그림체 LoRA 학습 자료의 기준으로 삼는다. 학습 자료 규칙(D-203): 직접 짠 문장, 'Original adult character'로 시작, 작가·작품 이름 없음.
사용: python style_t2i_drafts_s16.py <출력 폴더>
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

SUBJECTS = {
    "F": ("Original adult character: a woman in her twenties with shoulder-length chestnut hair tied with a ribbon, green eyes, wearing a "
          "navy coat over a white blouse and a long pleated skirt, standing in a sunlit greenhouse full of plants, one hand raised to "
          "brush the hair back, full body, looking at the viewer."),
    "M": ("Original adult character: a man in his late twenties with short spiky silver hair, amber eyes, wearing a black turtleneck, "
          "a grey long coat and dark trousers, walking along a rainy city street at night, full body, looking at the viewer."),
}
STYLES = {
    "S4w": ("highly detailed semi-realistic anime illustration, dense digital painting: delicate thin line art, anime face with realistic "
            "soft skin rendering, subtle glossy highlights and light blush, detailed eyes with reflections, very fine individual hair "
            "strands with loose flyaway wisps and strong backlight, deep realistic fabric folds and creases, detailed material texture, "
            "strong directional sunlight with sharp deep shadows and soft bounce light, high detail density, cinematic colour grading"),
    "S5a": ("vibrant high-quality anime illustration: crisp clean line art, glossy vivid colours with rich saturated tones, smooth cel "
            "shading with soft gradients, luminous eyes with bright highlights, shiny silky hair with sharp light streaks, dramatic "
            "glowing light with soft bloom, floating light particles, dynamic composition, polished key visual finish"),
    "S5b": ("polished semi-realistic anime illustration: clean thin line art, smooth flawless skin with soft airbrushed shading and a "
            "natural light blush, large expressive anime eyes with clear reflections, sleek glossy hair with smooth strands, soft "
            "lighting with a gentle rim light, clean fabric shading without heavy texture, refined high-resolution finish"),
}
TAIL = " No lettering, logo, watermark or multiple people."


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, STEPS
    from genai_lab.style_redraw_pipeline import LEGACY
    from style_redraw_worker import MEMORY_CAP_GIB
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(str(LEGACY["flux_model"]), torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for subject, text in SUBJECTS.items():
        for key, style in STYLES.items():
            prompt = f"{text} Rendered as a {style}.{TAIL}"
            t0 = time.time()
            image = pipe(prompt=prompt, width=768, height=1024, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                         max_sequence_length=MAX_SEQUENCE_LENGTH, generator=torch.Generator(device="cpu").manual_seed(42)).images[0]
            image.save(out / f"{subject}-{key}.png")
            records.append(dict(subject=subject, style=key, prompt=prompt, seconds=round(time.time() - t0, 1)))
            print(subject, key, records[-1]["seconds"], "s", flush=True)
    W, H = 330, 440
    sheet = Image.new("RGB", (W * 3, (H + 22) * 2), "white"); dr = ImageDraw.Draw(sheet)
    for r, subject in enumerate(SUBJECTS):
        for i, key in enumerate(STYLES):
            sheet.paste(Image.open(out / f"{subject}-{key}.png").convert("RGB").resize((W, H)), (i * W, r * (H + 22) + 22))
            dr.text((i * W + 4, r * (H + 22) + 5), f"{subject} {key}", fill="black")
    sheet.save(out / "t2i-drafts.png")
    (out / "t2i-drafts.json").write_text(json.dumps(dict(peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                                                         model="FLUX.2-klein-4B (Apache-2.0)", seed=42, records=records),
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
