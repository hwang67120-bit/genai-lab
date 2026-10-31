"""다른 GPU 일관성 시험(D-231) — Pod에서 FLUX 생성 단계만 이 PC와 똑같이 돈다(style_redraw_worker.generate_flux의 생성 부분).
입력: 이 PC의 일관성 시험(s20) run-a 폴더의 prepare.json(지시문)·character.png·garment.png. 설정: seed 42·43·44, 4단계, CFG 1.0,
문장 길이 256, 1024x1360, NF4(transformer·text_encoder), CPU 내려 두기, VAE 타일, 메모리 상한 6.5GiB(이 PC와 같게).
후처리(색 맞춤·태그·고정 특징)는 이 PC에서 같은 코드로 한다. 사용: python xgpu_generate.py <입력 폴더> <출력 폴더> <flux 폴더>
"""
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from PIL import Image

SEEDS, STEPS, GUIDANCE, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, CAP_GIB = (42, 43, 44), 4, 1.0, 256, (1024, 1360), 6.5


def main():
    source, out, flux = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    torch.cuda.set_per_process_memory_fraction(CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(str(flux), torch_dtype=torch.bfloat16, quantization_config=quant,
                                              low_cpu_mem_usage=True, local_files_only=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    report = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cases={})
    for case in sorted(p for p in source.iterdir() if p.is_dir()):
        prepared = json.loads((case / "prepare.json").read_text(encoding="utf-8"))
        refs = [Image.open(case / "character.png").convert("RGB")]
        if prepared.get("with_garment"):
            refs.append(Image.open(case / "garment.png").convert("RGB"))
        (out / case.name).mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            image = pipe(image=refs, prompt=prepared["prompt"], width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], num_inference_steps=STEPS,
                         guidance_scale=GUIDANCE, max_sequence_length=MAX_SEQUENCE_LENGTH,
                         generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
            image.save(out / case.name / f"result-raw-seed{seed}.png")
            rows.append(dict(seed=seed, seconds=round(time.time() - t0, 1)))
            print(case.name, seed, rows[-1]["seconds"], "s", flush=True)
        report["cases"][case.name] = dict(rows=rows, peak_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
    (out / "xgpu-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("XGPU_DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
