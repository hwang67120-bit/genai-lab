"""Run one isolated FLUX.2 Klein 4B multi-reference smoke test.

This runner does not import or modify the existing Stable Diffusion pipeline.
Official APIs:
https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
https://huggingface.co/docs/diffusers/quantization/bitsandbytes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", type=Path)
    parser.add_argument("--garment", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("D:/genai-cache/huggingface"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "flux2-klein-smoke.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "version": "flux2_klein_smoke_v1",
        "execution_scope": "runtime_smoke",
        "product_pipeline_equivalent": False,
        "final_return_eligible": False,
        "status": "started",
        "started_at": utc_now(),
        "model_id": args.model_id,
        "execution_mode": "local_diffusers_pytorch_cuda_no_inference_api",
        "cache_dir": str(args.cache_dir.resolve()),
        "local_files_only": args.local_files_only,
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "DIFFUSERS_OFFLINE": os.environ.get("DIFFUSERS_OFFLINE"),
            "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY"),
        },
        "quantization": "bitsandbytes_nf4_double_quant",
        "cpu_offload": "model",
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "prompt": args.prompt,
        "python": sys.version,
        "platform": platform.platform(),
    }
    started = time.perf_counter()
    try:
        import truststore
        truststore.inject_into_ssl()

        import torch
        import accelerate
        import diffusers
        import transformers
        from PIL import Image
        from diffusers import Flux2KleinPipeline
        from diffusers.quantizers import PipelineQuantizationConfig

        report["versions"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
        }
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this smoke test")

        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = {
            "name": properties.name,
            "total_vram_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
        if args.probe_only:
            report["status"] = "probe_passed"
            return 0

        if args.character is None or args.garment is None:
            raise ValueError("--character and --garment are required unless --probe-only is used")
        for label, path in (("character", args.character), ("garment", args.garment)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} reference not found: {path}")

        report["inputs"] = {
            "character": {"path": str(args.character.resolve()), "sha256": sha256(args.character)},
            "garment": {"path": str(args.garment.resolve()), "sha256": sha256(args.garment)},
        }
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            },
            components_to_quantize=["transformer", "text_encoder"],
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        pipeline = Flux2KleinPipeline.from_pretrained(
            args.model_id,
            cache_dir=str(args.cache_dir),
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        )
        pipeline.enable_model_cpu_offload()
        report["load_seconds"] = time.perf_counter() - load_started
        report["load_peak_vram_bytes"] = torch.cuda.max_memory_reserved(0)

        with Image.open(args.character) as source:
            character = source.convert("RGB").copy()
        with Image.open(args.garment) as source:
            garment = source.convert("RGB").copy()

        torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        result = pipeline(
            image=[character, garment],
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
            num_images_per_prompt=1,
        )
        report["inference_seconds"] = time.perf_counter() - inference_started
        report["inference_peak_vram_bytes"] = torch.cuda.max_memory_reserved(0)

        output_path = args.output_dir / "flux2-klein-smoke.png"
        result.images[0].save(output_path)
        report["output"] = {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
            "size": list(result.images[0].size),
        }
        report["status"] = "completed"
        character.close()
        garment.close()
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        report["finished_at"] = utc_now()
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path = write_report(args.output_dir, report)
        print(report_path, flush=True)

    return 0 if report["status"] in {"completed", "probe_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

