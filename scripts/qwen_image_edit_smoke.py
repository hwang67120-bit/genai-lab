"""Run one isolated low-VRAM Qwen-Image-Edit-2509 consistency test.

The runner follows the Qwen-Image paper's editing path: the approved Base is
encoded as the image to preserve and the isolated garment board is supplied as
a second visual condition. It produces a whole image and never composites
pixels.

Research and official implementations:
https://arxiv.org/abs/2508.02324
https://huggingface.co/Qwen/Qwen-Image-Edit-2509
https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2509.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_PROMPT = (
    "Figure 1 is the approved character Base. Figure 2 is an isolated garment-only "
    "reference. Preserve the exact identity, face, hair, animal ears, tail, pose, "
    "framing, and plain background of Figure 1. Change only the clothing design to "
    "match Figure 2. Do not transfer the person, face, body shape, hair, or gender "
    "from Figure 2. Keep one person and the original character species."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", type=Path)
    parser.add_argument("--garment", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=40)
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
    report_path = output_dir / "qwen-image-edit-smoke.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "version": "qwen_image_edit_2509_smoke_v1",
        "execution_scope": "runtime_smoke",
        "product_pipeline_equivalent": False,
        "final_return_eligible": False,
        "status": "started",
        "started_at": utc_now(),
        "model_id": args.model_id,
        "paper": "https://arxiv.org/abs/2508.02324",
        "consistency_mechanism": [
            "semantic_encoding",
            "vae_reconstruction_encoding",
            "i2i_reconstruction_training",
        ],
        "execution_mode": "local_diffsynth_low_vram_cuda_no_inference_api",
        "cache_dir": str(args.cache_dir.resolve()),
        "local_files_only": args.local_files_only,
        "pixel_composite_used": False,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "prompt": args.prompt,
        "python": sys.version,
        "platform": platform.platform(),
    }
    started = time.perf_counter()
    try:
        if args.local_files_only:
            os.environ.update({
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DIFFUSERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            })
        else:
            os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "false"
        os.environ["HF_HOME"] = str(args.cache_dir)
        os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(
            args.cache_dir / "diffsynth"
        )

        if importlib.util.find_spec("diffsynth") is None:
            raise RuntimeError(
                "DiffSynth-Studio is not installed; Qwen low-VRAM runtime is unavailable"
            )

        import truststore
        truststore.inject_into_ssl()

        import torch
        from PIL import Image
        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        report["torch"] = torch.__version__
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
            raise ValueError(
                "--character and --garment are required unless --probe-only is used"
            )
        for label, path in (("character", args.character), ("garment", args.garment)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} reference not found: {path}")
        report["inputs"] = {
            "character": {
                "path": str(args.character.resolve()),
                "sha256": sha256(args.character),
            },
            "garment": {
                "path": str(args.garment.resolve()),
                "sha256": sha256(args.garment),
            },
        }

        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        available_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024**3)
        report["vram_limit_gb"] = max(0.5, available_vram_gb - 0.5)
        load_started = time.perf_counter()
        pipeline = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(
                    model_id=args.model_id,
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                    **vram_config,
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="text_encoder/model*.safetensors",
                    **vram_config,
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                    **vram_config,
                ),
            ],
            processor_config=ModelConfig(
                model_id="Qwen/Qwen-Image-Edit",
                origin_file_pattern="processor/",
            ),
            vram_limit=report["vram_limit_gb"],
        )
        report["load_seconds"] = time.perf_counter() - load_started

        with Image.open(args.character) as source:
            character = source.convert("RGB").copy()
        with Image.open(args.garment) as source:
            garment = source.convert("RGB").copy()

        torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        result = pipeline(
            args.prompt,
            edit_image=[character, garment],
            seed=args.seed,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            edit_image_auto_resize=True,
        )
        report["inference_seconds"] = time.perf_counter() - inference_started
        report["inference_peak_vram_bytes"] = torch.cuda.max_memory_reserved(0)

        output_path = args.output_dir / "qwen-image-edit-smoke.png"
        result.save(output_path)
        report["output"] = {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
            "size": list(result.size),
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

