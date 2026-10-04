"""한 명의 캐릭터를 보존하면서 의상 영역만 편집하는 오늘용 실행기."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT = (
    "1boy, male focus, masculine silhouette, exactly one person, full body, "
    "fitted dark teal blue blazer, white collared shirt, very short matching "
    "pleated mini skirt, bare thighs, simple tailored outfit, preserve body pose, "
    "plain white background"
)
NEGATIVE = (
    "1girl, woman, female, breasts, pants, trousers, shorts, leggings, jeans, "
    "bodysuit, leotard, swimsuit, cape, cloak, long skirt, extra person, "
    "detached object, changed hair, changed animal ears, changed tail, scenery"
)
DEFAULT_PERSON_IMAGE = (
    Path.home() / "Downloads" / "\ucc38\uc870 \uc774\ubbf8\uc9c0"
    / "HFTK9dCbgAAkxKb.png"
)
DEFAULT_GARMENT_IMAGE = (
    Path.home() / "Downloads" / "\ucc38\uc870 \uc758\uc0c1"
    / "19f3058071f8221e4d563036a40a03b5.jpg"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-image", default=str(DEFAULT_PERSON_IMAGE))
    parser.add_argument("--garment-image", default=str(DEFAULT_GARMENT_IMAGE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1109384698, 1109384699])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-analysis", action="store_true")
    parser.add_argument("--inference-width", type=int, default=640)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--strength", type=float, default=0.68)
    parser.add_argument("--guidance", type=float, default=5.5)
    parser.add_argument("--adapter-scale", type=float, default=0.45)
    parser.add_argument("--minimum-y-ratio", type=float, default=0.20)
    parser.add_argument("--maximum-y-ratio", type=float, default=0.66)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE)
    parser.add_argument(
        "--analysis-python",
        default="D:/genai-cache/catvton-venv/Scripts/python.exe",
    )
    parser.add_argument(
        "--analysis-repository", default="D:/genai-cache/tools/CatVTON"
    )
    parser.add_argument(
        "--analysis-runner",
        default="//192.168.0.109/win_g/genai-lab/scripts/body_comparison_runner.py",
    )
    parser.add_argument("--cache-dir", default="D:/genai-cache/huggingface")
    parser.add_argument(
        "--base-model-id", default="cagliostrolab/animagine-xl-3.1",
        help="Local cached SDXL model. Network downloads are disabled.",
    )
    return parser.parse_args()


def load(path: Path, mode: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert(mode).copy()


def run_analysis(args: argparse.Namespace, out: Path) -> None:
    names = (
        "raw-mask.png", "protection-mask.png", "foreground-mask.png",
        "densepose.png", "body-analysis.json",
    )
    if args.reuse_analysis and all((out / name).is_file() for name in names):
        return
    command = [
        args.analysis_python, args.analysis_runner,
        "--repository-path", args.analysis_repository,
        "--person-image", args.person_image,
        "--clothing-type", "overall",
        "--output-raw-mask", str(out / names[0]),
        "--output-protection-mask", str(out / names[1]),
        "--output-foreground-mask", str(out / names[2]),
        "--output-densepose", str(out / names[3]),
        "--output-metadata-json", str(out / names[4]),
        "--cache-dir", args.cache_dir,
        "--width", "576", "--height", "1024",
        "--foreground-model-id", "isnet-anime",
    ]
    completed = subprocess.run(
        command, cwd=args.analysis_repository, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=1800, check=False,
    )
    if completed.returncode:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"의상·보호 마스크 분석 실패: {details}")


def mask_array(image: Image.Image) -> np.ndarray:
    return (np.asarray(image.convert("L"), dtype=np.uint8) >= 128).astype(np.uint8)


def create_masks(
    raw_image: Image.Image,
    protection_image: Image.Image,
    foreground_image: Image.Image,
    minimum_y_ratio: float,
    maximum_y_ratio: float,
    tail_protection_x_ratio: float = 0.58,
    tail_protection_y_ratio: float = 0.52,
) -> tuple[Image.Image, Image.Image, dict[str, float | int]]:
    if not (raw_image.size == protection_image.size == foreground_image.size):
        raise ValueError("분석 마스크 좌표가 서로 다릅니다.")
    raw = cv2.morphologyEx(
        mask_array(raw_image), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    raw = cv2.dilate(
        raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    )
    protection = cv2.dilate(
        mask_array(protection_image),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    foreground = cv2.dilate(
        mask_array(foreground_image),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    height, width = raw.shape
    vertical = np.zeros_like(raw, dtype=bool)
    start_y = round(height * minimum_y_ratio)
    end_y = round(height * maximum_y_ratio)
    vertical[start_y:end_y] = True
    safe = (raw > 0) & (protection == 0) & (foreground > 0) & vertical
    # 오늘 입력의 꼬리는 오른쪽 아래에 있다. SCHP가 윗부분을 소매나
    # 하의로 분류하므로 해당 구역은 추론 전에 명시적으로 보호한다.
    y_coordinates, x_coordinates = np.indices(raw.shape)
    tail_protection = (
        (x_coordinates >= round(width * tail_protection_x_ratio))
        & (y_coordinates >= round(height * tail_protection_y_ratio))
    )
    safe &= ~tail_protection
    hard_array = np.where(safe, 255, 0).astype(np.uint8)
    if not np.any(hard_array):
        raise RuntimeError("보호 영역을 제외한 의상 변경 마스크가 비었습니다.")
    hard = Image.fromarray(hard_array, mode="L")
    eroded = hard.filter(ImageFilter.MinFilter(17))
    blurred = eroded.filter(ImageFilter.GaussianBlur(8))
    soft = ImageChops.multiply(blurred, hard)
    eroded.close()
    blurred.close()
    return hard, soft, {
        "width": width,
        "height": height,
        "start_y": start_y,
        "end_y": end_y,
        "tail_protection_x": round(width * tail_protection_x_ratio),
        "tail_protection_y": round(height * tail_protection_y_ratio),
        "hard_mask_pixels": int(np.count_nonzero(hard_array)),
        "hard_mask_percent": round(np.count_nonzero(hard_array) / hard_array.size * 100, 6),
    }


def garment_condition(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    box = (
        round(image.width * 0.20), round(image.height * 0.20),
        round(image.width * 0.80), round(image.height * 0.57),
    )
    return image.crop(box).convert("RGB"), box


def overlay(source: Image.Image, mask: Image.Image) -> Image.Image:
    base = source.convert("RGBA")
    red = Image.new("RGBA", source.size, (255, 32, 32, 0))
    red.putalpha(mask.point(lambda value: 110 if value >= 128 else 0))
    result = Image.alpha_composite(base, red).convert("RGB")
    base.close()
    red.close()
    return result


def composite(
    source: Image.Image,
    proposed: Image.Image,
    hard: Image.Image,
    soft: Image.Image,
) -> tuple[Image.Image, int]:
    result = Image.composite(proposed.convert("RGB"), source.convert("RGB"), soft)
    source_array = np.asarray(source.convert("RGB"), dtype=np.uint8)
    result_array = np.asarray(result, dtype=np.uint8)
    outside = np.asarray(hard, dtype=np.uint8) < 128
    changed = int(np.count_nonzero(np.any(source_array != result_array, axis=2) & outside))
    if changed:
        result.close()
        raise RuntimeError(f"마스크 외부 픽셀이 {changed}개 변경됐습니다.")
    return result, changed


def contact_sheet(images: list[Image.Image]) -> Image.Image:
    cell = (360, 640)
    sheet = Image.new("RGB", (cell[0] * len(images), cell[1]), "white")
    for index, image in enumerate(images):
        preview = image.copy()
        preview.thumbnail(cell, Image.Resampling.LANCZOS)
        sheet.paste(
            preview,
            (index * cell[0] + (cell[0] - preview.width) // 2,
             (cell[1] - preview.height) // 2),
        )
        preview.close()
    return sheet


def generate(
    args: argparse.Namespace,
    source: Image.Image,
    garment: Image.Image,
    hard: Image.Image,
    soft: Image.Image,
    out: Path,
) -> tuple[list[Image.Image], list[dict[str, object]]]:
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline

    inference_size = (
        args.inference_width,
        max(256, round(source.height * args.inference_width / source.width / 8) * 8),
    )
    source_small = source.resize(inference_size, Image.Resampling.LANCZOS)
    mask_small = hard.resize(inference_size, Image.Resampling.NEAREST)
    model_kwargs = {
        "torch_dtype": torch.float16,
        "cache_dir": args.cache_dir,
        "use_safetensors": True,
        "local_files_only": True,
    }
    if args.base_model_id == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1":
        model_kwargs["variant"] = "fp16"
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.base_model_id, **model_kwargs,
    )
    if pipeline.unet.config.in_channels not in (4, 9):
        raise RuntimeError(
            f"Unsupported inpaint UNet channel count: {pipeline.unet.config.in_channels}"
        )
    adapter_snapshots = (
        Path(args.cache_dir)
        / "models--h94--IP-Adapter"
        / "snapshots"
    )
    snapshots = sorted(path for path in adapter_snapshots.iterdir() if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"로컬 IP-Adapter 스냅샷이 없습니다: {adapter_snapshots}")
    adapter_snapshot = snapshots[-1]
    pipeline.load_ip_adapter(
        str(adapter_snapshot), subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_folder="models/image_encoder",
        local_files_only=True,
    )
    pipeline.set_ip_adapter_scale(args.adapter_scale)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_tiling()
    results: list[Image.Image] = []
    records: list[dict[str, object]] = []
    try:
        for seed in args.seeds:
            started = perf_counter()
            raw = pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=source_small,
                mask_image=mask_small,
                ip_adapter_image=garment,
                width=inference_size[0], height=inference_size[1],
                strength=args.strength, num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0].convert("RGB")
            proposed = raw.resize(source.size, Image.Resampling.LANCZOS)
            raw.close()
            proposed.save(out / f"candidate-{seed}-raw.png")
            final, changed = composite(source, proposed, hard, soft)
            proposed.close()
            final.save(out / f"candidate-{seed}-final.png")
            results.append(final)
            records.append({
                "seed": seed,
                "elapsed_seconds": round(perf_counter() - started, 3),
                "outside_hard_mask_changed_pixels": changed,
                "output": f"candidate-{seed}-final.png",
            })
    finally:
        source_small.close()
        mask_small.close()
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results, records


def main() -> None:
    args = arguments()
    person_path = Path(args.person_image)
    garment_path = Path(args.garment_image)
    out = Path(args.output_dir)
    if not person_path.is_file() or not garment_path.is_file():
        raise FileNotFoundError("캐릭터 또는 의상 참고 이미지를 찾을 수 없습니다.")
    out.mkdir(parents=True, exist_ok=True)
    run_analysis(args, out)
    images: list[Image.Image] = []
    candidates: list[Image.Image] = []
    try:
        source = load(person_path, "RGB")
        garment_source = load(garment_path, "RGB")
        raw = load(out / "raw-mask.png", "L")
        protection = load(out / "protection-mask.png", "L")
        foreground = load(out / "foreground-mask.png", "L")
        images.extend((source, garment_source, raw, protection, foreground))
        hard, soft, mask_metrics = create_masks(
            raw, protection, foreground,
            args.minimum_y_ratio, args.maximum_y_ratio,
        )
        images.extend((hard, soft))
        hard.save(out / "hard-edit-mask.png")
        soft.save(out / "soft-edit-mask.png")
        mask_overlay = overlay(source, hard)
        images.append(mask_overlay)
        mask_overlay.save(out / "mask-overlay.png")
        garment, crop_box = garment_condition(garment_source)
        images.append(garment)
        garment.save(out / "garment-condition.png")
        generation_records: list[dict[str, object]] = []
        if not args.prepare_only:
            candidates, generation_records = generate(
                args, source, garment, hard, soft, out
            )
        sheet = contact_sheet([source, garment, mask_overlay, *candidates])
        sheet.save(out / "review-sheet.png")
        sheet.close()
        payload = {
            "contract": "today_reference_edit_v1",
            "person_image": str(person_path),
            "garment_image": str(garment_path),
            "manual_review_required": True,
            "outside_hard_mask_policy": "exact_source_pixels",
            "whole_image_regeneration": False,
            "mask_metrics": mask_metrics,
            "garment_crop_box": list(crop_box),
            "parameters": {
                "base_model_id": args.base_model_id,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "seeds": args.seeds,
                "inference_width": args.inference_width,
                "steps": args.steps,
                "strength": args.strength,
                "guidance": args.guidance,
                "adapter_scale": args.adapter_scale,
            },
            "candidates": generation_records,
        }
        (out / "run.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(out / "review-sheet.png")
    finally:
        for image in candidates:
            image.close()
        for image in images:
            image.close()


if __name__ == "__main__":
    main()
