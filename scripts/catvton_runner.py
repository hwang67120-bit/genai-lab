"""사용자가 승인한 Human-Agnostic 입력으로 CatVTON을 실행한다."""

import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageOps


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-path", required=True)
    parser.add_argument("--person-image", required=True)
    parser.add_argument("--approved-change-mask", required=True)
    parser.add_argument("--clothing-image", required=True)
    parser.add_argument(
        "--clothing-type",
        choices=("upper", "lower", "overall", "inner", "outer"),
        required=True,
    )
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-mask", required=True)
    parser.add_argument("--output-protection-mask", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument(
        "--mixed-precision",
        choices=("fp16", "bf16"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-safety-check", action="store_true")
    return parser.parse_args()


def load_rgb_image(image_path: str) -> Image.Image:
    """파일 핸들을 즉시 닫고 RGB 픽셀만 소유한 이미지로 반환한다."""
    with Image.open(image_path) as opened_image:
        return opened_image.convert("RGB")


def load_binary_mask(mask_path: str) -> Image.Image:
    """승인 마스크를 0 또는 255 값의 독립 L 이미지로 반환한다."""
    with Image.open(mask_path) as opened_image:
        grayscale_mask = opened_image.convert("L")
        try:
            return grayscale_mask.point(
                lambda pixel: 255 if pixel >= 128 else 0
            )
        finally:
            grayscale_mask.close()


def count_mask_pixels(mask_image: Image.Image) -> int:
    """128 이상인 마스크 픽셀 수를 계산한다."""
    histogram = mask_image.histogram()
    return sum(histogram[128:])


def main() -> None:
    """승인 이미지와 승인 마스크만 사용해 CatVTON 합성을 실행한다."""
    arguments = parse_arguments()
    repository_path = Path(arguments.repository_path).resolve()
    os.chdir(repository_path)
    sys.path.insert(0, str(repository_path))
    os.environ["HF_HOME"] = str(Path(arguments.cache_dir).resolve())

    import truststore

    truststore.inject_into_ssl()
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from huggingface_hub import snapshot_download

    from model.pipeline import CatVTONPipeline
    from utils import init_weight_dtype, resize_and_crop, resize_and_padding

    source_person_image = load_rgb_image(arguments.person_image)
    approved_change_mask = load_binary_mask(
        arguments.approved_change_mask
    )
    person_input_size = source_person_image.size
    approved_mask_size = approved_change_mask.size
    if person_input_size != approved_mask_size:
        source_person_image.close()
        approved_change_mask.close()
        raise RuntimeError(
            "CatVTON 인물 이미지와 승인 마스크 좌표가 다릅니다: "
            f"인물={person_input_size}, 마스크={approved_mask_size}"
        )

    approved_size = person_input_size
    approved_mask_pixel_count = count_mask_pixels(approved_change_mask)
    if approved_mask_pixel_count == 0:
        source_person_image.close()
        approved_change_mask.close()
        raise RuntimeError("승인된 CatVTON 변경 픽셀이 0개입니다.")

    person_image = resize_and_crop(
        source_person_image,
        (arguments.width, arguments.height),
    )
    resized_change_mask = resize_and_crop(
        approved_change_mask,
        (arguments.width, arguments.height),
    )
    grayscale_change_mask = resized_change_mask.convert("L")
    try:
        processed_change_mask = grayscale_change_mask.point(
            lambda pixel: 255 if pixel >= 128 else 0
        )
    finally:
        resized_change_mask.close()
        grayscale_change_mask.close()
    processed_mask_pixel_count = count_mask_pixels(processed_change_mask)
    if processed_mask_pixel_count == 0:
        source_person_image.close()
        approved_change_mask.close()
        person_image.close()
        processed_change_mask.close()
        raise RuntimeError(
            "처리 크기로 변환한 승인 CatVTON 변경 픽셀이 0개입니다."
        )

    original_clothing_image = load_rgb_image(arguments.clothing_image)
    try:
        clothing_image = resize_and_padding(
            original_clothing_image,
            (arguments.width, arguments.height),
        )
    finally:
        original_clothing_image.close()

    checkpoint_path = snapshot_download(
        repo_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
    )
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )
    model_mask = mask_processor.blur(
        processed_change_mask,
        blur_factor=9,
    )
    pipeline = CatVTONPipeline(
        base_ckpt=arguments.base_model_id,
        attn_ckpt=checkpoint_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(arguments.mixed_precision),
        skip_safety_check=arguments.skip_safety_check,
        use_tf32=True,
        device="cuda",
    )
    random_generator = torch.Generator(device="cuda").manual_seed(
        arguments.seed
    )
    result_image = pipeline(
        image=person_image,
        condition_image=clothing_image,
        mask=model_mask,
        num_inference_steps=arguments.inference_steps,
        guidance_scale=arguments.guidance_scale,
        generator=random_generator,
    )[0].convert("RGB")

    resized_output_image = result_image.resize(
        approved_size,
        Image.Resampling.LANCZOS,
    )
    identity_protection_mask = ImageOps.invert(approved_change_mask)
    metadata_payload = {
        "mask_source": "user_approved",
        "automasker_run_count": 0,
        "approved_image_width": approved_size[0],
        "approved_image_height": approved_size[1],
        "approved_mask_pixel_count": approved_mask_pixel_count,
        "processed_mask_pixel_count": processed_mask_pixel_count,
        "clothing_type": arguments.clothing_type,
        "safety_check_enabled": not arguments.skip_safety_check,
        "person_input_source": "generated_candidate",
        "person_input_width": approved_size[0],
        "person_input_height": approved_size[1],
    }
    try:
        resized_output_image.save(arguments.output_image)
        approved_change_mask.save(arguments.output_mask)
        identity_protection_mask.save(arguments.output_protection_mask)
        Path(arguments.output_metadata).write_text(
            json.dumps(metadata_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        resized_output_image.close()
        identity_protection_mask.close()
        source_person_image.close()
        approved_change_mask.close()
        person_image.close()
        processed_change_mask.close()
        clothing_image.close()
        model_mask.close()
        result_image.close()


if __name__ == "__main__":
    main()
