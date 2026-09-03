"""사용자가 승인한 Human-Agnostic 입력으로 CatVTON을 실행한다."""

import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from genai_lab.image_digest import calculate_image_pixel_sha256


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-path", required=True)
    parser.add_argument("--person-image", required=True)
    parser.add_argument("--approved-change-mask", required=True)
    parser.add_argument("--approved-model-mask", required=True)
    parser.add_argument("--clothing-image", required=True)
    parser.add_argument("--clothing-source-width", type=int, required=True)
    parser.add_argument("--clothing-source-height", type=int, required=True)
    parser.add_argument(
        "--clothing-alpha-pixel-count", type=int, required=True
    )
    parser.add_argument(
        "--clothing-alpha-coverage-percent", type=float, required=True
    )
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
    parser.add_argument("--mask-blur-factor", type=int, default=9)
    parser.add_argument("--expected-person-sha256")
    parser.add_argument("--expected-binary-mask-sha256")
    parser.add_argument("--expected-model-mask-sha256")
    parser.add_argument("--expected-clothing-sha256")
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


def load_grayscale_mask(mask_path: str) -> Image.Image:
    """0~255 블러 값을 보존한 독립 L 마스크를 반환한다."""
    with Image.open(mask_path) as opened_image:
        return opened_image.convert("L")


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

    model_mask = load_grayscale_mask(arguments.approved_model_mask)
    expected_model_mask_size = (arguments.width, arguments.height)
    if model_mask.size != expected_model_mask_size:
        source_person_image.close()
        approved_change_mask.close()
        person_image.close()
        processed_change_mask.close()
        model_mask.close()
        raise RuntimeError(
            "승인 CatVTON model_mask 처리 크기가 다릅니다: "
            f"승인={model_mask.size}, 설정={expected_model_mask_size}"
        )
    model_mask_pixel_count = int(
        sum(model_mask.histogram()[1:])
    )
    if model_mask_pixel_count == 0:
        source_person_image.close()
        approved_change_mask.close()
        person_image.close()
        processed_change_mask.close()
        model_mask.close()
        raise RuntimeError("승인 CatVTON model_mask가 0픽셀입니다.")

    original_clothing_image = load_rgb_image(arguments.clothing_image)
    clothing_input_size = original_clothing_image.size
    clothing_input_pixel_count = (
        clothing_input_size[0] * clothing_input_size[1]
    )
    if (
        arguments.clothing_source_width < clothing_input_size[0]
        or arguments.clothing_source_height < clothing_input_size[1]
    ):
        source_person_image.close()
        approved_change_mask.close()
        person_image.close()
        processed_change_mask.close()
        model_mask.close()
        original_clothing_image.close()
        raise RuntimeError(
            "의상 조건 이미지가 승인 추출본보다 클 수 없습니다: "
            f"원본={arguments.clothing_source_width}x"
            f"{arguments.clothing_source_height}, "
            f"조건={clothing_input_size[0]}x{clothing_input_size[1]}"
        )
    if not (
        0 < arguments.clothing_alpha_pixel_count <= clothing_input_pixel_count
    ):
        source_person_image.close()
        approved_change_mask.close()
        person_image.close()
        processed_change_mask.close()
        model_mask.close()
        original_clothing_image.close()
        raise RuntimeError(
            "의상 알파 픽셀 수가 조건 이미지 범위를 벗어났습니다: "
            f"알파={arguments.clothing_alpha_pixel_count}, "
            f"조건 전체={clothing_input_pixel_count}"
        )
    measured_clothing_coverage_percent = (
        arguments.clothing_alpha_pixel_count
        / clothing_input_pixel_count
        * 100.0
    )
    if abs(
        measured_clothing_coverage_percent
        - arguments.clothing_alpha_coverage_percent
    ) > 0.001:
        source_person_image.close()
        approved_change_mask.close()
        person_image.close()
        processed_change_mask.close()
        model_mask.close()
        original_clothing_image.close()
        raise RuntimeError(
            "의상 알파 점유율 기록이 실제 조건 이미지와 다릅니다: "
            f"계산={measured_clothing_coverage_percent:.6f}%, "
            f"기록={arguments.clothing_alpha_coverage_percent:.6f}%"
        )
    try:
        clothing_image = resize_and_padding(
            original_clothing_image,
            (arguments.width, arguments.height),
        )
    finally:
        original_clothing_image.close()

    actual_hashes = {
        "person": calculate_image_pixel_sha256(person_image, "RGB"),
        "binary_mask": calculate_image_pixel_sha256(
            processed_change_mask, "L"
        ),
        "model_mask": calculate_image_pixel_sha256(model_mask, "L"),
        "clothing": calculate_image_pixel_sha256(clothing_image, "RGB"),
    }
    expected_hashes = {
        "person": arguments.expected_person_sha256,
        "binary_mask": arguments.expected_binary_mask_sha256,
        "model_mask": arguments.expected_model_mask_sha256,
        "clothing": arguments.expected_clothing_sha256,
    }
    missing_expected_hashes = tuple(
        name for name, value in expected_hashes.items() if not value
    )
    if missing_expected_hashes:
        raise RuntimeError(
            "승인된 CatVTON Preflight 해시가 없습니다: "
            f"{missing_expected_hashes}"
        )
    mismatched_hashes = tuple(
        name
        for name, expected_value in expected_hashes.items()
        if actual_hashes[name] != expected_value
    )
    if mismatched_hashes:
        raise RuntimeError(
            "승인한 CatVTON Preflight 입력과 실제 모델 입력이 다릅니다: "
            f"{mismatched_hashes}"
        )
    checkpoint_path = snapshot_download(
        repo_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
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
        "model_mask_pixel_count": model_mask_pixel_count,
        "model_mask_source": "user_approved_preflight",
        "clothing_type": arguments.clothing_type,
        "safety_check_enabled": not arguments.skip_safety_check,
        "person_input_source": "generated_candidate",
        "person_input_width": approved_size[0],
        "person_input_height": approved_size[1],
        "clothing_source_width": arguments.clothing_source_width,
        "clothing_source_height": arguments.clothing_source_height,
        "clothing_input_width": clothing_input_size[0],
        "clothing_input_height": clothing_input_size[1],
        "clothing_alpha_pixel_count": arguments.clothing_alpha_pixel_count,
        "clothing_alpha_coverage_percent": (
            measured_clothing_coverage_percent
        ),
        "person_sha256": actual_hashes["person"],
        "binary_mask_sha256": actual_hashes["binary_mask"],
        "model_mask_sha256": actual_hashes["model_mask"],
        "clothing_sha256": actual_hashes["clothing"],
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
