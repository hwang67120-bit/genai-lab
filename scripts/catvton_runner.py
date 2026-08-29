"""GenAI Lab에서 CatVTON 공식 파이프라인을 한 번 실행하는 별도 프로세스."""

import argparse
import os
from pathlib import Path
import sys

from PIL import Image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-path", required=True)
    parser.add_argument("--person-image", required=True)
    parser.add_argument("--clothing-image", required=True)
    parser.add_argument(
        "--clothing-type",
        choices=("upper", "lower", "overall", "inner", "outer"),
        required=True,
    )
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-mask", required=True)
    parser.add_argument("--output-protection-mask", required=True)
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
    parser.add_argument("--mask-expansion-radius", type=int, required=True)
    parser.add_argument("--mask-closing-radius", type=int, required=True)
    return parser.parse_args()


def load_rgb_image(image_path: str) -> Image.Image:
    """파일 핸들을 즉시 닫고 RGB 픽셀만 소유한 이미지로 반환한다."""
    with Image.open(image_path) as opened_image:
        return opened_image.convert("RGB")


def main() -> None:
    """공식 CatVTON 자동 마스크와 합성을 실행하고 두 이미지를 반환한다."""
    arguments = parse_arguments()
    repository_path = Path(arguments.repository_path).resolve()
    os.chdir(repository_path)
    sys.path.insert(0, str(repository_path))
    os.environ["HF_HOME"] = str(Path(arguments.cache_dir).resolve())

    import truststore

    truststore.inject_into_ssl()
    import torch
    import numpy as np
    import cv2
    from diffusers.image_processor import VaeImageProcessor
    from huggingface_hub import snapshot_download

    from model.cloth_masker import (
        ATR_MAPPING,
        LIP_MAPPING,
        AutoMasker,
        part_mask_of,
    )
    from model.pipeline import CatVTONPipeline
    from utils import init_weight_dtype, resize_and_crop, resize_and_padding

    original_person_image = load_rgb_image(arguments.person_image)
    original_size = original_person_image.size
    person_image = resize_and_crop(
        original_person_image,
        (arguments.width, arguments.height),
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
    automatic_masker = AutoMasker(
        densepose_ckpt=str(Path(checkpoint_path) / "DensePose"),
        schp_ckpt=str(Path(checkpoint_path) / "SCHP"),
        device="cuda",
    )
    automatic_mask_result = automatic_masker(
        person_image,
        arguments.clothing_type,
    )
    raw_clothing_mask = automatic_mask_result["mask"].convert("L")

    # CharacterIdentityProtectionMask(캐릭터 신체 보호 마스크)
    # - 포함: 얼굴, 머리카락, 노출된 팔·다리, 손·발과 장신구 영역.
    # - 생성: CatVTON의 SCHP 사람 영역 분리 결과를 픽셀 규칙으로 결합한다.
    # - 처리: 별도 LLM 호출 없이 분류 모델 결과만 사용한다.
    # - 저장: 호출 프로세스의 임시 폴더에만 저장되고 작업 후 삭제된다.
    # - 다음 사용처: 의상 합성이 신체·캐릭터 특징을 덮지 못하게 제한한다.
    protected_parts = [
        "Hat", "Hair", "Sunglasses", "Face", "Left-arm", "Right-arm",
        "Left-leg", "Right-leg", "Left-shoe", "Right-shoe", "Glove",
        "Bag", "Scarf",
    ]
    schp_lip_mask = np.asarray(automatic_mask_result["schp_lip"])
    schp_atr_mask = np.asarray(automatic_mask_result["schp_atr"])
    identity_protection_array = (
        part_mask_of(protected_parts, schp_lip_mask, LIP_MAPPING)
        | part_mask_of(protected_parts, schp_atr_mask, ATR_MAPPING)
    )
    identity_protection_mask = Image.fromarray(
        (identity_protection_array > 0).astype(np.uint8) * 255
    ).convert("L")

    if not 5 <= arguments.mask_expansion_radius <= 15:
        raise RuntimeError("CatVTON 마스크 팽창 반경은 5~15픽셀이어야 합니다.")
    if not 1 <= arguments.mask_closing_radius <= 3:
        raise RuntimeError("CatVTON 마스크 닫기 반경은 1~3픽셀이어야 합니다.")

    raw_mask_array = np.where(
        np.asarray(raw_clothing_mask, dtype=np.uint8) >= 128,
        255,
        0,
    ).astype(np.uint8)
    closing_kernel_size = arguments.mask_closing_radius * 2 + 1
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size),
    )
    closed_mask_array = cv2.morphologyEx(
        raw_mask_array,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1,
    )
    expansion_kernel_size = arguments.mask_expansion_radius * 2 + 1
    expansion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (expansion_kernel_size, expansion_kernel_size),
    )
    expanded_mask_array = cv2.dilate(
        closed_mask_array,
        expansion_kernel,
        iterations=1,
    )
    inference_mask_array = np.where(
        identity_protection_array > 0,
        0,
        expanded_mask_array,
    ).astype(np.uint8)
    if int(np.count_nonzero(inference_mask_array)) == 0:
        raise RuntimeError("신체 보호 영역을 제외한 CatVTON 변경 픽셀이 0개입니다.")
    inference_clothing_mask = Image.fromarray(inference_mask_array, mode="L")

    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )
    model_mask = mask_processor.blur(inference_clothing_mask, blur_factor=9)
    pipeline = CatVTONPipeline(
        base_ckpt=arguments.base_model_id,
        attn_ckpt=checkpoint_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(arguments.mixed_precision),
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
        original_size,
        Image.Resampling.LANCZOS,
    )
    resized_output_mask = raw_clothing_mask.resize(
        original_size,
        Image.Resampling.NEAREST,
    )
    resized_protection_mask = identity_protection_mask.resize(
        original_size,
        Image.Resampling.NEAREST,
    )
    try:
        resized_output_image.save(arguments.output_image)
        resized_output_mask.save(arguments.output_mask)
        resized_protection_mask.save(arguments.output_protection_mask)
    finally:
        resized_output_image.close()
        resized_output_mask.close()
        resized_protection_mask.close()
        original_person_image.close()
        person_image.close()
        clothing_image.close()
        raw_clothing_mask.close()
        inference_clothing_mask.close()
        identity_protection_mask.close()
        model_mask.close()
        result_image.close()
        for result_name in ("mask", "densepose", "schp_lip", "schp_atr"):
            intermediate_image = automatic_mask_result.get(result_name)
            if intermediate_image is not None:
                intermediate_image.close()


if __name__ == "__main__":
    main()

