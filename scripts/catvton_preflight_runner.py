"""공식 CatVTON 전처리를 GPU 추론 없이 실행해 실제 모델 입력을 공개한다."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from genai_lab.image_digest import calculate_image_pixel_sha256
from genai_lab.guardrails import (
    create_catvton_preflight_guard_results,
    evaluate_guard_results,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-path", required=True)
    parser.add_argument("--person-image", required=True)
    parser.add_argument("--approved-change-mask", required=True)
    parser.add_argument("--clothing-image", required=True)
    parser.add_argument("--identity-protection-mask", required=True)
    parser.add_argument("--expanded-foreground-mask", required=True)
    parser.add_argument("--output-processed-person", required=True)
    parser.add_argument("--output-binary-mask", required=True)
    parser.add_argument("--output-raw-blurred-mask", required=True)
    parser.add_argument("--output-model-mask", required=True)
    parser.add_argument("--output-processed-clothing", required=True)
    parser.add_argument("--output-soft-overlap", required=True)
    parser.add_argument("--output-hard-overlap", required=True)
    parser.add_argument("--output-protected-overlap", required=True)
    parser.add_argument("--output-outside-foreground", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--mask-blur-factor", type=int, default=9)
    return parser.parse_args()


def load_image(path: str, mode: str) -> Image.Image:
    with Image.open(path) as opened_image:
        return opened_image.convert(mode)


def main() -> None:
    arguments = parse_arguments()
    repository_path = Path(arguments.repository_path).resolve()
    os.chdir(repository_path)
    sys.path.insert(0, str(repository_path))
    os.environ["HF_HOME"] = str(Path(arguments.cache_dir).resolve())

    from diffusers.image_processor import VaeImageProcessor
    from utils import resize_and_crop, resize_and_padding

    source_person = load_image(arguments.person_image, "RGB")
    source_mask = load_image(arguments.approved_change_mask, "L")
    source_clothing = load_image(arguments.clothing_image, "RGB")
    source_protection = load_image(arguments.identity_protection_mask, "L")
    source_foreground = load_image(arguments.expanded_foreground_mask, "L")
    processed_person = None
    resized_mask = None
    binary_mask = None
    raw_blurred_mask = None
    model_mask = None
    processed_clothing = None
    processed_protection = None
    processed_foreground = None
    soft_overlap = None
    hard_overlap = None
    protected_overlap = None
    outside_foreground = None
    try:
        original_size = source_person.size
        if any(
            image.size != original_size
            for image in (source_mask, source_protection, source_foreground)
        ):
            raise RuntimeError("Person·승인 마스크·보호 영역 좌표가 다릅니다.")

        target_size = (arguments.width, arguments.height)
        processed_person = resize_and_crop(source_person, target_size)
        resized_mask = resize_and_crop(source_mask, target_size).convert("L")
        binary_mask = resized_mask.point(
            lambda pixel: 255 if pixel >= 128 else 0
        )
        processed_clothing = resize_and_padding(source_clothing, target_size)
        processed_protection = resize_and_crop(
            source_protection, target_size
        ).convert("L").point(lambda pixel: 255 if pixel >= 128 else 0)
        processed_foreground = resize_and_crop(
            source_foreground, target_size
        ).convert("L").point(lambda pixel: 255 if pixel >= 128 else 0)

        mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        raw_blurred_mask = mask_processor.blur(
            binary_mask,
            blur_factor=arguments.mask_blur_factor,
        ).convert("L")

        binary_array = np.asarray(binary_mask, dtype=np.uint8) >= 128
        raw_blurred_array = np.asarray(
            raw_blurred_mask,
            dtype=np.uint8,
        ).copy()
        protection_array = (
            np.asarray(processed_protection, dtype=np.uint8) >= 128
        )
        foreground_array = (
            np.asarray(processed_foreground, dtype=np.uint8) >= 128
        )
        forbidden_array = protection_array | ~foreground_array
        soft_overlap_array = (
            (raw_blurred_array > 0)
            & (raw_blurred_array < 128)
            & forbidden_array
        )
        hard_overlap_array = (
            (raw_blurred_array >= 128)
            & forbidden_array
        )
        removed_array = (raw_blurred_array > 0) & forbidden_array
        guarded_model_array = raw_blurred_array.copy()
        guarded_model_array[forbidden_array] = 0
        model_mask = Image.fromarray(guarded_model_array, mode="L")
        model_array = guarded_model_array > 0
        protected_overlap_array = model_array & protection_array
        outside_foreground_array = model_array & ~foreground_array
        soft_overlap = Image.fromarray(
            soft_overlap_array.astype(np.uint8) * 255,
            mode="L",
        )
        hard_overlap = Image.fromarray(
            hard_overlap_array.astype(np.uint8) * 255,
            mode="L",
        )
        protected_overlap = Image.fromarray(
            protected_overlap_array.astype(np.uint8) * 255,
            mode="L",
        )
        outside_foreground = Image.fromarray(
            outside_foreground_array.astype(np.uint8) * 255,
            mode="L",
        )

        processed_mask_pixel_count = int(np.count_nonzero(binary_array))
        model_mask_pixel_count = int(np.count_nonzero(model_array))
        soft_overlap_pixel_count = int(
            np.count_nonzero(soft_overlap_array)
        )
        hard_overlap_pixel_count = int(
            np.count_nonzero(hard_overlap_array)
        )
        removed_pixel_count = int(np.count_nonzero(removed_array))
        protected_overlap_pixel_count = int(
            np.count_nonzero(protected_overlap_array)
        )
        outside_foreground_pixel_count = int(
            np.count_nonzero(outside_foreground_array)
        )
        guard_decision = evaluate_guard_results(
            create_catvton_preflight_guard_results(
                processed_mask_pixel_count=processed_mask_pixel_count,
                model_mask_pixel_count=model_mask_pixel_count,
                soft_overlap_pixel_count=soft_overlap_pixel_count,
                hard_overlap_pixel_count=hard_overlap_pixel_count,
                final_protected_overlap_pixel_count=(
                    protected_overlap_pixel_count
                ),
                final_outside_foreground_pixel_count=(
                    outside_foreground_pixel_count
                ),
            )
        )
        passed = guard_decision.approval_enabled
        if processed_mask_pixel_count == 0:
            reason_ko = "처리 크기 변환 후 승인 마스크가 0픽셀입니다."
        elif model_mask_pixel_count == 0:
            reason_ko = "금지 영역 제한 후 실제 모델 마스크가 0픽셀입니다."
        elif (
            protected_overlap_pixel_count != 0
            or outside_foreground_pixel_count != 0
        ):
            reason_ko = (
                "금지 영역 제한 후 최종 마스크 침범이 남았습니다: "
                f"보호={protected_overlap_pixel_count:,}픽셀, "
                f"외곽 밖={outside_foreground_pixel_count:,}픽셀"
            )
        elif hard_overlap_pixel_count != 0:
            reason_ko = (
                "blur 직후 128~255 강한 침범 "
                f"{hard_overlap_pixel_count:,}픽셀을 금지 영역에서 제거했습니다. "
                "최종 침범은 0픽셀입니다."
            )
        elif soft_overlap_pixel_count != 0:
            reason_ko = (
                "blur 직후 1~127 약한 침범 "
                f"{soft_overlap_pixel_count:,}픽셀을 금지 영역에서 제거했습니다. "
                "최종 침범은 0픽셀입니다."
            )
        else:
            reason_ko = (
                "최종 보호 영역·외곽 침범이 모두 0픽셀입니다."
            )

        processed_person.save(arguments.output_processed_person)
        binary_mask.save(arguments.output_binary_mask)
        raw_blurred_mask.save(arguments.output_raw_blurred_mask)
        model_mask.save(arguments.output_model_mask)
        processed_clothing.save(arguments.output_processed_clothing)
        soft_overlap.save(arguments.output_soft_overlap)
        hard_overlap.save(arguments.output_hard_overlap)
        protected_overlap.save(arguments.output_protected_overlap)
        outside_foreground.save(arguments.output_outside_foreground)
        Path(arguments.output_metadata).write_text(
            json.dumps(
                {
                    "width": arguments.width,
                    "height": arguments.height,
                    "blur_factor": arguments.mask_blur_factor,
                    "processed_mask_pixel_count": processed_mask_pixel_count,
                    "model_mask_pixel_count": model_mask_pixel_count,
                    "soft_overlap_pixel_count": soft_overlap_pixel_count,
                    "hard_overlap_pixel_count": hard_overlap_pixel_count,
                    "removed_pixel_count": removed_pixel_count,
                    "protected_overlap_pixel_count": protected_overlap_pixel_count,
                    "outside_foreground_pixel_count": outside_foreground_pixel_count,
                    "person_sha256": calculate_image_pixel_sha256(processed_person, "RGB"),
                    "binary_mask_sha256": calculate_image_pixel_sha256(binary_mask, "L"),
                    "model_mask_sha256": calculate_image_pixel_sha256(model_mask, "L"),
                    "clothing_sha256": calculate_image_pixel_sha256(processed_clothing, "RGB"),
                    "passed": passed,
                    "reason_ko": reason_ko,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        for image in (
            source_person, source_mask, source_clothing, source_protection,
            source_foreground, processed_person, resized_mask, binary_mask,
            raw_blurred_mask, model_mask, processed_clothing,
            processed_protection, processed_foreground, soft_overlap,
            hard_overlap, protected_overlap, outside_foreground,
        ):
            if image is not None:
                image.close()


if __name__ == "__main__":
    main()
