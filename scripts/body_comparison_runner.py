"""CatVTON의 SCHP·DensePose로 기준 후보의 의상·신체 영역을 분석한다."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-path", required=True)
    parser.add_argument("--person-image", required=True)
    parser.add_argument(
        "--clothing-type",
        choices=("upper", "lower", "overall", "inner", "outer"),
        required=True,
    )
    parser.add_argument("--output-raw-mask", required=True)
    parser.add_argument("--output-protection-mask", required=True)
    parser.add_argument("--output-foreground-mask", required=True)
    parser.add_argument("--output-densepose", required=True)
    parser.add_argument("--output-metadata-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--foreground-model-id", default="isnet-anime")
    parser.add_argument("--explicit-target-masks", action="store_true")
    return parser.parse_args()


def load_rgb_image(image_path: str) -> Image.Image:
    """파일 핸들을 즉시 닫고 RGB 픽셀만 소유한 이미지로 반환한다."""
    with Image.open(image_path) as opened_image:
        return opened_image.convert("RGB")


def extract_anime_character_foreground_mask(
    person_image: Image.Image,
    model_id: str,
    model_cache_dir: Path,
) -> Image.Image:
    """공식 isnetis ONNX로 캐릭터 전체 외곽의 알파 마스크를 반환한다."""
    if model_id != "isnet-anime":
        raise RuntimeError(
            "지원하지 않는 캐릭터 외곽 모델입니다: " f"{model_id}"
        )

    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="skytnt/anime-seg",
        filename="isnetis.onnx",
        cache_dir=model_cache_dir,
    )
    foreground_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    input_shape = foreground_session.get_inputs()[0].shape
    model_height = int(input_shape[2])
    model_width = int(input_shape[3])
    scale = min(
        model_width / person_image.width,
        model_height / person_image.height,
    )
    content_width = max(1, round(person_image.width * scale))
    content_height = max(1, round(person_image.height * scale))
    content_left = (model_width - content_width) // 2
    content_top = (model_height - content_height) // 2
    resized_image = person_image.resize(
        (content_width, content_height),
        Image.Resampling.LANCZOS,
    )
    model_canvas = Image.new("RGB", (model_width, model_height), (0, 0, 0))
    try:
        model_canvas.paste(resized_image, (content_left, content_top))
        model_input = np.asarray(model_canvas, dtype=np.float32)
        model_input = (
            model_input[:, :, ::-1]
            .transpose((2, 0, 1))[None, :, :, :]
            / 255.0
        )
        predicted_mask = foreground_session.run(
            [foreground_session.get_outputs()[0].name],
            {foreground_session.get_inputs()[0].name: model_input},
        )[0]
        predicted_mask = np.squeeze(predicted_mask)
        predicted_mask = np.clip(predicted_mask, 0.0, 1.0)
        model_mask = Image.fromarray(
            (predicted_mask * 255.0).round().astype(np.uint8),
            mode="L",
        )
        try:
            content_mask = model_mask.crop(
                (
                    content_left,
                    content_top,
                    content_left + content_width,
                    content_top + content_height,
                )
            )
            try:
                return content_mask.resize(
                    person_image.size,
                    Image.Resampling.LANCZOS,
                )
            finally:
                content_mask.close()
        finally:
            model_mask.close()
    finally:
        resized_image.close()
        model_canvas.close()


def main() -> None:
    """기준 후보에서 의상 마스크·신체 보호 영역·DensePose를 반환한다."""
    arguments = parse_arguments()
    started_at = perf_counter()
    repository_path = Path(arguments.repository_path).resolve()
    cache_dir = Path(arguments.cache_dir).resolve()
    os.environ["HF_HOME"] = str(cache_dir)
    sys.path.insert(0, str(repository_path))

    import truststore

    truststore.inject_into_ssl()
    import numpy as np
    from huggingface_hub import snapshot_download
    from model.cloth_masker import (
        ATR_MAPPING, LIP_MAPPING, AutoMasker, part_mask_of,
    )
    from utils import resize_and_crop

    original_person_image = None
    resized_person_image = None
    raw_mask = None
    identity_protection_mask = None
    character_foreground_mask = None
    densepose_preview = None
    automatic_mask_result = None
    try:
        original_person_image = load_rgb_image(arguments.person_image)
        original_size = original_person_image.size
        resized_person_image = resize_and_crop(
            original_person_image,
            (arguments.width, arguments.height),
        )

        foreground_started_at = perf_counter()
        character_foreground_mask = extract_anime_character_foreground_mask(
            resized_person_image,
            arguments.foreground_model_id,
            cache_dir,
        )
        foreground_elapsed_seconds = perf_counter() - foreground_started_at
        foreground_mask_array = np.asarray(
            character_foreground_mask,
            dtype=np.uint8,
        )
        foreground_pixel_count = int(
            np.count_nonzero(foreground_mask_array >= 128)
        )
        if foreground_pixel_count == 0:
            raise RuntimeError(
                "isnet-anime이 캐릭터 외곽 픽셀을 찾지 못했습니다."
            )
        foreground_percent = (
            foreground_pixel_count / foreground_mask_array.size * 100.0
        )

        checkpoint_path = snapshot_download(
            repo_id="zhengchong/CatVTON",
            cache_dir=cache_dir,
        )
        automatic_masker = AutoMasker(
            densepose_ckpt=str(Path(checkpoint_path) / "DensePose"),
            schp_ckpt=str(Path(checkpoint_path) / "SCHP"),
            device="cuda",
        )
        automatic_mask_result = automatic_masker(
            resized_person_image,
            arguments.clothing_type,
        )
        raw_mask = automatic_mask_result["mask"].convert("L")
        densepose_preview = automatic_mask_result["densepose"].convert("RGB")

        protected_parts = [
            "Hat", "Hair", "Sunglasses", "Face", "Left-arm", "Right-arm",
            "Left-leg", "Right-leg", "Left-shoe", "Right-shoe", "Glove",
            "Bag", "Scarf",
        ]
        if arguments.explicit_target_masks:
            # 교체 범위는 별도 사용자 승인 SAM2 마스크가 제한한다.
            # 신발을 무조건 보호하지 않되 얼굴·머리는 계속 보호한다.
            protected_parts = ["Hat", "Hair", "Sunglasses", "Face"]
        schp_lip_mask = np.asarray(automatic_mask_result["schp_lip"])
        schp_atr_mask = np.asarray(automatic_mask_result["schp_atr"])
        identity_protection_array = (
            part_mask_of(protected_parts, schp_lip_mask, LIP_MAPPING)
            | part_mask_of(protected_parts, schp_atr_mask, ATR_MAPPING)
        )
        identity_protection_mask = Image.fromarray(
            (identity_protection_array > 0).astype(np.uint8) * 255
        ).convert("L")

        resized_output = raw_mask.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        try:
            resized_output.save(arguments.output_raw_mask)
        finally:
            resized_output.close()
        resized_output = identity_protection_mask.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        try:
            resized_output.save(arguments.output_protection_mask)
        finally:
            resized_output.close()
        resized_output = character_foreground_mask.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )
        try:
            resized_foreground_array = np.asarray(
                resized_output,
                dtype=np.uint8,
            )
            foreground_pixel_count = int(
                np.count_nonzero(resized_foreground_array >= 128)
            )
            foreground_percent = (
                foreground_pixel_count
                / resized_foreground_array.size
                * 100.0
            )
            resized_output.save(arguments.output_foreground_mask)
        finally:
            resized_output.close()
        resized_output = densepose_preview.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        try:
            resized_output.save(arguments.output_densepose)
        finally:
            resized_output.close()
        Path(arguments.output_metadata_json).write_text(
            json.dumps(
                {
                    "model_ids": [
                        "zhengchong/CatVTON:SCHP+DensePose",
                        arguments.foreground_model_id,
                    ],
                    "foreground_model_id": arguments.foreground_model_id,
                    "explicit_target_masks": arguments.explicit_target_masks,
                    "foreground_pixel_count": foreground_pixel_count,
                    "foreground_percent": foreground_percent,
                    "foreground_elapsed_seconds": foreground_elapsed_seconds,
                    "elapsed_seconds": perf_counter() - started_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        for image in (
            original_person_image, resized_person_image, raw_mask,
            identity_protection_mask, character_foreground_mask,
            densepose_preview,
        ):
            if image is not None:
                image.close()
        if automatic_mask_result is not None:
            for result_name in ("mask", "densepose", "schp_lip", "schp_atr"):
                result_image = automatic_mask_result.get(result_name)
                if result_image is not None:
                    result_image.close()


if __name__ == "__main__":
    main()
