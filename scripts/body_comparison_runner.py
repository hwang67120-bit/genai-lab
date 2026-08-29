"""CatVTON 신체 분리와 DWPose 관절 분석을 한 번 실행한다."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image, ImageDraw


BODY_JOINT_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)

BODY_BONES = (
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 1), (0, 14), (14, 16), (0, 15), (15, 17),
)


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
    parser.add_argument("--output-densepose", required=True)
    parser.add_argument("--output-pose-preview", required=True)
    parser.add_argument("--output-pose-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--pose-device", choices=("cpu",), required=True)
    parser.add_argument("--minimum-pose-confidence", type=float, required=True)
    return parser.parse_args()


def load_rgb_image(image_path: str) -> Image.Image:
    """파일 핸들을 즉시 닫고 RGB 픽셀만 소유한 이미지로 반환한다."""
    with Image.open(image_path) as opened_image:
        return opened_image.convert("RGB")


def create_pose_preview_and_coordinates(
    source_image: Image.Image,
    pose_detector,
    minimum_confidence: float,
):
    """DWPose를 1회 실행해 18개 몸 관절 좌표와 확인 그림을 만든다."""
    import numpy as np

    normalized_source_image = source_image.convert("RGB")
    try:
        source_array = np.asarray(
            normalized_source_image,
            dtype=np.uint8,
        ).copy()
    finally:
        normalized_source_image.close()
    candidates, scores = pose_detector.pose_estimation(source_array)
    if candidates.ndim != 3 or scores.ndim != 2 or candidates.shape[0] == 0:
        raise RuntimeError("DWPose가 사람 관절 후보를 1명도 찾지 못했습니다.")

    body_scores = scores[:, :18]
    selected_person_index = int(np.argmax(np.mean(body_scores, axis=1)))
    selected_candidates = candidates[selected_person_index, :18]
    selected_scores = scores[selected_person_index, :18]

    preview_image = source_image.convert("RGB")
    preview_draw = ImageDraw.Draw(preview_image)
    for start_index, end_index in BODY_BONES:
        if (
            selected_scores[start_index] >= minimum_confidence
            and selected_scores[end_index] >= minimum_confidence
        ):
            preview_draw.line(
                (
                    (
                        float(selected_candidates[start_index][0]),
                        float(selected_candidates[start_index][1]),
                    ),
                    (
                        float(selected_candidates[end_index][0]),
                        float(selected_candidates[end_index][1]),
                    ),
                ),
                fill=(0, 255, 0),
                width=4,
            )

    joint_coordinates = []
    for joint_index, joint_name in enumerate(BODY_JOINT_NAMES):
        x_value = float(selected_candidates[joint_index][0])
        y_value = float(selected_candidates[joint_index][1])
        confidence_score = float(selected_scores[joint_index])
        detected = confidence_score >= minimum_confidence
        preview_color = (0, 220, 0) if detected else (255, 80, 80)
        preview_draw.ellipse(
            (x_value - 5, y_value - 5, x_value + 5, y_value + 5),
            fill=preview_color,
        )
        joint_coordinates.append(
            {
                "joint_name": joint_name,
                "x": x_value,
                "y": y_value,
                "confidence_score": confidence_score,
                "detected": detected,
            }
        )
    return preview_image, joint_coordinates


def main() -> None:
    """신체 분리 3개 결과와 DWPose 관절 좌표를 임시 파일로 반환한다."""
    arguments = parse_arguments()
    started_at = perf_counter()
    repository_path = Path(arguments.repository_path).resolve()
    cache_dir = Path(arguments.cache_dir).resolve()
    pose_cache_directory = cache_dir / "easy-dwpose"
    pose_cache_directory.mkdir(parents=True, exist_ok=True)

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

    try:
        from easy_dwpose import DWposeDetector
    except ImportError as error:
        raise RuntimeError(
            "DWPose 관절 분석용 easy-dwpose가 없습니다. "
            "CatVTON 전용 Python에 easy-dwpose와 onnxruntime을 설치하세요."
        ) from error

    original_person_image = None
    resized_person_image = None
    raw_mask = None
    identity_protection_mask = None
    densepose_preview = None
    pose_preview = None
    automatic_mask_result = None
    previous_directory = Path.cwd()
    try:
        original_person_image = load_rgb_image(arguments.person_image)
        original_size = original_person_image.size
        resized_person_image = resize_and_crop(
            original_person_image,
            (arguments.width, arguments.height),
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
        schp_lip_mask = np.asarray(automatic_mask_result["schp_lip"])
        schp_atr_mask = np.asarray(automatic_mask_result["schp_atr"])
        identity_protection_array = (
            part_mask_of(protected_parts, schp_lip_mask, LIP_MAPPING)
            | part_mask_of(protected_parts, schp_atr_mask, ATR_MAPPING)
        )
        identity_protection_mask = Image.fromarray(
            (identity_protection_array > 0).astype(np.uint8) * 255
        ).convert("L")

        os.chdir(pose_cache_directory)
        pose_detector = DWposeDetector(device=arguments.pose_device)
        pose_preview, joint_coordinates = create_pose_preview_and_coordinates(
            original_person_image,
            pose_detector,
            arguments.minimum_pose_confidence,
        )
        os.chdir(previous_directory)

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
        resized_output = densepose_preview.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        try:
            resized_output.save(arguments.output_densepose)
        finally:
            resized_output.close()
        pose_preview.save(arguments.output_pose_preview)
        Path(arguments.output_pose_json).write_text(
            json.dumps(
                {
                    "model_ids": [
                        "zhengchong/CatVTON:SCHP+DensePose",
                        "RedHash/DWPose:yolox_l+dw-ll_ucoco_384",
                    ],
                    "minimum_pose_confidence": arguments.minimum_pose_confidence,
                    "joint_coordinates": joint_coordinates,
                    "elapsed_seconds": perf_counter() - started_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        os.chdir(previous_directory)
        for image in (
            original_person_image, resized_person_image, raw_mask,
            identity_protection_mask, densepose_preview, pose_preview,
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
