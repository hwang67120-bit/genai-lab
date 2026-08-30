"""승인 자세 이미지에서 DWPose 관절과 ControlNet용 뼈대 지도를 만든다."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from PIL import Image

from body_comparison_runner import create_pose_preview_and_coordinates


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-image", required=True)
    parser.add_argument("--output-overlay", required=True)
    parser.add_argument("--output-control-map", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pose-device", choices=("cpu",), required=True)
    parser.add_argument(
        "--minimum-pose-confidence", type=float, required=True
    )
    return parser.parse_args()


def main() -> None:
    """DWPose를 CPU로 1회 실행하고 임시 결과 3개를 반환한다."""
    arguments = parse_arguments()
    if not 0.0 <= arguments.minimum_pose_confidence <= 1.0:
        raise RuntimeError("관절 기준 점수는 0.0~1.0이어야 합니다.")
    started_at = perf_counter()
    cache_dir = Path(arguments.cache_dir).resolve()
    pose_cache_directory = cache_dir / "easy-dwpose"
    pose_cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)

    import truststore

    truststore.inject_into_ssl()
    try:
        from easy_dwpose import DWposeDetector
    except ImportError as error:
        raise RuntimeError(
            "DWPose 관절 분석용 easy-dwpose가 없습니다. "
            "CatVTON 전용 Python에 easy-dwpose와 onnxruntime을 설치하세요."
        ) from error

    source_image = None
    overlay_image = None
    control_map_image = None
    previous_directory = Path.cwd()
    try:
        with Image.open(arguments.pose_image) as opened_image:
            source_image = opened_image.convert("RGB")
        os.chdir(pose_cache_directory)
        pose_detector = DWposeDetector(device=arguments.pose_device)
        overlay_image, joint_coordinates, control_map_image = (
            create_pose_preview_and_coordinates(
                source_image,
                pose_detector,
                arguments.minimum_pose_confidence,
                include_openpose_control_map=True,
            )
        )
        detected_joint_count = sum(
            1 for coordinate in joint_coordinates if coordinate["detected"]
        )
        overlay_image.save(arguments.output_overlay)
        control_map_image.save(arguments.output_control_map)
        Path(arguments.output_json).write_text(
            json.dumps(
                {
                    "model_ids": [
                        "RedHash/DWPose:yolox_l+dw-ll_ucoco_384"
                    ],
                    "source_width": source_image.width,
                    "source_height": source_image.height,
                    "minimum_pose_confidence": (
                        arguments.minimum_pose_confidence
                    ),
                    "joint_coordinates": joint_coordinates,
                    "detected_joint_count": detected_joint_count,
                    "missing_joint_count": 18 - detected_joint_count,
                    "elapsed_seconds": perf_counter() - started_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        os.chdir(previous_directory)
        for image in (source_image, overlay_image, control_map_image):
            if image is not None:
                image.close()


if __name__ == "__main__":
    main()
