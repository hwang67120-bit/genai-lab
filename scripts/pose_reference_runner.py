"""승인 자세 이미지에서 DWPose 관절과 ControlNet용 뼈대 지도를 만든다."""

import argparse
import json
import math
import os
from pathlib import Path
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


def normalize_pose_confidence_score(raw_confidence_score: float) -> float:
    """DWPose 원시 점수를 후속 단계의 0.0~1.0 범위로 맞춘다."""
    if not math.isfinite(raw_confidence_score) or raw_confidence_score < 0.0:
        return 0.0
    return min(raw_confidence_score, 1.0)


def create_pose_preview_and_coordinates(
    source_image: Image.Image,
    pose_detector,
    minimum_confidence: float,
):
    """DWPose를 1회 실행해 관절 좌표·확인 그림·표준 지도를 만든다."""
    import numpy as np
    from easy_dwpose.draw import draw_openpose

    normalized_source_image = source_image.convert("RGB")
    try:
        source_array = np.asarray(normalized_source_image, dtype=np.uint8).copy()
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
                    tuple(float(value) for value in selected_candidates[start_index]),
                    tuple(float(value) for value in selected_candidates[end_index]),
                ),
                fill=(0, 255, 0),
                width=4,
            )

    joint_coordinates = []
    for joint_index, joint_name in enumerate(BODY_JOINT_NAMES):
        x_value = float(selected_candidates[joint_index][0])
        y_value = float(selected_candidates[joint_index][1])
        raw_confidence_score = float(selected_scores[joint_index])
        confidence_score = normalize_pose_confidence_score(raw_confidence_score)
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
                "raw_confidence_score": (
                    raw_confidence_score
                    if math.isfinite(raw_confidence_score)
                    else str(raw_confidence_score)
                ),
                "confidence_normalized": confidence_score != raw_confidence_score,
            }
        )

    selected_all_candidates = candidates[
        selected_person_index:selected_person_index + 1
    ].copy()
    selected_all_scores = scores[
        selected_person_index:selected_person_index + 1
    ].copy()
    image_height, image_width = source_array.shape[:2]
    selected_all_candidates[..., 0] /= float(image_width)
    selected_all_candidates[..., 1] /= float(image_height)
    body_candidates = selected_all_candidates[:, :18].reshape(18, 2)
    body_scores = selected_all_scores[:, :18].copy()
    for joint_index in range(18):
        body_scores[0][joint_index] = (
            joint_index
            if body_scores[0][joint_index] >= minimum_confidence
            else -1
        )
    openpose_payload = {
        "bodies": body_candidates,
        "body_scores": body_scores,
        "hands": np.vstack([
            selected_all_candidates[:, 92:113],
            selected_all_candidates[:, 113:],
        ]),
        "hands_scores": np.vstack([
            selected_all_scores[:, 92:113],
            selected_all_scores[:, 113:],
        ]),
        "faces": selected_all_candidates[:, 24:92],
        "faces_scores": selected_all_scores[:, 24:92],
    }
    control_map_array = draw_openpose(
        openpose_payload,
        height=image_height,
        width=image_width,
        include_face=True,
        include_hands=True,
    )
    control_map_image = Image.fromarray(control_map_array).convert("RGB")
    return preview_image, joint_coordinates, control_map_image


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
