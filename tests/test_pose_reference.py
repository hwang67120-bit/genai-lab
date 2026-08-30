from pathlib import Path

import pytest
from PIL import Image

from genai_lab.pose_reference import (
    PoseReferenceSettings,
    PoseReferenceValidationError,
    approve_pose_reference_candidate,
    load_pose_reference_candidate,
)


def test_pose_reference_reports_numeric_input_values(tmp_path: Path) -> None:
    image_path = tmp_path / "pose.png"
    Image.new("RGB", (320, 640), "white").save(image_path)

    review_candidate = load_pose_reference_candidate(image_path)
    try:
        assert review_candidate.image_format == "PNG"
        assert review_candidate.width == 320
        assert review_candidate.height == 640
        assert review_candidate.pixel_count == 204_800
        assert review_candidate.aspect_ratio == 0.5
        assert review_candidate.file_size_bytes > 0
    finally:
        review_candidate.close()


def test_pose_reference_approval_owns_independent_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "pose.jpg"
    Image.new("RGB", (128, 256), "blue").save(image_path)
    review_candidate = load_pose_reference_candidate(image_path)

    approved_input = approve_pose_reference_candidate(review_candidate)
    review_candidate.close()
    try:
        assert approved_input.image.getpixel((0, 0))[2] > 200
        assert approved_input.width == 128
        assert approved_input.height == 256
    finally:
        approved_input.close()


def test_pose_reference_rejects_too_small_image(tmp_path: Path) -> None:
    image_path = tmp_path / "small.png"
    Image.new("RGB", (63, 128), "white").save(image_path)

    with pytest.raises(PoseReferenceValidationError, match="너무 작습니다"):
        load_pose_reference_candidate(image_path)


def test_pose_reference_rejects_pixel_limit(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    with pytest.raises(PoseReferenceValidationError, match="제한을 초과"):
        load_pose_reference_candidate(
            image_path,
            PoseReferenceSettings(maximum_pixel_count=9_999),
        )
