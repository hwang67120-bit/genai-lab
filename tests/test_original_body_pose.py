from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from genai_lab.original_body_pose import (
    OriginalBodyPoseError,
    approve_original_body_pose,
    prepare_original_body_pose_control,
)
from genai_lab.pose_estimation import (
    PoseEstimationReviewCandidate,
    PoseJointCoordinateCandidate,
)
from genai_lab.pose_fallback import PoseFallbackSettings


JOINT_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)


def create_candidate(source: Image.Image, detected_count: int = 18):
    coordinates = tuple(
        PoseJointCoordinateCandidate(
            joint_name=name,
            x=float(index + 1),
            y=float(index + 2),
            confidence_score=0.9 if index < detected_count else 0.1,
            detected=index < detected_count,
        )
        for index, name in enumerate(JOINT_NAMES)
    )
    control_map = Image.new("RGB", source.size, "black")
    ImageDraw.Draw(control_map).line((16, 8, 16, 56), fill="white", width=2)
    return PoseEstimationReviewCandidate(
        source_image=source.copy(),
        overlay_image=source.copy(),
        control_map_image=control_map,
        joint_coordinates=coordinates,
        detected_joint_count=detected_count,
        missing_joint_count=18 - detected_count,
        minimum_pose_confidence=0.3,
        model_ids=("test-dwpose",),
        elapsed_seconds=1.25,
    )


def quality_settings() -> PoseFallbackSettings:
    return PoseFallbackSettings(library_root=Path("unused"))


def test_original_body_pose_is_bound_to_exact_character_pixels() -> None:
    source = Image.new("RGB", (32, 64), (240, 220, 210))
    candidate = create_candidate(source)
    approved = approve_original_body_pose(
        source,
        candidate,
        quality_settings(),
    )
    try:
        assert approved.source_width == 32
        assert approved.source_height == 64
        assert len(approved.source_image_sha256) == 64
        assert approved.approved_pose.detected_joint_count == 18
        assert approved.required_group_pass_count == 4
        assert approved.required_group_count == 4
        assert approved.non_black_pixel_count > 0
        candidate.control_map_image.putpixel((16, 8), (255, 0, 0))
        assert approved.approved_pose.control_map_image.getpixel((16, 8)) == (
            255, 255, 255
        )
    finally:
        approved.close()
        candidate.close()
        source.close()


def test_original_body_pose_rejects_different_source_pixels() -> None:
    source = Image.new("RGB", (32, 64), "white")
    different = Image.new("RGB", (32, 64), "gray")
    candidate = create_candidate(different)
    try:
        with pytest.raises(OriginalBodyPoseError, match="원본 픽셀"):
            approve_original_body_pose(source, candidate, quality_settings())
    finally:
        candidate.close()
        source.close()
        different.close()


def test_original_body_pose_rejects_pose_quality_failure() -> None:
    source = Image.new("RGB", (32, 64), "white")
    candidate = create_candidate(source, detected_count=7)
    try:
        with pytest.raises(OriginalBodyPoseError, match="품질 미달"):
            approve_original_body_pose(source, candidate, quality_settings())
    finally:
        candidate.close()
        source.close()


def test_original_body_pose_is_reverified_before_controlnet_use() -> None:
    source = Image.new("RGB", (256, 512), "white")
    candidate = create_candidate(source)
    approved = approve_original_body_pose(source, candidate, quality_settings())
    changed = source.copy()
    changed.putpixel((0, 0), (0, 0, 0))
    try:
        prepared = prepare_original_body_pose_control(approved, source)
        try:
            assert prepared.control_map_image.size == source.size
            assert prepared.non_black_pixel_count > 0
        finally:
            prepared.close()
        with pytest.raises(OriginalBodyPoseError, match="직전.*픽셀"):
            prepare_original_body_pose_control(approved, changed)
    finally:
        changed.close()
        approved.close()
        candidate.close()
        source.close()
