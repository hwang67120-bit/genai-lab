from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from genai_lab.pose_estimation import (
    PoseEstimationReviewCandidate,
    PoseJointCoordinateCandidate,
    PoseReferenceEstimationError,
    approve_pose_estimation_candidate,
    prepare_pose_control_input,
)


def test_pose_estimation_approval_copies_control_map() -> None:
    coordinates = tuple(
        PoseJointCoordinateCandidate(
            joint_name=f"joint_{index}",
            x=float(index),
            y=float(index + 1),
            confidence_score=0.9,
            detected=True,
        )
        for index in range(18)
    )
    review_candidate = PoseEstimationReviewCandidate(
        source_image=Image.new("RGB", (64, 128), "white"),
        overlay_image=Image.new("RGB", (64, 128), "green"),
        control_map_image=Image.new("RGB", (64, 128), "black"),
        joint_coordinates=coordinates,
        detected_joint_count=18,
        missing_joint_count=0,
        minimum_pose_confidence=0.30,
        model_ids=("test-dwpose",),
        elapsed_seconds=1.25,
    )

    approved_input = approve_pose_estimation_candidate(review_candidate)
    review_candidate.close()
    try:
        assert approved_input.control_map_image.size == (64, 128)
        assert approved_input.detected_joint_count == 18
        assert approved_input.missing_joint_count == 0
        assert len(approved_input.joint_coordinates) == 18
    finally:
        approved_input.close()


def test_pose_control_preparation_preserves_ratio_with_black_padding() -> None:
    coordinates = tuple(
        PoseJointCoordinateCandidate(
            joint_name=f"joint_{index}",
            x=float(index),
            y=float(index + 1),
            confidence_score=0.9,
            detected=True,
        )
        for index in range(18)
    )
    control_map = Image.new("RGB", (100, 200), "black")
    ImageDraw.Draw(control_map).line((50, 20, 50, 180), fill="white", width=3)
    approved_input = approve_pose_estimation_candidate(
        PoseEstimationReviewCandidate(
            source_image=Image.new("RGB", (100, 200), "white"),
            overlay_image=Image.new("RGB", (100, 200), "green"),
            control_map_image=control_map,
            joint_coordinates=coordinates,
            detected_joint_count=18,
            missing_joint_count=0,
            minimum_pose_confidence=0.30,
            model_ids=("test-dwpose",),
            elapsed_seconds=1.0,
        )
    )
    try:
        prepared_input = prepare_pose_control_input(
            approved_input,
            target_width=768,
            target_height=1344,
        )
        try:
            assert prepared_input.control_map_image.size == (768, 1344)
            assert prepared_input.resize_scale == pytest.approx(6.72)
            assert prepared_input.padding_left == 48
            assert prepared_input.padding_right == 48
            assert prepared_input.padding_top == 0
            assert prepared_input.padding_bottom == 0
            assert prepared_input.non_black_pixel_count > 0
        finally:
            prepared_input.close()
    finally:
        approved_input.close()


def test_pose_control_preparation_rejects_empty_control_map() -> None:
    coordinates = tuple(
        PoseJointCoordinateCandidate(
            joint_name=f"joint_{index}",
            x=0.0,
            y=0.0,
            confidence_score=0.9,
            detected=True,
        )
        for index in range(18)
    )
    approved_input = approve_pose_estimation_candidate(
        PoseEstimationReviewCandidate(
            source_image=Image.new("RGB", (64, 128), "white"),
            overlay_image=Image.new("RGB", (64, 128), "green"),
            control_map_image=Image.new("RGB", (64, 128), "black"),
            joint_coordinates=coordinates,
            detected_joint_count=18,
            missing_joint_count=0,
            minimum_pose_confidence=0.30,
            model_ids=("test-dwpose",),
            elapsed_seconds=1.0,
        )
    )
    try:
        with pytest.raises(PoseReferenceEstimationError, match="뼈대 픽셀이 0개"):
            prepare_pose_control_input(
                approved_input,
                target_width=768,
                target_height=1344,
            )
    finally:
        approved_input.close()
