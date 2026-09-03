import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from genai_lab.pose_estimation import (
    PoseEstimationApprovedInput,
    PoseJointCoordinateCandidate,
)
from genai_lab.pose_fallback import (
    PoseFallbackError,
    PoseFallbackSettings,
    evaluate_pose_quality,
    load_default_approved_pose,
    save_default_approved_pose,
)


BODY_JOINT_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)


def create_pose(
    detected_names: set[str] | None = None,
) -> PoseEstimationApprovedInput:
    if detected_names is None:
        detected_names = set(BODY_JOINT_NAMES)
    coordinates = tuple(
        PoseJointCoordinateCandidate(
            joint_name=name,
            x=float(index * 2),
            y=float(index * 3),
            confidence_score=0.90 if name in detected_names else 0.10,
            detected=name in detected_names,
        )
        for index, name in enumerate(BODY_JOINT_NAMES)
    )
    control_map = Image.new("RGB", (64, 128), "black")
    ImageDraw.Draw(control_map).line((32, 10, 32, 118), fill="white", width=3)
    return PoseEstimationApprovedInput(
        control_map_image=control_map,
        joint_coordinates=coordinates,
        detected_joint_count=len(detected_names),
        missing_joint_count=18 - len(detected_names),
        minimum_pose_confidence=0.30,
        model_ids=("test-dwpose",),
    )


def test_pose_quality_accepts_full_body_groups(tmp_path: Path) -> None:
    detected_names = {
        "nose", "neck", "right_shoulder", "right_elbow",
        "right_wrist", "right_hip", "right_knee", "right_ankle",
    }
    pose = create_pose(detected_names)
    try:
        decision = evaluate_pose_quality(
            pose,
            PoseFallbackSettings(library_root=tmp_path),
        )
        assert decision.accepted is True
        assert decision.detected_joint_count == 8
        assert decision.required_group_pass_count == 4
        assert decision.required_group_count == 4
        assert decision.non_black_pixel_count > 0
    finally:
        pose.close()


def test_pose_quality_rejects_missing_ankle_group(tmp_path: Path) -> None:
    detected_names = set(BODY_JOINT_NAMES) - {"right_ankle", "left_ankle"}
    pose = create_pose(detected_names)
    try:
        decision = evaluate_pose_quality(
            pose,
            PoseFallbackSettings(library_root=tmp_path),
        )
        assert decision.accepted is False
        assert "필수 관절 그룹 누락=발목" in decision.rejection_reasons
    finally:
        pose.close()


def test_saved_pose_round_trip_preserves_hash_and_coordinates(
    tmp_path: Path,
) -> None:
    settings = PoseFallbackSettings(library_root=tmp_path)
    pose = create_pose()
    preview = Image.new("RGB", (64, 128), "white")
    try:
        digest = save_default_approved_pose(pose, preview, settings)
        loaded = load_default_approved_pose(settings)
        try:
            assert loaded.pose_id == "last-approved"
            assert loaded.control_map_sha256 == digest
            assert loaded.approved_pose.detected_joint_count == 18
            assert len(loaded.approved_pose.joint_coordinates) == 18
            assert loaded.source_preview_image.size == (64, 128)
        finally:
            loaded.close()
    finally:
        pose.close()
        preview.close()


def test_saved_pose_rejects_modified_control_map(tmp_path: Path) -> None:
    settings = PoseFallbackSettings(library_root=tmp_path)
    pose = create_pose()
    preview = Image.new("RGB", (64, 128), "white")
    try:
        save_default_approved_pose(pose, preview, settings)
        changed_map_path = tmp_path / "last-approved" / "control_map.png"
        Image.new("RGB", (64, 128), "red").save(changed_map_path)
        with pytest.raises(PoseFallbackError, match="SHA-256"):
            load_default_approved_pose(settings)
    finally:
        pose.close()
        preview.close()


def test_saved_pose_rejects_tampered_joint_count(tmp_path: Path) -> None:
    settings = PoseFallbackSettings(library_root=tmp_path)
    pose = create_pose()
    preview = Image.new("RGB", (64, 128), "white")
    try:
        save_default_approved_pose(pose, preview, settings)
        metadata_path = tmp_path / "last-approved" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["detected_joint_count"] = 17
        metadata["missing_joint_count"] = 1
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False),
            encoding="utf-8",
        )
        with pytest.raises(PoseFallbackError, match="현재 폴백 품질 기준"):
            load_default_approved_pose(settings)
    finally:
        pose.close()
        preview.close()


def test_saved_pose_rejects_tampered_detection_flag(tmp_path: Path) -> None:
    settings = PoseFallbackSettings(library_root=tmp_path)
    pose = create_pose()
    preview = Image.new("RGB", (64, 128), "white")
    try:
        save_default_approved_pose(pose, preview, settings)
        metadata_path = tmp_path / "last-approved" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["joint_coordinates"][0]["detected"] = False
        metadata["detected_joint_count"] = 17
        metadata["missing_joint_count"] = 1
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False),
            encoding="utf-8",
        )
        with pytest.raises(PoseFallbackError, match="현재 폴백 품질 기준"):
            load_default_approved_pose(settings)
    finally:
        pose.close()
        preview.close()


def test_missing_saved_pose_is_reported_without_creating_files(
    tmp_path: Path,
) -> None:
    settings = PoseFallbackSettings(library_root=tmp_path)
    with pytest.raises(PoseFallbackError, match="파일이 없습니다"):
        load_default_approved_pose(settings)
    assert list(tmp_path.iterdir()) == []
