import numpy as np
import pytest
from PIL import Image

from genai_lab.character_target_landmarks import (
    CharacterTargetLandmarkError,
    CharacterTargetLandmarkSettings,
    extract_character_target_landmarks,
)
from genai_lab.clothing import ClothingCategory
from genai_lab.pose_estimation import (
    PoseControlPreparedInput,
    PoseJointCoordinateCandidate,
)


def create_prepared_pose() -> PoseControlPreparedInput:
    return PoseControlPreparedInput(
        control_map_image=Image.new("RGB", (200, 200), "black"),
        source_width=100,
        source_height=200,
        target_width=200,
        target_height=200,
        resize_scale=1.0,
        padding_left=50,
        padding_top=0,
        padding_right=50,
        padding_bottom=0,
        non_black_pixel_count=100,
    )


def create_joint(
    name: str,
    x: float,
    y: float,
    confidence: float = 0.90,
    detected: bool = True,
) -> PoseJointCoordinateCandidate:
    return PoseJointCoordinateCandidate(
        joint_name=name,
        x=x,
        y=y,
        confidence_score=confidence,
        detected=detected,
    )


def create_top_joints() -> tuple[PoseJointCoordinateCandidate, ...]:
    return (
        create_joint("left_shoulder", 20, 30),
        create_joint("right_shoulder", 80, 30),
        create_joint("left_hip", 25, 110),
        create_joint("right_hip", 75, 110),
    )


def create_bottom_joints() -> tuple[PoseJointCoordinateCandidate, ...]:
    return (
        create_joint("left_hip", 25, 80),
        create_joint("right_hip", 75, 80),
        create_joint("left_knee", 30, 120),
        create_joint("right_knee", 70, 120),
        create_joint("left_ankle", 35, 160),
        create_joint("right_ankle", 65, 160),
    )


def create_mask(
    *,
    size: tuple[int, int] = (200, 200),
    box: tuple[int, int, int, int] = (60, 20, 141, 181),
) -> Image.Image:
    pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
    left, top, right, bottom = box
    pixels[top:bottom, left:right] = 255
    return Image.fromarray(pixels, mode="L")


def test_top_uses_shoulders_torso_midpoint_and_hips() -> None:
    prepared = create_prepared_pose()
    mask = create_mask()
    result = extract_character_target_landmarks(
        create_top_joints(),
        prepared,
        mask,
        ClothingCategory.TOP,
    )

    assert result.row_sources == (
        "dwpose_shoulders",
        "dwpose_torso_midpoint",
        "dwpose_hips",
    )
    assert result.selected_rows_y == (30, 70, 110)
    assert np.array_equal(
        result.points_xy,
        np.array(
            [[60, 30], [140, 30], [60, 70], [140, 70], [60, 110], [140, 110]],
            dtype=np.float32,
        ),
    )
    assert result.required_joint_names == (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )
    assert result.minimum_used_joint_confidence == pytest.approx(0.90)
    assert result.row_search_radius_pixels == 10
    mask.close()
    prepared.close()


def test_bottom_uses_hips_knees_and_ankles() -> None:
    prepared = create_prepared_pose()
    mask = create_mask(box=(60, 60, 141, 181))
    result = extract_character_target_landmarks(
        create_bottom_joints(),
        prepared,
        mask,
        ClothingCategory.BOTTOM,
    )

    assert result.selected_rows_y == (80, 120, 160)
    assert result.row_sources == (
        "dwpose_hips",
        "dwpose_knees",
        "dwpose_ankles",
    )
    mask.close()
    prepared.close()


@pytest.mark.parametrize(
    "category",
    [ClothingCategory.DRESS, ClothingCategory.FULL_BODY_OUTFIT],
)
def test_overall_categories_use_approved_mask_lower_90_percent(
    category: ClothingCategory,
) -> None:
    prepared = create_prepared_pose()
    mask = create_mask()
    result = extract_character_target_landmarks(
        create_top_joints(),
        prepared,
        mask,
        category,
    )

    assert result.selected_rows_y == (30, 110, 164)
    assert result.row_sources[-1] == "approved_mask_lower_90"
    assert result.approved_mask_bbox_xywh == (60, 20, 81, 161)
    mask.close()
    prepared.close()


def test_missing_exact_row_uses_nearest_approved_mask_row() -> None:
    prepared = create_prepared_pose()
    pixels = np.zeros((200, 200), dtype=np.uint8)
    pixels[20:181, 60:141] = 255
    pixels[30, :] = 0
    mask = Image.fromarray(pixels, mode="L")
    result = extract_character_target_landmarks(
        create_top_joints(),
        prepared,
        mask,
        ClothingCategory.TOP,
    )

    assert result.selected_rows_y[0] == 29
    mask.close()
    prepared.close()


def test_low_confidence_required_joint_is_rejected() -> None:
    prepared = create_prepared_pose()
    mask = create_mask()
    joints = list(create_top_joints())
    joints[0] = create_joint("left_shoulder", 20, 30, confidence=0.29)
    with pytest.raises(CharacterTargetLandmarkError, match="기준 미만"):
        extract_character_target_landmarks(
            tuple(joints),
            prepared,
            mask,
            ClothingCategory.TOP,
        )
    mask.close()
    prepared.close()


def test_duplicate_joint_name_is_rejected() -> None:
    prepared = create_prepared_pose()
    mask = create_mask()
    joints = create_top_joints() + (create_joint("left_shoulder", 21, 31),)
    with pytest.raises(CharacterTargetLandmarkError, match="중복"):
        extract_character_target_landmarks(
            joints,
            prepared,
            mask,
            ClothingCategory.TOP,
        )
    mask.close()
    prepared.close()


def test_mask_and_target_canvas_size_mismatch_is_rejected() -> None:
    prepared = create_prepared_pose()
    mask = create_mask(size=(100, 200), box=(10, 20, 91, 181))
    with pytest.raises(CharacterTargetLandmarkError, match="크기가 다릅니다"):
        extract_character_target_landmarks(
            create_top_joints(),
            prepared,
            mask,
            ClothingCategory.TOP,
        )
    mask.close()
    prepared.close()


@pytest.mark.parametrize(
    "category",
    [ClothingCategory.GLOVES, ClothingCategory.SHOES],
)
def test_unsupported_small_accessory_categories_are_rejected(
    category: ClothingCategory,
) -> None:
    prepared = create_prepared_pose()
    mask = create_mask()
    with pytest.raises(CharacterTargetLandmarkError, match="지원하지 않는"):
        extract_character_target_landmarks(
            create_top_joints(),
            prepared,
            mask,
            category,
        )
    mask.close()
    prepared.close()


def test_empty_approved_mask_is_rejected() -> None:
    prepared = create_prepared_pose()
    mask = Image.new("L", (200, 200), 0)
    with pytest.raises(CharacterTargetLandmarkError, match="0개"):
        extract_character_target_landmarks(
            create_top_joints(),
            prepared,
            mask,
            ClothingCategory.TOP,
        )
    mask.close()
    prepared.close()


def test_invalid_settings_are_rejected_before_coordinate_work() -> None:
    prepared = create_prepared_pose()
    mask = create_mask()
    with pytest.raises(CharacterTargetLandmarkError, match="0.0~1.0"):
        extract_character_target_landmarks(
            create_top_joints(),
            prepared,
            mask,
            ClothingCategory.TOP,
            CharacterTargetLandmarkSettings(minimum_joint_confidence=1.1),
        )
    mask.close()
    prepared.close()
