"""DWPose 자세와 승인 변경 마스크를 결합해 TPS 목표점을 만든다."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image

from genai_lab.clothing import ClothingCategory
from genai_lab.garment_landmarks import LANDMARK_ROLES
from genai_lab.pose_estimation import (
    PoseControlPreparedInput,
    PoseJointCoordinateCandidate,
)


SUPPORTED_CATEGORIES = frozenset(
    {
        ClothingCategory.TOP,
        ClothingCategory.BOTTOM,
        ClothingCategory.DRESS,
        ClothingCategory.FULL_BODY_OUTFIT,
    }
)


@dataclass(frozen=True)
class CharacterTargetLandmarkSettings:
    """관절 신뢰도와 승인 마스크 교차 탐색 수치."""

    minimum_joint_confidence: float = 0.30
    mask_alpha_threshold: int = 128
    row_search_ratio: float = 0.05
    maximum_row_search_pixels: int = 64


@dataclass(frozen=True)
class CharacterTargetLandmarkResult:
    """사용자 검토 전 캐릭터 의상 목표 좌표와 근거 수치."""

    points_xy: np.ndarray
    roles: tuple[str, ...]
    row_sources: tuple[str, str, str]
    selected_rows_y: tuple[int, int, int]
    target_canvas_size: tuple[int, int]
    approved_mask_foreground_pixels: int
    approved_mask_bbox_xywh: tuple[int, int, int, int]
    required_joint_names: tuple[str, ...]
    minimum_used_joint_confidence: float
    mean_used_joint_confidence: float
    row_search_radius_pixels: int
    horizontal_overlap_score: float
    extraction_method: str = "dwpose_mask_intersection_v1"


class CharacterTargetLandmarkError(ValueError):
    """DWPose·승인 마스크 목표 좌표 계약을 만족하지 못한 오류."""


def extract_character_target_landmarks(
    joint_coordinates: tuple[PoseJointCoordinateCandidate, ...],
    prepared_pose: PoseControlPreparedInput,
    approved_change_mask: Image.Image,
    clothing_category: ClothingCategory,
    settings: CharacterTargetLandmarkSettings | None = None,
) -> CharacterTargetLandmarkResult:
    """DWPose의 의미 Y축과 실제 승인 마스크의 좌우 경계를 결합한다."""
    resolved_settings = settings or CharacterTargetLandmarkSettings()
    _validate_settings(resolved_settings)
    if clothing_category not in SUPPORTED_CATEGORIES:
        raise CharacterTargetLandmarkError(
            "캐릭터 목표점 자동 추출이 지원하지 않는 의상 종류입니다: "
            f"{clothing_category.value}"
        )

    target_size = (prepared_pose.target_width, prepared_pose.target_height)
    if approved_change_mask.size != target_size:
        raise CharacterTargetLandmarkError(
            "승인 변경 마스크와 자세 목표 캔버스 크기가 다릅니다: "
            f"마스크={approved_change_mask.size}, 자세={target_size}"
        )
    if prepared_pose.source_width < 1 or prepared_pose.source_height < 1:
        raise CharacterTargetLandmarkError("DWPose 원본 크기는 1px 이상이어야 합니다.")
    if (
        not math.isfinite(prepared_pose.resize_scale)
        or prepared_pose.resize_scale <= 0.0
    ):
        raise CharacterTargetLandmarkError(
            f"자세 확대 비율이 유효하지 않습니다: {prepared_pose.resize_scale}"
        )

    mask_l = approved_change_mask.convert("L")
    try:
        alpha_array = np.asarray(mask_l, dtype=np.uint8)
    finally:
        mask_l.close()
    binary_mask = alpha_array >= resolved_settings.mask_alpha_threshold
    approved_mask_foreground_pixels = int(np.count_nonzero(binary_mask))
    if approved_mask_foreground_pixels == 0:
        raise CharacterTargetLandmarkError(
            "승인 변경 마스크의 임계값 이상 픽셀이 0개입니다: "
            f"임계값={resolved_settings.mask_alpha_threshold}"
        )
    bbox = cv2.boundingRect(binary_mask.astype(np.uint8))

    joints_by_name = _validate_and_index_joints(
        joint_coordinates=joint_coordinates,
        prepared_pose=prepared_pose,
        minimum_confidence=resolved_settings.minimum_joint_confidence,
        required_names=_required_joint_names(clothing_category),
    )
    mapped = {
        name: _map_joint_to_target(joint, prepared_pose)
        for name, joint in joints_by_name.items()
    }
    pose_pairs, row_sources = _create_pose_pairs(
        mapped=mapped,
        clothing_category=clothing_category,
        mask_bbox_xywh=bbox,
    )

    row_search_radius = min(
        resolved_settings.maximum_row_search_pixels,
        max(1, round(prepared_pose.target_height * resolved_settings.row_search_ratio)),
    )
    points: list[tuple[float, float]] = []
    selected_rows: list[int] = []
    overlap_scores: list[float] = []
    for left_pose, right_pose in pose_pairs:
        expected_left = min(left_pose[0], right_pose[0])
        expected_right = max(left_pose[0], right_pose[0])
        target_y = (left_pose[1] + right_pose[1]) / 2.0
        selected_y, selected_left, selected_right, overlap_score = (
            find_nearest_mask_span(
                binary_mask=binary_mask,
                target_y=target_y,
                expected_left=expected_left,
                expected_right=expected_right,
                search_radius=row_search_radius,
            )
        )
        points.extend(
            (
                (float(selected_left), float(selected_y)),
                (float(selected_right), float(selected_y)),
            )
        )
        selected_rows.append(selected_y)
        overlap_scores.append(overlap_score)

    points_array = np.asarray(points, dtype=np.float32)
    _validate_target_points(points_array, binary_mask)
    used_joints = tuple(joints_by_name[name] for name in joints_by_name)
    confidences = [joint.confidence_score for joint in used_joints]
    return CharacterTargetLandmarkResult(
        points_xy=points_array,
        roles=LANDMARK_ROLES,
        row_sources=row_sources,
        selected_rows_y=(
            selected_rows[0],
            selected_rows[1],
            selected_rows[2],
        ),
        target_canvas_size=target_size,
        approved_mask_foreground_pixels=approved_mask_foreground_pixels,
        approved_mask_bbox_xywh=bbox,
        required_joint_names=tuple(joints_by_name),
        minimum_used_joint_confidence=float(min(confidences)),
        mean_used_joint_confidence=float(np.mean(confidences)),
        row_search_radius_pixels=row_search_radius,
        horizontal_overlap_score=float(np.mean(overlap_scores)),
    )


def _required_joint_names(
    clothing_category: ClothingCategory,
) -> tuple[str, ...]:
    if clothing_category == ClothingCategory.BOTTOM:
        return (
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        )
    return (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )


def _validate_and_index_joints(
    joint_coordinates: tuple[PoseJointCoordinateCandidate, ...],
    prepared_pose: PoseControlPreparedInput,
    minimum_confidence: float,
    required_names: tuple[str, ...],
) -> dict[str, PoseJointCoordinateCandidate]:
    indexed: dict[str, PoseJointCoordinateCandidate] = {}
    duplicate_names: set[str] = set()
    for joint in joint_coordinates:
        if joint.joint_name in indexed:
            duplicate_names.add(joint.joint_name)
        indexed[joint.joint_name] = joint
    if duplicate_names:
        raise CharacterTargetLandmarkError(
            "DWPose 관절 이름이 중복되었습니다: "
            + ", ".join(sorted(duplicate_names))
        )

    validated: dict[str, PoseJointCoordinateCandidate] = {}
    for name in required_names:
        joint = indexed.get(name)
        if joint is None:
            raise CharacterTargetLandmarkError(f"필수 DWPose 관절이 없습니다: {name}")
        if not joint.detected or joint.confidence_score < minimum_confidence:
            raise CharacterTargetLandmarkError(
                "필수 DWPose 관절 신뢰도가 기준 미만입니다: "
                f"{name}={joint.confidence_score:.4f}, "
                f"기준={minimum_confidence:.4f}"
            )
        if (
            not math.isfinite(joint.x)
            or not math.isfinite(joint.y)
            or not math.isfinite(joint.confidence_score)
        ):
            raise CharacterTargetLandmarkError(
                f"필수 DWPose 관절에 NaN 또는 무한대가 있습니다: {name}"
            )
        if not (
            0.0 <= joint.x <= prepared_pose.source_width - 1
            and 0.0 <= joint.y <= prepared_pose.source_height - 1
        ):
            raise CharacterTargetLandmarkError(
                "필수 DWPose 관절이 원본 자세 캔버스 밖에 있습니다: "
                f"{name}=({joint.x:.2f}, {joint.y:.2f}), "
                f"캔버스={prepared_pose.source_width}×{prepared_pose.source_height}"
            )
        validated[name] = joint
    return validated


def _map_joint_to_target(
    joint: PoseJointCoordinateCandidate,
    prepared_pose: PoseControlPreparedInput,
) -> tuple[float, float]:
    return (
        joint.x * prepared_pose.resize_scale + prepared_pose.padding_left,
        joint.y * prepared_pose.resize_scale + prepared_pose.padding_top,
    )


def _create_pose_pairs(
    mapped: dict[str, tuple[float, float]],
    clothing_category: ClothingCategory,
    mask_bbox_xywh: tuple[int, int, int, int],
) -> tuple[
    tuple[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ],
    tuple[str, str, str],
]:
    if clothing_category == ClothingCategory.BOTTOM:
        return (
            (
                (mapped["left_hip"], mapped["right_hip"]),
                (mapped["left_knee"], mapped["right_knee"]),
                (mapped["left_ankle"], mapped["right_ankle"]),
            ),
            ("dwpose_hips", "dwpose_knees", "dwpose_ankles"),
        )

    shoulders = (mapped["left_shoulder"], mapped["right_shoulder"])
    hips = (mapped["left_hip"], mapped["right_hip"])
    if clothing_category == ClothingCategory.TOP:
        torso_middle = (
            _midpoint(shoulders[0], hips[0]),
            _midpoint(shoulders[1], hips[1]),
        )
        return (
            (shoulders, torso_middle, hips),
            ("dwpose_shoulders", "dwpose_torso_midpoint", "dwpose_hips"),
        )

    left, top, width, height = mask_bbox_xywh
    del left, width
    lower_y = float(top + round((height - 1) * 0.90))
    lower_pair = (
        (hips[0][0], lower_y),
        (hips[1][0], lower_y),
    )
    return (
        (shoulders, hips, lower_pair),
        ("dwpose_shoulders", "dwpose_hips", "approved_mask_lower_90"),
    )


def _midpoint(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def find_nearest_mask_span(
    binary_mask: np.ndarray,
    target_y: float,
    expected_left: float,
    expected_right: float,
    search_radius: int,
) -> tuple[int, int, int, float]:
    """목표 Y·X구간에 가장 가까운 승인 마스크 연속 구간을 반환한다."""
    height, width = binary_mask.shape
    center_x = (expected_left + expected_right) / 2.0
    target_row = min(height - 1, max(0, round(target_y)))
    for offset in range(search_radius + 1):
        rows = (target_row,) if offset == 0 else (target_row - offset, target_row + offset)
        for row_y in rows:
            if not 0 <= row_y < height:
                continue
            runs = _foreground_runs(binary_mask[row_y])
            if not runs:
                continue
            overlapping = [
                run
                for run in runs
                if run[1] >= expected_left and run[0] <= expected_right
            ]
            if overlapping:
                selected_left = min(run[0] for run in overlapping)
                selected_right = max(run[1] for run in overlapping)
            else:
                selected_left, selected_right = min(
                    runs,
                    key=lambda run: _distance_to_interval(center_x, run),
                )
            if selected_left >= selected_right:
                continue
            intersection = max(
                0.0,
                min(float(selected_right), expected_right)
                - max(float(selected_left), expected_left)
                + 1.0,
            )
            union = max(float(selected_right), expected_right) - min(
                float(selected_left), expected_left
            ) + 1.0
            return (
                row_y,
                selected_left,
                selected_right,
                intersection / union,
            )
    raise CharacterTargetLandmarkError(
        "DWPose 목표 행 주변 승인 마스크에서 좌우 경계를 찾지 못했습니다: "
        f"목표Y={target_row}px, 탐색반경={search_radius}px, "
        f"캔버스={width}×{height}"
    )


def _foreground_runs(row: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(row)
    if indices.size == 0:
        return []
    split_positions = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split_positions)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _distance_to_interval(value: float, interval: tuple[int, int]) -> float:
    if interval[0] <= value <= interval[1]:
        return 0.0
    return min(abs(value - interval[0]), abs(value - interval[1]))


def _validate_target_points(
    points_xy: np.ndarray,
    binary_mask: np.ndarray,
) -> None:
    if points_xy.shape != (6, 2):
        raise CharacterTargetLandmarkError(
            f"캐릭터 목표점은 6×2여야 합니다: shape={points_xy.shape}"
        )
    if np.unique(points_xy, axis=0).shape[0] != 6:
        raise CharacterTargetLandmarkError("캐릭터 목표점 6개 중 중복 좌표가 있습니다.")
    hull_area = float(cv2.contourArea(cv2.convexHull(points_xy)))
    if hull_area <= 0.0:
        raise CharacterTargetLandmarkError(
            "캐릭터 목표점의 볼록 껍질 면적이 0이라 TPS에 사용할 수 없습니다."
        )
    for x_value, y_value in points_xy:
        x = int(round(float(x_value)))
        y = int(round(float(y_value)))
        if not binary_mask[y, x]:
            raise CharacterTargetLandmarkError(
                f"캐릭터 목표점이 승인 마스크 밖에 있습니다: ({x}, {y})"
            )


def _validate_settings(settings: CharacterTargetLandmarkSettings) -> None:
    if (
        not math.isfinite(settings.minimum_joint_confidence)
        or not 0.0 <= settings.minimum_joint_confidence <= 1.0
    ):
        raise CharacterTargetLandmarkError(
            "최소 관절 신뢰도는 0.0~1.0 범위여야 합니다: "
            f"{settings.minimum_joint_confidence}"
        )
    if not 1 <= settings.mask_alpha_threshold <= 255:
        raise CharacterTargetLandmarkError(
            "마스크 알파 임계값은 1~255 범위여야 합니다: "
            f"{settings.mask_alpha_threshold}"
        )
    if not 0.0 < settings.row_search_ratio <= 0.50:
        raise CharacterTargetLandmarkError(
            "행 탐색 비율은 0 초과 0.50 이하여야 합니다: "
            f"{settings.row_search_ratio}"
        )
    if settings.maximum_row_search_pixels < 1:
        raise CharacterTargetLandmarkError(
            "최대 행 탐색 반경은 1px 이상이어야 합니다: "
            f"{settings.maximum_row_search_pixels}px"
        )
