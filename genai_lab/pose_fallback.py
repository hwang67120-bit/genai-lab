"""사용자 승인 자세를 저장하고 DWPose 실패 때 검증된 폴백으로 제공한다."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Protocol

from PIL import Image, UnidentifiedImageError

from genai_lab.image_digest import calculate_image_pixel_sha256
from genai_lab.pose_estimation import (
    EXPECTED_BODY_JOINT_COUNT,
    PoseEstimationApprovedInput,
    PoseJointCoordinateCandidate,
)


POSE_FALLBACK_SCHEMA_VERSION = 1
_SAFE_POSE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REQUIRED_JOINT_GROUPS = {
    "어깨": frozenset(("right_shoulder", "left_shoulder")),
    "골반": frozenset(("right_hip", "left_hip")),
    "무릎": frozenset(("right_knee", "left_knee")),
    "발목": frozenset(("right_ankle", "left_ankle")),
}
_EXPECTED_BODY_JOINT_NAMES = frozenset((
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
))


class PoseWithQualityFields(Protocol):
    """품질 판정에 필요한 승인 전·후 자세의 공통 필드."""

    control_map_image: Image.Image
    joint_coordinates: tuple[PoseJointCoordinateCandidate, ...]
    detected_joint_count: int
    missing_joint_count: int
    minimum_pose_confidence: float


@dataclass(frozen=True)
class PoseFallbackSettings:
    """저장 자세 폴백 경로와 초기 수치 정책."""

    library_root: Path
    enabled: bool = True
    default_pose_id: str = "last-approved"
    minimum_detected_joint_count: int = 8
    require_shoulder: bool = True
    require_hip: bool = True
    require_knee: bool = True
    require_ankle: bool = True
    require_user_approval: bool = True


@dataclass(frozen=True)
class PoseQualityDecision:
    """자세가 ControlNet 입력으로 충분한지 판정한 수치 결과."""

    accepted: bool
    rejection_reasons: tuple[str, ...]
    detected_joint_count: int
    required_group_pass_count: int
    required_group_count: int
    non_black_pixel_count: int


@dataclass
class SavedApprovedPose:
    """디스크 검증을 마치고 메모리 소유권을 가진 저장 자세."""

    pose_id: str
    source_preview_image: Image.Image
    approved_pose: PoseEstimationApprovedInput
    control_map_sha256: str
    saved_at: str

    def copy_approved_pose(self) -> PoseEstimationApprovedInput:
        """GUI가 독립적으로 소유할 승인 자세 복사본을 만든다."""
        return PoseEstimationApprovedInput(
            control_map_image=self.approved_pose.control_map_image.copy(),
            joint_coordinates=self.approved_pose.joint_coordinates,
            detected_joint_count=self.approved_pose.detected_joint_count,
            missing_joint_count=self.approved_pose.missing_joint_count,
            minimum_pose_confidence=self.approved_pose.minimum_pose_confidence,
            model_ids=self.approved_pose.model_ids,
        )

    def close(self) -> None:
        self.source_preview_image.close()
        self.approved_pose.close()


class PoseFallbackError(RuntimeError):
    """저장 자세가 없거나 데이터 계약·해시 검증을 통과하지 못한 오류."""


def evaluate_pose_quality(
    pose: PoseWithQualityFields,
    settings: PoseFallbackSettings,
) -> PoseQualityDecision:
    """관절 수·필수 그룹·뼈대 픽셀을 기준으로 자세를 판정한다."""
    _validate_settings(settings)
    reasons: list[str] = []
    coordinates = pose.joint_coordinates
    if len(coordinates) != EXPECTED_BODY_JOINT_COUNT:
        reasons.append(
            f"관절 좌표={len(coordinates)}/{EXPECTED_BODY_JOINT_COUNT}개"
        )
    detected_names = {
        coordinate.joint_name
        for coordinate in coordinates
        if coordinate.detected
    }
    coordinate_names = tuple(
        coordinate.joint_name for coordinate in coordinates
    )
    if frozenset(coordinate_names) != _EXPECTED_BODY_JOINT_NAMES:
        reasons.append("관절 이름 집합이 표준 몸 관절 18개와 다름")
    if len(set(coordinate_names)) != len(coordinate_names):
        reasons.append("중복 관절 이름이 1개 이상 존재")
    invalid_confidence_count = sum(
        1
        for coordinate in coordinates
        if not math.isfinite(coordinate.confidence_score)
        or not 0.0 <= coordinate.confidence_score <= 1.0
    )
    if invalid_confidence_count:
        reasons.append(
            f"0.0~1.0 밖 관절 신뢰도={invalid_confidence_count}개"
        )
    detection_flag_mismatch_count = sum(
        1
        for coordinate in coordinates
        if coordinate.detected
        != (
            coordinate.confidence_score
            >= pose.minimum_pose_confidence
        )
    )
    if detection_flag_mismatch_count:
        reasons.append(
            "신뢰도와 탐지 플래그 불일치="
            f"{detection_flag_mismatch_count}개"
        )
    detected_count = len(detected_names)
    if detected_count != pose.detected_joint_count:
        reasons.append(
            "탐지 관절 기록 불일치="
            f"좌표 {detected_count}개/기록 {pose.detected_joint_count}개"
        )
    if pose.detected_joint_count + pose.missing_joint_count != EXPECTED_BODY_JOINT_COUNT:
        reasons.append(
            "탐지+누락 관절 합계="
            f"{pose.detected_joint_count + pose.missing_joint_count}/"
            f"{EXPECTED_BODY_JOINT_COUNT}개"
        )
    if pose.detected_joint_count < settings.minimum_detected_joint_count:
        reasons.append(
            f"탐지 관절={pose.detected_joint_count}/18개, "
            f"최소={settings.minimum_detected_joint_count}/18개"
        )

    required_flags = {
        "어깨": settings.require_shoulder,
        "골반": settings.require_hip,
        "무릎": settings.require_knee,
        "발목": settings.require_ankle,
    }
    required_groups = tuple(
        name for name, required in required_flags.items() if required
    )
    passed_groups = tuple(
        name
        for name in required_groups
        if detected_names & _REQUIRED_JOINT_GROUPS[name]
    )
    missing_groups = tuple(
        name for name in required_groups if name not in passed_groups
    )
    if missing_groups:
        reasons.append("필수 관절 그룹 누락=" + ", ".join(missing_groups))

    grayscale_map = pose.control_map_image.convert("L")
    try:
        histogram = grayscale_map.histogram()
        non_black_pixel_count = (
            grayscale_map.width * grayscale_map.height - histogram[0]
        )
    finally:
        grayscale_map.close()
    if non_black_pixel_count < 1:
        reasons.append("ControlNet 뼈대 유효 픽셀=0px")

    return PoseQualityDecision(
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        detected_joint_count=pose.detected_joint_count,
        required_group_pass_count=len(passed_groups),
        required_group_count=len(required_groups),
        non_black_pixel_count=non_black_pixel_count,
    )


def save_default_approved_pose(
    approved_pose: PoseEstimationApprovedInput,
    source_preview_image: Image.Image,
    settings: PoseFallbackSettings,
) -> str:
    """승인 자세 PNG 2개와 JSON 1개를 마지막 승인 슬롯에 저장한다."""
    quality = evaluate_pose_quality(approved_pose, settings)
    if not quality.accepted:
        raise PoseFallbackError(
            "품질 기준을 통과하지 못한 자세는 폴백으로 저장할 수 없습니다: "
            + "; ".join(quality.rejection_reasons)
        )
    if source_preview_image.size != approved_pose.control_map_image.size:
        raise PoseFallbackError(
            "자세 원본과 뼈대 지도의 크기가 다릅니다: "
            f"원본={source_preview_image.size}, "
            f"뼈대={approved_pose.control_map_image.size}"
        )

    pose_directory = _pose_directory(settings)
    pose_directory.mkdir(parents=True, exist_ok=True)
    control_map_path = pose_directory / "control_map.png"
    source_preview_path = pose_directory / "source_preview.png"
    metadata_path = pose_directory / "metadata.json"
    control_map_temporary_path = pose_directory / "control_map.png.tmp"
    source_temporary_path = pose_directory / "source_preview.png.tmp"
    metadata_temporary_path = pose_directory / "metadata.json.tmp"

    control_map_sha256 = calculate_image_pixel_sha256(
        approved_pose.control_map_image,
        "RGB",
    )
    saved_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": POSE_FALLBACK_SCHEMA_VERSION,
        "pose_id": settings.default_pose_id,
        "saved_at": saved_at,
        "width": approved_pose.control_map_image.width,
        "height": approved_pose.control_map_image.height,
        "control_map_sha256": control_map_sha256,
        "detected_joint_count": approved_pose.detected_joint_count,
        "missing_joint_count": approved_pose.missing_joint_count,
        "minimum_pose_confidence": approved_pose.minimum_pose_confidence,
        "model_ids": list(approved_pose.model_ids),
        "joint_coordinates": [
            asdict(coordinate) for coordinate in approved_pose.joint_coordinates
        ],
        "quality": {
            "minimum_detected_joint_count": (
                settings.minimum_detected_joint_count
            ),
            "required_group_pass_count": quality.required_group_pass_count,
            "required_group_count": quality.required_group_count,
            "non_black_pixel_count": quality.non_black_pixel_count,
        },
    }
    converted_control_map = approved_pose.control_map_image.convert("RGB")
    converted_source_preview = source_preview_image.convert("RGB")
    try:
        converted_control_map.save(control_map_temporary_path, format="PNG")
        converted_source_preview.save(source_temporary_path, format="PNG")
        metadata_temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(control_map_temporary_path, control_map_path)
        os.replace(source_temporary_path, source_preview_path)
        os.replace(metadata_temporary_path, metadata_path)
    except OSError as error:
        raise PoseFallbackError(
            f"승인 자세 저장에 실패했습니다: {error}"
        ) from error
    finally:
        converted_control_map.close()
        converted_source_preview.close()
        for temporary_path in (
            control_map_temporary_path,
            source_temporary_path,
            metadata_temporary_path,
        ):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return control_map_sha256


def load_default_approved_pose(
    settings: PoseFallbackSettings,
) -> SavedApprovedPose:
    """마지막 승인 자세를 읽고 좌표·크기·SHA-256을 모두 검증한다."""
    _validate_settings(settings)
    pose_directory = _pose_directory(settings)
    metadata_path = pose_directory / "metadata.json"
    control_map_path = pose_directory / "control_map.png"
    source_preview_path = pose_directory / "source_preview.png"
    missing_paths = tuple(
        path.name
        for path in (metadata_path, control_map_path, source_preview_path)
        if not path.is_file()
    )
    if missing_paths:
        raise PoseFallbackError(
            "저장된 기본 자세 파일이 없습니다: " + ", ".join(missing_paths)
        )
    control_map_image = None
    source_preview_image = None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata["schema_version"]) != POSE_FALLBACK_SCHEMA_VERSION:
            raise ValueError("지원하지 않는 schema_version")
        if str(metadata["pose_id"]) != settings.default_pose_id:
            raise ValueError("pose_id 불일치")
        coordinates = tuple(
            PoseJointCoordinateCandidate(
                joint_name=str(payload["joint_name"]),
                x=float(payload["x"]),
                y=float(payload["y"]),
                confidence_score=float(payload["confidence_score"]),
                detected=bool(payload["detected"]),
                model_estimated=bool(payload.get("model_estimated", True)),
            )
            for payload in metadata["joint_coordinates"]
        )
        with Image.open(control_map_path) as opened_control_map:
            control_map_image = opened_control_map.convert("RGB")
        with Image.open(source_preview_path) as opened_source_preview:
            source_preview_image = opened_source_preview.convert("RGB")
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        UnidentifiedImageError,
    ) as error:
        if control_map_image is not None:
            control_map_image.close()
        if source_preview_image is not None:
            source_preview_image.close()
        raise PoseFallbackError(
            f"저장된 기본 자세를 읽을 수 없습니다: {error}"
        ) from error

    assert control_map_image is not None
    assert source_preview_image is not None
    approved_pose = None
    try:
        expected_size = (int(metadata["width"]), int(metadata["height"]))
        if control_map_image.size != expected_size:
            raise PoseFallbackError(
                "저장 뼈대 지도 크기가 메타데이터와 다릅니다: "
                f"이미지={control_map_image.size}, 기록={expected_size}"
            )
        if source_preview_image.size != expected_size:
            raise PoseFallbackError(
                "저장 자세 원본 크기가 메타데이터와 다릅니다: "
                f"이미지={source_preview_image.size}, 기록={expected_size}"
            )
        actual_sha256 = calculate_image_pixel_sha256(control_map_image, "RGB")
        expected_sha256 = str(metadata["control_map_sha256"])
        if actual_sha256 != expected_sha256:
            raise PoseFallbackError(
                "저장 뼈대 지도 SHA-256이 기록과 다릅니다."
            )
        approved_pose = PoseEstimationApprovedInput(
            control_map_image=control_map_image,
            joint_coordinates=coordinates,
            detected_joint_count=int(metadata["detected_joint_count"]),
            missing_joint_count=int(metadata["missing_joint_count"]),
            minimum_pose_confidence=float(metadata["minimum_pose_confidence"]),
            model_ids=tuple(str(value) for value in metadata["model_ids"]),
        )
        quality = evaluate_pose_quality(approved_pose, settings)
        if not quality.accepted:
            raise PoseFallbackError(
                "저장 자세가 현재 폴백 품질 기준을 통과하지 못했습니다: "
                + "; ".join(quality.rejection_reasons)
            )
        return SavedApprovedPose(
            pose_id=settings.default_pose_id,
            source_preview_image=source_preview_image,
            approved_pose=approved_pose,
            control_map_sha256=actual_sha256,
            saved_at=str(metadata["saved_at"]),
        )
    except Exception:
        source_preview_image.close()
        if approved_pose is not None:
            approved_pose.close()
        else:
            control_map_image.close()
        raise


def _pose_directory(settings: PoseFallbackSettings) -> Path:
    _validate_settings(settings)
    return settings.library_root / settings.default_pose_id


def _validate_settings(settings: PoseFallbackSettings) -> None:
    if not _SAFE_POSE_ID_PATTERN.fullmatch(settings.default_pose_id):
        raise PoseFallbackError(
            "기본 자세 ID는 영문·숫자로 시작하는 1~64자의 "
            "영문·숫자·점·밑줄·하이픈만 허용합니다."
        )
    if not 1 <= settings.minimum_detected_joint_count <= EXPECTED_BODY_JOINT_COUNT:
        raise PoseFallbackError("최소 탐지 관절 수는 1~18개여야 합니다.")
    if not settings.require_user_approval:
        raise PoseFallbackError(
            "저장 자세 폴백은 사용자 재승인 정책을 비활성화할 수 없습니다."
        )
