"""승인 자세 이미지의 DWPose 실행과 사용자 승인 데이터를 담당한다."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, UnidentifiedImageError

from genai_lab.pose_reference import PoseReferenceApprovedInput


EXPECTED_BODY_JOINT_COUNT = 18


@dataclass(frozen=True)
class PoseReferenceEstimationSettings:
    """DWPose 별도 실행 경로와 수치 제한."""

    python_executable: Path
    runner_path: Path
    temporary_root: Path
    cache_dir: Path
    timeout_seconds: int = 600
    pose_device: str = "cpu"
    minimum_pose_confidence: float = 0.30


@dataclass(frozen=True)
class PoseJointCoordinateCandidate:
    """DWPose가 추정했지만 사용자가 아직 승인하지 않은 관절 좌표."""

    joint_name: str
    x: float
    y: float
    confidence_score: float
    detected: bool
    model_estimated: bool = True


@dataclass(frozen=True)
class PoseEstimationReviewCandidate:
    """DWPose 실행이 끝났지만 사용자가 아직 승인하지 않은 자세 결과."""

    source_image: Image.Image
    overlay_image: Image.Image
    control_map_image: Image.Image
    joint_coordinates: tuple[PoseJointCoordinateCandidate, ...]
    detected_joint_count: int
    missing_joint_count: int
    minimum_pose_confidence: float
    model_ids: tuple[str, ...]
    elapsed_seconds: float

    def close(self) -> None:
        """자세 검토용 이미지 3개를 메모리에서 해제한다."""
        self.source_image.close()
        self.overlay_image.close()
        self.control_map_image.close()


@dataclass(frozen=True)
class PoseEstimationApprovedInput:
    """사용자가 ControlNet 전달 대상으로 승인한 뼈대 지도와 좌표."""

    control_map_image: Image.Image
    joint_coordinates: tuple[PoseJointCoordinateCandidate, ...]
    detected_joint_count: int
    missing_joint_count: int
    minimum_pose_confidence: float
    model_ids: tuple[str, ...]

    def close(self) -> None:
        """승인 뼈대 지도의 메모리 복사본을 해제한다."""
        self.control_map_image.close()


@dataclass(frozen=True)
class PoseControlPreparedInput:
    """생성 해상도에 맞춰 검사를 끝낸 ControlNet 자세 입력."""

    control_map_image: Image.Image
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resize_scale: float
    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int
    non_black_pixel_count: int

    def close(self) -> None:
        """생성용 뼈대 지도 메모리를 해제한다."""
        self.control_map_image.close()


class PoseReferenceEstimationError(RuntimeError):
    """DWPose 실행 또는 반환 계약을 확인하지 못한 오류."""


def execute_pose_reference_estimation(
    approved_pose_reference: PoseReferenceApprovedInput,
    settings: PoseReferenceEstimationSettings,
) -> PoseEstimationReviewCandidate:
    """승인 자세에 DWPose를 1회 실행하고 검토 후보를 반환한다.

    부수 효과:
        임시 파일만 만들며 ControlNet과 이미지 생성은 호출하지 않는다.
    """
    validate_pose_reference_estimation_settings(settings)
    settings.temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="genai-lab-pose-",
        dir=settings.temporary_root,
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        source_path = temporary_directory / "pose_source.png"
        overlay_path = temporary_directory / "pose_overlay.png"
        control_map_path = temporary_directory / "pose_control_map.png"
        result_json_path = temporary_directory / "pose_result.json"
        approved_pose_reference.image.save(source_path)
        command = [
            str(settings.python_executable),
            str(settings.runner_path),
            "--pose-image", str(source_path),
            "--output-overlay", str(overlay_path),
            "--output-control-map", str(control_map_path),
            "--output-json", str(result_json_path),
            "--cache-dir", str(settings.cache_dir),
            "--pose-device", settings.pose_device,
            "--minimum-pose-confidence",
            str(settings.minimum_pose_confidence),
        ]
        try:
            completed_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PoseReferenceEstimationError(
                f"DWPose 자세 분석을 시작하지 못했습니다: {error}"
            ) from error
        if completed_process.returncode != 0:
            execution_details = (
                completed_process.stderr.strip()
                or completed_process.stdout.strip()
                or "출력 없음"
            )
            raise PoseReferenceEstimationError(
                "DWPose 자세 분석에 실패했습니다. "
                f"별도 실행 출력: {execution_details}"
            )
        missing_outputs = tuple(
            path.name
            for path in (overlay_path, control_map_path, result_json_path)
            if not path.is_file()
        )
        if missing_outputs:
            raise PoseReferenceEstimationError(
                "DWPose 실행은 끝났지만 출력 파일이 없습니다: "
                f"{missing_outputs}"
            )
        try:
            with Image.open(overlay_path) as opened_image:
                overlay_image = opened_image.convert("RGB")
            with Image.open(control_map_path) as opened_image:
                control_map_image = opened_image.convert("RGB")
            result_payload = json.loads(
                result_json_path.read_text(encoding="utf-8")
            )
        except (OSError, UnidentifiedImageError, json.JSONDecodeError) as error:
            raise PoseReferenceEstimationError(
                f"DWPose 자세 결과를 읽을 수 없습니다: {error}"
            ) from error

    expected_size = approved_pose_reference.image.size
    if overlay_image.size != expected_size or control_map_image.size != expected_size:
        overlay_image.close()
        control_map_image.close()
        raise PoseReferenceEstimationError(
            "DWPose 미리보기와 뼈대 지도가 승인 자세 크기와 다릅니다."
        )
    try:
        joint_coordinates = tuple(
            PoseJointCoordinateCandidate(
                joint_name=str(payload["joint_name"]),
                x=float(payload["x"]),
                y=float(payload["y"]),
                confidence_score=float(payload["confidence_score"]),
                detected=bool(payload["detected"]),
            )
            for payload in result_payload["joint_coordinates"]
        )
        if len(joint_coordinates) != EXPECTED_BODY_JOINT_COUNT:
            raise ValueError(
                f"관절 수={len(joint_coordinates)}, "
                f"기대={EXPECTED_BODY_JOINT_COUNT}"
            )
        if any(
            not 0.0 <= coordinate.confidence_score <= 1.0
            for coordinate in joint_coordinates
        ):
            raise ValueError("관절 신뢰도가 0.0~1.0 범위를 벗어났습니다.")
        detected_joint_count = sum(
            1 for coordinate in joint_coordinates if coordinate.detected
        )
        missing_joint_count = EXPECTED_BODY_JOINT_COUNT - detected_joint_count
        if detected_joint_count != int(result_payload["detected_joint_count"]):
            raise ValueError("탐지 관절 수 기록이 좌표와 다릅니다.")
        if missing_joint_count != int(result_payload["missing_joint_count"]):
            raise ValueError("누락 관절 수 기록이 좌표와 다릅니다.")
        return PoseEstimationReviewCandidate(
            source_image=approved_pose_reference.image.copy(),
            overlay_image=overlay_image,
            control_map_image=control_map_image,
            joint_coordinates=joint_coordinates,
            detected_joint_count=detected_joint_count,
            missing_joint_count=missing_joint_count,
            minimum_pose_confidence=float(
                result_payload["minimum_pose_confidence"]
            ),
            model_ids=tuple(str(value) for value in result_payload["model_ids"]),
            elapsed_seconds=float(result_payload["elapsed_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        overlay_image.close()
        control_map_image.close()
        raise PoseReferenceEstimationError(
            f"DWPose 관절 결과 형식이 올바르지 않습니다: {error}"
        ) from error


def approve_pose_estimation_candidate(
    review_candidate: PoseEstimationReviewCandidate,
) -> PoseEstimationApprovedInput:
    """사용자가 확인한 뼈대 지도와 관절 좌표를 독립 복사한다."""
    return PoseEstimationApprovedInput(
        control_map_image=review_candidate.control_map_image.copy(),
        joint_coordinates=review_candidate.joint_coordinates,
        detected_joint_count=review_candidate.detected_joint_count,
        missing_joint_count=review_candidate.missing_joint_count,
        minimum_pose_confidence=review_candidate.minimum_pose_confidence,
        model_ids=review_candidate.model_ids,
    )


def prepare_pose_control_input(
    approved_pose_estimation: PoseEstimationApprovedInput,
    target_width: int,
    target_height: int,
) -> PoseControlPreparedInput:
    """승인 뼈대 지도를 자르지 않고 검은 여백으로 생성 크기에 맞춘다.

    반환값:
        ControlNet에 전달할 RGB 지도와 확대 비율·여백·유효 픽셀 수.

    오류:
        승인 관절 자료가 불완전하거나 뼈대 지도가 비어 있으면 중단한다.
    """
    if target_width < 256 or target_height < 256:
        raise PoseReferenceEstimationError(
            "자세 제어 목표 크기는 가로와 세로가 각각 256px 이상이어야 합니다."
        )
    if target_width % 8 != 0 or target_height % 8 != 0:
        raise PoseReferenceEstimationError(
            "자세 제어 목표 크기는 모델 처리 단위인 8의 배수여야 합니다."
        )
    if (
        approved_pose_estimation.detected_joint_count
        + approved_pose_estimation.missing_joint_count
        != EXPECTED_BODY_JOINT_COUNT
    ):
        raise PoseReferenceEstimationError(
            "승인 자세의 탐지 관절 수와 누락 관절 수 합계가 18개가 아닙니다."
        )
    if approved_pose_estimation.detected_joint_count < 1:
        raise PoseReferenceEstimationError(
            "승인 자세에서 탐지된 관절이 0개라 ControlNet을 실행할 수 없습니다."
        )

    source_map = approved_pose_estimation.control_map_image.convert("RGB")
    source_width, source_height = source_map.size
    if source_width < 1 or source_height < 1:
        source_map.close()
        raise PoseReferenceEstimationError("승인 자세 지도의 크기가 비어 있습니다.")

    resize_scale = min(
        target_width / source_width,
        target_height / source_height,
    )
    resized_width = max(1, round(source_width * resize_scale))
    resized_height = max(1, round(source_height * resize_scale))
    resized_map = source_map.resize(
        (resized_width, resized_height),
        Image.Resampling.NEAREST,
    )
    source_map.close()

    padding_left = (target_width - resized_width) // 2
    padding_top = (target_height - resized_height) // 2
    padding_right = target_width - resized_width - padding_left
    padding_bottom = target_height - resized_height - padding_top
    prepared_map = Image.new("RGB", (target_width, target_height), "black")
    prepared_map.paste(resized_map, (padding_left, padding_top))
    resized_map.close()

    grayscale_map = prepared_map.convert("L")
    try:
        histogram = grayscale_map.histogram()
        non_black_pixel_count = target_width * target_height - histogram[0]
    finally:
        grayscale_map.close()
    if non_black_pixel_count < 1:
        prepared_map.close()
        raise PoseReferenceEstimationError(
            "승인 자세 지도에 검은 배경 외의 뼈대 픽셀이 0개입니다."
        )

    return PoseControlPreparedInput(
        control_map_image=prepared_map,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        resize_scale=resize_scale,
        padding_left=padding_left,
        padding_top=padding_top,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
        non_black_pixel_count=non_black_pixel_count,
    )


def validate_pose_reference_estimation_settings(
    settings: PoseReferenceEstimationSettings,
) -> None:
    """DWPose 실행 전 경로 4개와 수치 3개를 검사한다."""
    for path_name, required_path in (
        ("DWPose 전용 Python", settings.python_executable),
        ("DWPose 실행 파일", settings.runner_path),
    ):
        if not required_path.is_file():
            raise PoseReferenceEstimationError(
                f"{path_name}을 찾을 수 없습니다: {required_path}"
            )
    if settings.timeout_seconds < 60:
        raise PoseReferenceEstimationError(
            "DWPose 제한 시간은 60초 이상이어야 합니다."
        )
    if settings.pose_device != "cpu":
        raise PoseReferenceEstimationError(
            "현재 DWPose 자세 참조 추출은 CPU만 허용합니다."
        )
    if not 0.0 <= settings.minimum_pose_confidence <= 1.0:
        raise PoseReferenceEstimationError(
            "관절 좌표 기준 점수는 0.0~1.0이어야 합니다."
        )
