"""캐릭터 신체 분석 결과로 의상 변경 마스크를 안전하게 준비한다."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class CharacterBodyComparisonSettings:
    """SCHP·DensePose·DWPose 별도 실행과 마스크 보정 설정."""

    python_executable: Path
    repository_path: Path
    runner_path: Path
    temporary_root: Path
    cache_dir: Path
    width: int
    height: int
    timeout_seconds: int
    pose_device: str = "cpu"
    minimum_pose_confidence: float = 0.30
    mask_expansion_ratio: float = 0.01
    minimum_mask_expansion_pixels: int = 5
    maximum_mask_expansion_pixels: int = 15
    mask_closing_radius_pixels: int = 2


@dataclass(frozen=True)
class CharacterJointCoordinateCandidate:
    """DWPose가 추정했지만 사용자가 아직 확인하지 않은 관절 좌표."""

    joint_name: str
    x: float
    y: float
    confidence_score: float
    detected: bool
    model_estimated: bool = True


@dataclass(frozen=True)
class CharacterClothingMaskRefinement:
    """닫기·팽창·보호 영역 차감이 끝난 임시 의상 마스크."""

    raw_mask: Image.Image
    closed_mask: Image.Image
    expanded_mask: Image.Image
    safe_change_mask: Image.Image
    identity_protection_mask: Image.Image
    expansion_radius_pixels: int
    closing_radius_pixels: int
    attempted_protected_overlap_pixels: int
    safe_change_pixel_count: int
    safe_change_percent: float

    def close(self) -> None:
        """이 결과 객체가 소유한 PIL 이미지 5개를 해제한다."""
        self.raw_mask.close()
        self.closed_mask.close()
        self.expanded_mask.close()
        self.safe_change_mask.close()
        self.identity_protection_mask.close()


@dataclass(frozen=True)
class HumanAgnosticImageCandidate:
    """기존 의상 영역을 중립색으로 가린 사용자 승인 전 이미지."""

    neutralized_image: Image.Image
    neutral_rgb: tuple[int, int, int]
    neutralized_pixel_count: int
    neutralized_percent: float
    raw_mask_pixel_count: int
    raw_mask_coverage_percent: float
    changed_pixel_count_outside_mask: int

    def close(self) -> None:
        """후보가 소유한 중립화 이미지 1개를 해제한다."""
        self.neutralized_image.close()


@dataclass(frozen=True)
class OriginalClothingRemovalVerification:
    """탐지된 기존 의상이 변경 영역 밖에 남았는지 검사한 결과."""

    remaining_clothing_mask: Image.Image
    detected_clothing_pixel_count: int
    removed_clothing_pixel_count: int
    remaining_clothing_pixel_count: int
    removal_percent: float
    passed: bool
    reason_ko: str

    def close(self) -> None:
        """검토 화면에 사용하는 기존 의상 잔여 마스크를 해제한다."""
        self.remaining_clothing_mask.close()


@dataclass(frozen=True)
class CharacterBodyComparisonCandidate:
    """신체·관절·변경 영역 분석이 끝난 사용자 승인 전 후보."""

    source_image: Image.Image
    densepose_preview_image: Image.Image
    pose_preview_image: Image.Image
    mask_refinement: CharacterClothingMaskRefinement
    human_agnostic_candidate: HumanAgnosticImageCandidate
    clothing_removal_verification: OriginalClothingRemovalVerification
    joint_coordinates: tuple[CharacterJointCoordinateCandidate, ...]
    model_ids: tuple[str, ...]
    detected_joint_count: int
    missing_joint_count: int
    elapsed_seconds: float

    def close(self) -> None:
        """GUI 검토에 사용한 PIL 이미지를 모두 해제한다."""
        self.source_image.close()
        self.densepose_preview_image.close()
        self.pose_preview_image.close()
        self.mask_refinement.close()
        self.human_agnostic_candidate.close()
        self.clothing_removal_verification.close()


@dataclass(frozen=True)
class ConfirmedCharacterBodyComparison:
    """사용자가 확인한 신체 비교 수치와 정확한 중립화 입력."""

    clothing_type: str
    approved_human_agnostic_image: Image.Image
    approved_change_mask: Image.Image
    neutral_rgb: tuple[int, int, int]
    neutralized_pixel_count: int
    neutralized_percent: float
    raw_mask_coverage_percent: float
    remaining_clothing_pixel_count: int
    clothing_removal_percent: float
    changed_pixel_count_outside_mask: int
    expansion_radius_pixels: int
    closing_radius_pixels: int
    attempted_protected_overlap_pixels: int
    safe_change_pixel_count: int
    safe_change_percent: float
    detected_joint_count: int
    missing_joint_count: int
    model_ids: tuple[str, ...]

    def close(self) -> None:
        """승인된 중립화 이미지와 교체 마스크를 해제한다."""
        self.approved_human_agnostic_image.close()
        self.approved_change_mask.close()


class CharacterBodyComparisonError(RuntimeError):
    """신체·관절 분석이나 마스크 보정을 안전하게 완료하지 못한 오류."""


def calculate_mask_expansion_radius(
    image_size: tuple[int, int],
    expansion_ratio: float = 0.01,
    minimum_pixels: int = 5,
    maximum_pixels: int = 15,
) -> int:
    """긴 변의 비율값을 계산하고 지정된 픽셀 범위로 제한한다."""
    width, height = image_size
    if width < 1 or height < 1:
        raise CharacterBodyComparisonError(
            "마스크 팽창값을 계산할 이미지 크기는 1픽셀 이상이어야 합니다."
        )
    if not 0.0 < expansion_ratio <= 1.0:
        raise CharacterBodyComparisonError(
            "마스크 팽창 비율은 0보다 크고 1 이하여야 합니다."
        )
    if not 1 <= minimum_pixels <= maximum_pixels:
        raise CharacterBodyComparisonError(
            "마스크 최소 팽창값은 1픽셀 이상이며 최대값 이하여야 합니다."
        )

    calculated_pixels = round(max(width, height) * expansion_ratio)
    return max(minimum_pixels, min(maximum_pixels, calculated_pixels))


def refine_character_clothing_change_mask(
    raw_clothing_mask: Image.Image,
    identity_protection_mask: Image.Image,
    expansion_radius_pixels: int,
    closing_radius_pixels: int = 2,
) -> CharacterClothingMaskRefinement:
    """마스크 구멍을 닫고 외곽을 팽창한 뒤 신체 보호 영역을 뺀다.

    반환값:
        원본·닫기·팽창·안전 마스크와 픽셀 측정값.

    오류:
        마스크 크기·반경이 잘못됐거나 최종 변경 영역이 0픽셀이면 중단한다.
    """
    if raw_clothing_mask.size != identity_protection_mask.size:
        raise CharacterBodyComparisonError(
            "의상 마스크와 신체 보호 마스크의 크기가 다릅니다: "
            f"{raw_clothing_mask.size}, {identity_protection_mask.size}"
        )
    if not 5 <= expansion_radius_pixels <= 15:
        raise CharacterBodyComparisonError(
            "의상 마스크 팽창 반경은 5~15픽셀이어야 합니다."
        )
    if not 1 <= closing_radius_pixels <= 3:
        raise CharacterBodyComparisonError(
            "마스크 닫기 반경은 1~3픽셀이어야 합니다."
        )

    normalized_raw_mask = raw_clothing_mask.convert("L")
    normalized_protection_mask = identity_protection_mask.convert("L")
    try:
        raw_mask_array = np.asarray(normalized_raw_mask, dtype=np.uint8)
        protection_mask_array = np.asarray(
            normalized_protection_mask,
            dtype=np.uint8,
        )
    finally:
        normalized_raw_mask.close()
        normalized_protection_mask.close()

    binary_clothing_mask = np.where(
        raw_mask_array >= 128,
        255,
        0,
    ).astype(np.uint8)
    closing_kernel_size = closing_radius_pixels * 2 + 1
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size),
    )
    closed_mask_array = cv2.morphologyEx(
        binary_clothing_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1,
    )

    expansion_kernel_size = expansion_radius_pixels * 2 + 1
    expansion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (expansion_kernel_size, expansion_kernel_size),
    )
    expanded_mask_array = cv2.dilate(
        closed_mask_array,
        expansion_kernel,
        iterations=1,
    )
    protected_pixels = protection_mask_array >= 128
    attempted_protected_overlap_pixels = int(
        np.count_nonzero((expanded_mask_array > 0) & protected_pixels)
    )
    safe_change_mask_array = np.where(
        protected_pixels,
        0,
        expanded_mask_array,
    ).astype(np.uint8)
    safe_change_pixel_count = int(np.count_nonzero(safe_change_mask_array))
    if safe_change_pixel_count == 0:
        raise CharacterBodyComparisonError(
            "마스크를 다듬고 신체 보호 영역을 제외한 뒤 "
            "변경 가능한 의상 픽셀이 0개입니다."
        )
    total_pixel_count = raw_clothing_mask.width * raw_clothing_mask.height

    return CharacterClothingMaskRefinement(
        raw_mask=Image.fromarray(binary_clothing_mask, mode="L"),
        closed_mask=Image.fromarray(closed_mask_array, mode="L"),
        expanded_mask=Image.fromarray(expanded_mask_array, mode="L"),
        safe_change_mask=Image.fromarray(safe_change_mask_array, mode="L"),
        identity_protection_mask=Image.fromarray(
            np.where(protected_pixels, 255, 0).astype(np.uint8),
            mode="L",
        ),
        expansion_radius_pixels=expansion_radius_pixels,
        closing_radius_pixels=closing_radius_pixels,
        attempted_protected_overlap_pixels=attempted_protected_overlap_pixels,
        safe_change_pixel_count=safe_change_pixel_count,
        safe_change_percent=(
            safe_change_pixel_count / total_pixel_count * 100.0
        ),
    )


def create_human_agnostic_image_candidate(
    source_image: Image.Image,
    clothing_erasure_mask: Image.Image,
    raw_clothing_mask: Image.Image,
    neutral_rgb: tuple[int, int, int] = (127, 127, 127),
) -> HumanAgnosticImageCandidate:
    """승인 전 의상 영역을 RGB 중립색으로 바꾼 이미지를 만든다.

    반환값:
        중립화 이미지와 원본 마스크 포함률·영역 밖 변경 픽셀 수.

    오류:
        이미지 크기나 RGB 값이 잘못됐거나 중립화할 픽셀이 0개면 중단한다.
    """
    if (
        source_image.size != clothing_erasure_mask.size
        or source_image.size != raw_clothing_mask.size
    ):
        raise CharacterBodyComparisonError(
            "원본·의상 제거 마스크·원본 마스크의 크기가 다릅니다: "
            f"{source_image.size}, {clothing_erasure_mask.size}, "
            f"{raw_clothing_mask.size}"
        )
    if len(neutral_rgb) != 3 or any(
        not 0 <= channel_value <= 255 for channel_value in neutral_rgb
    ):
        raise CharacterBodyComparisonError(
            "중립화 RGB 값 3개는 각각 0~255여야 합니다."
        )

    normalized_source_image = source_image.convert("RGB")
    normalized_erasure_mask = clothing_erasure_mask.convert("L")
    normalized_raw_mask = raw_clothing_mask.convert("L")
    try:
        source_array = np.asarray(
            normalized_source_image, dtype=np.uint8
        ).copy()
        erasure_mask_array = (
            np.asarray(normalized_erasure_mask, dtype=np.uint8).copy() > 0
        )
        raw_mask_array = (
            np.asarray(normalized_raw_mask, dtype=np.uint8).copy() > 0
        )
    finally:
        normalized_source_image.close()
        normalized_erasure_mask.close()
        normalized_raw_mask.close()
    neutralized_pixel_count = int(np.count_nonzero(erasure_mask_array))
    if neutralized_pixel_count == 0:
        raise CharacterBodyComparisonError(
            "Inpainting Mask Neutralization 대상 픽셀이 0개입니다."
        )

    neutralized_array = source_array.copy()
    neutralized_array[erasure_mask_array] = np.asarray(
        neutral_rgb, dtype=np.uint8
    )
    changed_pixel_array = np.any(neutralized_array != source_array, axis=2)
    changed_pixel_count_outside_mask = int(np.count_nonzero(
        changed_pixel_array & ~erasure_mask_array
    ))
    if changed_pixel_count_outside_mask != 0:
        raise CharacterBodyComparisonError(
            "중립화 마스크 밖 픽셀이 "
            f"{changed_pixel_count_outside_mask}개 변경됐습니다."
        )

    raw_mask_pixel_count = int(np.count_nonzero(raw_mask_array))
    covered_raw_mask_pixel_count = int(np.count_nonzero(
        raw_mask_array & erasure_mask_array
    ))
    total_pixel_count = source_image.width * source_image.height
    raw_mask_coverage_percent = (
        covered_raw_mask_pixel_count / raw_mask_pixel_count * 100.0
        if raw_mask_pixel_count > 0
        else 0.0
    )
    return HumanAgnosticImageCandidate(
        neutralized_image=Image.fromarray(neutralized_array, mode="RGB"),
        neutral_rgb=neutral_rgb,
        neutralized_pixel_count=neutralized_pixel_count,
        neutralized_percent=neutralized_pixel_count / total_pixel_count * 100.0,
        raw_mask_pixel_count=raw_mask_pixel_count,
        raw_mask_coverage_percent=raw_mask_coverage_percent,
        changed_pixel_count_outside_mask=changed_pixel_count_outside_mask,
    )


def verify_original_clothing_removal(
    raw_clothing_mask: Image.Image,
    approved_change_mask: Image.Image,
) -> OriginalClothingRemovalVerification:
    """탐지된 기존 의상이 승인 변경 영역에 전부 포함됐는지 검사한다.

    반환값:
        기존 의상 탐지·포함·잔여 픽셀 수와 잔여 위치 마스크.

    오류:
        두 마스크 크기가 다르거나 기존 의상 탐지 픽셀이 0개면 중단한다.
    """
    if raw_clothing_mask.size != approved_change_mask.size:
        raise CharacterBodyComparisonError(
            "기존 의상 마스크와 승인 변경 마스크 크기가 다릅니다: "
            f"기존 의상={raw_clothing_mask.size}, "
            f"승인 영역={approved_change_mask.size}"
        )

    normalized_raw_mask = raw_clothing_mask.convert("L")
    normalized_approved_mask = approved_change_mask.convert("L")
    try:
        raw_mask_array = (
            np.asarray(normalized_raw_mask, dtype=np.uint8).copy() >= 128
        )
        approved_mask_array = (
            np.asarray(normalized_approved_mask, dtype=np.uint8).copy() >= 128
        )
    finally:
        normalized_raw_mask.close()
        normalized_approved_mask.close()

    detected_clothing_pixel_count = int(np.count_nonzero(raw_mask_array))
    if detected_clothing_pixel_count == 0:
        raise CharacterBodyComparisonError(
            "기존 의상으로 탐지된 픽셀이 0개입니다."
        )

    removed_clothing_pixels = raw_mask_array & approved_mask_array
    remaining_clothing_pixels = raw_mask_array & ~approved_mask_array
    removed_clothing_pixel_count = int(
        np.count_nonzero(removed_clothing_pixels)
    )
    remaining_clothing_pixel_count = int(
        np.count_nonzero(remaining_clothing_pixels)
    )
    removal_percent = (
        removed_clothing_pixel_count
        / detected_clothing_pixel_count
        * 100.0
    )
    passed = remaining_clothing_pixel_count == 0
    reason_ko = (
        "탐지된 기존 의상 전체가 변경 영역에 포함됐습니다."
        if passed
        else (
            "기존 의상 일부가 변경 영역 밖에 남았습니다: "
            f"{remaining_clothing_pixel_count:,}픽셀"
        )
    )

    return OriginalClothingRemovalVerification(
        remaining_clothing_mask=Image.fromarray(
            np.where(remaining_clothing_pixels, 255, 0).astype(np.uint8),
            mode="L",
        ),
        detected_clothing_pixel_count=detected_clothing_pixel_count,
        removed_clothing_pixel_count=removed_clothing_pixel_count,
        remaining_clothing_pixel_count=remaining_clothing_pixel_count,
        removal_percent=removal_percent,
        passed=passed,
        reason_ko=reason_ko,
    )


def execute_character_body_comparison(
    character_image: Image.Image,
    clothing_type: str,
    settings: CharacterBodyComparisonSettings,
) -> CharacterBodyComparisonCandidate:
    """별도 환경에서 SCHP·DensePose·DWPose를 실행하고 마스크를 보정한다.

    반환값:
        사용자 확인 전 원본·중간 마스크·관절 좌표와 측정 수치.

    부수 효과:
        로컬 임시 폴더에 중간 파일을 만들고 함수 종료 시 모두 제거한다.
    """
    validate_character_body_comparison_settings(settings)
    settings.temporary_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="genai-lab-body-comparison-",
        dir=settings.temporary_root,
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        person_path = temporary_directory / "person.png"
        raw_mask_path = temporary_directory / "raw_mask.png"
        protection_mask_path = temporary_directory / "protection_mask.png"
        densepose_path = temporary_directory / "densepose.png"
        pose_preview_path = temporary_directory / "pose.png"
        pose_json_path = temporary_directory / "pose.json"

        normalized_character_image = character_image.convert("RGB")
        try:
            normalized_character_image.save(person_path)
        finally:
            normalized_character_image.close()
        command = [
            str(settings.python_executable), str(settings.runner_path),
            "--repository-path", str(settings.repository_path),
            "--person-image", str(person_path),
            "--clothing-type", clothing_type,
            "--output-raw-mask", str(raw_mask_path),
            "--output-protection-mask", str(protection_mask_path),
            "--output-densepose", str(densepose_path),
            "--output-pose-preview", str(pose_preview_path),
            "--output-pose-json", str(pose_json_path),
            "--cache-dir", str(settings.cache_dir),
            "--width", str(settings.width),
            "--height", str(settings.height),
            "--pose-device", settings.pose_device,
            "--minimum-pose-confidence", str(settings.minimum_pose_confidence),
        ]
        try:
            completed_process = subprocess.run(
                command,
                cwd=settings.repository_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CharacterBodyComparisonError(
                f"캐릭터 신체 비교 별도 실행을 시작하지 못했습니다: {error}"
            ) from error

        if completed_process.returncode != 0:
            execution_details = (
                completed_process.stderr.strip()
                or completed_process.stdout.strip()
                or "출력 없음"
            )
            raise CharacterBodyComparisonError(
                "SCHP·DensePose·DWPose 신체 비교 실행에 실패했습니다. "
                f"별도 실행 출력: {execution_details}"
            )
        required_outputs = (
            raw_mask_path, protection_mask_path, densepose_path,
            pose_preview_path, pose_json_path,
        )
        missing_outputs = tuple(
            path.name for path in required_outputs if not path.is_file()
        )
        if missing_outputs:
            raise CharacterBodyComparisonError(
                "신체 비교 실행은 끝났지만 출력 파일이 없습니다: "
                f"{missing_outputs}"
            )

        try:
            with Image.open(raw_mask_path) as opened_image:
                raw_mask = opened_image.convert("L")
            with Image.open(protection_mask_path) as opened_image:
                protection_mask = opened_image.convert("L")
            with Image.open(densepose_path) as opened_image:
                densepose_preview = opened_image.convert("RGB")
            with Image.open(pose_preview_path) as opened_image:
                pose_preview = opened_image.convert("RGB")
            pose_payload = json.loads(
                pose_json_path.read_text(encoding="utf-8")
            )
        except (OSError, UnidentifiedImageError, json.JSONDecodeError) as error:
            raise CharacterBodyComparisonError(
                f"신체 비교 중간 결과를 읽을 수 없습니다: {error}"
            ) from error

    expansion_radius_pixels = calculate_mask_expansion_radius(
        character_image.size,
        settings.mask_expansion_ratio,
        settings.minimum_mask_expansion_pixels,
        settings.maximum_mask_expansion_pixels,
    )
    try:
        mask_refinement = refine_character_clothing_change_mask(
            raw_clothing_mask=raw_mask,
            identity_protection_mask=protection_mask,
            expansion_radius_pixels=expansion_radius_pixels,
            closing_radius_pixels=settings.mask_closing_radius_pixels,
        )
    finally:
        raw_mask.close()
        protection_mask.close()

    try:
        human_agnostic_candidate = create_human_agnostic_image_candidate(
            source_image=character_image,
            clothing_erasure_mask=mask_refinement.safe_change_mask,
            raw_clothing_mask=mask_refinement.raw_mask,
        )
    except Exception:
        densepose_preview.close()
        pose_preview.close()
        mask_refinement.close()
        raise

    try:
        clothing_removal_verification = verify_original_clothing_removal(
            raw_clothing_mask=mask_refinement.raw_mask,
            approved_change_mask=mask_refinement.safe_change_mask,
        )
    except Exception:
        densepose_preview.close()
        pose_preview.close()
        mask_refinement.close()
        human_agnostic_candidate.close()
        raise

    try:
        joint_coordinates = tuple(
            CharacterJointCoordinateCandidate(
                joint_name=str(joint_payload["joint_name"]),
                x=float(joint_payload["x"]),
                y=float(joint_payload["y"]),
                confidence_score=float(joint_payload["confidence_score"]),
                detected=bool(joint_payload["detected"]),
            )
            for joint_payload in pose_payload["joint_coordinates"]
        )
        return CharacterBodyComparisonCandidate(
            source_image=character_image.convert("RGB"),
            densepose_preview_image=densepose_preview,
            pose_preview_image=pose_preview,
            mask_refinement=mask_refinement,
            human_agnostic_candidate=human_agnostic_candidate,
            clothing_removal_verification=clothing_removal_verification,
            joint_coordinates=joint_coordinates,
            model_ids=tuple(str(value) for value in pose_payload["model_ids"]),
            detected_joint_count=sum(
                1 for coordinate in joint_coordinates if coordinate.detected
            ),
            missing_joint_count=sum(
                1 for coordinate in joint_coordinates if not coordinate.detected
            ),
            elapsed_seconds=float(pose_payload["elapsed_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        densepose_preview.close()
        pose_preview.close()
        mask_refinement.close()
        human_agnostic_candidate.close()
        clothing_removal_verification.close()
        raise CharacterBodyComparisonError(
            f"DWPose 관절 좌표 형식이 올바르지 않습니다: {error}"
        ) from error


def validate_character_body_comparison_settings(
    settings: CharacterBodyComparisonSettings,
) -> None:
    """신체 비교 별도 실행 전에 경로와 수치 12개를 검사한다."""
    for path_name, required_path in (
        ("별도 Python", settings.python_executable),
        ("CatVTON 저장소", settings.repository_path),
        ("신체 비교 실행 파일", settings.runner_path),
    ):
        if not required_path.exists():
            raise CharacterBodyComparisonError(
                f"{path_name}을 찾을 수 없습니다: {required_path}"
            )
    if settings.width < 256 or settings.height < 256:
        raise CharacterBodyComparisonError(
            "신체 비교 크기는 가로와 세로 모두 256픽셀 이상이어야 합니다."
        )
    if settings.width % 8 or settings.height % 8:
        raise CharacterBodyComparisonError(
            "신체 비교 가로와 세로는 모두 8의 배수여야 합니다."
        )
    if settings.timeout_seconds < 60:
        raise CharacterBodyComparisonError(
            "신체 비교 제한 시간은 60초 이상이어야 합니다."
        )
    if settings.pose_device != "cpu":
        raise CharacterBodyComparisonError(
            "현재 DWPose는 GPU 메모리 충돌 방지를 위해 CPU만 허용합니다."
        )
    if not 0.0 <= settings.minimum_pose_confidence <= 1.0:
        raise CharacterBodyComparisonError(
            "관절 좌표 기준 점수는 0.0~1.0이어야 합니다."
        )
    calculate_mask_expansion_radius(
        (settings.width, settings.height),
        settings.mask_expansion_ratio,
        settings.minimum_mask_expansion_pixels,
        settings.maximum_mask_expansion_pixels,
    )
    if not 1 <= settings.mask_closing_radius_pixels <= 3:
        raise CharacterBodyComparisonError(
            "마스크 닫기 반경은 1~3픽셀이어야 합니다."
        )
