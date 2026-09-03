"""캐릭터 신체 분석 결과로 의상 변경 마스크를 안전하게 준비한다."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from genai_lab.target_masks import ApprovedTargetMasks


@dataclass(frozen=True)
class CharacterBodyComparisonSettings:
    """SCHP·DensePose 별도 실행과 마스크 보정 설정."""

    python_executable: Path
    repository_path: Path
    runner_path: Path
    temporary_root: Path
    cache_dir: Path
    width: int
    height: int
    timeout_seconds: int
    mask_expansion_ratio: float = 0.01
    minimum_mask_expansion_pixels: int = 5
    maximum_mask_expansion_pixels: int = 15
    mask_closing_radius_pixels: int = 2
    foreground_model_id: str = "isnet-anime"
    foreground_expansion_pixels: int = 15


@dataclass(frozen=True)
class CharacterForegroundMaskCandidate:
    """AI가 추출했지만 사용자가 아직 승인하지 않은 캐릭터 전체 외곽."""

    mask_image: Image.Image
    foreground_pixel_count: int
    foreground_percent: float
    model_id: str
    elapsed_seconds: float

    def close(self) -> None:
        """검토 화면에 사용하는 캐릭터 외곽 마스크를 해제한다."""
        self.mask_image.close()


@dataclass(frozen=True)
class CharacterClothingMaskRefinement:
    """닫기·팽창·보호 영역 차감이 끝난 임시 의상 마스크."""

    raw_mask: Image.Image
    closed_mask: Image.Image
    expanded_mask: Image.Image
    character_foreground_mask: Image.Image
    expanded_foreground_mask: Image.Image
    safe_change_mask: Image.Image
    identity_protection_mask: Image.Image
    foreground_expansion_pixels: int
    outside_foreground_rejected_pixel_count: int
    expansion_radius_pixels: int
    closing_radius_pixels: int
    attempted_protected_overlap_pixels: int
    safe_change_pixel_count: int
    safe_change_percent: float

    def close(self) -> None:
        """이 결과 객체가 소유한 PIL 이미지 7개를 해제한다."""
        self.raw_mask.close()
        self.closed_mask.close()
        self.expanded_mask.close()
        self.character_foreground_mask.close()
        self.expanded_foreground_mask.close()
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

    outside_foreground_mask: Image.Image
    remaining_clothing_mask: Image.Image
    detected_clothing_pixel_count: int
    outside_foreground_pixel_count: int
    outside_foreground_percent: float
    protected_overlap_pixel_count: int
    verifiable_clothing_pixel_count: int
    removed_clothing_pixel_count: int
    remaining_clothing_pixel_count: int
    removal_percent: float | None
    passed: bool
    reason_ko: str
    status: str = "covered"
    protected_conflict_mask: Image.Image | None = None

    def close(self) -> None:
        """검토 화면에 사용하는 외곽 오탐·의상 잔여 마스크를 해제한다."""
        self.outside_foreground_mask.close()
        self.remaining_clothing_mask.close()
        if self.protected_conflict_mask is not None:
            self.protected_conflict_mask.close()


@dataclass(frozen=True)
class CharacterBodyComparisonCandidate:
    """신체 보호 영역과 의상 변경 영역 분석이 끝난 사용자 승인 전 후보."""

    source_image: Image.Image
    densepose_preview_image: Image.Image
    foreground_candidate: CharacterForegroundMaskCandidate
    mask_refinement: CharacterClothingMaskRefinement
    human_agnostic_candidate: HumanAgnosticImageCandidate
    clothing_removal_verification: OriginalClothingRemovalVerification
    model_ids: tuple[str, ...]
    elapsed_seconds: float

    mask_source: str = "automasker_candidate"
    automatic_change_mask: Image.Image | None = None

    def close(self) -> None:
        """GUI 검토에 사용한 PIL 이미지를 모두 해제한다."""
        self.source_image.close()
        self.densepose_preview_image.close()
        self.foreground_candidate.close()
        self.mask_refinement.close()
        self.human_agnostic_candidate.close()
        self.clothing_removal_verification.close()
        if self.automatic_change_mask is not None:
            self.automatic_change_mask.close()


@dataclass(frozen=True)
class ConfirmedCharacterBodyComparison:
    """사용자가 확인한 신체 비교 수치와 정확한 중립화 입력."""

    clothing_type: str
    approved_human_agnostic_image: Image.Image
    approved_change_mask: Image.Image
    approved_model_mask: Image.Image
    neutral_rgb: tuple[int, int, int]
    neutralized_pixel_count: int
    neutralized_percent: float
    raw_mask_coverage_percent: float
    outside_foreground_pixel_count: int
    outside_foreground_percent: float
    remaining_clothing_pixel_count: int
    clothing_removal_percent: float
    changed_pixel_count_outside_mask: int
    expansion_radius_pixels: int
    closing_radius_pixels: int
    attempted_protected_overlap_pixels: int
    safe_change_pixel_count: int
    safe_change_percent: float
    model_ids: tuple[str, ...]
    preflight_person_sha256: str
    preflight_binary_mask_sha256: str
    preflight_model_mask_sha256: str
    preflight_clothing_sha256: str
    preflight_protected_overlap_pixel_count: int
    preflight_outside_foreground_pixel_count: int
    preflight_soft_overlap_pixel_count: int
    preflight_hard_overlap_pixel_count: int
    preflight_removed_pixel_count: int

    def close(self) -> None:
        """승인된 중립화 이미지와 원본·모델 교체 마스크를 해제한다."""
        self.approved_human_agnostic_image.close()
        self.approved_change_mask.close()
        self.approved_model_mask.close()


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
    character_foreground_mask: Image.Image,
    expansion_radius_pixels: int,
    foreground_expansion_pixels: int = 15,
    closing_radius_pixels: int = 2,
    preserve_approved_boundary: bool = False,
) -> CharacterClothingMaskRefinement:
    """의상 마스크를 캐릭터 외곽 안으로 제한하고 보호 영역을 뺀다.

    반환값:
        원본·닫기·팽창·안전 마스크와 픽셀 측정값.

    오류:
        마스크 크기·반경이 잘못됐거나 최종 변경 영역이 0픽셀이면 중단한다.
    """
    if (
        raw_clothing_mask.size != identity_protection_mask.size
        or raw_clothing_mask.size != character_foreground_mask.size
    ):
        raise CharacterBodyComparisonError(
            "의상·신체 보호·캐릭터 외곽 마스크의 크기가 다릅니다: "
            f"의상={raw_clothing_mask.size}, "
            f"보호={identity_protection_mask.size}, "
            f"외곽={character_foreground_mask.size}"
        )
    if not (preserve_approved_boundary and expansion_radius_pixels == 0) and not 5 <= expansion_radius_pixels <= 15:
        raise CharacterBodyComparisonError(
            "의상 마스크 팽창 반경은 5~15픽셀이어야 합니다."
        )
    if not (preserve_approved_boundary and closing_radius_pixels == 0) and not 1 <= closing_radius_pixels <= 3:
        raise CharacterBodyComparisonError(
            "마스크 닫기 반경은 1~3픽셀이어야 합니다."
        )
    if not 1 <= foreground_expansion_pixels <= 30:
        raise CharacterBodyComparisonError(
            "캐릭터 외곽 팽창 반경은 1~30픽셀이어야 합니다."
        )

    normalized_raw_mask = raw_clothing_mask.convert("L")
    normalized_protection_mask = identity_protection_mask.convert("L")
    normalized_foreground_mask = character_foreground_mask.convert("L")
    try:
        raw_mask_array = np.asarray(
            normalized_raw_mask, dtype=np.uint8
        ).copy()
        protection_mask_array = np.asarray(
            normalized_protection_mask,
            dtype=np.uint8,
        ).copy()
        foreground_mask_array = np.asarray(
            normalized_foreground_mask,
            dtype=np.uint8,
        ).copy()
    finally:
        normalized_raw_mask.close()
        normalized_protection_mask.close()
        normalized_foreground_mask.close()

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
    binary_foreground_mask = np.where(
        foreground_mask_array >= 128,
        255,
        0,
    ).astype(np.uint8)
    foreground_kernel_size = foreground_expansion_pixels * 2 + 1
    foreground_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (foreground_kernel_size, foreground_kernel_size),
    )
    expanded_foreground_mask_array = cv2.dilate(
        binary_foreground_mask,
        foreground_kernel,
        iterations=1,
    )
    protected_pixels = protection_mask_array >= 128
    foreground_pixels = expanded_foreground_mask_array > 0
    outside_foreground_pixels = (
        (expanded_mask_array > 0) & ~foreground_pixels
    )
    outside_foreground_rejected_pixel_count = int(
        np.count_nonzero(outside_foreground_pixels)
    )
    attempted_protected_overlap_pixels = int(
        np.count_nonzero((expanded_mask_array > 0) & protected_pixels)
    )
    safe_change_mask_array = np.where(
        foreground_pixels & ~protected_pixels,
        expanded_mask_array,
        0,
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
        character_foreground_mask=Image.fromarray(
            binary_foreground_mask,
            mode="L",
        ),
        expanded_foreground_mask=Image.fromarray(
            expanded_foreground_mask_array,
            mode="L",
        ),
        foreground_expansion_pixels=foreground_expansion_pixels,
        outside_foreground_rejected_pixel_count=(
            outside_foreground_rejected_pixel_count
        ),
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
    identity_protection_mask: Image.Image,
    expanded_character_foreground_mask: Image.Image,
) -> OriginalClothingRemovalVerification:
    """캐릭터 외곽 안의 기존 의상이 변경 영역에 포함됐는지 검사한다.

    반환값:
        외곽 밖 오탐·보호 겹침·검증 대상·포함·잔여 수와 위치 마스크.

    오류:
        네 마스크 크기가 다르거나 픽셀 분류 합계가 다르면 중단한다.
    """
    if (
        raw_clothing_mask.size != approved_change_mask.size
        or raw_clothing_mask.size != identity_protection_mask.size
        or raw_clothing_mask.size != expanded_character_foreground_mask.size
    ):
        raise CharacterBodyComparisonError(
            "기존 의상·승인 변경·신체 보호·캐릭터 외곽 "
            "마스크 크기가 다릅니다: "
            f"기존 의상={raw_clothing_mask.size}, "
            f"승인 영역={approved_change_mask.size}, "
            f"보호 영역={identity_protection_mask.size}, "
            f"캐릭터 외곽={expanded_character_foreground_mask.size}"
        )

    normalized_raw_mask = raw_clothing_mask.convert("L")
    normalized_approved_mask = approved_change_mask.convert("L")
    normalized_protection_mask = identity_protection_mask.convert("L")
    normalized_foreground_mask = expanded_character_foreground_mask.convert("L")
    try:
        raw_mask_array = (
            np.asarray(normalized_raw_mask, dtype=np.uint8).copy() >= 128
        )
        approved_mask_array = (
            np.asarray(normalized_approved_mask, dtype=np.uint8).copy() >= 128
        )
        protection_mask_array = (
            np.asarray(normalized_protection_mask, dtype=np.uint8).copy() >= 128
        )
        foreground_mask_array = (
            np.asarray(normalized_foreground_mask, dtype=np.uint8).copy() >= 128
        )
    finally:
        normalized_raw_mask.close()
        normalized_approved_mask.close()
        normalized_protection_mask.close()
        normalized_foreground_mask.close()

    detected_clothing_pixel_count = int(np.count_nonzero(raw_mask_array))
    outside_foreground_pixels = raw_mask_array & ~foreground_mask_array
    protected_overlap_pixels = (
        raw_mask_array & foreground_mask_array & protection_mask_array
    )
    verifiable_clothing_pixels = (
        raw_mask_array & foreground_mask_array
    )
    removed_clothing_pixels = (
        verifiable_clothing_pixels & approved_mask_array & ~protection_mask_array
    )
    remaining_clothing_pixels = verifiable_clothing_pixels & ~removed_clothing_pixels
    outside_foreground_pixel_count = int(
        np.count_nonzero(outside_foreground_pixels)
    )
    protected_overlap_pixel_count = int(
        np.count_nonzero(protected_overlap_pixels)
    )
    verifiable_clothing_pixel_count = int(
        np.count_nonzero(verifiable_clothing_pixels)
    )
    removed_clothing_pixel_count = int(
        np.count_nonzero(removed_clothing_pixels)
    )
    remaining_clothing_pixel_count = int(
        np.count_nonzero(remaining_clothing_pixels)
    )
    classified_clothing_pixel_count = (
        outside_foreground_pixel_count
        + verifiable_clothing_pixel_count
    )
    if classified_clothing_pixel_count != detected_clothing_pixel_count:
        raise CharacterBodyComparisonError(
            "기존 의상 픽셀 분류 합계가 일치하지 않습니다: "
            f"탐지={detected_clothing_pixel_count:,}픽셀, "
            f"분류={classified_clothing_pixel_count:,}픽셀"
        )
    outside_foreground_percent = (
        outside_foreground_pixel_count
        / max(detected_clothing_pixel_count, 1)
        * 100.0
    )
    removal_percent = (
        removed_clothing_pixel_count
        / verifiable_clothing_pixel_count
        * 100.0
        if verifiable_clothing_pixel_count > 0
        else None
    )
    if not verifiable_clothing_pixel_count:
        status = "not_evaluable"
        reason_ko = "검사할 기존 의상 영역이 없습니다. 제거 완료가 아닌 계산 불가입니다."
    elif protected_overlap_pixel_count:
        status = "needs_review"
        reason_ko = (
            f"교체 의상과 보호 영역이 {protected_overlap_pixel_count:,}px 겹칩니다. "
            "충돌 표시를 확인하고 교체·보호 마스크를 다시 선택하세요."
        )
    elif remaining_clothing_pixel_count:
        status = "incomplete"
        reason_ko = f"교체할 기존 의상이 변경 영역에서 {remaining_clothing_pixel_count:,}px 누락됐습니다."
    else:
        status = "covered"
        reason_ko = "검사 대상 의상이 변경 영역에 포함됐습니다. 생성 품질 완료를 뜻하지 않습니다."
    passed = status == "covered"

    return OriginalClothingRemovalVerification(
        outside_foreground_mask=Image.fromarray(
            np.where(outside_foreground_pixels, 255, 0).astype(np.uint8),
            mode="L",
        ),
        remaining_clothing_mask=Image.fromarray(
            np.where(remaining_clothing_pixels, 255, 0).astype(np.uint8),
            mode="L",
        ),
        detected_clothing_pixel_count=detected_clothing_pixel_count,
        outside_foreground_pixel_count=outside_foreground_pixel_count,
        outside_foreground_percent=outside_foreground_percent,
        protected_overlap_pixel_count=protected_overlap_pixel_count,
        verifiable_clothing_pixel_count=verifiable_clothing_pixel_count,
        removed_clothing_pixel_count=removed_clothing_pixel_count,
        remaining_clothing_pixel_count=remaining_clothing_pixel_count,
        removal_percent=removal_percent,
        passed=passed,
        reason_ko=reason_ko,
        status=status,
        protected_conflict_mask=Image.fromarray(
            protected_overlap_pixels.astype(np.uint8) * 255
        ),
    )


def execute_character_body_comparison(
    character_image: Image.Image,
    clothing_type: str,
    settings: CharacterBodyComparisonSettings,
    approved_target_masks: ApprovedTargetMasks | None = None,
) -> CharacterBodyComparisonCandidate:
    """별도 환경에서 SCHP·DensePose를 실행하고 의상 마스크를 보정한다.

    반환값:
        사용자 확인 전 원본·중간 마스크와 측정 수치.

    부수 효과:
        로컬 임시 폴더에 중간 파일을 만들고 함수 종료 시 모두 제거한다.
    """
    validate_character_body_comparison_settings(settings)
    if approved_target_masks is not None:
        approved_target_masks.validate_source(character_image)
    settings.temporary_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="genai-lab-body-comparison-",
        dir=settings.temporary_root,
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        person_path = temporary_directory / "person.png"
        raw_mask_path = temporary_directory / "raw_mask.png"
        protection_mask_path = temporary_directory / "protection_mask.png"
        foreground_mask_path = temporary_directory / "foreground_mask.png"
        densepose_path = temporary_directory / "densepose.png"
        metadata_json_path = temporary_directory / "metadata.json"

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
            "--output-foreground-mask", str(foreground_mask_path),
            "--output-densepose", str(densepose_path),
            "--output-metadata-json", str(metadata_json_path),
            "--cache-dir", str(settings.cache_dir),
            "--width", str(settings.width),
            "--height", str(settings.height),
            "--foreground-model-id", settings.foreground_model_id,
        ]
        if approved_target_masks is not None:
            command.append("--explicit-target-masks")
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
                "SCHP·DensePose 신체 비교 실행에 실패했습니다. "
                f"별도 실행 출력: {execution_details}"
            )
        required_outputs = (
            raw_mask_path, protection_mask_path, foreground_mask_path,
            densepose_path,
            metadata_json_path,
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
            with Image.open(foreground_mask_path) as opened_image:
                foreground_mask = opened_image.convert("L")
            with Image.open(densepose_path) as opened_image:
                densepose_preview = opened_image.convert("RGB")
            metadata_payload = json.loads(
                metadata_json_path.read_text(encoding="utf-8")
            )
        except (OSError, UnidentifiedImageError, json.JSONDecodeError) as error:
            raise CharacterBodyComparisonError(
                f"신체 비교 중간 결과를 읽을 수 없습니다: {error}"
            ) from error

    try:
        foreground_model_id = str(metadata_payload["foreground_model_id"])
        foreground_pixel_count = int(
            metadata_payload["foreground_pixel_count"]
        )
        foreground_percent = float(metadata_payload["foreground_percent"])
        foreground_elapsed_seconds = float(
            metadata_payload["foreground_elapsed_seconds"]
        )
        measured_foreground_pixel_count = int(
            np.count_nonzero(
                np.asarray(foreground_mask, dtype=np.uint8) >= 128
            )
        )
        if foreground_model_id != settings.foreground_model_id:
            raise ValueError(
                "설정과 실행 기록의 외곽 모델 ID가 다릅니다."
            )
        if foreground_pixel_count != measured_foreground_pixel_count:
            raise ValueError(
                "실행 기록과 실제 캐릭터 외곽 픽셀 수가 다릅니다."
            )
        foreground_candidate = CharacterForegroundMaskCandidate(
            mask_image=foreground_mask,
            foreground_pixel_count=foreground_pixel_count,
            foreground_percent=foreground_percent,
            model_id=foreground_model_id,
            elapsed_seconds=foreground_elapsed_seconds,
        )
    except (KeyError, TypeError, ValueError) as error:
        raw_mask.close()
        protection_mask.close()
        foreground_mask.close()
        densepose_preview.close()
        raise CharacterBodyComparisonError(
            f"캐릭터 외곽 분석 기록 형식이 올바르지 않습니다: {error}"
        ) from error

    automatic_change_mask = None
    if approved_target_masks is not None:
        automatic_change_mask = raw_mask
        raw_mask = approved_target_masks.clothing_mask.copy()
        merged_protection = np.maximum(
            np.asarray(protection_mask, dtype=np.uint8),
            np.asarray(approved_target_masks.special_protection_mask, dtype=np.uint8),
        )
        protection_mask.close()
        protection_mask = Image.fromarray(merged_protection)
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
            character_foreground_mask=foreground_candidate.mask_image,
            expansion_radius_pixels=(0 if approved_target_masks else expansion_radius_pixels),
            foreground_expansion_pixels=(
                settings.foreground_expansion_pixels
            ),
            closing_radius_pixels=(0 if approved_target_masks else settings.mask_closing_radius_pixels),
            preserve_approved_boundary=approved_target_masks is not None,
        )
    except Exception:
        densepose_preview.close()
        foreground_candidate.close()
        if automatic_change_mask is not None:
            automatic_change_mask.close()
        raise
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
        foreground_candidate.close()
        mask_refinement.close()
        if automatic_change_mask is not None:
            automatic_change_mask.close()
        raise

    try:
        clothing_removal_verification = verify_original_clothing_removal(
            raw_clothing_mask=mask_refinement.raw_mask,
            approved_change_mask=mask_refinement.safe_change_mask,
            identity_protection_mask=mask_refinement.identity_protection_mask,
            expanded_character_foreground_mask=(
                mask_refinement.expanded_foreground_mask
            ),
        )
    except Exception:
        densepose_preview.close()
        foreground_candidate.close()
        mask_refinement.close()
        human_agnostic_candidate.close()
        if automatic_change_mask is not None:
            automatic_change_mask.close()
        raise

    try:
        return CharacterBodyComparisonCandidate(
            source_image=character_image.convert("RGB"),
            densepose_preview_image=densepose_preview,
            foreground_candidate=foreground_candidate,
            mask_refinement=mask_refinement,
            human_agnostic_candidate=human_agnostic_candidate,
            clothing_removal_verification=clothing_removal_verification,
            model_ids=tuple(
                str(value) for value in metadata_payload["model_ids"]
            ),
            elapsed_seconds=float(metadata_payload["elapsed_seconds"]),
            mask_source=("user_selected_target_sam2" if approved_target_masks else "automasker_candidate"),
            automatic_change_mask=automatic_change_mask,
        )
    except (KeyError, TypeError, ValueError) as error:
        densepose_preview.close()
        foreground_candidate.close()
        mask_refinement.close()
        human_agnostic_candidate.close()
        clothing_removal_verification.close()
        if automatic_change_mask is not None:
            automatic_change_mask.close()
        raise CharacterBodyComparisonError(
            f"신체 마스크 분석 기록 형식이 올바르지 않습니다: {error}"
        ) from error


def validate_character_body_comparison_settings(
    settings: CharacterBodyComparisonSettings,
) -> None:
    """신체 비교 별도 실행 전에 경로와 마스크 수치를 검사한다."""
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
    if not settings.foreground_model_id.strip():
        raise CharacterBodyComparisonError(
            "캐릭터 외곽 모델 ID가 비어 있습니다."
        )
    if not 1 <= settings.foreground_expansion_pixels <= 30:
        raise CharacterBodyComparisonError(
            "캐릭터 외곽 팽창 반경은 1~30픽셀이어야 합니다."
        )
