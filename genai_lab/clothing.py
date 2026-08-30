"""의상 참조 합성 결과가 캐릭터의 보호 영역을 바꾸지 못하게 제한한다."""

from dataclasses import dataclass
import json
import os
import subprocess
import tempfile

from enum import Enum
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageChops, ImageFilter, ImageOps, UnidentifiedImageError

from genai_lab.body_comparison import (
    calculate_mask_expansion_radius,
)


class ClothingCategory(str, Enum):
    """의상 참조가 바꿀 수 있는 의상 종류."""

    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    FULL_BODY_OUTFIT = "full_body_outfit"
    GLOVES = "gloves"
    SHOES = "shoes"


@dataclass(frozen=True)
class ClothingReferenceInput:
    """사용자가 선택한 의상 참조 입력."""

    image_path: Path
    category: ClothingCategory
    region_box_xyxy: tuple[int, int, int, int] | None = None
    approved_image: Image.Image | None = None


@dataclass(frozen=True)
class CharacterAgnosticApprovedInput:
    """사용자가 승인한 기존 의상 제거 이미지와 정확한 변경 마스크."""

    human_agnostic_image: Image.Image
    approved_change_mask: Image.Image
    clothing_type: str
    approved_mask_pixel_count: int

    def close(self) -> None:
        """작업 스레드가 소유한 승인 이미지 2개를 해제한다."""
        self.human_agnostic_image.close()
        self.approved_change_mask.close()


@dataclass(frozen=True)
class CatVTONExecutionMetadata:
    """별도 CatVTON 프로세스가 실제 사용한 입력 경계를 증명한다."""

    mask_source: str
    automasker_run_count: int
    approved_image_width: int
    approved_image_height: int
    approved_mask_pixel_count: int
    processed_mask_pixel_count: int
    safety_check_enabled: bool
    person_input_source: str
    person_input_width: int
    person_input_height: int
    clothing_source_width: int
    clothing_source_height: int
    clothing_input_width: int
    clothing_input_height: int
    clothing_alpha_pixel_count: int
    clothing_alpha_coverage_percent: float


@dataclass(frozen=True)
class CatVTONClothingConditionImage:
    """승인 추출본에서 투명 여백을 제거한 모델 전용 의상 입력."""

    image: Image.Image
    source_size: tuple[int, int]
    crop_box_xyxy: tuple[int, int, int, int]
    alpha_pixel_count: int
    alpha_coverage_percent: float


@dataclass(frozen=True)
class CharacterClothingTryOnRequest:
    """의상 합성 모델에 전달할 캐릭터와 승인된 의상 참조."""

    base_character_image: Image.Image
    clothing_reference_image: Image.Image
    clothing_category: ClothingCategory


@dataclass(frozen=True)
class CharacterTryOnProtectionPlan:
    """의상 합성에서 변경 허용 영역과 보호 영역을 분리한 계획."""

    clothing_change_mask: Image.Image
    identity_protection_mask: Image.Image
    boundary_blend_mask: Image.Image


@dataclass(frozen=True)
class CharacterTryOnVerification:
    """보호 영역 불변 검사가 끝난 의상 합성 검증 결과."""

    passed: bool
    changed_pixel_count_outside_clothing: int
    reason_ko: str


@dataclass(frozen=True)
class CharacterClothingTryOnCandidate:
    """의상 합성 후 보호 영역 검사를 마쳤지만 아직 승인되지 않은 후보."""

    image: Image.Image
    verification: CharacterTryOnVerification
    clothing_category: ClothingCategory


class CharacterClothingTryOnEngine(Protocol):
    """CatVTON 같은 의상 합성 구현체가 지켜야 하는 최소 계약."""

    def generate_clothing_try_on_image(
        self,
        request: CharacterClothingTryOnRequest,
    ) -> Image.Image:
        """의상 합성 원본을 반환하며 파일에는 저장하지 않는다."""


class CharacterClothingProtectionError(ValueError):
    """의상 합성 보호 계획이나 결과를 안전하게 처리할 수 없는 오류."""


def load_clothing_reference_image(
    clothing_reference_input: ClothingReferenceInput,
) -> Image.Image:
    """의상 참조에서 모델에 전달할 여백 제거 RGB 이미지를 반환한다."""
    return prepare_catvton_clothing_condition_image(
        clothing_reference_input
    ).image


def prepare_catvton_clothing_condition_image(
    clothing_reference_input: ClothingReferenceInput,
) -> CatVTONClothingConditionImage:
    """승인 알파 영역의 경계 상자로 잘라 CatVTON 조건 입력을 만든다.

    반환값:
        원본 추출 증거를 변경하지 않은 모델 전용 복사본과 수치 기록.

    오류:
        승인 이미지의 의상 알파 픽셀이 0개면 실행을 중단한다.
    """
    if clothing_reference_input.approved_image is not None:
        approved_rgba_image = clothing_reference_input.approved_image.convert(
            "RGBA"
        )
        alpha_channel = approved_rgba_image.getchannel("A")
        try:
            crop_box = alpha_channel.getbbox()
            if crop_box is None:
                raise CharacterClothingProtectionError(
                    "승인된 의상 추출본의 알파 픽셀이 0개입니다."
                )
            alpha_histogram = alpha_channel.histogram()
            alpha_pixel_count = sum(alpha_histogram[1:])
            cropped_rgba_image = approved_rgba_image.crop(crop_box)
            white_background = Image.new(
                "RGBA", cropped_rgba_image.size, (255, 255, 255, 255)
            )
            try:
                condition_image = Image.alpha_composite(
                    white_background,
                    cropped_rgba_image,
                ).convert("RGB")
            finally:
                cropped_rgba_image.close()
                white_background.close()
            crop_pixel_count = condition_image.width * condition_image.height
            coverage_percent = (
                alpha_pixel_count / crop_pixel_count * 100.0
            )
            return CatVTONClothingConditionImage(
                image=condition_image,
                source_size=approved_rgba_image.size,
                crop_box_xyxy=crop_box,
                alpha_pixel_count=alpha_pixel_count,
                alpha_coverage_percent=coverage_percent,
            )
        finally:
            alpha_channel.close()
            approved_rgba_image.close()

    image_path = clothing_reference_input.image_path
    if not image_path.is_file():
        raise CharacterClothingProtectionError(
            f"의상 참조 이미지를 찾을 수 없습니다: {image_path}"
        )
    try:
        with Image.open(image_path) as opened_image:
            opened_image.load()
            normalized_image = ImageOps.exif_transpose(opened_image).convert("RGB")
            region_box = clothing_reference_input.region_box_xyxy
            if region_box is None:
                width, height = normalized_image.size
                return CatVTONClothingConditionImage(
                    image=normalized_image,
                    source_size=(width, height),
                    crop_box_xyxy=(0, 0, width, height),
                    alpha_pixel_count=width * height,
                    alpha_coverage_percent=100.0,
                )

            x1, y1, x2, y2 = region_box
            if not (
                0 <= x1 < x2 <= normalized_image.width
                and 0 <= y1 < y2 <= normalized_image.height
            ):
                normalized_image.close()
                raise CharacterClothingProtectionError(
                    "승인된 의상 영역이 이미지 경계를 벗어났습니다. "
                    f"이미지={opened_image.width}x{opened_image.height}, "
                    f"영역={region_box}"
                )
            cropped_image = normalized_image.crop(region_box)
            normalized_image.close()
            return CatVTONClothingConditionImage(
                image=cropped_image,
                source_size=(opened_image.width, opened_image.height),
                crop_box_xyxy=region_box,
                alpha_pixel_count=cropped_image.width * cropped_image.height,
                alpha_coverage_percent=100.0,
            )
    except (UnidentifiedImageError, OSError) as error:
        raise CharacterClothingProtectionError(
            f"의상 참조 이미지를 읽을 수 없습니다: {image_path}"
        ) from error


def create_character_try_on_protection_plan(
    clothing_change_mask: Image.Image,
    identity_protection_mask: Image.Image,
    boundary_blur_radius: float = 4.0,
) -> CharacterTryOnProtectionPlan:
    """의상 영역에서 얼굴·손·발·피부 보호 영역을 뺀 합성 계획을 만든다.

    반환값:
        의상 변경 가능 영역, 변경 금지 영역과 경계 혼합 영역.

    오류:
        마스크 크기가 다르거나 실제 변경 가능 영역이 없으면 중단한다.
    """
    validate_same_image_size(
        clothing_change_mask,
        identity_protection_mask,
        "의상 영역 마스크와 신체 보호 마스크",
    )
    if boundary_blur_radius < 0:
        raise CharacterClothingProtectionError(
            "의상 경계 흐림 반경은 0 이상이어야 합니다."
        )

    normalized_clothing_mask = clothing_change_mask.convert("L")
    normalized_identity_mask = identity_protection_mask.convert("L")

    # CharacterTryOnProtectionPlan(캐릭터 의상 합성 보호 계획)
    # - 포함: 의상 변경 가능 영역, 얼굴·머리·손·발·피부·배경 보호 영역과 경계.
    # - 생성: 사람·의상 분리 모델의 마스크를 규칙으로 결합해 만든다.
    # - 처리: 이 함수에는 AI 호출이 없고 픽셀 규칙만 사용한다.
    # - 저장: 마스크는 임시 값이며 자동 저장하지 않는다.
    # - 다음 사용처: 의상 합성 원본에서 허용 픽셀만 가져올 때 사용한다.
    safe_clothing_change_mask = ImageChops.multiply(
        normalized_clothing_mask,
        ImageOps.invert(normalized_identity_mask),
    )
    if safe_clothing_change_mask.getbbox() is None:
        raise CharacterClothingProtectionError(
            "신체 보호 영역을 제외한 뒤 변경 가능한 의상 영역이 없습니다."
        )

    strict_identity_protection_mask = ImageOps.invert(
        safe_clothing_change_mask
    )
    blurred_boundary_mask = safe_clothing_change_mask.filter(
        ImageFilter.GaussianBlur(radius=boundary_blur_radius)
    )
    boundary_blend_mask = ImageChops.multiply(
        blurred_boundary_mask,
        safe_clothing_change_mask,
    )
    return CharacterTryOnProtectionPlan(
        clothing_change_mask=safe_clothing_change_mask,
        identity_protection_mask=strict_identity_protection_mask,
        boundary_blend_mask=boundary_blend_mask,
    )


def apply_protected_clothing_try_on(
    try_on_engine: CharacterClothingTryOnEngine,
    try_on_request: CharacterClothingTryOnRequest,
    protection_plan: CharacterTryOnProtectionPlan,
) -> CharacterClothingTryOnCandidate:
    """의상 합성 원본에서 허용된 의상 픽셀만 기본 캐릭터에 합성한다.

    반환값:
        보호 영역 불변 검사를 통과한 사용자 승인 전 의상 후보.

    오류:
        합성기가 크기를 바꾸거나 보호 영역 픽셀이 하나라도 달라지면 중단한다.
    """
    raw_try_on_image = try_on_engine.generate_clothing_try_on_image(
        try_on_request
    ).convert("RGB")
    base_character_image = try_on_request.base_character_image.convert("RGB")
    try:
        validate_same_image_size(
            base_character_image,
            raw_try_on_image,
            "기본 캐릭터와 의상 합성 결과",
        )
        validate_protection_plan_size(
            protection_plan,
            base_character_image.size,
        )
        protected_candidate_image = Image.composite(
            raw_try_on_image,
            base_character_image,
            protection_plan.boundary_blend_mask,
        )
    finally:
        raw_try_on_image.close()
        base_character_image.close()

    verification = verify_character_try_on_protection(
        try_on_request.base_character_image,
        protected_candidate_image,
        protection_plan,
    )
    if not verification.passed:
        protected_candidate_image.close()
        raise CharacterClothingProtectionError(verification.reason_ko)

    return CharacterClothingTryOnCandidate(
        image=protected_candidate_image,
        verification=verification,
        clothing_category=try_on_request.clothing_category,
    )


def verify_character_try_on_protection(
    base_character_image: Image.Image,
    protected_candidate_image: Image.Image,
    protection_plan: CharacterTryOnProtectionPlan,
) -> CharacterTryOnVerification:
    """의상 변경 허용 영역 밖의 픽셀이 그대로인지 검사한다."""
    validate_same_image_size(
        base_character_image,
        protected_candidate_image,
        "기본 캐릭터와 보호 합성 후보",
    )
    validate_protection_plan_size(
        protection_plan,
        base_character_image.size,
    )

    difference_image = ImageChops.difference(
        base_character_image.convert("RGB"),
        protected_candidate_image.convert("RGB"),
    ).convert("L")
    protected_difference_image = ImageChops.multiply(
        difference_image,
        protection_plan.identity_protection_mask,
    )
    changed_pixel_count = sum(protected_difference_image.histogram()[1:])
    protected_difference_image.close()
    difference_image.close()

    if changed_pixel_count:
        return CharacterTryOnVerification(
            passed=False,
            changed_pixel_count_outside_clothing=changed_pixel_count,
            reason_ko=(
                "의상 밖의 얼굴·신체·배경 영역이 "
                f"{changed_pixel_count}픽셀 변경되어 의상 후보를 폐기합니다."
            ),
        )
    return CharacterTryOnVerification(
        passed=True,
        changed_pixel_count_outside_clothing=0,
        reason_ko="의상 변경 허용 영역 밖의 픽셀이 모두 유지되었습니다.",
    )


def validate_protection_plan_size(
    protection_plan: CharacterTryOnProtectionPlan,
    expected_size: tuple[int, int],
) -> None:
    """보호 계획의 모든 마스크가 캐릭터 이미지와 같은 크기인지 검사한다."""
    for mask_name, mask_image in (
        ("의상 변경", protection_plan.clothing_change_mask),
        ("신체 보호", protection_plan.identity_protection_mask),
        ("경계 혼합", protection_plan.boundary_blend_mask),
    ):
        if mask_image.size != expected_size:
            raise CharacterClothingProtectionError(
                f"{mask_name} 마스크 크기 {mask_image.size}가 "
                f"캐릭터 이미지 크기 {expected_size}와 다릅니다."
            )


def validate_same_image_size(
    first_image: Image.Image,
    second_image: Image.Image,
    image_description: str,
) -> None:
    """두 이미지 크기가 다르면 안전한 픽셀 합성을 중단한다."""
    if first_image.size != second_image.size:
        raise CharacterClothingProtectionError(
            f"{image_description}의 크기가 다릅니다: "
            f"{first_image.size}, {second_image.size}"
        )



@dataclass(frozen=True)
class CatVTONLocalSettings:
    """별도 Python 환경에 설치한 CatVTON 로컬 실행 설정."""

    python_executable: Path
    repository_path: Path
    runner_path: Path
    temporary_root: Path
    cache_dir: Path
    model_id: str
    base_model_id: str
    width: int
    height: int
    inference_steps: int
    guidance_scale: float
    mixed_precision: str
    timeout_seconds: int
    safety_check_enabled: bool = False
    mask_expansion_ratio: float = 0.01
    minimum_mask_expansion_pixels: int = 5
    maximum_mask_expansion_pixels: int = 15
    mask_closing_radius_pixels: int = 2


@dataclass(frozen=True)
class CatVTONClothingTryOnResult:
    """CatVTON 합성과 보호 검사가 끝난 사용자 승인 전 결과."""

    candidate: CharacterClothingTryOnCandidate
    clothing_change_mask: Image.Image
    execution_metadata: CatVTONExecutionMetadata


@dataclass(frozen=True)
class PreparedClothingTryOnEngine:
    """이미 실행된 CatVTON 원본을 보호 합성 함수에 전달한다."""

    raw_try_on_image: Image.Image

    def generate_clothing_try_on_image(
        self,
        request: CharacterClothingTryOnRequest,
    ) -> Image.Image:
        return self.raw_try_on_image.copy()


def execute_catvton_clothing_try_on(
    base_character_image: Image.Image,
    clothing_reference_input: ClothingReferenceInput,
    approved_agnostic_input: CharacterAgnosticApprovedInput,
    settings: CatVTONLocalSettings,
    seed: int,
) -> CatVTONClothingTryOnResult:
    """별도 CatVTON 프로세스를 실행하고 의상 허용 영역만 합성한다.

    반환값:
        마스크 밖 픽셀 불변 검사를 통과한 의상 후보와 확인용 마스크.

    오류:
        별도 환경이 없거나 CatVTON 실행·보호 검사에 실패하면 중단한다.

    부수 효과:
        로컬 임시 폴더에 입력과 중간 결과를 만들고 함수 종료 시 제거한다.
    """
    validate_catvton_local_settings(settings)
    catvton_clothing_type = find_catvton_clothing_type(
        clothing_reference_input.category
    )
    validate_character_agnostic_approved_input(
        approved_agnostic_input,
        expected_clothing_type=catvton_clothing_type,
    )
    validate_catvton_approved_coordinates(
        base_character_image=base_character_image,
        approved_agnostic_input=approved_agnostic_input,
    )
    clothing_condition = prepare_catvton_clothing_condition_image(
        clothing_reference_input
    )
    clothing_reference_image = clothing_condition.image
    settings.temporary_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="genai-lab-catvton-",
        dir=settings.temporary_root,
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        person_input_path = temporary_directory / "base_character.png"
        approved_mask_input_path = temporary_directory / "approved_change_mask.png"
        clothing_input_path = temporary_directory / "clothing.png"
        raw_output_path = temporary_directory / "raw_try_on.png"
        mask_output_path = temporary_directory / "clothing_mask.png"
        protection_output_path = temporary_directory / "identity_protection_mask.png"
        metadata_output_path = temporary_directory / "execution_metadata.json"

        person_input_image = base_character_image.convert("RGB")
        approved_mask_input_image = (
            approved_agnostic_input.approved_change_mask.convert("L")
        )
        try:
            person_input_image.save(person_input_path)
            approved_mask_input_image.save(approved_mask_input_path)
        finally:
            person_input_image.close()
            approved_mask_input_image.close()
        clothing_reference_image.save(clothing_input_path)
        clothing_reference_image.close()

        command = [
            str(settings.python_executable),
            str(settings.runner_path),
            "--repository-path",
            str(settings.repository_path),
            "--person-image",
            str(person_input_path),
            "--approved-change-mask",
            str(approved_mask_input_path),
            "--clothing-image",
            str(clothing_input_path),
            "--clothing-source-width",
            str(clothing_condition.source_size[0]),
            "--clothing-source-height",
            str(clothing_condition.source_size[1]),
            "--clothing-alpha-pixel-count",
            str(clothing_condition.alpha_pixel_count),
            "--clothing-alpha-coverage-percent",
            f"{clothing_condition.alpha_coverage_percent:.6f}",
            "--clothing-type",
            catvton_clothing_type,
            "--output-image",
            str(raw_output_path),
            "--output-mask",
            str(mask_output_path),
            "--output-protection-mask",
            str(protection_output_path),
            "--output-metadata",
            str(metadata_output_path),
            "--model-id",
            settings.model_id,
            "--base-model-id",
            settings.base_model_id,
            "--cache-dir",
            str(settings.cache_dir),
            "--width",
            str(settings.width),
            "--height",
            str(settings.height),
            "--inference-steps",
            str(settings.inference_steps),
            "--guidance-scale",
            str(settings.guidance_scale),
            "--mixed-precision",
            settings.mixed_precision,
            "--seed",
            str(seed),
        ]
        if not settings.safety_check_enabled:
            command.append("--skip-safety-check")
        execution_environment = os.environ.copy()
        execution_environment["HF_HOME"] = str(settings.cache_dir)
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
                env=execution_environment,
            )
        except subprocess.TimeoutExpired as error:
            raise CharacterClothingProtectionError(
                "의상 합성 제한 시간을 초과했습니다. "
                f"제한={settings.timeout_seconds}초"
            ) from error
        except OSError as error:
            raise CharacterClothingProtectionError(
                f"CatVTON 별도 실행을 시작하지 못했습니다: {error}"
            ) from error

        if completed_process.returncode != 0:
            execution_details = (
                completed_process.stderr.strip()
                or completed_process.stdout.strip()
                or "상세 출력 없음"
            )
            raise CharacterClothingProtectionError(
                "CatVTON 의상 합성에 실패했습니다. "
                f"별도 실행 출력: {execution_details}"
            )
        if (
            not raw_output_path.is_file()
            or not mask_output_path.is_file()
            or not protection_output_path.is_file()
            or not metadata_output_path.is_file()
        ):
            raise CharacterClothingProtectionError(
                "CatVTON 실행은 끝났지만 합성 이미지 또는 보호 마스크가 없습니다."
            )

        with Image.open(raw_output_path) as opened_raw_image:
            raw_try_on_image = opened_raw_image.convert("RGB").copy()
        with Image.open(mask_output_path) as opened_mask_image:
            clothing_change_mask = opened_mask_image.convert("L").copy()
        validate_runner_mask_matches_approved_input(
            clothing_change_mask,
            approved_agnostic_input.approved_change_mask,
        )

        with Image.open(protection_output_path) as opened_protection_image:
            identity_protection_mask = (
                opened_protection_image.convert("L").copy()
            )
        execution_metadata = load_catvton_execution_metadata(
            metadata_output_path,
            approved_agnostic_input,
            expected_safety_check_enabled=settings.safety_check_enabled,
        )
    original_size = base_character_image.size
    if raw_try_on_image.size != original_size:
        resized_raw_try_on_image = raw_try_on_image.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )
        raw_try_on_image.close()
        raw_try_on_image = resized_raw_try_on_image
    if clothing_change_mask.size != original_size:
        resized_clothing_change_mask = clothing_change_mask.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        clothing_change_mask.close()
        clothing_change_mask = resized_clothing_change_mask

    if identity_protection_mask.size != original_size:
        resized_identity_protection_mask = identity_protection_mask.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        identity_protection_mask.close()
        identity_protection_mask = resized_identity_protection_mask
    protection_plan = create_character_try_on_protection_plan(
        clothing_change_mask,
        identity_protection_mask,
        boundary_blur_radius=4.0,
    )
    safe_clothing_change_mask = protection_plan.clothing_change_mask.copy()
    try_on_clothing_reference_image = load_clothing_reference_image(
        clothing_reference_input
    )
    try:
        try_on_candidate = apply_protected_clothing_try_on(
            PreparedClothingTryOnEngine(raw_try_on_image),
            CharacterClothingTryOnRequest(
                base_character_image=base_character_image,
                clothing_reference_image=try_on_clothing_reference_image,
                clothing_category=clothing_reference_input.category,
            ),
            protection_plan,
        )
    finally:
        raw_try_on_image.close()
        try_on_clothing_reference_image.close()
        clothing_change_mask.close()
        identity_protection_mask.close()
        protection_plan.clothing_change_mask.close()
        protection_plan.identity_protection_mask.close()
        protection_plan.boundary_blend_mask.close()

    return CatVTONClothingTryOnResult(
        candidate=try_on_candidate,
        clothing_change_mask=safe_clothing_change_mask,
        execution_metadata=execution_metadata,
    )


def count_binary_mask_pixels(mask_image: Image.Image) -> int:
    """128 이상인 변경 허용 픽셀 수를 반환한다."""
    normalized_mask = mask_image.convert("L")
    try:
        histogram = normalized_mask.histogram()
        return sum(histogram[128:])
    finally:
        normalized_mask.close()


def validate_character_agnostic_approved_input(
    approved_input: CharacterAgnosticApprovedInput,
    expected_clothing_type: str,
) -> None:
    """승인 이미지·마스크·의상 종류·픽셀 수가 같은 계약인지 검사한다."""
    if (
        approved_input.human_agnostic_image.size
        != approved_input.approved_change_mask.size
    ):
        raise CharacterClothingProtectionError(
            "승인 Human-Agnostic 이미지와 변경 마스크 크기가 다릅니다."
        )
    if approved_input.clothing_type != expected_clothing_type:
        raise CharacterClothingProtectionError(
            "승인 당시 의상 종류와 현재 의상 종류가 다릅니다."
        )
    actual_pixel_count = count_binary_mask_pixels(
        approved_input.approved_change_mask
    )
    if actual_pixel_count == 0:
        raise CharacterClothingProtectionError(
            "승인된 의상 변경 픽셀이 0개입니다."
        )
    if actual_pixel_count != approved_input.approved_mask_pixel_count:
        raise CharacterClothingProtectionError(
            "승인 마스크 픽셀 수가 기록값과 다릅니다: "
            f"실제={actual_pixel_count}, "
            f"기록={approved_input.approved_mask_pixel_count}"
        )


def validate_catvton_approved_coordinates(
    base_character_image: Image.Image,
    approved_agnostic_input: CharacterAgnosticApprovedInput,
) -> None:
    """생성 후보와 승인 이미지·마스크가 같은 좌표인지 검사한다."""
    expected_size = base_character_image.size
    if approved_agnostic_input.human_agnostic_image.size != expected_size:
        raise CharacterClothingProtectionError(
            "생성 후보와 Human-Agnostic 이미지 크기가 다릅니다: "
            f"후보={expected_size}, "
            "승인 이미지="
            f"{approved_agnostic_input.human_agnostic_image.size}"
        )
    if approved_agnostic_input.approved_change_mask.size != expected_size:
        raise CharacterClothingProtectionError(
            "생성 후보와 승인 마스크 크기가 다릅니다: "
            f"후보={expected_size}, "
            f"승인 마스크={approved_agnostic_input.approved_change_mask.size}"
        )


def validate_runner_mask_matches_approved_input(
    runner_mask: Image.Image,
    approved_mask: Image.Image,
) -> None:
    """실행기가 승인 마스크를 바꾸거나 다시 계산하지 않았는지 검사한다."""
    if runner_mask.size != approved_mask.size:
        raise CharacterClothingProtectionError(
            "CatVTON 반환 마스크 크기가 승인 마스크와 다릅니다."
        )
    runner_binary = runner_mask.convert("L").point(
        lambda pixel: 255 if pixel >= 128 else 0
    )
    approved_binary = approved_mask.convert("L").point(
        lambda pixel: 255 if pixel >= 128 else 0
    )
    try:
        if (
            ImageChops.difference(runner_binary, approved_binary).getbbox()
            is not None
        ):
            raise CharacterClothingProtectionError(
                "CatVTON 반환 마스크가 사용자 승인 마스크와 다릅니다."
            )
    finally:
        runner_binary.close()
        approved_binary.close()


def load_catvton_execution_metadata(
    metadata_path: Path,
    approved_input: CharacterAgnosticApprovedInput,
    expected_safety_check_enabled: bool,
) -> CatVTONExecutionMetadata:
    """실행 기록을 읽고 AutoMasker 미실행과 승인 입력 사용을 검증한다."""
    try:
        metadata_payload = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        safety_check_enabled = metadata_payload["safety_check_enabled"]
        if not isinstance(safety_check_enabled, bool):
            raise TypeError(
                "safety_check_enabled는 true 또는 false여야 합니다."
            )
        execution_metadata = CatVTONExecutionMetadata(
            mask_source=str(metadata_payload["mask_source"]),
            automasker_run_count=int(
                metadata_payload["automasker_run_count"]
            ),
            approved_image_width=int(
                metadata_payload["approved_image_width"]
            ),
            approved_image_height=int(
                metadata_payload["approved_image_height"]
            ),
            approved_mask_pixel_count=int(
                metadata_payload["approved_mask_pixel_count"]
            ),
            processed_mask_pixel_count=int(
                metadata_payload["processed_mask_pixel_count"]
            ),
            safety_check_enabled=safety_check_enabled,
            person_input_source=str(
                metadata_payload["person_input_source"]
            ),
            person_input_width=int(metadata_payload["person_input_width"]),
            person_input_height=int(metadata_payload["person_input_height"]),
            clothing_source_width=int(
                metadata_payload["clothing_source_width"]
            ),
            clothing_source_height=int(
                metadata_payload["clothing_source_height"]
            ),
            clothing_input_width=int(
                metadata_payload["clothing_input_width"]
            ),
            clothing_input_height=int(
                metadata_payload["clothing_input_height"]
            ),
            clothing_alpha_pixel_count=int(
                metadata_payload["clothing_alpha_pixel_count"]
            ),
            clothing_alpha_coverage_percent=float(
                metadata_payload["clothing_alpha_coverage_percent"]
            ),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise CharacterClothingProtectionError(
            f"CatVTON 실행 증명 기록을 읽지 못했습니다: {error}"
        ) from error

    expected_size = approved_input.human_agnostic_image.size
    if execution_metadata.person_input_source != "generated_candidate":
        raise CharacterClothingProtectionError(
            "CatVTON 인물 입력이 실제 생성 후보가 아닙니다."
        )
    clothing_input_pixel_count = (
        execution_metadata.clothing_input_width
        * execution_metadata.clothing_input_height
    )
    if (
        execution_metadata.clothing_source_width
        < execution_metadata.clothing_input_width
        or execution_metadata.clothing_source_height
        < execution_metadata.clothing_input_height
        or clothing_input_pixel_count <= 0
    ):
        raise CharacterClothingProtectionError(
            "CatVTON 의상 조건 이미지 크기 기록이 잘못됐습니다."
        )
    if not (
        0
        < execution_metadata.clothing_alpha_pixel_count
        <= clothing_input_pixel_count
    ):
        raise CharacterClothingProtectionError(
            "CatVTON 의상 알파 픽셀 기록이 조건 이미지 범위를 벗어났습니다."
        )
    measured_coverage_percent = (
        execution_metadata.clothing_alpha_pixel_count
        / clothing_input_pixel_count
        * 100.0
    )
    if abs(
        measured_coverage_percent
        - execution_metadata.clothing_alpha_coverage_percent
    ) > 0.001:
        raise CharacterClothingProtectionError(
            "CatVTON 의상 알파 점유율 기록이 실제 수치와 다릅니다."
        )
    if (
        execution_metadata.person_input_width,
        execution_metadata.person_input_height,
    ) != expected_size:
        raise CharacterClothingProtectionError(
            "CatVTON 인물 입력 크기 기록이 승인 좌표와 다릅니다."
        )
    if execution_metadata.mask_source != "user_approved":
        raise CharacterClothingProtectionError(
            "CatVTON이 사용자 승인 마스크를 사용하지 않았습니다."
        )
    if execution_metadata.automasker_run_count != 0:
        raise CharacterClothingProtectionError(
            "CatVTON 내부 AutoMasker가 다시 실행됐습니다."
        )
    if (
        execution_metadata.approved_image_width,
        execution_metadata.approved_image_height,
    ) != expected_size:
        raise CharacterClothingProtectionError(
            "CatVTON 승인 이미지 크기 기록이 실제 입력과 다릅니다."
        )
    if (
        execution_metadata.approved_mask_pixel_count
        != approved_input.approved_mask_pixel_count
    ):
        raise CharacterClothingProtectionError(
            "CatVTON 승인 마스크 픽셀 기록이 실제 입력과 다릅니다."
        )
    if (
        execution_metadata.safety_check_enabled
        is not expected_safety_check_enabled
    ):
        raise CharacterClothingProtectionError(
            "CatVTON 안전 검사 실행 기록이 요청 설정과 다릅니다."
        )
    return execution_metadata


def validate_catvton_local_settings(
    settings: CatVTONLocalSettings,
) -> None:
    """CatVTON 별도 실행에 필요한 경로와 값이 준비됐는지 검사한다."""
    if not settings.python_executable.is_file():
        raise CharacterClothingProtectionError(
            "CatVTON 전용 Python이 없습니다: "
            f"{settings.python_executable}"
        )
    if not (settings.repository_path / "model" / "pipeline.py").is_file():
        raise CharacterClothingProtectionError(
            "CatVTON 저장소를 찾을 수 없습니다: "
            f"{settings.repository_path}"
        )
    if not settings.runner_path.is_file():
        raise CharacterClothingProtectionError(
            f"CatVTON 연결 실행 파일이 없습니다: {settings.runner_path}"
        )
    if settings.width < 256 or settings.height < 256:
        raise CharacterClothingProtectionError(
            "CatVTON 처리 크기는 가로와 세로 모두 256 이상이어야 합니다."
        )
    if settings.width % 8 or settings.height % 8:
        raise CharacterClothingProtectionError(
            "CatVTON 처리 크기는 8의 배수여야 합니다."
        )
    if settings.inference_steps < 1:
        raise CharacterClothingProtectionError(
            "CatVTON 반복 횟수는 1 이상이어야 합니다."
        )
    if settings.guidance_scale <= 0:
        raise CharacterClothingProtectionError(
            "CatVTON 문장 반영 강도는 0보다 커야 합니다."
        )
    if settings.mixed_precision not in ("fp16", "bf16"):
        raise CharacterClothingProtectionError(
            "CatVTON 계산 형식은 fp16 또는 bf16이어야 합니다."
        )
    if settings.timeout_seconds < 60:
        raise CharacterClothingProtectionError(
            "CatVTON 제한 시간은 60초 이상이어야 합니다."
        )
    calculate_mask_expansion_radius(
        (settings.width, settings.height),
        settings.mask_expansion_ratio,
        settings.minimum_mask_expansion_pixels,
        settings.maximum_mask_expansion_pixels,
    )
    if not 1 <= settings.mask_closing_radius_pixels <= 3:
        raise CharacterClothingProtectionError(
            "CatVTON 마스크 닫기 반경은 1~3픽셀이어야 합니다."
        )


def find_catvton_clothing_type(
    clothing_category: ClothingCategory,
) -> str:
    """프로젝트 의상 종류를 CatVTON이 받는 종류로 변환한다."""
    category_mapping = {
        ClothingCategory.TOP: "upper",
        ClothingCategory.BOTTOM: "lower",
        ClothingCategory.DRESS: "overall",
        ClothingCategory.FULL_BODY_OUTFIT: "overall",
    }
    catvton_clothing_type = category_mapping.get(clothing_category)
    if catvton_clothing_type is None:
        raise CharacterClothingProtectionError(
            "현재 CatVTON 연결은 상의, 하의, 드레스와 전신 의상만 지원합니다."
        )
    return catvton_clothing_type

