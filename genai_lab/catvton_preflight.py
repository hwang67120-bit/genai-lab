"""CatVTON 추론 전에 실제 모델 입력 변환 결과를 공개하고 검증한다."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, UnidentifiedImageError

from genai_lab.image_digest import calculate_image_pixel_sha256


@dataclass(frozen=True)
class CatVTONPreflightSettings:
    """CatVTON 공식 전처리를 별도 환경에서 실행하는 설정."""

    python_executable: Path
    repository_path: Path
    runner_path: Path
    temporary_root: Path
    cache_dir: Path
    width: int
    height: int
    timeout_seconds: int
    mask_blur_factor: int = 9


@dataclass(frozen=True)
class CatVTONInputSnapshot:
    """CatVTON 전처리 전에 GUI가 그대로 공개하는 입력 5개와 수치."""

    person_image: Image.Image
    approved_change_mask: Image.Image
    clothing_condition_image: Image.Image
    identity_protection_mask: Image.Image
    expanded_foreground_mask: Image.Image
    person_sha256: str
    change_mask_sha256: str
    clothing_sha256: str
    change_mask_pixel_count: int
    protected_overlap_pixel_count: int
    outside_foreground_pixel_count: int
    passed: bool
    reason_ko: str

    def close(self) -> None:
        """입력 검토 화면이 소유한 이미지 5개를 해제한다."""
        self.person_image.close()
        self.approved_change_mask.close()
        self.clothing_condition_image.close()
        self.identity_protection_mask.close()
        self.expanded_foreground_mask.close()


@dataclass(frozen=True)
class CatVTONPreflightCandidate:
    """사용자 승인 전 실제 CatVTON 모델 입력과 측정 결과."""

    processed_person_image: Image.Image
    processed_binary_mask: Image.Image
    raw_blurred_mask_image: Image.Image
    model_mask_image: Image.Image
    processed_clothing_image: Image.Image
    soft_overlap_mask: Image.Image
    hard_overlap_mask: Image.Image
    protected_overlap_mask: Image.Image
    outside_foreground_mask: Image.Image
    processed_mask_pixel_count: int
    model_mask_pixel_count: int
    soft_overlap_pixel_count: int
    hard_overlap_pixel_count: int
    removed_pixel_count: int
    protected_overlap_pixel_count: int
    outside_foreground_pixel_count: int
    person_sha256: str
    binary_mask_sha256: str
    model_mask_sha256: str
    clothing_sha256: str
    width: int
    height: int
    blur_factor: int
    passed: bool
    reason_ko: str

    def close(self) -> None:
        """승인창에서 사용한 모델 입력·진단 이미지 9개를 해제한다."""
        self.processed_person_image.close()
        self.processed_binary_mask.close()
        self.raw_blurred_mask_image.close()
        self.model_mask_image.close()
        self.processed_clothing_image.close()
        self.soft_overlap_mask.close()
        self.hard_overlap_mask.close()
        self.protected_overlap_mask.close()
        self.outside_foreground_mask.close()


class CatVTONPreflightError(RuntimeError):
    """CatVTON 입력 전처리 또는 실행 전 검증을 완료하지 못한 오류."""


def create_catvton_input_snapshot(
    person_image: Image.Image,
    approved_change_mask: Image.Image,
    clothing_condition_image: Image.Image,
    identity_protection_mask: Image.Image,
    expanded_foreground_mask: Image.Image,
) -> CatVTONInputSnapshot:
    """전처리 전 입력 좌표와 변경 허용 영역을 수치로 검증하고 복사한다."""
    expected_size = person_image.size
    coordinate_images = {
        "변경 마스크": approved_change_mask,
        "보호 마스크": identity_protection_mask,
        "캐릭터 외곽": expanded_foreground_mask,
    }
    for input_name, input_image in coordinate_images.items():
        if input_image.size != expected_size:
            raise CatVTONPreflightError(
                f"{input_name} 좌표 불일치: 캐릭터={expected_size}, "
                f"{input_name}={input_image.size}"
            )

    normalized_change_mask = approved_change_mask.convert("L")
    normalized_protection_mask = identity_protection_mask.convert("L")
    normalized_foreground_mask = expanded_foreground_mask.convert("L")
    try:
        change_array = (
            np.asarray(normalized_change_mask, dtype=np.uint8) >= 128
        )
        protection_array = (
            np.asarray(normalized_protection_mask, dtype=np.uint8) >= 128
        )
        foreground_array = (
            np.asarray(normalized_foreground_mask, dtype=np.uint8) >= 128
        )
        change_mask_pixel_count = int(np.count_nonzero(change_array))
        protected_overlap_pixel_count = int(
            np.count_nonzero(change_array & protection_array)
        )
        outside_foreground_pixel_count = int(
            np.count_nonzero(change_array & ~foreground_array)
        )
    finally:
        normalized_change_mask.close()
        normalized_protection_mask.close()
        normalized_foreground_mask.close()

    passed = (
        change_mask_pixel_count > 0
        and protected_overlap_pixel_count == 0
        and outside_foreground_pixel_count == 0
    )
    if change_mask_pixel_count == 0:
        reason_ko = "변경 마스크가 0픽셀입니다."
    elif protected_overlap_pixel_count != 0:
        reason_ko = (
            "변경 마스크가 보호 영역 "
            f"{protected_overlap_pixel_count:,}픽셀을 침범했습니다."
        )
    elif outside_foreground_pixel_count != 0:
        reason_ko = (
            "변경 마스크가 캐릭터 외곽 밖 "
            f"{outside_foreground_pixel_count:,}픽셀을 침범했습니다."
        )
    else:
        reason_ko = "입력 좌표와 변경 영역 검증을 통과했습니다."

    return CatVTONInputSnapshot(
        person_image=person_image.convert("RGB"),
        approved_change_mask=approved_change_mask.convert("L"),
        clothing_condition_image=clothing_condition_image.convert("RGB"),
        identity_protection_mask=identity_protection_mask.convert("L"),
        expanded_foreground_mask=expanded_foreground_mask.convert("L"),
        person_sha256=calculate_image_pixel_sha256(person_image, "RGB"),
        change_mask_sha256=calculate_image_pixel_sha256(
            approved_change_mask,
            "L",
        ),
        clothing_sha256=calculate_image_pixel_sha256(
            clothing_condition_image,
            "RGB",
        ),
        change_mask_pixel_count=change_mask_pixel_count,
        protected_overlap_pixel_count=protected_overlap_pixel_count,
        outside_foreground_pixel_count=outside_foreground_pixel_count,
        passed=passed,
        reason_ko=reason_ko,
    )


def execute_catvton_preflight(
    person_image: Image.Image,
    approved_change_mask: Image.Image,
    clothing_condition_image: Image.Image,
    identity_protection_mask: Image.Image,
    expanded_foreground_mask: Image.Image,
    settings: CatVTONPreflightSettings,
) -> CatVTONPreflightCandidate:
    """공식 CatVTON 변환 함수를 실행하되 GPU 추론은 실행하지 않는다."""
    expected_size = person_image.size
    input_sizes = (
        approved_change_mask.size,
        identity_protection_mask.size,
        expanded_foreground_mask.size,
    )
    if any(input_size != expected_size for input_size in input_sizes):
        raise CatVTONPreflightError(
            "CatVTON Preflight의 인물·마스크·보호 영역 좌표가 다릅니다: "
            f"인물={expected_size}, 나머지={input_sizes}"
        )
    if settings.width < 256 or settings.height < 256:
        raise CatVTONPreflightError(
            "CatVTON Preflight 처리 크기는 가로·세로 256픽셀 이상이어야 합니다."
        )
    if settings.width % 8 != 0 or settings.height % 8 != 0:
        raise CatVTONPreflightError(
            "CatVTON Preflight 처리 크기는 8의 배수여야 합니다."
        )
    if settings.mask_blur_factor < 0:
        raise CatVTONPreflightError("마스크 블러 값은 0 이상이어야 합니다.")

    settings.temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="genai-lab-catvton-preflight-",
        dir=settings.temporary_root,
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        input_paths = {
            "person": temporary_directory / "person.png",
            "mask": temporary_directory / "approved_mask.png",
            "clothing": temporary_directory / "clothing.png",
            "protection": temporary_directory / "protection.png",
            "foreground": temporary_directory / "foreground.png",
        }
        output_paths = {
            "person": temporary_directory / "processed_person.png",
            "binary_mask": temporary_directory / "processed_binary_mask.png",
            "raw_blurred_mask": temporary_directory / "raw_blurred_mask.png",
            "model_mask": temporary_directory / "model_mask.png",
            "clothing": temporary_directory / "processed_clothing.png",
            "soft_overlap": temporary_directory / "soft_overlap.png",
            "hard_overlap": temporary_directory / "hard_overlap.png",
            "protected_overlap": temporary_directory / "protected_overlap.png",
            "outside_foreground": temporary_directory / "outside_foreground.png",
            "metadata": temporary_directory / "metadata.json",
        }

        inputs = (
            (person_image, "RGB", input_paths["person"]),
            (approved_change_mask, "L", input_paths["mask"]),
            (clothing_condition_image, "RGB", input_paths["clothing"]),
            (identity_protection_mask, "L", input_paths["protection"]),
            (expanded_foreground_mask, "L", input_paths["foreground"]),
        )
        for input_image, mode, input_path in inputs:
            normalized_image = input_image.convert(mode)
            try:
                normalized_image.save(input_path, format="PNG")
            finally:
                normalized_image.close()

        command = [
            str(settings.python_executable),
            str(settings.runner_path),
            "--repository-path", str(settings.repository_path),
            "--person-image", str(input_paths["person"]),
            "--approved-change-mask", str(input_paths["mask"]),
            "--clothing-image", str(input_paths["clothing"]),
            "--identity-protection-mask", str(input_paths["protection"]),
            "--expanded-foreground-mask", str(input_paths["foreground"]),
            "--output-processed-person", str(output_paths["person"]),
            "--output-binary-mask", str(output_paths["binary_mask"]),
            "--output-raw-blurred-mask", str(output_paths["raw_blurred_mask"]),
            "--output-model-mask", str(output_paths["model_mask"]),
            "--output-processed-clothing", str(output_paths["clothing"]),
            "--output-soft-overlap", str(output_paths["soft_overlap"]),
            "--output-hard-overlap", str(output_paths["hard_overlap"]),
            "--output-protected-overlap", str(output_paths["protected_overlap"]),
            "--output-outside-foreground", str(output_paths["outside_foreground"]),
            "--output-metadata", str(output_paths["metadata"]),
            "--cache-dir", str(settings.cache_dir),
            "--width", str(settings.width),
            "--height", str(settings.height),
            "--mask-blur-factor", str(settings.mask_blur_factor),
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
            raise CatVTONPreflightError(
                f"CatVTON Preflight 실행을 시작하지 못했습니다: {error}"
            ) from error
        if completed_process.returncode != 0:
            details = (
                completed_process.stderr.strip()
                or completed_process.stdout.strip()
                or "출력 없음"
            )
            raise CatVTONPreflightError(
                "CatVTON Preflight 실행에 실패했습니다: " f"{details}"
            )
        missing_outputs = tuple(
            output_path.name
            for output_path in output_paths.values()
            if not output_path.is_file()
        )
        if missing_outputs:
            raise CatVTONPreflightError(
                "CatVTON Preflight 출력 파일이 없습니다: " f"{missing_outputs}"
            )

        try:
            opened_images = {}
            for output_name, mode in (
                ("person", "RGB"),
                ("binary_mask", "L"),
                ("raw_blurred_mask", "L"),
                ("model_mask", "L"),
                ("clothing", "RGB"),
                ("soft_overlap", "L"),
                ("hard_overlap", "L"),
                ("protected_overlap", "L"),
                ("outside_foreground", "L"),
            ):
                with Image.open(output_paths[output_name]) as opened_image:
                    opened_images[output_name] = opened_image.convert(mode)
            metadata = json.loads(
                output_paths["metadata"].read_text(encoding="utf-8")
            )
            return CatVTONPreflightCandidate(
                processed_person_image=opened_images["person"],
                processed_binary_mask=opened_images["binary_mask"],
                raw_blurred_mask_image=opened_images["raw_blurred_mask"],
                model_mask_image=opened_images["model_mask"],
                processed_clothing_image=opened_images["clothing"],
                soft_overlap_mask=opened_images["soft_overlap"],
                hard_overlap_mask=opened_images["hard_overlap"],
                protected_overlap_mask=opened_images["protected_overlap"],
                outside_foreground_mask=opened_images["outside_foreground"],
                processed_mask_pixel_count=int(metadata["processed_mask_pixel_count"]),
                model_mask_pixel_count=int(metadata["model_mask_pixel_count"]),
                soft_overlap_pixel_count=int(metadata["soft_overlap_pixel_count"]),
                hard_overlap_pixel_count=int(metadata["hard_overlap_pixel_count"]),
                removed_pixel_count=int(metadata["removed_pixel_count"]),
                protected_overlap_pixel_count=int(metadata["protected_overlap_pixel_count"]),
                outside_foreground_pixel_count=int(metadata["outside_foreground_pixel_count"]),
                person_sha256=str(metadata["person_sha256"]),
                binary_mask_sha256=str(metadata["binary_mask_sha256"]),
                model_mask_sha256=str(metadata["model_mask_sha256"]),
                clothing_sha256=str(metadata["clothing_sha256"]),
                width=int(metadata["width"]),
                height=int(metadata["height"]),
                blur_factor=int(metadata["blur_factor"]),
                passed=bool(metadata["passed"]),
                reason_ko=str(metadata["reason_ko"]),
            )
        except (OSError, UnidentifiedImageError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            for opened_image in locals().get("opened_images", {}).values():
                opened_image.close()
            raise CatVTONPreflightError(
                f"CatVTON Preflight 결과를 읽을 수 없습니다: {error}"
            ) from error
