from dataclasses import dataclass
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from genai_lab.clothing import (
    CharacterAgnosticApprovedInput,
    CharacterClothingProtectionError,
    CharacterClothingTryOnRequest,
    ClothingCategory,
    ClothingReferenceInput,
    apply_protected_clothing_try_on,
    create_character_try_on_protection_plan,
    find_catvton_clothing_type,
    load_clothing_reference_image,
    load_catvton_execution_metadata,
    prepare_catvton_clothing_condition_image,
    validate_character_agnostic_approved_input,
    validate_catvton_approved_coordinates,
    validate_runner_mask_matches_approved_input,
)


@dataclass(frozen=True)
class FixedTryOnEngine:
    """테스트에서 고정된 의상 합성 원본을 반환한다."""

    output_image: Image.Image

    def generate_clothing_try_on_image(
        self,
        request: CharacterClothingTryOnRequest,
    ) -> Image.Image:
        return self.output_image.copy()


def create_mask(
    size: tuple[int, int],
    rectangle: tuple[int, int, int, int],
) -> Image.Image:
    mask_image = Image.new("L", size, 0)
    ImageDraw.Draw(mask_image).rectangle(rectangle, fill=255)
    return mask_image


def test_approved_agnostic_input_accepts_matching_mask_contract() -> None:
    approved_mask = create_mask((8, 8), (2, 2, 5, 5))
    approved_input = CharacterAgnosticApprovedInput(
        human_agnostic_image=Image.new("RGB", (8, 8), (127, 127, 127)),
        approved_change_mask=approved_mask,
        clothing_type="upper",
        approved_mask_pixel_count=16,
    )

    validate_character_agnostic_approved_input(approved_input, "upper")


def test_approved_agnostic_input_rejects_recorded_pixel_mismatch() -> None:
    approved_input = CharacterAgnosticApprovedInput(
        human_agnostic_image=Image.new("RGB", (8, 8), (127, 127, 127)),
        approved_change_mask=create_mask((8, 8), (2, 2, 5, 5)),
        clothing_type="upper",
        approved_mask_pixel_count=15,
    )

    with pytest.raises(
        CharacterClothingProtectionError,
        match="픽셀 수",
    ):
        validate_character_agnostic_approved_input(approved_input, "upper")


def test_runner_mask_must_equal_user_approved_mask() -> None:
    approved_mask = create_mask((8, 8), (2, 2, 5, 5))
    changed_runner_mask = create_mask((8, 8), (1, 1, 5, 5))

    with pytest.raises(
        CharacterClothingProtectionError,
        match="승인 마스크와 다릅니다",
    ):
        validate_runner_mask_matches_approved_input(
            changed_runner_mask,
            approved_mask,
        )


def test_catvton_metadata_confirms_disabled_safety_check(tmp_path) -> None:
    metadata_path = tmp_path / "execution_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "mask_source": "user_approved",
                "automasker_run_count": 0,
                "approved_image_width": 8,
                "approved_image_height": 8,
                "approved_mask_pixel_count": 16,
                "processed_mask_pixel_count": 64,
                "safety_check_enabled": False,
                "person_input_source": "generated_candidate",
                "person_input_width": 8,
                "person_input_height": 8,
                "clothing_source_width": 10,
                "clothing_source_height": 12,
                "clothing_input_width": 5,
                "clothing_input_height": 8,
                "clothing_alpha_pixel_count": 20,
                "clothing_alpha_coverage_percent": 50.0,
            }
        ),
        encoding="utf-8",
    )
    approved_input = CharacterAgnosticApprovedInput(
        human_agnostic_image=Image.new("RGB", (8, 8), "gray"),
        approved_change_mask=create_mask((8, 8), (2, 2, 5, 5)),
        clothing_type="upper",
        approved_mask_pixel_count=16,
    )

    execution_metadata = load_catvton_execution_metadata(
        metadata_path,
        approved_input,
        expected_safety_check_enabled=False,
    )

    assert execution_metadata.safety_check_enabled is False
    assert execution_metadata.person_input_source == "generated_candidate"


def test_catvton_metadata_rejects_unexpected_safety_check_state(
    tmp_path,
) -> None:
    metadata_path = tmp_path / "execution_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "mask_source": "user_approved",
                "automasker_run_count": 0,
                "approved_image_width": 8,
                "approved_image_height": 8,
                "approved_mask_pixel_count": 16,
                "processed_mask_pixel_count": 64,
                "safety_check_enabled": True,
                "person_input_source": "generated_candidate",
                "person_input_width": 8,
                "person_input_height": 8,
                "clothing_source_width": 10,
                "clothing_source_height": 12,
                "clothing_input_width": 5,
                "clothing_input_height": 8,
                "clothing_alpha_pixel_count": 20,
                "clothing_alpha_coverage_percent": 50.0,
            }
        ),
        encoding="utf-8",
    )
    approved_input = CharacterAgnosticApprovedInput(
        human_agnostic_image=Image.new("RGB", (8, 8), "gray"),
        approved_change_mask=create_mask((8, 8), (2, 2, 5, 5)),
        clothing_type="upper",
        approved_mask_pixel_count=16,
    )

    with pytest.raises(
        CharacterClothingProtectionError,
        match="안전 검사 실행 기록",
    ):
        load_catvton_execution_metadata(
            metadata_path,
            approved_input,
            expected_safety_check_enabled=False,
        )


def test_clothing_try_on_changes_only_allowed_clothing_pixels() -> None:
    base_character_image = Image.new("RGB", (8, 8), "red")
    raw_try_on_image = Image.new("RGB", (8, 8), "blue")
    clothing_change_mask = create_mask((8, 8), (1, 1, 6, 6))
    identity_protection_mask = create_mask((8, 8), (3, 3, 4, 4))
    protection_plan = create_character_try_on_protection_plan(
        clothing_change_mask,
        identity_protection_mask,
        boundary_blur_radius=0,
    )

    candidate = apply_protected_clothing_try_on(
        FixedTryOnEngine(raw_try_on_image),
        CharacterClothingTryOnRequest(
            base_character_image=base_character_image,
            clothing_reference_image=Image.new("RGB", (4, 4), "white"),
            clothing_category=ClothingCategory.DRESS,
        ),
        protection_plan,
    )

    assert candidate.verification.passed is True
    assert candidate.image.getpixel((1, 1)) == (0, 0, 255)
    assert candidate.image.getpixel((3, 3)) == (255, 0, 0)
    assert candidate.image.getpixel((0, 0)) == (255, 0, 0)


def test_clothing_try_on_rejects_different_output_size() -> None:
    base_character_image = Image.new("RGB", (8, 8), "red")
    protection_plan = create_character_try_on_protection_plan(
        create_mask((8, 8), (1, 1, 6, 6)),
        Image.new("L", (8, 8), 0),
        boundary_blur_radius=0,
    )

    with pytest.raises(
        CharacterClothingProtectionError,
        match="크기가 다릅니다",
    ):
        apply_protected_clothing_try_on(
            FixedTryOnEngine(Image.new("RGB", (4, 4), "blue")),
            CharacterClothingTryOnRequest(
                base_character_image=base_character_image,
                clothing_reference_image=Image.new("RGB", (4, 4), "white"),
                clothing_category=ClothingCategory.TOP,
            ),
            protection_plan,
        )



@pytest.mark.parametrize(
    ("clothing_category", "expected_catvton_type"),
    [
        (ClothingCategory.TOP, "upper"),
        (ClothingCategory.BOTTOM, "lower"),
        (ClothingCategory.DRESS, "overall"),
        (ClothingCategory.FULL_BODY_OUTFIT, "overall"),
    ],
)
def test_clothing_category_maps_to_catvton_type(
    clothing_category: ClothingCategory,
    expected_catvton_type: str,
) -> None:
    assert (
        find_catvton_clothing_type(clothing_category)
        == expected_catvton_type
    )


def test_unsupported_clothing_category_is_rejected() -> None:
    with pytest.raises(
        CharacterClothingProtectionError,
        match="상의, 하의",
    ):
        find_catvton_clothing_type(ClothingCategory.GLOVES)


def test_loaded_clothing_reference_remains_open_after_file_is_closed(
    tmp_path,
) -> None:
    clothing_path = tmp_path / "clothing.png"
    Image.new("RGB", (8, 8), "white").save(clothing_path)

    clothing_reference_image = load_clothing_reference_image(
        ClothingReferenceInput(
            image_path=clothing_path,
            category=ClothingCategory.TOP,
        )
    )

    assert clothing_reference_image.getpixel((0, 0)) == (255, 255, 255)
    clothing_reference_image.save(tmp_path / "copied.png")


def test_clothing_reference_uses_only_approved_region(tmp_path) -> None:
    clothing_path = tmp_path / "clothing-region.png"
    source_image = Image.new("RGB", (10, 8), "red")
    ImageDraw.Draw(source_image).rectangle((2, 1, 7, 5), fill="blue")
    source_image.save(clothing_path)
    source_image.close()

    clothing_reference_image = load_clothing_reference_image(
        ClothingReferenceInput(
            image_path=clothing_path,
            category=ClothingCategory.TOP,
            region_box_xyxy=(2, 1, 8, 6),
        )
    )

    try:
        assert clothing_reference_image.size == (6, 5)
        assert clothing_reference_image.getpixel((0, 0)) == (0, 0, 255)
    finally:
        clothing_reference_image.close()


def test_catvton_condition_crops_transparent_approved_margin() -> None:
    approved_image = Image.new("RGBA", (12, 10), (0, 0, 0, 0))
    ImageDraw.Draw(approved_image).rectangle(
        (3, 2, 8, 7), fill=(10, 20, 30, 255)
    )
    condition = prepare_catvton_clothing_condition_image(
        ClothingReferenceInput(
            image_path=Path("unused.png"),
            category=ClothingCategory.TOP,
            approved_image=approved_image,
        )
    )

    try:
        assert condition.source_size == (12, 10)
        assert condition.crop_box_xyxy == (3, 2, 9, 8)
        assert condition.image.size == (6, 6)
        assert condition.alpha_pixel_count == 36
        assert condition.alpha_coverage_percent == 100.0
        assert condition.image.getpixel((0, 0)) == (10, 20, 30)
    finally:
        condition.image.close()
        approved_image.close()


def test_catvton_condition_rejects_empty_approved_alpha() -> None:
    approved_image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    try:
        with pytest.raises(
            CharacterClothingProtectionError,
            match="알파 픽셀이 0개",
        ):
            prepare_catvton_clothing_condition_image(
                ClothingReferenceInput(
                    image_path=Path("unused.png"),
                    category=ClothingCategory.TOP,
                    approved_image=approved_image,
                )
            )
    finally:
        approved_image.close()


def test_catvton_coordinates_accept_same_generated_candidate_size() -> None:
    base_image = Image.new("RGB", (8, 12), "white")
    approved_input = CharacterAgnosticApprovedInput(
        human_agnostic_image=Image.new("RGB", (8, 12), "gray"),
        approved_change_mask=Image.new("L", (8, 12), 255),
        clothing_type="overall",
        approved_mask_pixel_count=96,
    )
    try:
        validate_catvton_approved_coordinates(base_image, approved_input)
    finally:
        base_image.close()
        approved_input.close()


def test_catvton_coordinates_reject_reference_sized_mask() -> None:
    base_image = Image.new("RGB", (8, 12), "white")
    approved_input = CharacterAgnosticApprovedInput(
        human_agnostic_image=Image.new("RGB", (10, 16), "gray"),
        approved_change_mask=Image.new("L", (10, 16), 255),
        clothing_type="overall",
        approved_mask_pixel_count=160,
    )
    try:
        with pytest.raises(
            CharacterClothingProtectionError,
            match="Human-Agnostic 이미지 크기가 다릅니다",
        ):
            validate_catvton_approved_coordinates(base_image, approved_input)
    finally:
        base_image.close()
        approved_input.close()
