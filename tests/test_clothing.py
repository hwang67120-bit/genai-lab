from dataclasses import dataclass

import pytest
from PIL import Image, ImageDraw

from genai_lab.clothing import (
    CharacterClothingProtectionError,
    CharacterClothingTryOnRequest,
    ClothingCategory,
    ClothingReferenceInput,
    apply_protected_clothing_try_on,
    create_character_try_on_protection_plan,
    find_catvton_clothing_type,
    load_clothing_reference_image,
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
