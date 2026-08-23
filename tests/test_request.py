from pathlib import Path

import pytest
from PIL import Image

from genai_lab.request import (
    CharacterFramingType,
    CharacterGenerationInput,
    CharacterGenerationPreparationError,
    CharacterGenerationSettings,
    prepare_character_generation_request,
)


def character_generation_settings() -> CharacterGenerationSettings:
    """테스트에서 공통으로 사용하는 정상 설정을 반환한다."""
    return CharacterGenerationSettings(
        model_id="cagliostrolab/animagine-xl-3.1",
        reference_adapter_id=(
            "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
        ),
        inference_steps=20,
        guidance_scale=5.5,
        reference_image_strength=0.55,
        default_negative_prompt="low quality",
    )


def create_reference_image(reference_image_path: Path) -> None:
    """요청 변환 테스트에 사용할 작은 RGB 이미지를 만든다."""
    Image.new("RGB", (16, 16), "white").save(reference_image_path)


def test_full_body_input_becomes_model_ready_request(
    tmp_path: Path,
) -> None:
    reference_image_path = tmp_path / "character.png"
    create_reference_image(reference_image_path)

    generation_request = prepare_character_generation_request(
        CharacterGenerationInput(
            reference_image_path=reference_image_path,
            framing_type=CharacterFramingType.FULL_BODY,
        ),
        character_generation_settings(),
        candidate_number=1,
        seed=1234,
    )

    assert generation_request.reference_image.mode == "RGB"
    assert generation_request.reference_image_name == "character.png"
    assert (generation_request.width, generation_request.height) == (576, 896)
    assert generation_request.seed == 1234
    assert "head to toe" in generation_request.prompt
    assert "feet visible" in generation_request.prompt
    assert "cropped" in generation_request.negative_prompt
    assert "feet out of frame" in generation_request.negative_prompt


def test_missing_reference_image_is_rejected(tmp_path: Path) -> None:
    missing_image_path = tmp_path / "missing.png"

    with pytest.raises(
        CharacterGenerationPreparationError,
        match="참조 이미지가 없습니다",
    ):
        prepare_character_generation_request(
            CharacterGenerationInput(
                reference_image_path=missing_image_path,
                framing_type=CharacterFramingType.FACE,
            ),
            character_generation_settings(),
            candidate_number=1,
            seed=1234,
        )


def test_candidate_number_outside_one_to_three_is_rejected(
    tmp_path: Path,
) -> None:
    reference_image_path = tmp_path / "character.png"
    create_reference_image(reference_image_path)

    with pytest.raises(
        CharacterGenerationPreparationError,
        match="1번부터 3번",
    ):
        prepare_character_generation_request(
            CharacterGenerationInput(
                reference_image_path=reference_image_path,
                framing_type=CharacterFramingType.UPPER_BODY,
            ),
            character_generation_settings(),
            candidate_number=4,
            seed=1234,
        )
