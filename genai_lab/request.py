"""사용자 입력을 AI 모델 실행 요청으로 변환한다."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from secrets import randbelow
from typing import Mapping

from PIL import Image, UnidentifiedImageError


class CharacterFramingType(str, Enum):
    """사용자가 선택할 수 있는 캐릭터 화면 범위."""

    FULL_BODY = "full_body"
    UPPER_BODY = "upper_body"
    FACE = "face"


@dataclass(frozen=True)
class CharacterGenerationInput:
    """GUI에서 받은 원본 입력."""

    reference_image_path: Path
    framing_type: CharacterFramingType


@dataclass(frozen=True)
class CharacterGenerationSettings:
    """YAML 설정에서 읽어 확정한 이미지 생성 설정."""

    model_id: str
    reference_adapter_id: str
    inference_steps: int
    guidance_scale: float
    reference_image_strength: float
    default_negative_prompt: str


@dataclass(frozen=True)
class CharacterFramingRule:
    """화면 범위 선택에 따라 적용할 크기와 프롬프트 규칙."""

    width: int
    height: int
    prompt: str
    negative_prompt: str


@dataclass(frozen=True)
class CharacterGenerationRequest:
    """검사를 마치고 AI 모델에 전달할 생성 요청."""

    reference_image: Image.Image
    reference_image_name: str
    framing_type: CharacterFramingType
    width: int
    height: int
    prompt: str
    negative_prompt: str
    seed: int
    candidate_number: int
    inference_steps: int
    guidance_scale: float
    reference_image_strength: float
    model_id: str
    reference_adapter_id: str


class CharacterGenerationPreparationError(ValueError):
    """생성 요청을 준비할 수 없을 때 발생하는 오류."""


CHARACTER_FRAMING_RULES: Mapping[
    CharacterFramingType,
    CharacterFramingRule,
] = {
    CharacterFramingType.FULL_BODY: CharacterFramingRule(
        width=576,
        height=896,
        prompt=(
            "full body, standing, head to toe, feet visible, "
            "entire character in frame, centered composition, long shot"
        ),
        negative_prompt=(
            "cropped, out of frame, close-up, upper body, cowboy shot, "
            "feet out of frame, head out of frame"
        ),
    ),
    CharacterFramingType.UPPER_BODY: CharacterFramingRule(
        width=768,
        height=768,
        prompt="upper body, waist up, centered composition, face visible",
        negative_prompt="full body, close-up, cropped head, out of frame",
    ),
    CharacterFramingType.FACE: CharacterFramingRule(
        width=768,
        height=768,
        prompt=(
            "portrait, close-up, face focus, "
            "head and shoulders, centered composition"
        ),
        negative_prompt=(
            "full body, upper body, wide shot, cropped face, out of frame"
        ),
    ),
}


def prepare_character_generation_request(
    character_generation_input: CharacterGenerationInput,
    character_generation_settings: CharacterGenerationSettings,
    candidate_number: int,
    seed: int | None = None,
) -> CharacterGenerationRequest:
    """GUI 입력을 검사하고 AI 모델 실행 요청으로 변환한다.

    반환값:
        모델이 이미지 한 장을 생성할 때 사용할 확정 요청.

    오류:
        참조 이미지가 없거나 설정값이 잘못되면 한글 오류를 발생시킨다.

    부수 효과:
        로컬 참조 이미지 파일을 읽지만 이미지 생성과 저장은 하지 않는다.
    """
    validate_candidate_number(candidate_number)
    validate_character_generation_settings(character_generation_settings)

    framing_rule = find_character_framing_rule(
        character_generation_input.framing_type
    )
    reference_image = load_reference_image_as_rgb(
        character_generation_input.reference_image_path
    )
    generation_seed = seed if seed is not None else randbelow(2**31)
    validate_generation_seed(generation_seed)

    positive_prompt = ", ".join(
        (
            "1girl",
            "solo",
            "masterpiece",
            "best quality",
            framing_rule.prompt,
            "white background",
            "simple background",
        )
    )
    negative_prompt = join_negative_prompts(
        character_generation_settings.default_negative_prompt,
        framing_rule.negative_prompt,
    )

    # CharacterGenerationRequest(캐릭터 생성 확정 요청)
    # - 포함: RGB 참조 이미지, 화면 범위, 크기, 프롬프트와 모델 설정.
    # - 생성: GUI 입력과 YAML 설정을 검사한 뒤 만든다.
    # - 처리: 규칙으로만 만들며 AI 판단이나 외부 API 호출은 없다.
    # - 저장: 이 객체 자체는 저장하지 않는다.
    # - 다음 사용처: GUI 호환 구간을 거쳐 generator.py의 모델 실행에 사용한다.
    return CharacterGenerationRequest(
        reference_image=reference_image,
        reference_image_name=character_generation_input.reference_image_path.name,
        framing_type=character_generation_input.framing_type,
        width=framing_rule.width,
        height=framing_rule.height,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=generation_seed,
        candidate_number=candidate_number,
        inference_steps=character_generation_settings.inference_steps,
        guidance_scale=character_generation_settings.guidance_scale,
        reference_image_strength=(
            character_generation_settings.reference_image_strength
        ),
        model_id=character_generation_settings.model_id,
        reference_adapter_id=(
            character_generation_settings.reference_adapter_id
        ),
    )


def find_character_framing_rule(
    framing_type: CharacterFramingType,
) -> CharacterFramingRule:
    """화면 범위에 해당하는 크기와 프롬프트 규칙을 반환한다."""
    framing_rule = CHARACTER_FRAMING_RULES.get(framing_type)
    if framing_rule is None:
        raise CharacterGenerationPreparationError(
            f"지원하지 않는 화면 범위입니다: {framing_type}"
        )
    return framing_rule


def load_reference_image_as_rgb(reference_image_path: Path) -> Image.Image:
    """로컬 참조 이미지를 읽어 RGB 이미지로 반환한다."""
    if not reference_image_path.exists():
        raise CharacterGenerationPreparationError(
            f"참조 이미지가 없습니다: {reference_image_path}"
        )
    if not reference_image_path.is_file():
        raise CharacterGenerationPreparationError(
            f"참조 이미지 경로가 파일이 아닙니다: {reference_image_path}"
        )

    try:
        with Image.open(reference_image_path) as opened_image:
            opened_image.load()
            return opened_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise CharacterGenerationPreparationError(
            f"참조 이미지를 읽을 수 없습니다: {reference_image_path}"
        ) from error


def validate_candidate_number(candidate_number: int) -> None:
    """후보 번호가 1번부터 3번 사이인지 검사한다."""
    if candidate_number not in (1, 2, 3):
        raise CharacterGenerationPreparationError(
            "후보 번호는 1번부터 3번까지만 사용할 수 있습니다."
        )


def validate_generation_seed(seed: int) -> None:
    """재현에 사용하는 시드 범위를 검사한다."""
    if seed < 0 or seed >= 2**31:
        raise CharacterGenerationPreparationError(
            "시드는 0 이상 2의 31승 미만이어야 합니다."
        )


def validate_character_generation_settings(
    settings: CharacterGenerationSettings,
) -> None:
    """이미지 생성에 필요한 설정값을 규칙으로 검사한다."""
    if not settings.model_id.strip():
        raise CharacterGenerationPreparationError(
            "베이스 모델 이름이 비어 있습니다."
        )
    if not settings.reference_adapter_id.strip():
        raise CharacterGenerationPreparationError(
            "참조 이미지 적용 모델 이름이 비어 있습니다."
        )
    if settings.inference_steps < 1:
        raise CharacterGenerationPreparationError(
            "생성 반복 횟수는 1 이상이어야 합니다."
        )
    if settings.guidance_scale <= 0:
        raise CharacterGenerationPreparationError(
            "프롬프트 반영 강도는 0보다 커야 합니다."
        )
    if not 0.0 <= settings.reference_image_strength <= 1.0:
        raise CharacterGenerationPreparationError(
            "참조 이미지 반영 강도는 0부터 1 사이여야 합니다."
        )


def join_negative_prompts(
    default_negative_prompt: str,
    framing_negative_prompt: str,
) -> str:
    """기본 제외 조건과 화면 범위 제외 조건을 합친다."""
    negative_prompt_parts = (
        default_negative_prompt.strip(),
        framing_negative_prompt.strip(),
    )
    return ", ".join(
        prompt_part
        for prompt_part in negative_prompt_parts
        if prompt_part
    )
