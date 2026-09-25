"""Clean contracts between the Animagine Base and native refinement stages.

The character Base is generated from character evidence only. Garment design
evidence remains available for final validation, but it is not injected into
the Base prompt or Base IP-Adapter condition. FLUX receives an isolated garment board and a native natural-language
instruction instead of the Animagine tag prompt.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from genai_lab.clothing_reference_generation import BASE_GENDER_CONDITION_TAGS
from genai_lab.garment_topology import resolve_garment_topology, topology_guidance
from genai_lab.hair_structure import hair_structure_guidance, resolve_hair_structure
from genai_lab.hair_transfer_contract import build_hair_transfer_contract
from genai_lab.reference_tag_policy import (
    CHARACTER_GENDERS,
    normalize_tag,
    resolve_character_gender,
)
from genai_lab.request import CharacterFramingType, CharacterGenerationRequest
from scripts.generation_inputs import (
    _token_count,
    _tokenizer_limit,
    prepare_prompt_for_clip,
)


CONTRACT_VERSION = "native_pipeline_contract_v5"
BASE_PROFILE = "character_only"
GARMENT_SOURCE = "isolated_garment_board"
SDXL_LOCAL_REFINEMENT = "sdxl_local"
REFINEMENT_MODES = {SDXL_LOCAL_REFINEMENT}
OPTIONAL_CHARACTER_PART_LABELS = {
    "human_ears": "human ears",
    "animal_ears": "animal ears",
    "tail": "tail",
}


class NativePipelineConfigurationError(ValueError):
    """Native v2 설정이 부분 활성화되거나 계약과 다를 때 발생한다."""


def native_pipeline_enabled(config: Mapping[str, Any] | None) -> bool:
    """완전한 character-only 계약일 때만 Native v2를 활성화한다."""
    section = (config or {}).get("native_pipeline_v2", {})
    if section is None:
        return False
    if not isinstance(section, Mapping):
        raise NativePipelineConfigurationError(
            "native_pipeline_v2는 항목 묶음이어야 합니다."
        )
    enabled = section.get("enabled", False)
    if enabled is False:
        return False
    if enabled is not True:
        raise NativePipelineConfigurationError(
            "native_pipeline_v2.enabled는 true 또는 false여야 합니다."
        )
    required = {
        "base_profile": BASE_PROFILE,
        "base_garment_prompt_enabled": False,
        "base_garment_adapter_enabled": False,
        "native_garment_source": GARMENT_SOURCE,
        "reuse_animagine_prompt_in_native_stage": False,
        "raw_garment_person_image_allowed": False,
    }
    mismatches = [
        f"{key}={section.get(key)!r}"
        for key, expected in required.items()
        if section.get(key) != expected
    ]
    if mismatches:
        raise NativePipelineConfigurationError(
            "Native v2 character-only 계약이 불완전합니다: "
            + ", ".join(mismatches)
        )
    return True


def native_primary_route(config: Mapping[str, Any] | None) -> str:
    section = (config or {}).get("native_pipeline_v2", {})
    if not isinstance(section, Mapping):
        raise NativePipelineConfigurationError(
            "native_pipeline_v2는 항목 묶음이어야 합니다."
        )
    route = str(section.get("primary_route", "approved_animagine_base"))
    if route != "approved_animagine_base":
        raise NativePipelineConfigurationError(
            f"승인 Animagine Base만 지원하는 주 실행 경로입니다: {route}"
        )
    return route


def resolve_refinement_mode(config: Mapping[str, Any] | None) -> str | None:
    """Return the one sealed final-refinement mode for this request."""
    if not native_pipeline_enabled(config):
        return None
    section = (config or {}).get("refinement_execution", {})
    if not isinstance(section, Mapping):
        raise NativePipelineConfigurationError(
            "refinement_execution은 항목 묶음이어야 합니다."
        )
    if section.get("enabled") is not True:
        raise NativePipelineConfigurationError(
            "Native 파이프라인에는 refinement_execution.enabled=true가 필요합니다."
        )
    mode = section.get("mode")
    if mode not in REFINEMENT_MODES:
        raise NativePipelineConfigurationError(
            "refinement_execution.mode는 sdxl_local이어야 합니다."
        )
    return str(mode)


def final_refinement_enabled(config: Mapping[str, Any] | None) -> bool:
    return resolve_refinement_mode(config) is not None







def validate_native_pipeline_config(config: Mapping[str, Any]) -> None:
    """Validate one explicit finalizer without activating both engines."""
    native_active = native_pipeline_enabled(config)
    execution = config.get("refinement_execution", {})
    if execution is None:
        execution = {}
    if not isinstance(execution, Mapping):
        raise NativePipelineConfigurationError(
            "refinement_execution은 항목 묶음이어야 합니다."
        )
    execution_flag = execution.get("enabled", False)
    if execution_flag not in (True, False):
        raise NativePipelineConfigurationError(
            "refinement_execution.enabled는 true 또는 false여야 합니다."
        )
    if native_active != (execution_flag is True):
        raise NativePipelineConfigurationError(
            "native_pipeline_v2와 refinement_execution은 함께 활성화하거나 "
            "함께 비활성화해야 합니다."
        )
    if not native_active:
        return
    resolve_refinement_mode(config)
    if native_primary_route(config) != "approved_animagine_base":
        raise NativePipelineConfigurationError(
            "SDXL 국소 정밀화는 승인 Animagine Base 경로에서만 실행할 수 있습니다."
        )
    garment = config.get("staged_reference_generation", {}).get("garment", {})
    if not isinstance(garment, Mapping) or garment.get("enabled") is not True:
        raise NativePipelineConfigurationError(
            "정밀화 실행 계약이 불완전합니다: staged_reference_generation.garment.enabled"
        )


def _unique_tags(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        # Unspecified gender is absence of a condition, not the prompt "none".
        if value is None:
            continue
        for part in str(value).split(","):
            tag = normalize_tag(part)
            if tag and tag not in result:
                result.append(tag)
    return tuple(result)


def _framing_tags(framing: CharacterFramingType) -> tuple[str, ...]:
    return {
        CharacterFramingType.FULL_BODY: (
            "full body",
            "head to toe",
            "feet visible",
            "entire character inside frame",
        ),
        CharacterFramingType.UPPER_BODY: (
            "upper body",
            "head to waist",
            "entire head visible",
        ),
        CharacterFramingType.FACE: (
            "portrait",
            "head and shoulders",
            "entire head visible",
        ),
    }[framing]


def _build_character_prompt(
    tokenizers: Iterable[Any],
    required: Iterable[str],
    optional: Iterable[str],
) -> tuple[str, dict[str, Any]]:
    tokenizers = tuple(tokenizers)
    if not tokenizers or any(tokenizer is None for tokenizer in tokenizers):
        raise ValueError("Animagine Base 생성용 토크나이저가 필요합니다.")
    required_tags = _unique_tags(required)
    optional_tags = tuple(
        tag for tag in _unique_tags(optional) if tag not in required_tags
    )
    required_prompt = ", ".join(required_tags)
    limits = tuple(_tokenizer_limit(tokenizer) for tokenizer in tokenizers)

    def counts(text: str) -> tuple[int, ...]:
        return tuple(_token_count(tokenizer, text) for tokenizer in tokenizers)

    required_counts = counts(required_prompt)
    if any(count > limit for count, limit in zip(required_counts, limits)):
        raise ValueError(
            "필수 캐릭터 Base 조건이 토큰 한도를 초과했습니다. "
            f"필요={required_counts}, 한도={limits}"
        )
    kept: list[str] = []
    omitted: list[str] = []
    for tag in optional_tags:
        candidate = ", ".join((*required_tags, *kept, tag))
        if all(count <= limit for count, limit in zip(counts(candidate), limits)):
            kept.append(tag)
        else:
            omitted.append(tag)
    original = ", ".join((*required_tags, *optional_tags))
    effective = ", ".join((*required_tags, *kept))
    return effective, {
        "policy_version": CONTRACT_VERSION,
        "profile": BASE_PROFILE,
        "required_prompt": required_prompt,
        "required_token_counts": list(required_counts),
        "tokenizer_limits": list(limits),
        "original": original,
        "effective": effective,
        "original_token_counts": list(counts(original)),
        "effective_token_counts": list(counts(effective)),
        "omitted_optional": omitted,
        "truncated": bool(omitted),
        "garment_tags_included": False,
        "garment_adapter_included": False,
        "long_prompt": {
            "policy_version": "single_clip_context_v1",
            "chunk_count": 1,
        },
    }


def prepare_character_only_base_request(
    request: CharacterGenerationRequest,
    tokenizers: Iterable[Any],
    *,
    character_tags: Iterable[str],
    character_gender: str,
) -> tuple[CharacterGenerationRequest, dict[str, Any]]:
    """Build an Animagine request without garment tags or garment conditioning."""

    resolved_tags, removed_gender_tags = resolve_character_gender(
        character_tags, character_gender
    )
    selected_gender_tag = CHARACTER_GENDERS[character_gender]
    gender_support = BASE_GENDER_CONDITION_TAGS.get(character_gender, ())
    normalized_tags = _unique_tags(resolved_tags)
    specific_ears = any(
        tag.endswith(" ears") and tag not in {"animal ears", "human ears"}
        for tag in normalized_tags
    )
    specific_tail = any(
        tag.endswith(" tail") and tag != "tail"
        for tag in normalized_tags
    )
    removed_redundant_features = tuple(
        tag for tag in normalized_tags
        if (tag == "animal ears" and specific_ears)
        or (tag == "tail" and specific_tail)
    )
    routed_tags = tuple(
        tag for tag in normalized_tags
        if tag not in removed_redundant_features
        and tag != selected_gender_tag
        and tag not in gender_support
    )
    animal_features = tuple(
        tag for tag in routed_tags
        if tag in {"animal ears", "human ears", "tail"}
        or tag.endswith(" ears") or tag.endswith(" tail")
    )
    identity_features = tuple(
        tag for tag in routed_tags if tag not in animal_features
    )
    animal_features = tuple(sorted(
        animal_features,
        key=lambda tag: (tag.endswith(" tail") or tag == "tail", tag),
    ))
    required = _unique_tags((
        selected_gender_tag,
        *gender_support,
        *_framing_tags(request.framing_type),
        *identity_features,
        *animal_features,
    ))
    positive, positive_record = _build_character_prompt(
        tokenizers,
        required,
        ("solo", "simple background", "coherent anatomy", "best quality"),
    )

    terms = [
        term.strip() for term in request.negative_prompt.split(",") if term.strip()
    ]
    garment_specific_negative = {
        "different outfit",
        "mismatched colors",
        "unnatural clothing folds",
        "warped clothing",
    }
    retained = [
        term
        for term in terms
        if normalize_tag(term) not in garment_specific_negative
    ]
    gender_guard = {"male": "1girl", "female": "1boy"}.get(character_gender)
    if gender_guard:
        retained = [gender_guard] + [
            term for term in retained if normalize_tag(term) != gender_guard
        ]
    negative, negative_record = prepare_prompt_for_clip(
        ", ".join(retained), tuple(tokenizers)
    )
    if (
        selected_gender_tag
        and positive.split(",", 1)[0].strip() != selected_gender_tag
    ):
        raise ValueError("캐릭터 Base 성별 태그가 프롬프트 선두에서 누락됐습니다.")
    if gender_guard and negative.split(",", 1)[0].strip() != gender_guard:
        raise ValueError("캐릭터 Base 성별 네거티브 조건이 누락됐습니다.")
    record = {
        "version": CONTRACT_VERSION,
        "mode": "animagine_character_only_base",
        "approved_character_tags": list(resolved_tags),
        "base_prompt_tag_order": list(required),
        "removed_redundant_character_tags": list(
            removed_redundant_features),
        "removed_character_gender_tags": list(removed_gender_tags),
        "removed_gender_negative_terms": [],
        "removed_conflicting_negative_terms": sorted(garment_specific_negative),
        "excluded_non_design_tags": [],
        "garment_gender_policy": "not_used_character_only_base",
        "character_gender": character_gender,
        "gender_source": "user",
        "base_gender_condition_tags": list(gender_support),
        "gender_negative_guard": gender_guard,
        "positive": positive_record,
        "negative": negative_record,
        "approved_tags": [],
        "approved_detail_tags": [],
        "garment_image_adapter": False,
        "garment_tags_enabled": False,
        "garment_source": None,
        "body_restoration": False,
        "pixel_preservation_guaranteed": False,
    }
    return replace(request, prompt=positive, negative_prompt=negative), record












def _approved_garment_digest(approved_run: Mapping[str, Any]) -> str:
    parts = approved_run.get("parts", ())
    matches = [
        part for part in parts
        if isinstance(part, Mapping) and part.get("name") == "garment"
    ]
    if len(matches) != 1:
        raise ValueError("승인 조건에 격리 의상 보드 기록이 정확히 1개 필요합니다.")
    expected = matches[0].get("rgb")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("승인 조건의 격리 의상 보드 다이제스트가 잘못됐습니다.")
    return expected


def garment_board_image_digest(path: str | Path) -> str:
    board = Path(path)
    with Image.open(board) as opened:
        opened.load()
        digest = hashlib.sha256(json.dumps(
            (opened.mode, opened.size),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(opened.tobytes())
    return digest.hexdigest()




def require_isolated_garment_board(
    path: str | Path,
    approved_run: Mapping[str, Any] | None = None,
) -> Path:
    board = Path(path)
    if not board.is_file():
        raise ValueError(f"격리된 의상 보드 파일이 없습니다: {board}")
    if board.name != "input_garment.png":
        raise ValueError(
            "Native 정밀화에는 생성 배치가 저장한 input_garment.png만 사용할 수 있습니다."
        )
    if approved_run is not None:
        expected = _approved_garment_digest(approved_run)
        actual = garment_board_image_digest(board)
        if actual != expected:
            raise ValueError(
                "격리 의상 보드가 승인 시점의 픽셀 다이제스트와 일치하지 않습니다. "
                f"기대={expected[:12]}, 실제={actual[:12]}"
            )
    return board
