from copy import deepcopy

import pytest

from genai_lab.native_pipeline_contract import (
    FLUX_WHOLE_IMAGE_REFINEMENT,
    SDXL_LOCAL_REFINEMENT,
    NativePipelineConfigurationError,
    resolve_refinement_mode,
    validate_native_pipeline_config,
)


def valid_config():
    return {
        "native_pipeline_v2": {
            "enabled": True,
            "base_profile": "character_only",
            "base_garment_prompt_enabled": False,
            "base_garment_adapter_enabled": False,
            "native_garment_source": "isolated_garment_board",
            "reuse_animagine_prompt_in_native_stage": False,
            "raw_garment_person_image_allowed": False,
        },
        "refinement_execution": {
            "enabled": True,
            "mode": FLUX_WHOLE_IMAGE_REFINEMENT,
        },
        "native_refinement": {
            "enabled": True,
            "mode": "gui_local_gpu",
            "image_merge_enabled": False,
            "hard_paste_enabled": False,
            "mask_composite_enabled": False,
            "failed_candidates_returned": False,
            "strategy": "flux2_klein_only",
            "candidate_policy": "whole_image_native_output",
            "all_failed_action": "keep_approved_base_and_seed",
            "engines": ["flux2_klein"],
            "minimum_similarity": 0.65,
            "minimum_color_similarity": 0.55,
            "width": 512,
            "height": 512,
            "flux_steps": 4,
            "timeout_seconds": 120,
            "allow_download": False,
            "python_executable": "python.exe",
            "runner_path": "runner.py",
            "model_cache_dir": "cache",
        },
        "staged_reference_generation": {"garment": {"enabled": True}},
    }


def test_refinement_mode_is_required_and_explicit():
    config = valid_config()
    del config["refinement_execution"]["mode"]
    with pytest.raises(NativePipelineConfigurationError, match="sdxl_local"):
        resolve_refinement_mode(config)


def test_flux_mode_validates_as_single_finalizer():
    config = valid_config()
    validate_native_pipeline_config(config)
    assert resolve_refinement_mode(config) == FLUX_WHOLE_IMAGE_REFINEMENT


def test_sdxl_local_mode_uses_staged_garment_contract():
    config = valid_config()
    config["refinement_execution"]["mode"] = SDXL_LOCAL_REFINEMENT
    validate_native_pipeline_config(config)
    assert resolve_refinement_mode(config) == SDXL_LOCAL_REFINEMENT


def test_sdxl_local_rejects_source_direct_route():
    config = valid_config()
    config["refinement_execution"]["mode"] = SDXL_LOCAL_REFINEMENT
    config["native_pipeline_v2"]["primary_route"] = "source_character_direct"
    with pytest.raises(NativePipelineConfigurationError, match="승인 Animagine Base"):
        validate_native_pipeline_config(config)
