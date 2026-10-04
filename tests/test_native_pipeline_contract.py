from copy import deepcopy

from PIL import Image
import pytest

from genai_lab.approved_reference_run import image_digest
from genai_lab.native_pipeline_contract import (
    NativePipelineConfigurationError,
    native_pipeline_enabled,
    final_refinement_enabled,
    require_isolated_garment_board,
    validate_native_pipeline_config,
)


def native_config():
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
            "mode": "sdxl_local",
        },
        "staged_reference_generation": {"garment": {"enabled": True}},
        "native_refinement": {
            "enabled": True,
            "mode": "gui_local_gpu",
            "python_executable": "python.exe",
            "runner_path": "runner.py",
            "model_cache_dir": "cache",
            "allow_download": False,
            "timeout_seconds": 1800,
            "strategy": "flux2_klein_only",
            "engines": ["flux2_klein"],
            "candidate_policy": "whole_image_native_output",
            "image_merge_enabled": False,
            "hard_paste_enabled": False,
            "mask_composite_enabled": False,
            "failed_candidates_returned": False,
            "all_failed_action": "keep_approved_base_and_seed",
            "minimum_similarity": 0.65,
            "minimum_color_similarity": 0.55,
            "width": 320,
            "height": 512,
            "flux_steps": 4,
        },
    }


def test_native_activation_requires_complete_character_only_contract():
    config = native_config()
    assert native_pipeline_enabled(config)
    assert final_refinement_enabled(config)
    validate_native_pipeline_config(config)

    invalid = deepcopy(config)
    invalid["native_pipeline_v2"]["base_garment_adapter_enabled"] = True
    with pytest.raises(NativePipelineConfigurationError, match="불완전"):
        native_pipeline_enabled(invalid)




def test_garment_board_pixels_must_match_approved_digest(tmp_path):
    board = tmp_path / "input_garment.png"
    approved_image = Image.new("RGB", (16, 16), "blue")
    expected = image_digest(approved_image)
    approved_image.save(board)
    approved_image.close()
    approved_run = {
        "parts": [{"name": "garment", "rgb": expected}],
    }

    assert require_isolated_garment_board(board, approved_run) == board

    changed = Image.new("RGB", (16, 16), "red")
    changed.save(board)
    changed.close()
    with pytest.raises(ValueError, match="픽셀 다이제스트"):
        require_isolated_garment_board(board, approved_run)
