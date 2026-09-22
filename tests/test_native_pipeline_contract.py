from copy import deepcopy

from PIL import Image
import pytest

from genai_lab.approved_reference_run import image_digest
from genai_lab.native_pipeline_contract import (
    NativePipelineConfigurationError,
    build_native_refinement_instruction,
    native_direct_enabled,
    native_pipeline_enabled,
    native_refinement_enabled,
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
            "mode": "flux_whole_image",
        },
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
    assert native_refinement_enabled(config)
    validate_native_pipeline_config(config)

    invalid = deepcopy(config)
    invalid["native_pipeline_v2"]["base_garment_adapter_enabled"] = True
    with pytest.raises(NativePipelineConfigurationError, match="불완전"):
        native_pipeline_enabled(invalid)


def test_native_pipeline_and_refinement_must_be_enabled_together():
    config = native_config()
    config["native_refinement"]["enabled"] = False
    assert not native_refinement_enabled(config)
    with pytest.raises(NativePipelineConfigurationError, match="native_refinement.enabled"):
        validate_native_pipeline_config(config)


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

def test_direct_native_route_requires_explicit_animagine_fallback():
    config = native_config()
    config["native_pipeline_v2"].update({
        "primary_route": "source_character_direct",
        "animagine_fallback": "explicit_only",
    })
    config["native_refinement"][
        "all_failed_action"
    ] = "require_explicit_animagine_fallback"

    assert native_direct_enabled(config)
    validate_native_pipeline_config(config)


def _approved_native_run(character_tags, **extra):
    return {
        "approved_character_tags": character_tags,
        "approved_tags": ["blue jacket"],
        "approved_detail_tags": ["white shirt"],
        "gender": "female",
        **extra,
    }


def test_native_instruction_preserves_confirmed_two_piece_topology():
    instruction, contract = build_native_refinement_instruction(
        _approved_native_run(
            ["short black hair", "blue eyes"],
            approved_tags=["striped bikini"],
            approved_detail_tags=[],
        )
    )

    assert contract["garment_topology"]["topology"] == "two_piece"
    assert contract["garment_topology_guidance_applied"] is True
    assert "two visually separate components" in instruction
    assert "abdomen between them visibly uncovered" in instruction


def test_native_instruction_does_not_guess_generic_swimsuit_topology():
    instruction, contract = build_native_refinement_instruction(
        _approved_native_run(
            ["short black hair", "blue eyes"],
            approved_tags=["swimsuit"],
            approved_detail_tags=[],
        )
    )

    assert contract["garment_topology"]["status"] == "UNRESOLVED"
    assert contract["garment_topology_guidance_applied"] is False
    assert "two visually separate components" not in instruction


def _native_authority_properties(instruction):
    clause = instruction.split("authority for ", 1)[1].split(
        ". Preserve those properties", 1
    )[0]
    return tuple(item.strip() for item in clause.split(","))


def test_native_instruction_omits_unapproved_optional_character_parts():
    instruction, contract = build_native_refinement_instruction(
        _approved_native_run(["short black hair", "blue eyes"])
    )

    assert contract["character_parts"] == []
    assert "human ears" not in _native_authority_properties(instruction)
    assert "animal ears" not in _native_authority_properties(instruction)
    assert "tail" not in _native_authority_properties(instruction)
    assert all(
        not item["present"]
        for item in contract["character_part_evidence"].values()
    )


def test_native_instruction_includes_only_approved_animal_ears():
    instruction, contract = build_native_refinement_instruction(
        _approved_native_run(["long white hair", "cat ears"])
    )

    assert contract["character_parts"] == ["animal_ears"]
    assert "animal ears" in _native_authority_properties(instruction)
    assert "tail" not in _native_authority_properties(instruction)
    assert "human ears" not in _native_authority_properties(instruction)
    assert contract["character_part_evidence"]["animal_ears"]["sources"] == [
        "approved_character_tags"
    ]


def test_native_instruction_uses_recorded_ear_and_confirmed_tail_presence():
    instruction, contract = build_native_refinement_instruction(
        _approved_native_run(
            ["long white hair"],
            ear_contract={
                "human": {"state": "unknown"},
                "animal": {"state": "present"},
            },
            condition_snapshot={
                "observations": {
                    "tail": {
                        "detection": {
                            "status": "detected",
                            "accepted_masks": 1,
                            "semantic_status": "confirmed",
                            "target_status": "resolved",
                            "automatic_conditioning": True,
                        }
                    }
                }
            },
        )
    )

    assert contract["character_parts"] == ["animal_ears", "tail"]
    assert "animal ears" in _native_authority_properties(instruction)
    assert "tail" in _native_authority_properties(instruction)
    assert "human ears" not in _native_authority_properties(instruction)
    assert contract["character_part_evidence"]["animal_ears"]["sources"] == [
        "ear_contract_present"
    ]
    assert contract["character_part_evidence"]["tail"]["sources"] == [
        "condition_snapshot_confirmed"
    ]


def test_native_instruction_does_not_promote_unresolved_tail_mask():
    instruction, contract = build_native_refinement_instruction(
        _approved_native_run(
            ["short black hair", "blue eyes"],
            condition_snapshot={
                "observations": {
                    "tail": {
                        "detection": {
                            "status": "detected",
                            "accepted_masks": 1,
                            "semantic_status": "unresolved",
                            "target_status": "unresolved_cross_class_conflict",
                            "automatic_conditioning": False,
                            "outcome_reason": (
                                "cross_class_overlap_without_approved_tag"
                            ),
                        }
                    }
                }
            },
        )
    )

    assert contract["character_parts"] == []
    assert "tail" not in _native_authority_properties(instruction)
    evidence = contract["character_part_evidence"]["tail"]
    assert evidence["present"] is False
    assert "semantic_status_unconfirmed" in evidence["detection_block_reasons"]
    assert "cross_class_conflict" in evidence["detection_block_reasons"]


def test_native_instruction_keeps_human_and_animal_ears_independent():
    instruction, contract = build_native_refinement_instruction(
        _approved_native_run(
            ["short brown hair"],
            ear_contract={
                "human": {"state": "present"},
                "animal": {"state": "absent"},
            },
        )
    )

    assert contract["character_parts"] == ["human_ears"]
    assert "human ears" in _native_authority_properties(instruction)
    assert "animal ears" not in _native_authority_properties(instruction)
    assert "tail" not in _native_authority_properties(instruction)



def test_native_instruction_uses_only_confirmed_hair_parts_as_soft_guidance():
    instruction, contract = build_native_refinement_instruction(
        _approved_native_run(
            ["short_hair", "blunt_bangs", "blue_hair"],
            hair_detail_analysis={
                "views": [{"name": "rear_silhouette_candidate"}],
            },
        )
    )

    assert contract["hair_structure_guidance_applied"] is True
    assert "overall length (short hair)" in instruction
    assert "front hair (blunt bangs)" in instruction
    assert "hair color (blue hair)" in instruction
    assert contract["hair_structure"]["components"]["side"]["state"] == "unknown"
    assert (
        contract["hair_structure"]["components"]["rear_silhouette"]
        ["semantic_identity_confirmed"]
        is False
    )
