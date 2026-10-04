from pathlib import Path

import pytest

from scripts.run_contract_scope_gpu_regression import (
    normalize_character_canvas,
    refresh_current_analysis_policy,
    reset_input_bound_approval_state,
    validate_input_roles,
)


def test_role_validation_keeps_character_and_garment_sources_separate(tmp_path: Path):
    character_root = tmp_path / "characters"
    garment_root = tmp_path / "garments"
    character_root.mkdir()
    garment_root.mkdir()
    character = character_root / "character.png"
    garment = garment_root / "garment.png"
    character.write_bytes(b"character")
    garment.write_bytes(b"garment")

    record = validate_input_roles(character, garment, garment_root)

    assert record["character_role"] == "character_reference_only"
    assert record["garment_role"] == "garment_reference_only"
    assert record["character_sha256"] != record["garment_sha256"]


def test_role_validation_rejects_character_from_garment_root(tmp_path: Path):
    garment_root = tmp_path / "garments"
    garment_root.mkdir()
    character = garment_root / "character.png"
    garment = garment_root / "garment.png"
    character.write_bytes(b"character")
    garment.write_bytes(b"garment")

    with pytest.raises(ValueError, match="character reference"):
        validate_input_roles(character, garment, garment_root)


def test_role_validation_rejects_garment_outside_root(tmp_path: Path):
    character_root = tmp_path / "characters"
    garment_root = tmp_path / "garments"
    other_root = tmp_path / "other"
    character_root.mkdir()
    garment_root.mkdir()
    other_root.mkdir()
    character = character_root / "character.png"
    garment = other_root / "garment.png"
    character.write_bytes(b"character")
    garment.write_bytes(b"garment")

    with pytest.raises(ValueError, match="garment reference"):
        validate_input_roles(character, garment, garment_root)

def test_character_canvas_is_padded_to_model_multiple_without_resizing():
    from PIL import Image
    with Image.new("RGB", (735, 1387), (12, 34, 56)) as source:
        normalized, record = normalize_character_canvas(source)
    try:
        assert normalized.size == (736, 1392)
        assert record["original_size"] == [735, 1387]
        assert record["normalized_size"] == [736, 1392]
        assert record["padding_ltrb"] == [0, 2, 1, 3]
        assert record["content_resized"] is False
        assert normalized.getpixel((735, 1391)) == (12, 34, 56)
    finally:
        normalized.close()





def test_input_bound_approval_state_does_not_inherit_old_character_or_garment():
    section = {
        "approved_character_tags": ("raccoon ears", "raccoon tail"),
        "character_gender": "unspecified",
        "approved_tags": ("leotard", "pantyhose"),
        "approved_detail_tags": ("star print",),
        "garment_detail_report": {"evidence": ["old"]},
        "character_feature_routing": {"selected": ("raccoon ears",)},
        "eye_color_report": {"status": "old"},
        "hair_detail_report": {"status": "old"},
    }

    record = reset_input_bound_approval_state(
        section,
        character_tags=("1boy", "white hair", "short hair"),
        character_gender="male",
        garment_name="current-garment.jpg",
        garment_tags=("oversized jacket", "black trousers"),
        garment_detail_tags=("cable knit",),
    )

    assert section["approved_character_tags"] == (
        "1boy", "white hair", "short hair")
    assert section["approved_tags"] == ("oversized jacket", "black trousers")
    assert section["approved_detail_tags"] == ("cable knit",)
    assert section["garment_detail_report"] is None
    assert section["approved_garment_topology"] is None
    assert section["hair_prompt_delivery"] is None
    assert section["eye_color_report"] is None
    assert section["hair_detail_report"] is None
    assert section["character_feature_routing"]["selected"] == (
        "1boy", "white hair", "short hair")
    assert "raccoon ears" not in section["character_feature_routing"]["selected"]
    assert section["source_name"] == "current-garment.jpg"
    assert record["version"] == "input_bound_approval_reset_v1"
    assert record["garment_tags"] == ["oversized jacket", "black trousers"]
    assert record["garment_detail_tags"] == ["cable knit"]


def test_runtime_refresh_replaces_only_current_analysis_gate_sections():
    config = {
        "model": {"repository": "keep-me"},
        "reference_face_observation": {"enabled": False},
        "reference_hair_observation": {"enabled": False},
        "reference_base_mask_validation": {"block_on_review": False},
        "reference_analysis": {
            "part_detection": {
                "maximum_foreground_area_ratios": {"human_ears": 0.99},
            },
        },
    }

    report = refresh_current_analysis_policy(config)

    assert config["model"] == {"repository": "keep-me"}
    assert config["reference_face_observation"]["enabled"] is True
    assert config["reference_hair_observation"]["enabled"] is True
    assert config["reference_base_mask_validation"]["block_on_review"] is True
    limits = config["reference_analysis"]["part_detection"][
        "maximum_foreground_area_ratios"
    ]
    assert limits["human_ears"] == 0.12
    assert report["model_seed_and_input_contract_changed"] is False
