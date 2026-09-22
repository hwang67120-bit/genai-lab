import hashlib
import json
from pathlib import Path

from genai_lab.body_proportion_presets import (
    ASSET_ROOT,
    BODY_PROPORTION_PRESETS,
    CANVAS_SIZE,
    get_body_proportion_preset,
    load_body_proportion_control,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_four_families_have_six_reviewable_presets():
    assert {item.family for item in BODY_PROPORTION_PRESETS} == {
        "super_deformed",
        "child_five_head",
        "standard_adult",
        "fashion_adult",
    }
    assert len(BODY_PROPORTION_PRESETS) == 6


def test_sd_and_child_are_distinct_concepts():
    sd = get_body_proportion_preset("sd_3h_neutral")
    child = get_body_proportion_preset("child_5h_neutral")
    assert sd.head_count == 3.0
    assert child.head_count == 5.0
    assert sd.life_stage == "stylized_age_neutral"
    assert child.life_stage == "child"


def test_only_adult_families_have_shape_variants():
    for family in ("standard_adult", "fashion_adult"):
        variants = {
            item.body_form for item in BODY_PROPORTION_PRESETS
            if item.family == family
        }
        assert variants == {"shoulder_dominant", "hip_dominant"}
    assert get_body_proportion_preset("sd_3h_neutral").body_form == "neutral"
    assert get_body_proportion_preset("child_5h_neutral").body_form == "neutral"


def test_presets_do_not_bind_gender_or_openpose():
    for preset in BODY_PROPORTION_PRESETS:
        record = preset.record()
        assert record["gender_conditioning_applied"] is False
        assert record["gender_identity_source"] == "separate_character_request"
        assert record["openpose_used"] is False
        assert record["application_stage"] == "animagine_base_only"
        assert record["model_conditioning_applied"] is True


def test_generated_control_assets_match_manifest():
    manifest_path = PROJECT_ROOT / ASSET_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "approved_asset"
    assert manifest["family_count"] == 4
    assert manifest["preset_count"] == 6
    assert manifest["render_profile"] == "project_generated_human_form_sheet_v3"
    assert manifest["source_type"] == "project_generated_original"
    assert manifest["reference_usage"] == "structure_only"
    assert manifest["copied_character_design"] is False
    source_path = PROJECT_ROOT / manifest["source_path"]
    assert source_path.is_file()
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == manifest["source_sha256"]
    by_id = {item["preset_id"]: item for item in manifest["presets"]}
    for preset in BODY_PROPORTION_PRESETS:
        with load_body_proportion_control(PROJECT_ROOT, preset.preset_id) as image:
            assert image.size == CANVAS_SIZE
            assert image.getbbox() is not None
            width, height = image.size
            assert image.crop((0, 0, width, 12)).getbbox() is None
            assert image.crop((0, height - 12, width, height)).getbbox() is None
            assert image.crop((0, 0, 12, height)).getbbox() is None
            assert image.crop((width - 12, 0, width, height)).getbbox() is None
        path = PROJECT_ROOT / preset.control_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            by_id[preset.preset_id]["control_sha256"]
        )
