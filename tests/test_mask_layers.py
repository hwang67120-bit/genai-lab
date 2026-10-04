import numpy as np
from PIL import Image
import pytest

from genai_lab.mask_layers import synthesize_mask_layers
from genai_lab.target_masks import accumulate_mask, approve_target_masks


def test_accumulation_preserves_existing_and_soft_alpha():
    with Image.new("L", (8, 8), 0) as old, Image.new("L", (8, 8), 0) as new:
        old.putpixel((1, 1), 220)
        old.putpixel((2, 2), 80)
        new.putpixel((2, 2), 160)
        new.putpixel((3, 3), 255)
        merged = accumulate_mask(old, new)
        try:
            assert merged.getpixel((1, 1)) == 220
            assert merged.getpixel((2, 2)) == 160
            assert merged.getpixel((3, 3)) == 255
            assert old.getpixel((2, 2)) == 80
        finally:
            merged.close()


@pytest.mark.parametrize("automatic", [True, False])
def test_layer_priority_and_diagnostic_positions(automatic):
    source = Image.new("RGB", (16, 16), (100, 100, 100))
    masks = [Image.new("L", source.size, 0) for _ in range(7)]
    auto, protection, identity, manual, keep, exclude, fg = masks
    fg.paste(255, (1, 1, 16, 16))
    for p in ((2, 4), (2, 5)):
        auto.putpixel(p, 255)
    for p in ((6, 5), (6, 2), (6, 6), (0, 0)):
        manual.putpixel(p, 255)
        protection.putpixel(p, 255)
    identity.putpixel((6, 2), 255)
    keep.putpixel((6, 6), 255)
    exclude.putpixel((2, 5), 255)
    original = np.array(source)
    result = synthesize_mask_layers(source, *masks, use_automatic_base=automatic)
    try:
        final = result.images["final_mask"]
        assert np.array_equal(np.asarray(result.images['face_hair_reference']), np.asarray(identity))
        assert result.images['face_hair_reference'].getpixel((6, 6)) == 0
        assert result.images['face_hair_reference'].getpixel((2, 5)) == 0
        assert final.getpixel((2, 4)) == (255 if automatic else 0)
        assert final.getpixel((6, 5)) == 255  # explicit add overrides auto only
        for p in ((6, 2), (6, 6), (0, 0), (2, 5), (14, 14)):
            assert final.getpixel(p) == 0
        assert result.metrics["manual_conflict_pixels"] == 2
        assert result.metrics["manual_override_pixels"] == 1
        assert result.metrics["manual_outside_pixels"] == 1
        assert result.images["effective_protection"].getpixel((6, 5)) == 0
        assert np.array_equal(np.array(source), original)
    finally:
        result.close()
        source.close()
        for mask in masks:
            mask.close()


def test_auto_only_approval_and_copy_keep_settings():
    with Image.new("RGB", (16, 16)) as source, Image.new("L", source.size, 0) as zero:
        approved = approve_target_masks(
            source, zero, zero, use_automatic_base=True, excluded_mask=zero)
        copied = approved.copy()
        approved.close()
        try:
            assert copied.use_automatic_base and copied.layered_priority
            assert copied.excluded_mask.getpixel((0, 0)) == 0
            copied.validate_source(source)
        finally:
            copied.close()


def test_semantic_layers_restore_shoe_candidate_but_keep_face_hair_separate():
    from scripts.body_comparison_runner import create_layered_semantic_masks
    with Image.new("L", (8, 8), 0) as auto, Image.new("L", (8, 8), 0) as lip, \
            Image.new("L", (8, 8), 0) as atr:
        mapping = {"Face": 1, "Hair": 2, "Left-shoe": 3, "Pants": 4, "Left-arm": 5}
        for x, label in enumerate((1, 2, 3, 4, 5), 1):
            lip.putpixel((x, 1), label)
        identity, clothes = create_layered_semantic_masks(auto, lip, atr, mapping, mapping)
        try:
            assert identity.getpixel((1, 1)) == 255
            assert identity.getpixel((2, 1)) == 255
            assert clothes.getpixel((3, 1)) == 255
            assert clothes.getpixel((4, 1)) == 255
            assert clothes.getpixel((5, 1)) == 0
            assert clothes.getpixel((1, 1)) == 0
        finally:
            identity.close()
            clothes.close()


def test_approved_skirt_contract_excludes_pants_socks_and_shoes_from_mask():
    from scripts.body_comparison_runner import create_layered_semantic_masks
    with Image.new("L", (8, 8), 255) as auto, Image.new("L", (8, 8), 0) as lip, \
            Image.new("L", (8, 8), 0) as atr:
        mapping = {
            "Face": 1, "Hair": 2, "Upper-clothes": 3, "Coat": 4,
            "Skirt": 5, "Pants": 6, "Socks": 7, "Left-shoe": 8,
            "Right-shoe": 9,
        }
        for x, label in enumerate((1, 2, 3, 5, 6, 7, 8), 1):
            lip.putpixel((x, 1), label)
        identity, clothes = create_layered_semantic_masks(
            auto, lip, atr, mapping, mapping,
            approved_garment_tags=("blue jacket", "white shirt", "blue skirt"),
        )
        try:
            assert clothes.getpixel((3, 1)) == 255
            assert clothes.getpixel((4, 1)) == 255
            assert clothes.getpixel((5, 1)) == 0
            assert clothes.getpixel((6, 1)) == 0
            assert clothes.getpixel((7, 1)) == 0
            assert clothes.getpixel((0, 0)) == 0
        finally:
            identity.close()
            clothes.close()


def test_cross_parser_leg_veto_wins_over_approved_garment_label():
    from scripts.body_comparison_runner import create_layered_semantic_masks
    with Image.new("L", (8, 8), 0) as auto, Image.new("L", (8, 8), 0) as lip, \
            Image.new("L", (8, 8), 0) as atr:
        mapping = {
            "Face": 1, "Hair": 2, "Upper-clothes": 3, "Coat": 4,
            "Skirt": 5, "Pants": 6, "Socks": 7, "Left-shoe": 8,
            "Right-shoe": 9, "Left-leg": 10, "Right-leg": 11,
        }
        lip.putpixel((3, 3), mapping["Skirt"])
        atr.putpixel((3, 3), mapping["Left-leg"])
        lip.putpixel((4, 3), mapping["Skirt"])
        atr.putpixel((4, 3), mapping["Skirt"])
        lip.putpixel((1, 1), mapping["Face"])
        identity, clothes = create_layered_semantic_masks(
            auto, lip, atr, mapping, mapping,
            approved_garment_tags=("blue jacket", "blue skirt"),
        )
        try:
            assert clothes.getpixel((3, 3)) == 0
            assert clothes.getpixel((4, 3)) == 255
        finally:
            identity.close()
            clothes.close()


def test_approved_garment_removes_tiny_disconnected_parser_fragments():
    from scripts.body_comparison_runner import create_layered_semantic_masks
    size = (64, 64)
    with Image.new("L", size, 0) as auto, Image.new("L", size, 0) as lip, \
            Image.new("L", size, 0) as atr:
        mapping = {
            "Face": 1, "Hair": 2, "Upper-clothes": 3, "Coat": 4,
            "Skirt": 5, "Pants": 6, "Socks": 7, "Left-shoe": 8,
            "Right-shoe": 9, "Left-leg": 10, "Right-leg": 11,
        }
        lip.paste(mapping["Skirt"], (16, 12, 48, 48))
        atr.paste(mapping["Skirt"], (16, 12, 48, 48))
        # 두 파서가 모두 잘못 의상으로 본 작은 꼬리/다리 조각을 흉내 낸다.
        lip.paste(mapping["Skirt"], (2, 58, 4, 60))
        atr.paste(mapping["Skirt"], (2, 58, 4, 60))
        lip.putpixel((1, 1), mapping["Face"])
        identity, clothes = create_layered_semantic_masks(
            auto, lip, atr, mapping, mapping,
            approved_garment_tags=("blue jacket", "blue skirt"),
        )
        try:
            assert clothes.getpixel((24, 24)) == 255
            assert clothes.getpixel((2, 58)) == 0
        finally:
            identity.close()
            clothes.close()


def test_layered_preprocessing_retains_priority_through_neutralization(tmp_path, monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    from genai_lab.body_comparison import CharacterBodyComparisonSettings, execute_character_body_comparison

    def runner(command, **kwargs):
        assert "--layered-target-masks" in command
        outputs = {
            "--output-raw-mask": ("L", (8, 10, 25, 50)),
            "--output-protection-mask": ("L", (30, 10, 40, 40)),
            "--output-face-hair-mask": ("L", (30, 10, 40, 15)),
            "--output-foreground-mask": ("L", (1, 1, 64, 64)),
            "--output-densepose": ("RGB", None),
        }
        for flag, (mode, box) in outputs.items():
            with Image.new(mode, (64, 64), 0) as image:
                if box:
                    image.paste(255, box)
                image.save(command[command.index(flag) + 1])
        Path(command[command.index("--output-metadata-json") + 1]).write_text(json.dumps({
            "foreground_model_id": "isnet-anime", "foreground_pixel_count": 63 * 63,
            "foreground_percent": 63 * 63 / 4096 * 100,
            "foreground_elapsed_seconds": 0, "model_ids": ["test"], "elapsed_seconds": 0,
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("genai_lab.body_comparison.subprocess.run", runner)
    settings = CharacterBodyComparisonSettings(
        tmp_path, tmp_path, tmp_path, tmp_path / "temp", tmp_path, 256, 256, 60)
    with Image.new("RGB", (64, 64), "red") as source, Image.new("L", source.size, 0) as add, \
            Image.new("L", source.size, 0) as keep, Image.new("L", source.size, 0) as exclude:
        add.paste(255, (32, 12, 38, 30))  # face conflict and non-face auto protection overlap
        exclude.paste(255, (10, 20, 15, 30))  # exposed hand selected incorrectly by auto
        keep.paste(255, (16, 20, 20, 30))
        approved = approve_target_masks(source, add, keep, use_automatic_base=True,
                                        excluded_mask=exclude)
        result = execute_character_body_comparison(source, "overall", settings, approved)
        try:
            final = result.mask_refinement.safe_change_mask
            assert final.getpixel((9, 15)) == 255
            assert final.getpixel((33, 20)) == 255
            assert result.mask_refinement.identity_protection_mask.getpixel((33, 20)) == 0
            for p in ((33, 12), (12, 25), (18, 25)):
                assert final.getpixel(p) == 0
                assert result.human_agnostic_candidate.neutralized_image.getpixel(p) == (255, 0, 0)
            assert result.human_agnostic_candidate.neutralized_image.getpixel((33, 20)) == (127, 127, 127)
            assert result.mask_layers.metrics["manual_override_pixels"] > 0
            assert result.mask_layers.metrics["manual_conflict_pixels"] > 0
            assert result.mask_source == "automatic_with_user_layers"
        finally:
            approved.close()
            result.close()
