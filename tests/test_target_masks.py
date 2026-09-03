import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from genai_lab.body_comparison import (
    CharacterBodyComparisonSettings, execute_character_body_comparison,
    refine_character_clothing_change_mask, verify_original_clothing_removal,
)
from genai_lab.target_masks import approve_target_masks


def test_approved_masks_are_owned_and_bound_to_source_pixels():
    with Image.new("RGB", (8, 8), "white") as source, Image.new("L", (8, 8), 255) as clothes, Image.new("L", (8, 8), 0) as special:
        approved = approve_target_masks(source, clothes, special)
        copied = approved.copy()
        try:
            clothes.paste(0, (0, 0, 8, 8))
            assert copied.clothing_mask.getpixel((1, 1)) == 255
            approved.validate_source(source)
            source.putpixel((0, 0), (0, 0, 0))
            with pytest.raises(ValueError, match="재승인"):
                approved.validate_source(source)
        finally:
            approved.close()
            copied.close()


@pytest.mark.parametrize("case", ["empty", "conflict", "size"])
def test_reject_invalid_target_masks(case):
    with Image.new("RGB", (8, 8)) as source, Image.new("L", (8, 8), 0 if case == "empty" else 255) as clothes, Image.new("L", (4, 4) if case == "size" else (8, 8), 255 if case == "conflict" else 0) as special:
        with pytest.raises(ValueError):
            approve_target_masks(source, clothes, special)


@pytest.mark.parametrize("outside", [False, True])
def test_no_verifiable_target_is_not_evaluable(outside):
    with Image.new("L", (8, 8), 255 if outside else 0) as raw, Image.new("L", (8, 8), 0) as zero:
        result = verify_original_clothing_removal(raw, zero, zero, zero)
        try:
            assert not result.passed
            assert result.status == "not_evaluable"
            assert result.removal_percent is None
        finally:
            result.close()


def test_fully_protected_target_is_not_removed_even_if_mask_overlaps():
    with Image.new("L", (8, 8), 255) as full:
        result = verify_original_clothing_removal(full, full, full, full)
        try:
            assert result.status == "needs_review"
            assert result.removal_percent == 0
            assert result.remaining_clothing_pixel_count == 64
        finally:
            result.close()


def test_explicit_boundary_does_not_expand_into_tail_or_background():
    with Image.new("L", (64, 64), 0) as clothes, Image.new("L", (64, 64), 0) as special, Image.new("L", (64, 64), 255) as foreground:
        clothes.paste(255, (15, 20, 25, 45))
        clothes.putpixel((30, 60), 255)  # selected shoe
        special.paste(255, (26, 30, 40, 50))  # tail
        result = refine_character_clothing_change_mask(
            clothes, special, foreground, 0, closing_radius_pixels=0,
            preserve_approved_boundary=True,
        )
        try:
            assert np.array_equal(np.asarray(result.safe_change_mask), np.asarray(clothes))
            assert result.safe_change_mask.getpixel((30, 60)) == 255
            assert result.safe_change_mask.getpixel((26, 30)) == 0
        finally:
            result.close()


def test_execution_uses_selected_mask_and_keeps_auto_mask_diagnostic(tmp_path, monkeypatch):
    captured = []
    def fake_runner(command, **kwargs):
        captured.extend(command)
        def save(flag, mode, color):
            with Image.new(mode, (64, 64), color) as image:
                image.save(command[command.index(flag) + 1])
        save("--output-raw-mask", "L", 255)
        save("--output-protection-mask", "L", 0)
        save("--output-foreground-mask", "L", 255)
        save("--output-densepose", "RGB", "black")
        Path(command[command.index("--output-metadata-json") + 1]).write_text(json.dumps({
            "foreground_model_id": "isnet-anime", "foreground_pixel_count": 4096,
            "foreground_percent": 100, "foreground_elapsed_seconds": 0,
            "model_ids": ["test"], "elapsed_seconds": 0,
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("genai_lab.body_comparison.subprocess.run", fake_runner)
    settings = CharacterBodyComparisonSettings(
        tmp_path, tmp_path, tmp_path, tmp_path / "temp", tmp_path,
        256, 256, 60,
    )
    with Image.new("RGB", (64, 64), "white") as source, Image.new("L", (64, 64), 0) as clothes, Image.new("L", (64, 64), 0) as special:
        clothes.paste(255, (12, 12, 24, 40))
        clothes.putpixel((20, 60), 255)
        special.paste(255, (30, 30, 50, 55))
        approved = approve_target_masks(source, clothes, special)
        result = execute_character_body_comparison(source, "upper", settings, approved)
        try:
            assert "--explicit-target-masks" in captured
            assert result.mask_source == "user_selected_target_sam2"
            assert np.array_equal(np.asarray(result.mask_refinement.raw_mask), np.asarray(clothes))
            assert result.automatic_change_mask.getpixel((0, 0)) == 255
            assert result.human_agnostic_candidate.neutralized_image.getpixel((20, 60)) == (127, 127, 127)
            assert result.human_agnostic_candidate.neutralized_image.getpixel((35, 35)) == (255, 255, 255)
            assert result.clothing_removal_verification.passed
        finally:
            result.close()
            approved.close()
