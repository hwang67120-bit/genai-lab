import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from genai_lab.reference_regions import (
    load_reference_regions, analyze_reference_regions, run_reference_process,
)
from scripts.body_comparison_runner import (
    save_reference_regions, prepare_reference_canvas,
)


def write_regions(directory, *, original_size=(8, 8), approved_tags=()):
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[:2, 2:6] = 2
    labels[2:3, 2:6] = 13
    labels[3:7, 1:7] = 5
    result = {
        "schp_lip": Image.fromarray(labels).convert("P"),
        "schp_atr": Image.new("L", (8, 8)),
        "mask": Image.new("L", (8, 8)),
    }
    result["schp_lip"].putpalette([255] * 768)
    foreground = Image.new("L", (8, 8), 255)
    try:
        save_reference_regions(result, foreground, original_size, directory,
            {"Face": 13, "Hair": 2, "Upper-clothes": 5},
            {"Face": 11, "Hair": 2}, approved_garment_tags=approved_tags)
    finally:
        foreground.close()
        for mask in result.values():
            mask.close()


def test_reference_outputs_only_and_palette_class_ids_preserved(tmp_path):
    write_regions(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "identity.png", "hair.png", "face.png", "garment.png",
        "foreground.png", "regions.json"}
    regions = load_reference_regions(tmp_path, (8, 8))
    try:
        assert regions.record["pixel_counts"]["hair"] == 8
        assert regions.record["pixel_counts"]["face"] == 4
        assert regions.record["neutralization_used"] is False
        assert regions.record["removal_verification_used"] is False
        assert "tail" in regions.record["unresolved"]
        assert not np.any(np.asarray(regions.masks["identity"]) & np.asarray(regions.masks["garment"]))
    finally:
        regions.close()


def test_reference_record_preserves_approved_garment_mask_policy(tmp_path):
    write_regions(tmp_path, approved_tags=("blue jacket", "blue skirt"))
    regions = load_reference_regions(tmp_path, (8, 8))
    try:
        policy = regions.record["garment_mask_policy"]
        assert policy["mode"] == "approved_semantic_classes"
        assert policy["approved_tags"] == ["blue jacket", "blue skirt"]
        assert policy["included_classes"] == ["Upper-clothes", "Coat", "Skirt"]
        assert "Pants" in policy["excluded_classes"]
        assert "Socks" in policy["excluded_classes"]
        assert "Left-shoe" in policy["excluded_classes"]
    finally:
        regions.close()


@pytest.mark.parametrize(
    "failure", ["count", "size", "mode", "overlap", "hair", "face", "foreground"])
def test_corrupt_reference_outputs_fail_closed(tmp_path, failure):
    write_regions(tmp_path)
    path = tmp_path / "regions.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if failure == "count":
        record["pixel_counts"]["hair"] = 999
    elif failure == "size":
        record["size"] = [9, 8]
    elif failure == "mode":
        record["mode"] = "legacy_body"
    else:
        name = {
            "overlap": "garment", "hair": "hair", "face": "face",
            "foreground": "foreground",
        }[failure]
        value = 0 if failure == "foreground" else 255
        with Image.new("L", (8, 8), value) as image:
            image.save(tmp_path / f"{name}.png")
        record["pixel_counts"][name] = 0 if value == 0 else 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        load_reference_regions(tmp_path, (8, 8))


def test_canvas_preserves_entire_source_and_restores_coordinates(tmp_path):
    with Image.new("RGB", (8, 4), "red") as source:
        canvas, box = prepare_reference_canvas(source, (8, 8))
    assert box == (0, 2, 8, 6)
    assert canvas.getpixel((0, 2)) == (255, 0, 0)
    assert canvas.getpixel((0, 0)) == (255, 255, 255)
    canvas.close()
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[2, :] = 2
    labels[3:6, :] = 5
    masks = {"schp_lip": Image.fromarray(labels),
             "schp_atr": Image.new("L", (8, 8)), "mask": Image.new("L", (8, 8))}
    foreground = Image.new("L", (8, 8), 255)
    try:
        save_reference_regions(masks, foreground, (8, 4), tmp_path,
            {"Face": 13, "Hair": 2, "Upper-clothes": 5}, {"Face": 11, "Hair": 2}, box)
        regions = load_reference_regions(tmp_path, (8, 4))
        assert regions.record["pixel_counts"]["hair"] == 8
        assert regions.record["pixel_counts"]["garment"] == 24
        regions.close()
    finally:
        foreground.close()
        for mask in masks.values():
            mask.close()


def test_reference_parent_calls_only_reference_runner_and_ignores_removal_settings(tmp_path, monkeypatch):
    seen = []
    def run(command, cwd, timeout, cancelled):
        output = Path(command[command.index("--reference-output-dir") + 1])
        write_regions(output, original_size=(16, 16))
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr("genai_lab.reference_regions.run_reference_process", run)
    cfg = {"character_body_comparison": {
        "python_executable": sys.executable, "runner_path": __file__,
        "repository_path": str(tmp_path), "temporary_root": str(tmp_path),
        "cache_dir": str(tmp_path), "width": 256, "height": 256,
        "timeout_seconds": 60, "mask_expansion_ratio": "no_longer_used"},
        "clothing_reference_generation": {
            "approved_tags": ("blue jacket", "blue skirt"),
        }}
    with Image.new("RGB", (16, 16), "white") as source:
        regions = analyze_reference_regions(source, cfg, tmp_path)
    assert len(seen) == 1
    assert "--reference-output-dir" in seen[0]
    assert "--allow-parser-foreground-fallback" not in seen[0]
    tag_flag = seen[0].index("--approved-garment-tags-json")
    assert json.loads(seen[0][tag_flag + 1]) == ["blue jacket", "blue skirt"]
    assert not list(tmp_path.glob("genai-reference-regions-*"))
    regions.close()


def test_generated_output_analysis_enables_parser_foreground_fallback(
        tmp_path, monkeypatch):
    seen = []

    def run(command, cwd, timeout, cancelled):
        output = Path(command[command.index("--reference-output-dir") + 1])
        write_regions(output, original_size=(16, 16))
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "genai_lab.reference_regions.run_reference_process", run)
    cfg = {"character_body_comparison": {
        "python_executable": sys.executable, "runner_path": __file__,
        "repository_path": str(tmp_path), "temporary_root": str(tmp_path),
        "cache_dir": str(tmp_path), "width": 256, "height": 256,
        "timeout_seconds": 60}}
    with Image.new("RGB", (16, 16), "white") as source:
        regions = analyze_reference_regions(
            source, cfg, tmp_path,
            allow_parser_foreground_fallback=True)
    assert "--allow-parser-foreground-fallback" in seen[0]
    regions.close()



def test_base_output_scope_ignores_target_garment_tags_and_allows_empty_garment(
        tmp_path, monkeypatch):
    seen = []

    def run(command, cwd, timeout, cancelled):
        output = Path(command[command.index("--reference-output-dir") + 1])
        write_regions(output, original_size=(16, 16))
        with Image.new("L", (16, 16)) as empty:
            empty.save(output / "garment.png")
        record_path = output / "regions.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["pixel_counts"]["garment"] = 0
        record_path.write_text(json.dumps(record), encoding="utf-8")
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "genai_lab.reference_regions.run_reference_process", run)
    cfg = {"character_body_comparison": {
        "python_executable": sys.executable, "runner_path": __file__,
        "repository_path": str(tmp_path), "temporary_root": str(tmp_path),
        "cache_dir": str(tmp_path), "width": 256, "height": 256,
        "timeout_seconds": 60},
        "clothing_reference_generation": {
            "approved_tags": ("leotard", "pantyhose"),
        }}
    with Image.new("RGB", (16, 16), "white") as source:
        regions = analyze_reference_regions(
            source, cfg, tmp_path, analysis_scope="base_output")
    try:
        assert "--approved-garment-tags-json" not in seen[0]
        assert regions.record["analysis_scope"] == "base_output"
        assert regions.record["required_regions"] == ["identity"]
        assert regions.record["target_garment_tags_applied"] is False
    finally:
        regions.close()


def test_final_output_scope_applies_target_garment_tags(tmp_path, monkeypatch):
    seen = []

    def run(command, cwd, timeout, cancelled):
        output = Path(command[command.index("--reference-output-dir") + 1])
        write_regions(
            output, original_size=(16, 16), approved_tags=("leotard",))
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "genai_lab.reference_regions.run_reference_process", run)
    cfg = {"character_body_comparison": {
        "python_executable": sys.executable, "runner_path": __file__,
        "repository_path": str(tmp_path), "temporary_root": str(tmp_path),
        "cache_dir": str(tmp_path), "width": 256, "height": 256,
        "timeout_seconds": 60},
        "clothing_reference_generation": {
            "approved_tags": ("leotard",),
        }}
    with Image.new("RGB", (16, 16), "white") as source:
        regions = analyze_reference_regions(
            source, cfg, tmp_path, analysis_scope="final_output")
    try:
        tag_flag = seen[0].index("--approved-garment-tags-json")
        assert json.loads(seen[0][tag_flag + 1]) == ["leotard"]
        assert regions.record["required_regions"] == ["identity", "garment"]
        assert regions.record["target_garment_tags_applied"] is True
    finally:
        regions.close()


def test_cancel_prevents_process_start(tmp_path, monkeypatch):
    monkeypatch.setattr("genai_lab.reference_regions.subprocess.Popen",
                        lambda *a, **k: pytest.fail("취소 후 실행하면 안 됩니다."))
    with pytest.raises(InterruptedError):
        run_reference_process(["unused"], tmp_path, 10, lambda: True)


def test_owned_process_is_stopped_on_timeout(tmp_path):
    with pytest.raises(TimeoutError):
        run_reference_process([sys.executable, "-c", "import time; time.sleep(10)"],
                              tmp_path, .1, lambda: False)
