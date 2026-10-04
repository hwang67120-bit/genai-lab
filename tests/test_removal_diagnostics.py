import json

import numpy as np
from PIL import Image
import pytest

from genai_lab import removal_diagnostics as diagnostics
from genai_lab.body_comparison import create_human_agnostic_image_candidate


def records():
    paths = list(diagnostics.DIAGNOSTIC_ROOT.glob("*.jsonl"))
    return paths, [json.loads(line) for path in paths for line in
                   path.read_text(encoding="utf-8").splitlines()]


def test_mask_holes_are_measured_without_modifying_input():
    array = np.zeros((20, 20), dtype=np.uint8)
    array[2:18, 2:18] = 255
    array[8:10, 8:10] = 0
    with Image.fromarray(array) as image:
        evidence = diagnostics.image_evidence(image)
        assert evidence["selected_pixels"] == 252
        assert evidence["enclosed_zero_pixels"] == 4
        assert evidence["enclosed_zero_components"] == 1
        assert evidence["largest_enclosed_zero_regions"][0]["bbox_xywh"] == [8, 8, 2, 2]
        assert np.array_equal(np.asarray(image), array)


def test_nested_stages_share_run_id_and_failure_is_preserved():
    @diagnostics.trace_removal("inner")
    def inner():
        raise ValueError("test failure")

    @diagnostics.trace_removal("outer")
    def outer():
        inner()

    with pytest.raises(ValueError, match="test failure"):
        outer()
    paths, events = records()
    assert len(paths) == 1
    assert [(e["stage"], e["event"]) for e in events] == [
        ("outer", "start"), ("inner", "start"), ("inner", "failed"), ("outer", "failed")]
    assert len({e["run_id"] for e in events}) == 1
    assert diagnostics._active.get() is None


def test_real_neutralization_records_input_and_output_evidence():
    with Image.new("RGB", (16, 16), "red") as source, Image.new("L", (16, 16), 0) as mask:
        mask.paste(255, (4, 4, 12, 12))
        result = create_human_agnostic_image_candidate(source, mask, mask)
        try:
            assert result.neutralized_image.getpixel((6, 6)) == (127, 127, 127)
            assert result.neutralized_image.getpixel((0, 0)) == (255, 0, 0)
            _, events = records()
            assert events[0]["source_rgb_sha256"]
            assert events[0]["inputs"]["clothing_erasure_mask"]["selected_pixels"] == 64
            assert events[-1]["outputs"]["neutralized_pixel_count"] == 64
            assert events[-1]["outputs"]["changed_pixel_count_outside_mask"] == 0
            assert events[0]["coverage_meaning"] == "selected_mask_coverage_not_visual_clothing_removal"
        finally:
            result.close()


def test_diagnostic_disk_failure_does_not_change_result(tmp_path, monkeypatch):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(diagnostics, "DIAGNOSTIC_ROOT", blocked)
    sentinel = object()

    @diagnostics.trace_removal("test")
    def operation():
        return sentinel

    assert operation() is sentinel


def test_separate_calls_get_different_ids():
    @diagnostics.trace_removal("test")
    def operation():
        return 1

    operation()
    operation()
    paths, events = records()
    assert len(paths) == 2
    assert len({e["run_id"] for e in events}) == 2
