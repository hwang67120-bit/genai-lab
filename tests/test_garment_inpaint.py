from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest
from PIL import Image

from genai_lab.garment_inpaint import (
    GarmentGenerationEngine,
    GarmentInpaintError,
    GarmentInpaintSettings,
    SubprocessSDXLGarmentGenerationEngine,
    approve_garment_inpaint_review,
    execute_garment_inpaint,
)


def test_default_subprocess_engine_implements_generation_contract():
    assert issubclass(
        SubprocessSDXLGarmentGenerationEngine,
        GarmentGenerationEngine,
    )


def _inputs(size=(32, 32)):
    base = Image.new("RGB", size, (245, 245, 245))
    mask_array = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask_array[8:24, 7:25] = 255
    mask = Image.fromarray(mask_array)
    human_agnostic = base.copy()
    human_agnostic.paste((127, 127, 127), (7, 8, 25, 24))
    garment = Image.new("RGBA", (20, 24), (0, 0, 0, 0))
    garment.paste((20, 80, 180, 255), (3, 3, 17, 21))
    return base, human_agnostic, mask, garment


def _settings(tmp_path):
    python_path = tmp_path / "python.exe"
    runner_path = tmp_path / "runner.py"
    python_path.write_bytes(b"")
    runner_path.write_text("# test", encoding="utf-8")
    return GarmentInpaintSettings(
        python_executable=python_path,
        runner_path=runner_path,
        temporary_root=tmp_path / "runtime",
        cache_dir=tmp_path / "cache",
    )


def _success(command, **kwargs):
    output = Path(command[command.index("--output-image") + 1])
    width = int(command[command.index("--width") + 1])
    height = int(command[command.index("--height") + 1])
    Image.new("RGB", (width, height), (10, 20, 30)).save(output)
    if "--prompt-record-file" in command:
        prompt_record = Path(
            command[command.index("--prompt-record-file") + 1]
        )
        prompt_record.write_text(
            json.dumps({
                "adapter_dimensions": {
                    "image_encoder_hidden_size": 1280,
                    "adapter_projection_input_size": 1280,
                },
                "prompt": {
                    "original": "blue tailored jacket",
                    "effective": "blue tailored jacket",
                    "truncated": False,
                },
            }),
            encoding="utf-8",
        )
    return subprocess.CompletedProcess(command, 0, "ok", "")


def _close(inputs):
    for item in inputs:
        item.close()


def test_execution_protects_outside_and_exposes_eight_previews(
    tmp_path, monkeypatch
):
    inputs = _inputs()
    monkeypatch.setattr("genai_lab.garment_inpaint.subprocess.run", _success)
    candidate = execute_garment_inpaint(
        *inputs, prompt="blue tailored jacket", negative_prompt="blurred",
        seed=42, settings=_settings(tmp_path),
    )
    try:
        assert candidate.raw_changed_outside_mask_pixels > 0
        assert candidate.protected_changed_outside_mask_pixels == 0
        assert candidate.protected_changed_inside_mask_pixels > 0
        assert candidate.benchmark_file_count == 10
        assert candidate.benchmark_directory.is_dir()
        assert (candidate.benchmark_directory / "raw_output_A.png").is_file()
        assert (candidate.benchmark_directory / "protected_output.png").is_file()
        metadata = json.loads(
            (candidate.benchmark_directory / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["status"] == "completed"
        assert metadata["schema_version"] == 4
        assert metadata["initial_image_source"] == "approved_human_agnostic"
        assert metadata["tps_rgb_composite_enabled"] is False
        assert metadata["garment_controlnet_enabled"] is False
        assert metadata["base_model_id"] == (
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        )
        assert metadata["model_variant"] == "fp16"
        assert metadata["adapter_image_encoder_subfolder"] == (
            "models/image_encoder"
        )
        assert metadata["prompt_execution"]["adapter_dimensions"] == {
            "image_encoder_hidden_size": 1280,
            "adapter_projection_input_size": 1280,
        }
        assert metadata["garment_reference_board"][
            "retained_component_count"
        ] == 1
        assert metadata["seed"] == 42
        assert metadata["input_files"]["mask"]["sha256"]
        assert metadata["input_files"]["garment_board"]["sha256"]
        assert metadata["audit_files"]["garment_source"]["sha256"]
        assert candidate.automatic_save_count == 0
        assert candidate.garment_reference_preview.getpixel((0, 0)) == (
            255, 255, 255,
        )
        assert candidate.garment_reference_board_preview.size == (1024, 1024)
        assert candidate.garment_retained_component_count == 1
        assert candidate.garment_board_occupied_pixel_count > 0
    finally:
        candidate.close()
        _close(inputs)


def test_initial_image_is_approved_human_agnostic_image(tmp_path, monkeypatch):
    inputs = _inputs()
    monkeypatch.setattr("genai_lab.garment_inpaint.subprocess.run", _success)
    candidate = execute_garment_inpaint(
        *inputs, prompt="blue jacket", negative_prompt="", seed=1,
        settings=_settings(tmp_path),
    )
    try:
        assert candidate.base_character_preview.getpixel((10, 10)) == (
            245, 245, 245,
        )
        assert candidate.human_agnostic_preview.getpixel((10, 10)) == (
            127, 127, 127,
        )
    finally:
        candidate.close()
        _close(inputs)


def test_neutral_residual_warning_does_not_block_gray_garment(tmp_path, monkeypatch):
    inputs = _inputs()
    def partial_generation(command, **kwargs):
        initial_path = command[command.index("--initial-image") + 1]
        output_path = command[command.index("--output-image") + 1]
        with Image.open(initial_path) as initial:
            initial.putpixel((10, 10), (20, 80, 180))
            initial.save(output_path)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("genai_lab.garment_inpaint.subprocess.run", partial_generation)
    candidate = execute_garment_inpaint(*inputs, prompt="gray jacket", negative_prompt="", seed=1, settings=_settings(tmp_path))
    try:
        assert candidate.neutral_residual.suspected_pixel_count > 0
        approved = approve_garment_inpaint_review(candidate)
        approved.close()
        metadata = json.loads((candidate.benchmark_directory / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["neutral_residual"]["severity"] == "warning_only"
        assert metadata["neutral_residual"]["suspected_pixel_count"] > 0
    finally:
        candidate.close()
        _close(inputs)


def test_runner_command_uses_no_tps_lineart_or_controlnet(tmp_path, monkeypatch):
    inputs = _inputs()
    captured: list[str] = []

    def capture(command, **kwargs):
        captured.extend(command)
        return _success(command, **kwargs)

    monkeypatch.setattr("genai_lab.garment_inpaint.subprocess.run", capture)
    candidate = execute_garment_inpaint(
        *inputs, prompt="blue jacket", negative_prompt="", seed=1,
        settings=_settings(tmp_path),
    )
    try:
        assert "--control-image" not in captured
        assert "--controlnet-model-id" not in captured
        assert "--controlnet-conditioning-scale" not in captured
        assert "--initial-image" in captured
        assert "--garment-image" in captured
        assert "--model-variant" in captured
        assert captured[
            captured.index("--adapter-image-encoder-subfolder") + 1
        ] == "models/image_encoder"
        assert "--prompt-record-file" in captured
        assert captured[captured.index("--garment-image") + 1].endswith(
            "garment_board.png"
        )
    finally:
        candidate.close()
        _close(inputs)


def test_size_mismatch_is_blocked_before_process(tmp_path):
    inputs = list(_inputs())
    inputs[2].close()
    inputs[2] = Image.new("L", (31, 32), 255)
    try:
        with pytest.raises(GarmentInpaintError, match="크기가"):
            execute_garment_inpaint(
                *inputs, prompt="jacket", negative_prompt="", seed=1,
                settings=_settings(tmp_path),
            )
    finally:
        _close(inputs)


def test_human_agnostic_change_outside_mask_is_blocked(tmp_path):
    inputs = _inputs()
    inputs[1].putpixel((0, 0), (127, 127, 127))
    try:
        with pytest.raises(GarmentInpaintError, match="1px"):
            execute_garment_inpaint(
                *inputs, prompt="jacket", negative_prompt="", seed=1,
                settings=_settings(tmp_path),
            )
    finally:
        _close(inputs)


def test_subprocess_failure_exposes_runner_error(tmp_path, monkeypatch):
    inputs = _inputs()
    monkeypatch.setattr(
        "genai_lab.garment_inpaint.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 2, "", "CUDA failure"
        ),
    )
    try:
        with pytest.raises(GarmentInpaintError, match="CUDA failure"):
            execute_garment_inpaint(
                *inputs, prompt="jacket", negative_prompt="", seed=1,
                settings=_settings(tmp_path),
            )
    finally:
        _close(inputs)


def test_missing_output_is_blocked(tmp_path, monkeypatch):
    inputs = _inputs()
    monkeypatch.setattr(
        "genai_lab.garment_inpaint.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    try:
        with pytest.raises(GarmentInpaintError, match="결과 이미지를"):
            execute_garment_inpaint(
                *inputs, prompt="jacket", negative_prompt="", seed=1,
                settings=_settings(tmp_path),
            )
    finally:
        _close(inputs)


@pytest.mark.parametrize(
    "change",
    [
        {"strength": 1.1},
        {"inference_steps": 0},
        {"ip_adapter_scale": -0.1},
        {"dtype": "float32"},
    ],
)
def test_invalid_settings_are_blocked(tmp_path, change):
    inputs = _inputs()
    try:
        with pytest.raises(GarmentInpaintError):
            execute_garment_inpaint(
                *inputs, prompt="jacket", negative_prompt="", seed=1,
                settings=replace(_settings(tmp_path), **change),
            )
    finally:
        _close(inputs)


def test_explicit_approval_returns_owned_copy(tmp_path, monkeypatch):
    inputs = _inputs()
    monkeypatch.setattr("genai_lab.garment_inpaint.subprocess.run", _success)
    candidate = execute_garment_inpaint(
        *inputs, prompt="jacket", negative_prompt="", seed=1,
        settings=_settings(tmp_path),
    )
    approved = approve_garment_inpaint_review(candidate)
    try:
        assert approved.image is not candidate.protected_output
        assert approved.changed_outside_mask_pixels == 0
        assert approved.changed_inside_mask_pixels > 0
    finally:
        approved.close()
        candidate.close()
        _close(inputs)


def test_empty_prompt_is_blocked(tmp_path):
    inputs = _inputs()
    try:
        with pytest.raises(GarmentInpaintError, match="비어"):
            execute_garment_inpaint(
                *inputs, prompt=" ", negative_prompt="", seed=1,
                settings=_settings(tmp_path),
            )
    finally:
        _close(inputs)


def test_unchanged_initial_output_cannot_be_approved(tmp_path, monkeypatch):
    inputs = _inputs()

    def unchanged(command, **kwargs):
        source = Path(command[command.index("--initial-image") + 1])
        output = Path(command[command.index("--output-image") + 1])
        with Image.open(source) as opened:
            opened.convert("RGB").save(output)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("genai_lab.garment_inpaint.subprocess.run", unchanged)
    candidate = execute_garment_inpaint(
        *inputs, prompt="jacket", negative_prompt="", seed=1,
        settings=_settings(tmp_path),
    )
    try:
        assert candidate.protected_changed_inside_mask_pixels > 0
        assert candidate.raw_changed_from_initial_inside_mask_pixels == 0
        with pytest.raises(GarmentInpaintError, match="Human-Agnostic"):
            approve_garment_inpaint_review(candidate)
    finally:
        candidate.close()
        _close(inputs)


def test_progress_events_are_collected_without_changing_output(
    tmp_path,
    monkeypatch,
):
    inputs = _inputs()
    received = []

    def success_with_progress(command, **kwargs):
        progress_path = Path(command[command.index("--progress-file") + 1])
        events = (
            {
                "phase": "runner_started",
                "phase_elapsed_seconds": 0.0,
                "total_elapsed_seconds": 0.0,
                "message": "started",
            },
            {
                "phase": "pipeline_loading",
                "phase_elapsed_seconds": 3.0,
                "total_elapsed_seconds": 3.0,
                "message": "completed",
            },
            {
                "phase": "ip_adapter_loading",
                "phase_elapsed_seconds": 1.0,
                "total_elapsed_seconds": 4.0,
                "message": "completed",
            },
            {
                "phase": "diffusion_running",
                "phase_elapsed_seconds": 10.0,
                "total_elapsed_seconds": 14.0,
                "current_step": 18,
                "configured_steps": 28,
                "message": "completed",
            },
            {
                "phase": "output_saving",
                "phase_elapsed_seconds": 0.5,
                "total_elapsed_seconds": 14.5,
                "message": "completed",
            },
            {
                "phase": "completed",
                "phase_elapsed_seconds": 14.5,
                "total_elapsed_seconds": 14.5,
                "current_step": 18,
                "configured_steps": 28,
                "message": "completed",
            },
        )
        with progress_path.open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event) + "\n")
            stream.write("{invalid-json\n")
        return _success(command, **kwargs)

    monkeypatch.setattr(
        "genai_lab.garment_inpaint.subprocess.run",
        success_with_progress,
    )
    candidate = execute_garment_inpaint(
        *inputs,
        prompt="jacket",
        negative_prompt="",
        seed=1,
        settings=_settings(tmp_path),
        progress_callback=received.append,
    )
    try:
        metrics = candidate.execution_metrics
        assert metrics.pipeline_load_seconds == 3.0
        assert metrics.ip_adapter_load_seconds == 1.0
        assert metrics.diffusion_seconds == 10.0
        assert metrics.output_save_seconds == 0.5
        assert metrics.runner_total_seconds == 14.5
        assert metrics.parent_total_seconds > 0.0
        assert metrics.progress_event_count == 6
        assert metrics.invalid_progress_event_count == 1
        assert metrics.heartbeat_count == 0
        assert len(received) == 6
        assert received[-1].current_step == 18
        assert candidate.raw_inpaint_output.getpixel((0, 0)) == (10, 20, 30)
        assert candidate.protected_changed_outside_mask_pixels == 0
    finally:
        candidate.close()
        _close(inputs)


def test_progress_heartbeat_is_emitted_within_configured_interval(
    tmp_path,
    monkeypatch,
):
    inputs = _inputs()
    received = []

    def slow_success(command, **kwargs):
        time.sleep(0.04)
        return _success(command, **kwargs)

    monkeypatch.setattr(
        "genai_lab.garment_inpaint.PROGRESS_POLL_INTERVAL_SECONDS",
        0.005,
    )
    monkeypatch.setattr(
        "genai_lab.garment_inpaint.PROGRESS_HEARTBEAT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        "genai_lab.garment_inpaint.subprocess.run",
        slow_success,
    )
    candidate = execute_garment_inpaint(
        *inputs,
        prompt="jacket",
        negative_prompt="",
        seed=1,
        settings=_settings(tmp_path),
        progress_callback=received.append,
    )
    try:
        assert candidate.execution_metrics.heartbeat_count >= 1
        assert any(event.message == "heartbeat" for event in received)
    finally:
        candidate.close()
        _close(inputs)


def test_progress_monitor_keeps_existing_subprocess_timeout(
    tmp_path,
    monkeypatch,
):
    inputs = _inputs()

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 1800
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(
        "genai_lab.garment_inpaint.subprocess.run",
        timeout,
    )
    try:
        with pytest.raises(GarmentInpaintError, match="1800초 초과"):
            execute_garment_inpaint(
                *inputs,
                prompt="jacket",
                negative_prompt="",
                seed=1,
                settings=_settings(tmp_path),
            )
        benchmark_directories = tuple(
            (tmp_path / "runtime" / "debug_benchmark").iterdir()
        )
        assert len(benchmark_directories) == 1
        benchmark_directory = benchmark_directories[0]
        metadata = json.loads(
            (benchmark_directory / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["status"] == "timeout"
        assert metadata["seed"] == 1
        assert (benchmark_directory / "initial.png").is_file()
        assert (benchmark_directory / "mask.png").is_file()
        assert not (benchmark_directory / "lineart.png").exists()
        assert (benchmark_directory / "garment_source.png").is_file()
        assert (benchmark_directory / "garment_board.png").is_file()
        assert (benchmark_directory / "stdout.log").is_file()
        assert (benchmark_directory / "stderr.log").is_file()
    finally:
        _close(inputs)
