import json
import sys

import pytest
from pathlib import Path

from PIL import Image

from genai_lab.native_refinement_execution import (
    NativeRefinementExecutionError,
    NativeRefinementExecutionSettings,
    build_native_refinement_command,
    execute_native_refinement,
)


def create_settings(
    tmp_path: Path,
    runner_path: Path,
) -> NativeRefinementExecutionSettings:
    cache = tmp_path / "cache"
    cache.mkdir()
    return NativeRefinementExecutionSettings(
        python_executable=Path(sys.executable),
        runner_path=runner_path,
        model_cache_dir=cache,
        width=32,
        height=48,
        flux_steps=4,
        minimum_similarity=0.65,
        minimum_color_similarity=0.55,
        timeout_seconds=30,
        local_files_only=True,
    )


def write_fake_runner(path: Path, status: str) -> None:
    path.write_text(
        """import json
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
def value(name):
    return Path(args[args.index(name) + 1])

output = value("--output-dir")
output.mkdir(parents=True, exist_ok=True)
base = value("--base-image")
if %r == "PASS":
    selected = output / "selected.png"
    shutil.copyfile(base, selected)
    report = {
        "status": "PASS",
        "selected_output": str(selected),
        "decision": {"selected_engine": "flux2_klein"},
        "attempts": {"flux2_klein": {"gate": {"status": "PASS"}}},
    }
    code = 0
else:
    report = {
        "status": "BASE_LOCKED",
        "decision": {"selected_engine": None},
        "attempts": {
            "flux2_klein": {"gate": {"status": "FAIL"}},
        },
    }
    code = 2
(output / "run.json").write_text(
    json.dumps(report), encoding="utf-8"
)
raise SystemExit(code)
"""
        % status,
        encoding="utf-8",
    )


def create_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    base = tmp_path / "base.png"
    garment = tmp_path / "input_garment.png"
    candidate = tmp_path / "candidate.json"
    approval = tmp_path / "approval.json"
    Image.new("RGB", (16, 24), "blue").save(base)
    Image.new("RGB", (16, 24), "green").save(garment)
    candidate.write_text("{}", encoding="utf-8")
    approval.write_text("{}", encoding="utf-8")
    return base, garment, candidate, approval


def test_command_is_offline_by_default(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("", encoding="utf-8")
    settings = create_settings(tmp_path, runner)
    base, garment, candidate, approval = create_inputs(tmp_path)

    command = build_native_refinement_command(
        settings,
        base_image=base,
        candidate_record=candidate,
        approved_run=approval,
        garment_reference=garment,
        output_directory=tmp_path / "output",
    )

    assert "--allow-download" not in command
    assert command[0] == sys.executable
    assert command[2] == str(runner)
    assert command[command.index("--model-cache-dir") + 1] == str(
        settings.model_cache_dir
    )
    assert command[command.index("--garment-reference") + 1] == str(garment)


def test_execution_returns_flux_pass_result(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    write_fake_runner(runner, "PASS")
    settings = create_settings(tmp_path, runner)
    base, garment, candidate, approval = create_inputs(tmp_path)
    messages: list[str] = []

    result = execute_native_refinement(
        settings,
        project_root=tmp_path,
        base_image=base,
        candidate_record=candidate,
        approved_run=approval,
        garment_reference=garment,
        output_directory=tmp_path / "output",
        status_callback=messages.append,
    )

    assert result.status == "PASS"
    assert result.selected_engine == "flux2_klein"
    assert result.selected_image_path.is_file()
    assert result.report_path.is_file()
    assert messages


def test_execution_returns_approved_base_when_all_fail(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    write_fake_runner(runner, "BASE_LOCKED")
    settings = create_settings(tmp_path, runner)
    base, garment, candidate, approval = create_inputs(tmp_path)

    result = execute_native_refinement(
        settings,
        project_root=tmp_path,
        base_image=base,
        candidate_record=candidate,
        approved_run=approval,
        garment_reference=garment,
        output_directory=tmp_path / "output",
    )

    assert result.status == "BASE_LOCKED"
    assert result.selected_engine is None
    assert result.selected_image_path == base



def test_command_records_direct_source_input_mode(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("", encoding="utf-8")
    settings = create_settings(tmp_path, runner)
    base, garment, candidate, approval = create_inputs(tmp_path)

    command = build_native_refinement_command(
        settings,
        base_image=base,
        candidate_record=candidate,
        approved_run=approval,
        garment_reference=garment,
        output_directory=tmp_path / "output",
        input_mode="source_character_direct",
    )

    assert command[command.index("--input-mode") + 1] == (
        "source_character_direct"
    )


def test_execution_cancellation_terminates_subprocess(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import time\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    settings = create_settings(tmp_path, runner)
    base, garment, candidate, approval = create_inputs(tmp_path)

    with pytest.raises(
        NativeRefinementExecutionError,
        match="사용자가 로컬 정밀화를 취소",
    ):
        execute_native_refinement(
            settings,
            project_root=tmp_path,
            base_image=base,
            candidate_record=candidate,
            approved_run=approval,
            garment_reference=garment,
            output_directory=tmp_path / "output",
            cancelled=lambda: True,
        )
