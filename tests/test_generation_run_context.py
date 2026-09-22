import json

import pytest

from genai_lab.generation_run_context import (
    GenerationRunContext,
    GenerationRunContextError,
    sha256_file,
)


def test_run_context_creates_isolated_request_directories(tmp_path):
    context = GenerationRunContext.create(
        {
            "generation_run_context": {
                "root": str(tmp_path / "outputs"),
                "training_roots": [str(tmp_path / "training")],
            }
        },
        tmp_path,
        refinement_mode="sdxl_local",
    )

    directories = {
        context.inputs_directory,
        context.base_directory,
        context.masks_directory,
        context.refinement_directory,
        context.diagnostics_directory,
    }
    assert len(directories) == 5
    assert all(path.is_dir() for path in directories)
    assert all(path.parent == context.run_directory for path in directories)
    assert context.directory_for_mode("sdxl_local").is_dir()
    with pytest.raises(GenerationRunContextError, match="봉인"):
        context.directory_for_mode("flux_whole_image")


def test_run_context_rejects_training_output_overlap(tmp_path):
    shared = tmp_path / "shared"
    with pytest.raises(GenerationRunContextError, match="겹칩니다"):
        GenerationRunContext.create(
            {
                "generation_run_context": {
                    "root": str(shared / "outputs"),
                    "training_roots": [str(shared)],
                }
            },
            tmp_path,
            refinement_mode="flux_whole_image",
        )


def test_run_context_records_artifact_digest(tmp_path):
    context = GenerationRunContext.create(
        {
            "generation_run_context": {
                "root": str(tmp_path / "outputs"),
                "training_roots": [str(tmp_path / "training")],
            }
        },
        tmp_path,
        refinement_mode="flux_whole_image",
    )
    artifact = context.inputs_directory / "reference.bin"
    artifact.write_bytes(b"sealed-reference")
    context.record_artifact("reference", artifact, digest=True)

    record = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert record["artifacts"]["reference"]["sha256"] == sha256_file(artifact)


def test_stage_event_does_not_mark_whole_run_completed(tmp_path):
    context = GenerationRunContext.create(
        {
            "generation_run_context": {
                "root": str(tmp_path / "outputs"),
                "training_roots": [str(tmp_path / "training")],
            }
        },
        tmp_path,
        refinement_mode="flux_whole_image",
    )

    context.event("select_base_candidate", "completed", candidate_number=1)

    record = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert record["status"] == "created"
    assert record["events"][-1]["status"] == "completed"

    context.finish("completed")
    record = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert "finished_at" in record
