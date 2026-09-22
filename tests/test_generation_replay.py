from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from genai_lab.approved_reference_run import ApprovedReferenceRun, canonical
from genai_lab.generation_replay import (
    load_generation_replay_bundle,
    save_generation_replay_bundle,
)
from genai_lab.request import (
    CharacterFramingType,
    CharacterGenerationRequest,
)
from genai_lab.visual_reference import VisualInputs


def make_request(reference):
    return CharacterGenerationRequest(
        reference_image=reference,
        reference_image_name="character.png",
        reference_enhancement_applied=False,
        reference_enhancement_model_id=None,
        reference_quality_status="good",
        framing_type=CharacterFramingType.FULL_BODY,
        width=8,
        height=8,
        prompt="approved prompt",
        negative_prompt="approved negative",
        seed=1234,
        candidate_number=1,
        inference_steps=20,
        guidance_scale=7.0,
        original_image_change_strength=0.25,
        reference_image_strength=0.8,
        model_id="model",
        reference_adapter_id="adapter",
        body_proportion_preset_id="standard_7_5h_shoulder",
    )


def test_generation_replay_round_trip_preserves_images_and_approval(
        tmp_path, monkeypatch):
    source = Image.new("RGB", (8, 8), "white")
    identity_mask = Image.new("L", (8, 8))
    garment_mask = Image.new("L", (8, 8))
    identity_mask.putpixel((1, 1), 255)
    garment_mask.putpixel((6, 6), 255)
    inputs = VisualInputs(
        source=source.copy(),
        identity=Image.new("RGB", (4, 4), "red"),
        garment=Image.new("RGB", (4, 4), "blue"),
        identity_mask=identity_mask,
        garment_mask=garment_mask,
        analysis_record={},
        color_evidence={},
        part_color_descriptions=(),
    )
    payload_record = {"version": "test-approval"}
    payload = canonical(payload_record)
    approval = ApprovedReferenceRun(
        payload, hashlib.sha256(payload).hexdigest())
    inputs.approved_generation = approval
    request = make_request(source)
    config = {
        "paths": {"output_dir": "replays"},
        "clothing_reference_generation": {
            "part_color_descriptions": (),
            "approved_part_color_descriptions": (),
        },
    }

    directory = save_generation_replay_bundle(
        inputs, config, request, tmp_path)
    monkeypatch.setattr(
        "genai_lab.approved_reference_run.require_reference_run",
        lambda loaded, loaded_config, loaded_request: approval,
    )
    loaded, loaded_config, loaded_request, loaded_approval = (
        load_generation_replay_bundle(directory)
    )
    try:
        assert loaded.source.tobytes() == inputs.source.tobytes()
        assert loaded.identity_mask.tobytes() == inputs.identity_mask.tobytes()
        assert loaded.garment_mask.tobytes() == inputs.garment_mask.tobytes()
        assert loaded_request.seed == request.seed
        assert loaded_request.prompt == request.prompt
        assert loaded_request.body_morphology == request.body_morphology
        assert loaded_request.body_proportion_preset_id == request.body_proportion_preset_id
        assert loaded_config["paths"]["output_dir"] == "replays"
        assert loaded_approval.fingerprint == approval.fingerprint
        assert (directory / "bundle.json").is_file()
        assert (directory / "approval.json").is_file()
        assert (directory / "config.json").is_file()
        bundle_record = json.loads(
            (directory / "bundle.json").read_text(encoding="utf-8"))
        assert "config.json" in bundle_record["artifact_sha256"]
        assert "source.png" in bundle_record["artifact_sha256"]
    finally:
        loaded.close()
        loaded_request.reference_image.close()
        inputs.close()
        request.reference_image.close()


def test_restore_approved_inputs_does_not_create_new_approval(tmp_path):
    from genai_lab.generation_orchestrator import (
        GenerationOrchestrator,
        GenerationPhase,
    )

    approval = SimpleNamespace(fingerprint="b" * 64)
    inputs = SimpleNamespace(approved_generation=approval)
    calls = []
    orchestrator = GenerationOrchestrator(
        {},
        SimpleNamespace(),
        tmp_path,
        SimpleNamespace(write_stage=lambda *args: None),
        prepare_visual_inputs_fn=lambda *args: None,
        generate_visual_batch_fn=lambda *args, **kwargs: None,
        approve_reference_run_fn=lambda *args: calls.append("approve"),
        require_reference_run_fn=lambda *args: calls.append("require") or approval,
        save_replay_bundle_fn=lambda *args: calls.append("save"),
    )

    assert orchestrator.restore_approved_inputs(inputs) == "b" * 64
    assert orchestrator.phase is GenerationPhase.INPUTS_APPROVED
    assert calls == ["require"]


def test_generation_replay_rejects_changed_artifact(tmp_path, monkeypatch):
    source = Image.new("RGB", (8, 8), "white")
    identity_mask = Image.new("L", (8, 8))
    garment_mask = Image.new("L", (8, 8))
    identity_mask.putpixel((1, 1), 255)
    garment_mask.putpixel((6, 6), 255)
    inputs = VisualInputs(
        source=source.copy(),
        identity=Image.new("RGB", (4, 4), "red"),
        garment=Image.new("RGB", (4, 4), "blue"),
        identity_mask=identity_mask,
        garment_mask=garment_mask,
        analysis_record={},
        color_evidence={},
    )
    payload = canonical({"version": "tamper-test"})
    inputs.approved_generation = ApprovedReferenceRun(
        payload, hashlib.sha256(payload).hexdigest())
    request = make_request(source)
    config = {"paths": {"output_dir": "replays"},
              "clothing_reference_generation": {}}
    directory = save_generation_replay_bundle(
        inputs, config, request, tmp_path)
    (directory / "source.png").write_bytes(b"changed")
    with pytest.raises(Exception, match="변경됐습니다"):
        load_generation_replay_bundle(directory)
    inputs.close()
    request.reference_image.close()


def test_json_safe_preserves_analysis_keys_reserved_only_for_config():
    from genai_lab.generation_replay import _json_safe

    record = {
        "part_color_descriptions": [{"part_name": "animal_ears"}],
        "nested": {
            "approved_part_color_descriptions": [{"part_name": "tail"}],
        },
    }

    assert _json_safe(record) == record
    assert _json_safe(record, omit_config_objects=True) == {
        "nested": {},
    }


