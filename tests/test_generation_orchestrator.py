from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image
import pytest

from genai_lab.generation_orchestrator import (
    BaseCandidateSelection,
    GenerationOrchestrationError,
    GenerationOrchestrator,
    GenerationPhase,
)
from genai_lab.final_candidate_review import (
    APPROVED,
    create_final_review_decision,
)
from genai_lab.final_result_storage import (
    SAVED,
    create_storage_record,
    write_storage_record,
)


class Log:
    def __init__(self):
        self.entries = []

    def write_stage(self, stage, detail):
        self.entries.append((stage, detail))


def test_product_orchestrator_enforces_shared_base_flow(tmp_path):
    order = []
    inputs = SimpleNamespace(approved_generation=None)
    approval = SimpleNamespace(fingerprint="a" * 64)

    def prepare(*args):
        order.append("prepare")
        return inputs

    def approve(value, config, request):
        order.append("approve")
        value.approved_generation = approval

    def require(value, config, request):
        order.append("require")
        assert value.approved_generation is approval
        return approval

    batch = SimpleNamespace(
        directory=tmp_path,
        review_stage="native_base",
        candidates=[object()],
    )

    def generate(*args, **kwargs):
        order.append("generate")
        assert kwargs["defer_final_refinement"] is True
        return batch

    config = {
        "native_pipeline_v2": {
            "enabled": True,
            "base_profile": "character_only",
            "base_garment_prompt_enabled": False,
            "base_garment_adapter_enabled": False,
            "native_garment_source": "isolated_garment_board",
            "reuse_animagine_prompt_in_native_stage": False,
            "raw_garment_person_image_allowed": False,
        },
        "refinement_execution": {"enabled": True, "mode": "flux_whole_image"},
        "native_refinement": {"enabled": True},
    }
    request = SimpleNamespace(seed=1234, model_id="animagine")
    orchestrator = GenerationOrchestrator(
        config,
        request,
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=prepare,
        generate_visual_batch_fn=generate,
        approve_reference_run_fn=approve,
        require_reference_run_fn=require,
        save_replay_bundle_fn=lambda *args: tmp_path / "replay",
    )

    assert orchestrator.prepare_visual_inputs(object(), object()) is inputs
    assert orchestrator.approve_visual_inputs(inputs) == "a" * 64
    assert orchestrator.generate_base_candidates(object(), inputs) is batch
    assert order == ["prepare", "approve", "require", "generate"]
    assert orchestrator.phase is GenerationPhase.BASE_COMPLETED

    record = json.loads(
        (tmp_path / "product_execution.json").read_text(encoding="utf-8")
    )
    assert record["execution_scope"] == "product_pipeline"
    assert record["runtime_smoke"] is False
    assert record["status"] == "BASE_GATE_PASS"
    assert record["final_return_eligible"] is False
    assert record["approval_fingerprint"] == "a" * 64


def test_product_orchestrator_rejects_generation_before_approval(tmp_path):
    inputs = SimpleNamespace(approved_generation=None)
    orchestrator = GenerationOrchestrator(
        {},
        SimpleNamespace(),
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=lambda *args: inputs,
        generate_visual_batch_fn=lambda *args, **kwargs: pytest.fail(
            "승인 전에 생성하면 안 됩니다."
        ),
        approve_reference_run_fn=lambda *args: None,
        require_reference_run_fn=lambda *args: None,
    )
    orchestrator.prepare_visual_inputs(object(), object())
    with pytest.raises(GenerationOrchestrationError, match="단계 순서"):
        orchestrator.generate_base_candidates(object(), inputs)


def test_product_orchestrator_rejects_different_input_object(tmp_path):
    inputs = SimpleNamespace(approved_generation=None)
    orchestrator = GenerationOrchestrator(
        {},
        SimpleNamespace(),
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=lambda *args: inputs,
        generate_visual_batch_fn=lambda *args, **kwargs: None,
        approve_reference_run_fn=lambda *args: None,
        require_reference_run_fn=lambda *args: None,
    )
    orchestrator.prepare_visual_inputs(object(), object())
    with pytest.raises(GenerationOrchestrationError, match="준비하지 않은"):
        orchestrator.approve_visual_inputs(
            SimpleNamespace(approved_generation=None)
        )


def test_native_stage_writes_final_product_evidence(tmp_path, monkeypatch):
    selected = tmp_path / "selected.png"
    selected.write_bytes(b"image")
    native_report = tmp_path / "run.json"
    native_report.write_text("{}", encoding="utf-8")
    result = SimpleNamespace(
        status="PASS",
        selected_engine="flux2_klein",
        selected_image_path=selected,
        report_path=native_report,
    )

    monkeypatch.setattr(
        "genai_lab.native_refinement_execution.execute_native_refinement",
        lambda *args, **kwargs: result,
    )
    approved = tmp_path / "approved_generation.json"
    approved.write_text("{}", encoding="utf-8")
    output = tmp_path / "native"

    returned = GenerationOrchestrator._execute_native_refinement_stage(
        SimpleNamespace(),
        project_root=tmp_path,
        base_image=tmp_path / "base.png",
        candidate_record=tmp_path / "base.json",
        approved_run=approved,
        garment_reference=tmp_path / "garment.png",
        output_directory=output,
    )

    assert returned is result
    record = json.loads(
        (output / "product_execution.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "FINAL_GATE_PASS"
    assert record["final_return_eligible"] is True
    assert record["runtime_smoke"] is False


def test_product_orchestrator_selects_and_finalizes_one_candidate(
        tmp_path, monkeypatch):
    inputs = SimpleNamespace(approved_generation=None)
    approval = SimpleNamespace(fingerprint="a" * 64)

    def approve(value, config, request):
        value.approved_generation = approval

    base_image = tmp_path / "candidate_1.png"
    candidate_record = base_image.with_suffix(".json")
    approved_run = tmp_path / "approved_generation.json"
    garment_reference = tmp_path / "input_garment.png"
    for artifact in (
        base_image,
        candidate_record,
        approved_run,
        garment_reference,
    ):
        artifact.write_bytes(b"artifact")
    candidate = SimpleNamespace(seed=77, candidate_number=1)
    batch = SimpleNamespace(
        directory=tmp_path,
        review_stage="native_base",
        candidates=[candidate],
        paths=[base_image],
    )
    config = {
        "native_pipeline_v2": {
            "enabled": True,
            "base_profile": "character_only",
            "base_garment_prompt_enabled": False,
            "base_garment_adapter_enabled": False,
            "native_garment_source": "isolated_garment_board",
            "reuse_animagine_prompt_in_native_stage": False,
            "raw_garment_person_image_allowed": False,
        },
        "refinement_execution": {"enabled": True, "mode": "flux_whole_image"},
        "native_refinement": {"enabled": True},
    }
    orchestrator = GenerationOrchestrator(
        config,
        SimpleNamespace(seed=1234, model_id="animagine"),
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=lambda *args: inputs,
        generate_visual_batch_fn=lambda *args, **kwargs: batch,
        approve_reference_run_fn=approve,
        require_reference_run_fn=lambda *args: approval,
        save_replay_bundle_fn=lambda *args: tmp_path / "replay",
    )
    monkeypatch.setattr(
        "genai_lab.native_refinement_execution.resolve_native_refinement_settings",
        lambda config, project_root: SimpleNamespace(),
    )
    selected_image = tmp_path / "selected.png"
    selected_image.write_bytes(b"selected")
    result = SimpleNamespace(
        status="PASS",
        selected_engine="flux2_klein",
        selected_image_path=selected_image,
        report_path=tmp_path / "native" / "run.json",
        output_directory=tmp_path / "native",
        report={},
    )
    captured = {}

    def execute(settings, **kwargs):
        captured.update(kwargs)
        return result

    orchestrator._execute_native_refinement_stage = execute
    orchestrator.prepare_visual_inputs(object(), object())
    orchestrator.approve_visual_inputs(inputs)
    assert orchestrator.generate_base_candidates(object(), inputs) is batch

    selection = orchestrator.select_base_candidate(batch, 0)
    assert selection.base_image.name == "approved-base.png"
    assert selection.candidate_record.name == "approved-base.json"
    assert selection.approved_run.name == "approved-generation.json"
    assert selection.garment_reference.name == "input_garment.png"
    assert selection.base_image.read_bytes() == base_image.read_bytes()
    assert selection.candidate_record.read_bytes() == candidate_record.read_bytes()
    assert orchestrator.phase is GenerationPhase.CANDIDATE_SELECTED

    returned = orchestrator.finalize_selected_candidate(
        selection,
        output_directory=tmp_path / "native",
    )

    assert returned is result
    assert captured["base_image"] == selection.base_image
    assert captured["candidate_record"] == selection.candidate_record
    assert captured["approved_run"] == selection.approved_run
    assert captured["garment_reference"] == selection.garment_reference
    assert orchestrator.phase is GenerationPhase.COMPLETED

    evidence_path = tmp_path / "native" / "stage8-review-evidence.json"
    assert evidence_path.is_file()
    evidence_record = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence_record["candidate_id"] == "candidate-1"
    assert evidence_record["guidance"]["status"] == "not_reported"
    assert result.report["final_review"]["status"] == "pending_user_decision"

    decision = create_final_review_decision(
        orchestrator.final_review_evidence,
        APPROVED,
    )
    decision_path = orchestrator.record_final_review_decision(decision)
    assert decision_path.is_file()
    product = json.loads(
        (tmp_path / "native" / "product_execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert product["user_approval_status"] == APPROVED
    assert product["next_action"] == "save_result"

    storage = create_storage_record(
        candidate_id=decision.candidate_id,
        stage8_decision=APPROVED,
        status=SAVED,
        storage_class="final_result",
        selected_output_root=tmp_path / "chosen",
        image_path=tmp_path / "chosen" / "result.png",
        metadata_path=tmp_path / "chosen" / "result.json",
    )
    storage_path = write_storage_record(storage, tmp_path / "native")
    orchestrator.record_final_result_storage(storage, storage_path)
    product = json.loads(
        (tmp_path / "native" / "product_execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert product["user_save_status"] == SAVED
    assert product["saved_result"]["storage_class"] == "final_result"
    assert product["stage9_storage_decision"] == str(storage_path)


def test_product_orchestrator_seals_direct_source_without_animagine(tmp_path):
    source = Image.new("RGB", (24, 32), "blue")
    garment = Image.new("RGB", (24, 32), "green")

    class Approval:
        fingerprint = "e" * 64

        def record(self):
            return {
                "parts": [],
                "gender": "male",
            }

    approval = Approval()
    inputs = SimpleNamespace(
        source=source,
        garment=garment,
        approved_generation=approval,
    )
    config = {
        "paths": {"output_dir": "outputs"},
        "native_pipeline_v2": {
            "enabled": True,
            "primary_route": "source_character_direct",
            "animagine_fallback": "explicit_only",
            "base_profile": "character_only",
            "base_garment_prompt_enabled": False,
            "base_garment_adapter_enabled": False,
            "native_garment_source": "isolated_garment_board",
            "reuse_animagine_prompt_in_native_stage": False,
            "raw_garment_person_image_allowed": False,
        },
        "refinement_execution": {"enabled": True, "mode": "flux_whole_image"},
        "native_refinement": {"enabled": True},
    }
    orchestrator = GenerationOrchestrator(
        config,
        SimpleNamespace(seed=23, candidate_number=1),
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=lambda *args: inputs,
        generate_visual_batch_fn=lambda *args, **kwargs: pytest.fail(
            "direct route must not generate an Animagine Base"
        ),
        approve_reference_run_fn=lambda *args: None,
        require_reference_run_fn=lambda *args: approval,
        save_replay_bundle_fn=lambda *args: tmp_path / "replay",
    )
    orchestrator.restore_approved_inputs(inputs)

    selection = orchestrator.select_direct_source(inputs)

    assert selection.input_mode == "source_character_direct"
    assert selection.base_image.name == "approved-source.png"
    assert selection.garment_reference.name == "input_garment.png"
    assert selection.base_image.is_file()
    assert selection.garment_reference.is_file()
    source_record = json.loads(
        selection.candidate_record.read_text(encoding="utf-8")
    )
    approved_record = json.loads(
        selection.approved_run.read_text(encoding="utf-8")
    )
    assert source_record["approved_generation_fingerprint"] == "e" * 64
    assert approved_record["approval_fingerprint"] == "e" * 64
    assert orchestrator.phase is GenerationPhase.CANDIDATE_SELECTED
    source.close()
    garment.close()


def test_product_orchestrator_uses_only_sdxl_local_finalizer(
        tmp_path, monkeypatch):
    config = {
        "native_pipeline_v2": {
            "enabled": True,
            "base_profile": "character_only",
            "base_garment_prompt_enabled": False,
            "base_garment_adapter_enabled": False,
            "native_garment_source": "isolated_garment_board",
            "reuse_animagine_prompt_in_native_stage": False,
            "raw_garment_person_image_allowed": False,
        },
        "refinement_execution": {"enabled": True, "mode": "sdxl_local"},
        "native_refinement": {"enabled": True},
        "generation_run_context": {
            "root": str(tmp_path / "runs"),
            "training_roots": [str(tmp_path / "training")],
        },
        "person_count_diagnostic": {"enabled": False},
    }
    orchestrator = GenerationOrchestrator(
        config,
        SimpleNamespace(seed=17),
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=lambda *args: None,
        generate_visual_batch_fn=lambda *args, **kwargs: None,
        approve_reference_run_fn=lambda *args: None,
        require_reference_run_fn=lambda *args: None,
    )
    base = orchestrator.run_context.base_directory / "approved-base.png"
    record = orchestrator.run_context.base_directory / "approved-base.json"
    approved = orchestrator.run_context.inputs_directory / "approved-generation.json"
    garment = orchestrator.run_context.inputs_directory / "input_garment.png"
    Image.new("RGB", (24, 32), "blue").save(base)
    Image.new("RGB", (24, 32), "green").save(garment)
    record.write_text("{}", encoding="utf-8")
    approved.write_text("{}", encoding="utf-8")
    selection = BaseCandidateSelection(
        index=0,
        candidate_number=1,
        base_image=base,
        candidate_record=record,
        approved_run=approved,
        garment_reference=garment,
        batch_directory=tmp_path,
    )
    orchestrator._selected_base = selection
    orchestrator._generation_pipeline = SimpleNamespace()
    orchestrator.phase = GenerationPhase.CANDIDATE_SELECTED
    output = tmp_path / "local-output"
    selected = output / "selected.png"
    report_path = output / "local-report.json"
    called = []

    def local_finalizer(pipeline, **kwargs):
        called.append("sdxl_local")
        output.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 32), "navy").save(selected)
        report_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            status="PASS",
            selected_engine="sdxl_local",
            selected_image_path=selected,
            report_path=report_path,
            output_directory=output,
            report={},
        )

    orchestrator._execute_sdxl_local_refinement_stage = local_finalizer
    orchestrator._execute_native_refinement_stage = lambda *args, **kwargs: pytest.fail(
        "FLUX finalizer must not run in sdxl_local mode"
    )

    result = orchestrator.finalize_selected_candidate(
        selection,
        output_directory=output,
    )

    assert result.selected_engine == "sdxl_local"
    assert called == ["sdxl_local"]
    assert orchestrator.phase is GenerationPhase.COMPLETED
    evidence = json.loads(
        (output / "product_execution.json").read_text(encoding="utf-8")
    )
    assert evidence["refinement_mode"] == "sdxl_local"
    assert evidence["person_count_diagnostic"]["status"] == "DISABLED"


def test_product_orchestrator_resets_runtime_when_base_generation_fails(tmp_path):
    inputs = SimpleNamespace(approved_generation=None)
    approval = SimpleNamespace(fingerprint="b" * 64)

    def approve(value, config, request):
        value.approved_generation = approval

    class Pipeline:
        def __init__(self):
            self._interrupt = True
            self.scales = []
            self.release_count = 0

        def set_ip_adapter_scale(self, value):
            self.scales.append(value)

        def maybe_free_model_hooks(self):
            self.release_count += 1

    pipeline = Pipeline()
    config = {
        "native_pipeline_v2": {
            "enabled": True,
            "base_profile": "character_only",
            "base_garment_prompt_enabled": False,
            "base_garment_adapter_enabled": False,
            "native_garment_source": "isolated_garment_board",
            "reuse_animagine_prompt_in_native_stage": False,
            "raw_garment_person_image_allowed": False,
        },
        "refinement_execution": {"enabled": True, "mode": "sdxl_local"},
        "native_refinement": {"enabled": True},
        "generation_run_context": {
            "root": str(tmp_path / "runs"),
            "training_roots": [str(tmp_path / "training")],
        },
    }
    orchestrator = GenerationOrchestrator(
        config,
        SimpleNamespace(seed=9),
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=lambda *args: inputs,
        generate_visual_batch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("base failed")
        ),
        approve_reference_run_fn=approve,
        require_reference_run_fn=lambda *args: approval,
        save_replay_bundle_fn=lambda *args: tmp_path / "replay",
    )
    orchestrator.prepare_visual_inputs(object(), object())
    orchestrator.approve_visual_inputs(inputs)

    with pytest.raises(RuntimeError, match="base failed"):
        orchestrator.generate_base_candidates(pipeline, inputs)

    assert orchestrator.phase is GenerationPhase.FAILED
    assert orchestrator._generation_pipeline is None
    assert pipeline._interrupt is False
    assert pipeline.scales == [0.0, 0.0]
    manifest = json.loads(
        orchestrator.run_context.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["runtime_reset"]["boundary"] == "request_failed"


def test_product_orchestrator_closes_unfinished_request(tmp_path):
    pipeline = SimpleNamespace(_interrupt=True)
    orchestrator = GenerationOrchestrator(
        {
            "generation_run_context": {
                "root": str(tmp_path / "runs"),
                "training_roots": [str(tmp_path / "training")],
            }
        },
        SimpleNamespace(),
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=lambda *args: None,
        generate_visual_batch_fn=lambda *args, **kwargs: None,
        approve_reference_run_fn=lambda *args: None,
        require_reference_run_fn=lambda *args: None,
    )
    orchestrator.phase = GenerationPhase.BASE_COMPLETED

    report = orchestrator.close_request(
        reason="base_candidate_not_selected",
        pipeline=pipeline,
    )

    assert report["boundary"] == "request_end"
    assert pipeline._interrupt is False
    assert orchestrator.phase is GenerationPhase.CANCELLED
    manifest = json.loads(
        orchestrator.run_context.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["status"] == "cancelled"
    assert manifest["reason"] == "base_candidate_not_selected"


def test_post_refinement_failure_uses_request_failed_boundary(
    tmp_path,
    monkeypatch,
):
    config = {
        "native_pipeline_v2": {
            "enabled": True,
            "base_profile": "character_only",
            "base_garment_prompt_enabled": False,
            "base_garment_adapter_enabled": False,
            "native_garment_source": "isolated_garment_board",
            "reuse_animagine_prompt_in_native_stage": False,
            "raw_garment_person_image_allowed": False,
        },
        "refinement_execution": {
            "enabled": True,
            "mode": "flux_whole_image",
        },
        "native_refinement": {"enabled": True},
        "generation_run_context": {
            "root": str(tmp_path / "runs"),
            "training_roots": [str(tmp_path / "training")],
        },
        "person_count_diagnostic": {"enabled": True},
    }
    orchestrator = GenerationOrchestrator(
        config,
        SimpleNamespace(seed=17),
        tmp_path,
        Log(),
        prepare_visual_inputs_fn=lambda *args: None,
        generate_visual_batch_fn=lambda *args, **kwargs: None,
        approve_reference_run_fn=lambda *args: None,
        require_reference_run_fn=lambda *args: None,
    )
    base = orchestrator.run_context.base_directory / "approved-base.png"
    candidate_record = (
        orchestrator.run_context.base_directory / "approved-base.json"
    )
    approved_run = (
        orchestrator.run_context.inputs_directory / "approved-generation.json"
    )
    garment = orchestrator.run_context.inputs_directory / "input_garment.png"
    selected = orchestrator.run_context.refinement_directory / "selected.png"
    Image.new("RGB", (24, 32), "blue").save(base)
    Image.new("RGB", (24, 32), "green").save(garment)
    Image.new("RGB", (24, 32), "navy").save(selected)
    candidate_record.write_text("{}", encoding="utf-8")
    approved_run.write_text("{}", encoding="utf-8")
    selection = BaseCandidateSelection(
        index=0,
        candidate_number=1,
        base_image=base,
        candidate_record=candidate_record,
        approved_run=approved_run,
        garment_reference=garment,
        batch_directory=tmp_path,
    )
    orchestrator._selected_base = selection
    orchestrator.phase = GenerationPhase.CANDIDATE_SELECTED
    result = SimpleNamespace(
        status="PASS",
        selected_engine="flux2_klein",
        selected_image_path=selected,
        report_path=orchestrator.run_context.refinement_directory / "run.json",
        output_directory=orchestrator.run_context.refinement_directory,
        report={},
    )
    result.report_path.write_text("{}", encoding="utf-8")
    orchestrator._execute_native_refinement_stage = (
        lambda *args, **kwargs: result
    )
    monkeypatch.setattr(
        "genai_lab.native_refinement_execution.resolve_native_refinement_settings",
        lambda config, project_root: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "genai_lab.person_count_diagnostic.evaluate_person_count",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("diagnostic failed")
        ),
    )
    boundaries = []

    def reset_runtime(pipeline, *, boundary):
        boundaries.append(boundary)
        return {
            "version": "request_runtime_reset_v1",
            "boundary": boundary,
            "status": "completed",
            "actions": [],
            "errors": [],
        }

    monkeypatch.setattr(
        "genai_lab.request_runtime.reset_request_runtime",
        reset_runtime,
    )

    with pytest.raises(RuntimeError, match="diagnostic failed"):
        orchestrator.finalize_selected_candidate(selection)

    assert boundaries[-1] == "request_failed"
    manifest = json.loads(
        orchestrator.run_context.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
