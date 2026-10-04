from __future__ import annotations

import json
from types import SimpleNamespace

import scripts.run_product_generation as runner


def test_codex_runner_uses_product_orchestrator_for_final_selection(
        tmp_path, monkeypatch):
    calls = []
    inputs = SimpleNamespace(close=lambda: calls.append("inputs_close"))
    request_image = SimpleNamespace(close=lambda: calls.append("request_close"))
    request = SimpleNamespace(reference_image=request_image)
    approval = SimpleNamespace(fingerprint="c" * 64)
    batch = SimpleNamespace(
        directory=tmp_path,
        review_stage="native_base",
        candidates=[object()],
        paths=[tmp_path / "candidate_1.png"],
        close=lambda: calls.append("batch_close"),
    )
    pipeline = SimpleNamespace(
        maybe_free_model_hooks=lambda: calls.append("pipeline_free"))

    class FakeLog:
        file_path = tmp_path / "run.log"

        def write_stage(self, *args):
            calls.append("log_stage")

        def write_failure(self, *args):
            calls.append("log_failure")

        def close(self):
            calls.append("log_close")

    selection = object()
    native_result = SimpleNamespace(
        status="BASE_LOCKED",
        selected_engine=None,
        selected_image_path=tmp_path / "candidate_1.png",
        report_path=tmp_path / "native" / "run.json",
    )

    class FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            calls.append("orchestrator_init")

        def restore_approved_inputs(self, value):
            assert value is inputs
            calls.append("restore")

        def generate_base_candidates(self, value, prepared):
            assert value is pipeline and prepared is inputs
            calls.append("generate")
            return batch

        def select_base_candidate(self, value, index):
            assert value is batch and index == 0
            calls.append("select")
            return selection

        def finalize_selected_candidate(
                self, value, *, output_directory, status_callback):
            assert value is selection
            assert output_directory is None
            calls.append("finalize")
            return native_result

    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: SimpleNamespace(
            bundle=tmp_path / "bundle",
            selected_candidate=1,
            native_output_dir=None,
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_generation_replay_bundle",
        lambda path: (inputs, {}, request, approval),
    )
    monkeypatch.setattr(runner, "validate_config", lambda config: None)
    monkeypatch.setattr(runner, "configure_system_certificates", lambda: None)
    monkeypatch.setattr(
        runner,
        "check_environment",
        lambda: {"gpu": "mock", "vram_bytes": 8 * 1024**3},
    )
    monkeypatch.setattr(runner, "prepare_pipeline", lambda config: pipeline)
    monkeypatch.setattr(runner, "create_generation_run_log", lambda root: FakeLog())
    monkeypatch.setattr(runner, "GenerationOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    assert runner.main() == 0
    record = json.loads(
        (tmp_path / "codex_product_result.json").read_text(encoding="utf-8")
    )
    assert record["execution_scope"] == "product_pipeline"
    assert record["runtime_smoke"] is False
    assert record["status"] == "FINAL_BASE_LOCKED"
    assert record["native_status"] == "BASE_LOCKED"
    assert record["final_return_eligible"] is True
    assert calls[:3] == ["log_stage", "log_stage", "orchestrator_init"]
    assert "restore" in calls and "generate" in calls
    assert "select" in calls and "finalize" in calls
    assert "batch_close" in calls and "inputs_close" in calls


def test_codex_runner_records_failure_with_recovery_action(tmp_path, monkeypatch):
    calls = []

    class FakeLog:
        def write_failure(self, stage, error, recovery_action):
            calls.append((stage, type(error).__name__, recovery_action))

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: SimpleNamespace(
            bundle=tmp_path / "missing-bundle",
            selected_candidate=None,
            native_output_dir=None,
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_generation_replay_bundle",
        lambda _path: (_ for _ in ()).throw(ValueError("broken bundle")),
    )
    monkeypatch.setattr(
        runner, "create_generation_run_log", lambda _root: FakeLog())
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    assert runner.main() == 2
    assert calls[0][0:2] == ("Codex 제품 재생", "ValueError")
    assert "재생 번들 무결성" in calls[0][2]
    assert calls[-1] == "closed"
