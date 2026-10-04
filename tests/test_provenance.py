"""Provenance contracts, not image-quality tests."""
import copy
import json
import random
from types import SimpleNamespace

import pytest

from genai_lab.provenance import (
    RunProvenance, TrackedConfig, adapter_loaded, observe_attempt,
    observe_gate, plain, recorder, track_config,
)


def tracked(tmp_path, value):
    return TrackedConfig(value, recorder=RunProvenance(
        value, path=tmp_path / "missing.yaml", root=tmp_path))


def test_nested_reads_do_not_consume_unread_descendants(tmp_path):
    c = tracked(tmp_path, {"style": {"scale": .8, "unused": True},
                          "garment_inpaint": {"base_model_id": "unloaded"}})
    assert isinstance(c, dict) and isinstance(c["style"], dict)
    assert c["style"].get("scale") == .8
    result = recorder(c).record()["config"]
    assert result["read"]["style.scale"] == .8
    assert result["unread"] == [
        "garment_inpaint", "garment_inpaint.base_model_id", "style.unused"]


def test_copy_deepcopy_json_do_not_count_as_value_consumption(tmp_path):
    c = tracked(tmp_path, {"a": {"x": 1, "y": 2}, "b": 3})
    assert json.loads(json.dumps(c)) == {"a": {"x": 1, "y": 2}, "b": 3}
    shallow = copy.copy(c)
    deep = copy.deepcopy(c)
    assert recorder(c).read == {}
    assert recorder(shallow) is recorder(c) is recorder(deep)
    assert deep["a"]["x"] == 1
    assert "a.y" in recorder(c).record()["config"]["unread"]


def test_runtime_overwrite_retains_declared_and_read_history(tmp_path):
    c = tracked(tmp_path, {"garment": {"strength": .95}})
    assert c["garment"]["strength"] == .95
    c["garment"]["strength"] = .90
    assert c["garment"]["strength"] == .90
    out = recorder(c).record()["config"]
    assert out["declared"]["garment.strength"] == .95
    assert out["read"]["garment.strength"] == .90
    assert out["read_history"]["garment.strength"] == [.95, .90]


def test_bulk_consumers_defaults_and_mapping_semantics(tmp_path):
    c = tracked(tmp_path, {"a": 1, "b": 2})
    assert dict(c) == {"a": 1, "b": 2}
    assert recorder(c).read == {"a": 1, "b": 2}
    assert list(c.items()) == [("a", 1), ("b", 2)]
    assert list(c.values()) == [1, 2]
    assert list(c.items() & {("a", 1)}) == [("a", 1)]
    assert c.get("missing", .5) == .5
    with pytest.raises(KeyError):
        c["missing"]
    assert c.setdefault("a", 100) == 1
    assert c.pop("b") == 2
    assert c.pop("none", 10) == 10


def test_read_after_copy_and_new_subtree_remains_tracked(tmp_path):
    c = tracked(tmp_path, {"a": {"x": 1}})
    d = dict(c)
    assert d["a"]["x"] == 1
    c.update({"new": {"value": 5}})
    assert c["new"]["value"] == 5
    assert recorder(d) is recorder(c)
    assert recorder(c).read["new.value"] == 5


def test_failed_run_persists_without_swallowing_original_exception(tmp_path):
    c = tracked(tmp_path, {"x": 3, "unused": 4})
    rec = recorder(c)
    rec.bind(SimpleNamespace(run_id="failed", run_directory=tmp_path / "failed"))
    error = ValueError("original")
    with pytest.raises(ValueError) as caught:
        try:
            assert c["x"] == 3
            raise error
        except BaseException as original:
            rec.event("generate_base_candidates", "failed",
                      error_type=type(original).__name__)
            raise
    assert caught.value is error
    record = json.loads(rec.path.read_text())
    assert record["completed"] is False
    assert record["config"]["read"]["x"] == 3
    assert record["config"]["unread"] == ["unused"]


def test_retry_verdict_without_actual_call_stays_zero(tmp_path):
    c = tracked(tmp_path, {})
    observe_gate(c, "semantic", {"status": "RETRY", "action": "retry"}, "base:1")
    rec = recorder(c)
    rec.event("finalize_selected_candidate", "completed")
    assert rec.gates[0]["verdict"] == "RETRY"
    assert rec.gates[0]["action"] == "none"
    assert rec.gates[0]["retry_count"] == 0
    assert rec.gates[0]["aborted"] is False


def test_only_executed_retry_calls_increment_count(tmp_path):
    c = tracked(tmp_path, {})
    rec = recorder(c)
    observe_attempt(c, 1, 1, None)
    observe_gate(c, "semantic", {"status": "RETRY"}, "base:1")
    rec.action("semantic", "base:1", "quarantined")
    assert rec.gates[0]["retry_count"] == 0
    observe_attempt(c, 2, 1, "quality")
    assert rec.gates[0]["retry_count"] == 1
    assert rec.gates[0]["action"] == "retry"
    observe_gate(c, "integrity", {"status": "corrupted"}, "base:2")
    observe_attempt(c, 2, 2, "quality")
    assert rec.gates[0]["retry_count"] == 1
    assert rec.gates[1]["retry_count"] == 1


def test_distinct_requests_have_distinct_collectors(tmp_path):
    a = tracked(tmp_path, {"x": 1})
    b = tracked(tmp_path, {"x": 1})
    a["x"]
    assert not recorder(b).read
    assert recorder(a) is not recorder(b)


def test_observation_does_not_consume_python_or_torch_rng(tmp_path):
    torch = pytest.importorskip("torch")
    c = tracked(tmp_path, {"x": 1})
    py_state, torch_state = random.getstate(), torch.random.get_rng_state()
    c["x"]
    recorder(c).start()
    recorder(c).event("prepare", "failed")
    assert random.getstate() == py_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_actual_projection_hook_preserves_output_and_reads_tokens(tmp_path):
    torch = pytest.importorskip("torch")
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("anchor", torch.zeros(1))
        def forward(self, value):
            return value
    projection = Projection()
    unet = torch.nn.Module()
    unet.config = SimpleNamespace(in_channels=4)
    unet.encoder_hid_proj = SimpleNamespace(image_projection_layers=[projection])
    unet.attn_processors = {"adapter": SimpleNamespace(scale=[.8])}
    encoder = torch.nn.Module()
    encoder.config = SimpleNamespace(hidden_size=1664)
    pipe = SimpleNamespace(unet=unet, image_encoder=encoder,
                           config={"_name_or_path": "real-object-model"},
                           scheduler=SimpleNamespace())
    adapter_loaded(pipe, weight_name="misleading-plus-filename.bin")
    c = tracked(tmp_path, {})
    rec = recorder(c)
    rec.pipeline(pipe, "test")
    assert rec.loaded["ip_adapter_tokens"] is None  # no dummy forward
    value = torch.zeros(2, 4, 8)
    assert projection(value) is value
    assert rec.loaded["ip_adapter_tokens"] == 4
    assert rec.loaded["image_projection_outputs"][0]["shape"] == [2, 4, 8]
    assert rec.loaded["unet_in_channels"] == 4
    assert rec.loaded["ip_adapter_scale"] == {"adapter": [.8]}
    assert rec.loaded["lora"] == rec.loaded["controlnet"] == []
    assert rec.loaded["base_model"] == "real-object-model"
    rec.remove_hooks()
    assert not projection._forward_hooks


def test_declared_comes_from_file_not_effective_override(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"x": 0.95, "unused": 1}')
    c = track_config({"x": .9}, path, tmp_path)
    c["x"]
    assert recorder(c).declared["x"] == .95
    assert recorder(c).read["x"] == .9
    assert recorder(c).record()["config"]["unread"] == ["unused"]


def test_json_safe_replay_serialization_does_not_read_every_key(tmp_path):
    c = tracked(tmp_path, {"inactive": {"id": "never"}, "active": 1})
    from genai_lab.generation_replay import _json_safe
    assert _json_safe(c)["inactive"]["id"] == "never"
    assert recorder(c).read == {}


def test_plain_caller_alias_is_not_replaced_or_claimed_fully_tracked(tmp_path):
    from genai_lab.provenance import ensure_recorder
    section = {}
    config = {"approval": section}
    rec = ensure_recorder(config, root=tmp_path)
    section["approved"] = True
    assert config["approval"] is section
    assert config["approval"]["approved"] is True
    assert recorder(config) is rec
    assert rec.record()["config"]["tracking_available"] is False


def test_completed_tracked_config_gets_fresh_run_collector(tmp_path):
    c = tracked(tmp_path, {"a": 1})
    recorder(c).completed = True
    next_run = track_config(c)
    assert recorder(next_run) is not recorder(c)
    assert recorder(next_run).read == {}
