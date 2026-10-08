"""CPU contracts for the isolated head-contour trial; no GPU/model loading."""
from dataclasses import asdict, replace
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import head_contour_adapter_trial as runner
from scripts import head_contour_trial_inputs as contracts
from genai_lab.onepass_generation import OnePassInputs, OnePassGenerationError, UNetObservation
from genai_lab.onepass_generation_settings import OnePassGenerationSettings
from genai_lab.onepass_prompt import OnePassPrompt, EncoderChunkPlan, PromptChunk


def save_json(path, value):
    path.write_text(contracts.canonical(value), encoding="utf-8")


@pytest.fixture
def trial_request(tmp_path):
    control, face = tmp_path / "control.png", tmp_path / "face.png"
    Image.new("RGB", (8, 8), "black").save(control)
    Image.new("RGB", (8, 8), "white").save(face)
    chunk = PromptChunk("text", "fixed text", (1,) * 77)
    encoder = EncoderChunkPlan(2, 2, (chunk,), (chunk,))
    prompt = OnePassPrompt("fixed positive", "fixed negative", (encoder, encoder), {"fixed": True})
    inputs = OnePassInputs(prompt, control, face, contracts.sha(control), contracts.sha(face), .5,
                           pose_mode="with_pose")
    settings = replace(OnePassGenerationSettings(), width=8, height=8)
    return inputs, settings


def test_base_rehydration_preserves_every_input_byte(trial_request):
    inputs, settings = trial_request
    plan = {"inputs": {"BASE": asdict(inputs)}, "settings": asdict(settings)}
    # Model the actual JSON serialization, not a dataclass equality shortcut.
    import json
    plan = json.loads(contracts.canonical(plan))
    restored, restored_settings = contracts.restore_request(plan)
    assert contracts.canonical(asdict(restored)) == contracts.canonical(plan["inputs"]["BASE"])
    assert contracts.canonical(asdict(restored_settings)) == contracts.canonical(plan["settings"])


def test_multiadapter_loads_two_local_models_in_order(trial_request, tmp_path):
    _, settings = trial_request
    loaded = []
    class Adapter:
        @staticmethod
        def from_pretrained(path, **kwargs):
            item = (path, kwargs)
            loaded.append(item)
            return item
    pair = runner.load_adapter_pair(Adapter, tuple, settings, tmp_path, "fp16")
    assert pair == tuple(loaded) and len(pair) == 2
    assert pair[0][0] == str(settings.adapter_root)
    assert pair[1][0] == str(tmp_path)
    assert "variant" not in pair[0][1] and pair[1][1]["variant"] == "fp16"
    assert all(item[1]["local_files_only"] for item in pair)


@pytest.mark.parametrize("strength", [None, .9, .5])
def test_pair_call_keeps_face_and_applies_explicit_strengths(trial_request, strength):
    _, settings = trial_request
    captured = {}
    def pipe(**kwargs):
        captured.update(kwargs)
        return "result"
    pose, face, contour = object(), object(), object()
    assert runner.call_with_contour(pipe, {"prompt_embeds": "unchanged"}, (pose, face), contour,
                                   settings, "seed", {"callback": "cb"},
                                   **({} if strength is None else {"sketch_strength": strength})) == "result"
    assert captured["image"] == [pose, contour]
    assert captured["ip_adapter_image"] is face
    assert captured["adapter_conditioning_scale"] == [1.2, .9 if strength is None else strength]
    assert int(settings.steps * captured["adapter_conditioning_factor"]) == 11
    assert captured["prompt_embeds"] == "unchanged"


@pytest.fixture
def geometry(tmp_path):
    import cv2
    mask = np.zeros((100, 100), np.uint8)
    mask[10:61, 25:76] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = np.zeros((100, 100, 3), np.uint8)
    cv2.drawContours(rgb, contours, -1, (255, 255, 255), 4)
    mp, cp = tmp_path / "mask.png", tmp_path / "contour.png"
    Image.fromarray(mask).save(mp)
    Image.fromarray(rgb).save(cp)
    joints = [{"joint_name": name, "x": 50., "y": y, "detected": True, "confidence_score": .9}
              for name, y in (("nose", 35.), ("neck", 68.))]
    return cp, mp, joints


def test_locked_geometry_passes_without_transform(geometry):
    record = contracts.check_geometry(*geometry, size=(100, 100))
    assert record["boundary_max_distance_px"] <= 2
    assert record["neck_gap_px"] == 8
    assert record["coordinate_transform"].startswith("none")


@pytest.mark.parametrize("damage", ["shift", "grey", "empty", "missing_half", "canvas"])
def test_bad_contour_rejected(geometry, damage):
    cp, mp, joints = geometry
    with Image.open(cp) as im:
        rgb = np.array(im)
    if damage == "shift":
        rgb = np.roll(rgb, 10, axis=1)
    elif damage == "grey":
        rgb[20, 20] = 127
    elif damage == "empty":
        rgb[:] = 0
    elif damage == "missing_half":
        rgb[:45] = 0
    else:
        rgb = rgb[:-1]
    Image.fromarray(rgb).save(cp)
    with pytest.raises(ValueError):
        contracts.check_geometry(cp, mp, joints, size=(100, 100))


@pytest.mark.parametrize("joint_name,value", [("nose", 90.), ("neck", 95.)])
def test_misaligned_joints_rejected(geometry, joint_name, value):
    cp, mp, joints = geometry
    for point in joints:
        if point["joint_name"] == joint_name:
            point["y"] = value
    with pytest.raises(ValueError):
        contracts.check_geometry(cp, mp, joints, size=(100, 100))


@pytest.fixture
def download(tmp_path, monkeypatch):
    files, hashes = {}, {}
    for name, payload in (("config.json", b"{}"), ("diffusion_pytorch_model.fp16.safetensors", b"weights")):
        p = tmp_path / name
        p.write_bytes(payload)
        digest = contracts.sha(p)
        hashes[name] = digest
        files[name] = {"path": str(p), "match": True, "size": len(payload), "remote_size": len(payload),
                       "sha256": digest, "remote_lfs_sha256": digest if name.endswith("safetensors") else None}
    monkeypatch.setattr(contracts, "MODEL_HASHES", hashes)
    record = {"repo": contracts.SKETCH_ID, "revision": contracts.REVISION, "files": files}
    path = tmp_path / "download.json"
    save_json(path, record)
    return path, record


def test_download_verified_before_loading(download):
    path, _ = download
    root, record = contracts.verify_download(path)
    assert root == path.parent
    assert record["revision"] == contracts.REVISION


@pytest.mark.parametrize("damage", ["revision", "file", "failed", "remote_sha", "size"])
def test_bad_download_rejected(download, damage):
    path, record = download
    item = record["files"]["diffusion_pytorch_model.fp16.safetensors"]
    if damage == "revision":
        record["revision"] = "wrong"
    elif damage == "file":
        Path(item["path"]).write_bytes(b"changed")
    elif damage == "failed":
        item["match"] = False
    elif damage == "remote_sha":
        item["remote_lfs_sha256"] = "0" * 64
    else:
        item["remote_size"] += 1
    save_json(path, record)
    with pytest.raises(ValueError):
        contracts.verify_download(path)


class Cuda:
    reserved = 1
    allocated = 1
    def max_memory_reserved(self):
        return self.reserved
    def max_memory_allocated(self):
        return self.allocated


def emit_steps(observation, settings, bad_ip=False, missing_residual=False):
    for index in range(settings.steps):
        ip = .5 if index < 11 else .9
        module = SimpleNamespace(attn_processors={"ip": SimpleNamespace(scale=[0. if bad_ip else ip])})
        residual = [] if index < 11 and not missing_residual else None
        observation(module, (), {"down_intrablock_additional_residuals": residual}, None)


def test_guard_observes_exact_shared_schedule(trial_request):
    _, settings = trial_request
    observation = UNetObservation()
    emit_steps(runner.StepGuard(observation, settings, .5, Cuda()), settings)
    assert len(observation.calls) == 28
    assert [x["index"] for x in observation.calls if x["adapter_applied"]] == list(range(11))
    assert observation.calls[10]["ip_scales"] == [.5]
    assert observation.calls[11]["ip_scales"] == [.9]


@pytest.mark.parametrize("damage", ["memory", "ip", "residual"])
def test_guard_stops_on_first_bad_call(trial_request, damage):
    _, settings = trial_request
    observation, cuda = UNetObservation(), Cuda()
    if damage == "memory":
        cuda.reserved = 7 * 2**30
    guard = runner.StepGuard(observation, settings, .5, cuda)
    with pytest.raises(OnePassGenerationError):
        emit_steps(guard, settings, bad_ip=damage == "ip", missing_residual=damage == "residual")
    assert len(observation.calls) == 1


class FakeBackend:
    callback_mode = "legacy_callback"
    pipe = SimpleNamespace(model_cpu_offload_seq=runner.OFFLOAD)
    metrics = {"max_memory_allocated_bytes": 1, "max_memory_reserved_bytes": 1}
    forward_calls = [1]
    closed = False
    def __init__(self, settings):
        self.settings = settings
    def generate(self, inputs, images, seed, observation, cancelled):
        emit_steps(observation, self.settings)
        return Image.new("RGB", (8, 8), "white"), {"generation_seconds": 0., "max_memory_reserved_bytes": 1}
    def close(self):
        self.closed = True


@pytest.fixture
def report():
    return {"models": [{"id": "pose"}, {"id": "sketch"}], "control_shas": ["pose-sha", "sketch-sha"],
            "face_sha256": "face-sha"}


def test_failed_sha_preserves_raw_and_stops_batch(trial_request, report, tmp_path):
    inputs, settings = trial_request
    calls, backends = [], []
    def factory(settings, contour, report):
        calls.append(contour)
        backend = FakeBackend(settings)
        backends.append(backend)
        return backend
    dest = tmp_path / "trial"
    with pytest.raises(OnePassGenerationError, match="SHA"):
        runner.run_trial(dest, inputs, settings, report, factory)
    assert calls == [False] and backends[0].closed
    case = dest / "BASE_209212003"
    assert (case / "raw.png").exists()
    assert contracts.read(case / "run.json")["valid"] is False
    assert contracts.read(dest / "status.json")["status"] == "stopped"
    assert not (dest / "CONTOUR_209212003").exists()


def test_load_failure_leaves_case_record(trial_request, report, tmp_path):
    inputs, settings = trial_request
    def factory(*args):
        raise RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        runner.run_trial(tmp_path / "trial", inputs, settings, report, factory)
    record = contracts.read(tmp_path / "trial/BASE_209212003/run.json")
    assert not record["valid"] and record["error"] == "load failed"


def test_baseline_case_passes_same_input_object_to_existing_generator(trial_request, report, tmp_path, monkeypatch):
    inputs, settings = trial_request
    backend = FakeBackend(settings)
    captured = {}
    def generate(actual_backend, actual_inputs, seed, folder, **kwargs):
        captured.update(backend=actual_backend, inputs=actual_inputs, kwargs=kwargs)
        folder.mkdir()
        save_json(folder / "run.json", {"valid": True, "adapter_applied_calls": list(range(11))})
        return "result"
    monkeypatch.setattr(runner, "generate_onepass_image", generate)
    case = runner.cases()[0]
    assert runner.run_case(case, tmp_path, inputs, settings, report, lambda *args: backend) == "result"
    assert captured["inputs"] is inputs
    assert captured["kwargs"]["expected_sha256"] == contracts.BASE_SHAS[209212003]
    record = contracts.read(tmp_path / case["id"] / "run.json")
    assert record["trial"]["strengths"] == [1.2]
    assert record["trial"]["control_image_sha256"] == ["pose-sha"]
    assert backend.closed


def test_pair_record_lists_real_two_models(trial_request, report, tmp_path):
    _, settings = trial_request
    backend = FakeBackend(settings)
    backend.forward_calls = [1, 1]
    path = tmp_path / "run.json"
    save_json(path, {"valid": True, "adapter_applied_calls": list(range(11))})
    runner.enrich_record(path, report, True, backend)
    record = contracts.read(path)
    assert record["models"]["adapter"] == report["models"]
    assert record["trial"]["individual_adapter_forward_calls"] == [1, 1]
    assert record["trial"]["strengths"] == [1.2, .9]


def test_existing_output_not_overwritten(trial_request, report, tmp_path):
    inputs, settings = trial_request
    with pytest.raises(FileExistsError):
        runner.run_trial(tmp_path, inputs, settings, report)


def test_saved_cpu_report_detects_changed_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DESIGN", tmp_path)
    runner.save_preflight({"strengths": [1.2, .9]})
    runner.require_saved_preflight({"strengths": [1.2, .9]})
    with pytest.raises(ValueError, match="changed"):
        runner.require_saved_preflight({"strengths": [1.2, 1.]})


def test_baseline_factory_uses_production_backend_only(trial_request, monkeypatch):
    import sys
    _, settings = trial_request
    cuda = Cuda()
    cuda.reset_peak_memory_stats = lambda: None
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    called = []
    expected = FakeBackend(settings)
    def original(settings):
        called.append("single")
        return expected
    def forbidden(*args):
        pytest.fail("Baseline must not load sketch or MultiAdapter")
    monkeypatch.setattr(runner, "verify_runtime_models", lambda *args: None)
    monkeypatch.setattr(runner, "DiffusersOnePassBackend", original)
    monkeypatch.setattr(runner, "HeadContourBackend", forbidden)
    monkeypatch.setattr(runner, "verify_download", forbidden)
    monkeypatch.setattr(runner, "MonitoredBackend", lambda backend, settings, contour: backend)
    assert runner.create_backend(settings, False, {}) is expected
    assert called == ["single"]


def test_trial_cases_are_exactly_four_no_extra_seeds():
    cases = runner.cases()
    assert [(c["seed"], c["contour"]) for c in cases] == [
        (209212003, False), (209212003, True), (209212001, False), (209212001, True)]
    assert all(c["expected_sha256"] is None for c in cases if c["contour"])


def test_real_cpu_multiadapter_keeps_explicit_weights():
    import torch
    from diffusers import MultiAdapter
    seen = []
    class TinyAdapter(torch.nn.Module):
        downscale_factor = 16
        total_downscale_factor = 32
        def __init__(self, name):
            super().__init__()
            self.name = name
        def forward(self, x):
            seen.append(self.name)
            return [x.clone()]
    pair = MultiAdapter([TinyAdapter("pose"), TinyAdapter("sketch")])
    result = pair([torch.ones(1, 1, 1, 1), torch.full((1, 1, 1, 1), 2.)], [1.2, .9])
    assert seen == ["pose", "sketch"]
    assert result[0].device.type == "cpu"
    assert result[0].item() == pytest.approx(3.)


def test_final_memory_failure_preserves_completed_raw(trial_request, report, tmp_path):
    inputs, settings = trial_request
    class MemoryBackend(FakeBackend):
        metrics = {"max_memory_allocated_bytes": 1, "max_memory_reserved_bytes": 7 * 2**30}
        def generate(self, *args):
            image, timing = super().generate(*args)
            timing["max_memory_reserved_bytes"] = 7 * 2**30
            return image, timing
    backend = MemoryBackend(settings)
    case = {"id": "memory", "seed": 209212003, "contour": False, "expected_sha256": None}
    with pytest.raises(OnePassGenerationError, match="reserved"):
        runner.run_case(case, tmp_path, inputs, settings, report, lambda *args: backend)
    assert (tmp_path / "memory/raw.png").exists()
    assert contracts.read(tmp_path / "memory/run.json")["valid"] is False


def test_trial_model_recheck_rejects_changed_weights(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"old")
    manifest = {"models": [{"files": {"weights": {"path": str(path), "sha256": contracts.sha(path)}}}]}
    runner.verify_runtime_models(manifest, False)
    path.write_bytes(b"new")
    with pytest.raises(ValueError, match="before loading"):
        runner.verify_runtime_models(manifest, False)


@pytest.mark.parametrize("filename", ["preflight.json", "run.json", "status.json"])
def test_trial_writer_serializes_nested_paths(tmp_path, filename):
    from pathlib import PureWindowsPath
    path = tmp_path / filename
    value = {"settings": {"root": tmp_path},
             "models": [{"path": PureWindowsPath("G:/models/sketch")}],
             "inputs": (tmp_path / "control.png",)}
    runner.write_trial_record(path, value)
    saved = contracts.read(path)
    assert saved["settings"]["root"] == str(tmp_path)
    assert saved["models"][0]["path"] == str(PureWindowsPath("G:/models/sketch"))
    assert saved["inputs"] == [str(tmp_path / "control.png")]


def test_writer_rejects_unknown_object_without_truncating_file(tmp_path):
    path = tmp_path / "record.json"
    path.write_text("previous", encoding="utf-8")
    with pytest.raises(TypeError, match="Unsupported"):
        runner.write_trial_record(path, {"bad": object()})
    assert path.read_text() == "previous"


def test_preflight_roundtrip_with_real_request_path_types(trial_request, tmp_path, monkeypatch):
    inputs, settings = trial_request
    monkeypatch.setattr(runner, "DESIGN", tmp_path)
    report = {"settings": asdict(settings), "base_inputs": asdict(inputs)}
    path = runner.save_preflight(report)
    runner.require_saved_preflight(report)
    saved = contracts.read(path)
    assert saved["base_inputs"]["face_file"] == str(inputs.face_file)
    assert saved["settings"]["model_root"] == str(settings.model_root)
    assert contracts.canonical(saved) == contracts.canonical(report)


def test_complete_cpu_batch_persists_path_report_and_case_records(trial_request, report, tmp_path, monkeypatch):
    inputs, settings = trial_request
    report.update(settings=asdict(settings), base_inputs=asdict(inputs))
    for model in report["models"]:
        model["path"] = tmp_path / model["id"]
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    expected = hashlib.sha256(buf.getvalue()).hexdigest()
    monkeypatch.setattr(runner, "BASE_SHAS", {209212003: expected, 209212001: expected})
    loaded = []
    def factory(settings, contour, report):
        backend = FakeBackend(settings)
        backend.forward_calls = [1, 1] if contour else [1]
        loaded.append(backend)
        return backend
    dest = tmp_path / "cpu-only-batch"
    runner.run_trial(dest, inputs, settings, report, factory)
    assert len(loaded) == 4 and all(b.closed for b in loaded)
    saved = contracts.read(dest / "preflight.json")
    assert saved["base_inputs"]["control_file"] == str(inputs.control_file)
    assert saved["settings"]["adapter_root"] == str(settings.adapter_root)
    state = contracts.read(dest / "status.json")
    assert state["status"] == "completed_pending_visual_review"
    assert len(state["completed"]) == 4
    for case in runner.cases():
        folder = dest / case["id"]
        actual = contracts.read(folder / "run.json")
        assert actual["valid"] and (folder / "raw.png").exists()
        assert actual["trial"]["adapters"][0]["path"] == str(tmp_path / "pose")
        assert actual["prompt"] == asdict(inputs.prompt) or contracts.canonical(actual["prompt"]) == contracts.canonical(asdict(inputs.prompt))


def test_preflight_write_failure_records_stop_before_model_load(trial_request, report, tmp_path, monkeypatch):
    inputs, settings = trial_request
    original = runner.write_trial_record
    def failing(path, data):
        if path.name == "preflight.json":
            raise OSError("preflight write failed")
        return original(path, data)
    monkeypatch.setattr(runner, "write_trial_record", failing)
    def forbidden(*args):
        pytest.fail("Model must not load after preflight write failure")
    dest = tmp_path / "failed-before-model"
    with pytest.raises(OSError, match="preflight write failed"):
        runner.run_trial(dest, inputs, settings, report, forbidden)
    state = contracts.read(dest / "status.json")
    assert state["status"] == "stopped" and state["completed"] == []
    assert state["error_type"] == "OSError"


@pytest.fixture
def reused_baselines(trial_request, report, tmp_path, monkeypatch):
    inputs, settings = trial_request
    monkeypatch.setattr(runner, "DESIGN", tmp_path)
    report.update(settings=asdict(settings), base_inputs=asdict(inputs), strengths=[1.2, .9],
                  protected={str(Path(runner.__file__)): "old-runner", "production.py": "same"})
    source = tmp_path / "gpu-02"
    source.mkdir()
    runner.write_trial_record(source / "preflight.json", report)
    completed, expected = [], {}
    for seed in (209212003, 209212001):
        name = f"BASE_{seed}"
        folder = source / name
        folder.mkdir()
        Image.new("RGB", (8, 8), "white").save(folder / "raw.png")
        digest = contracts.sha(folder / "raw.png")
        expected[seed] = digest
        runner.write_trial_record(folder / "run.json", {
            "completed": True, "valid": True, "seed": seed, "raw_sha256": digest,
            "settings": asdict(settings), "prompt": asdict(inputs.prompt), "validation_errors": [],
            "max_reserved_gib": 1., "adapter_applied_calls": list(range(11)), "trial": {"strengths": [1.2]}})
        completed.append(name)
    runner.write_trial_record(source / "status.json", {"completed": completed})
    monkeypatch.setattr(runner, "BASE_SHAS", expected)
    report["protected"][str(Path(runner.__file__))] = "new-runner"
    return report, source


def test_contour_only_saves_actual_options_and_reuses_bases(trial_request, reused_baselines, tmp_path):
    inputs, settings = trial_request
    report, source = reused_baselines
    original = {str(p): contracts.sha(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "gpu-03"
    configured = runner.configure_trial(report, .5, True, output)
    lock = runner.save_preflight(configured)
    assert lock == tmp_path / "gpu-03_preflight.json" and not output.exists()
    runner.require_saved_preflight(configured)
    loaded = []
    def factory(settings, contour, actual_report):
        assert contour and actual_report["strengths"] == [1.2, .5]
        backend = FakeBackend(settings)
        backend.forward_calls = [1, 1]
        loaded.append(backend)
        return backend
    runner.run_trial(output, inputs, settings, configured, factory)
    assert len(loaded) == 2 and all(b.closed for b in loaded)
    state = contracts.read(output / "status.json")
    assert state["strengths"] == [1.2, .5]
    assert state["options"] == configured["options"]
    assert state["status"] == "completed_pending_visual_review"
    assert state["completed"] == ["CONTOUR_209212003", "CONTOUR_209212001"]
    assert len(state["reused_baselines"]["cases"]) == 2
    for name in state["completed"]:
        record = contracts.read(output / name / "run.json")
        assert record["trial"]["strengths"] == [1.2, .5]
        assert record["trial"]["options"] == state["options"]
        assert record["trial"]["reused_baselines"] == state["reused_baselines"]
        assert record["adapter_applied_calls"] == list(range(11))
        assert contracts.canonical(record["prompt"]) == contracts.canonical(asdict(inputs.prompt))
    assert original == {str(p): contracts.sha(p) for p in source.rglob("*") if p.is_file()}
    assert not (output / "BASE_209212003").exists()


@pytest.mark.parametrize("damage", ["raw", "invalid", "missing", "prompt", "settings", "code", "schedule", "memory"])
def test_bad_reused_base_stops_before_run(reused_baselines, tmp_path, damage):
    report, source = reused_baselines
    folder = source / "BASE_209212003"
    if damage == "raw":
        (folder / "raw.png").write_bytes(b"wrong")
    elif damage == "missing":
        (folder / "run.json").unlink()
    elif damage == "code":
        report["protected"]["production.py"] = "changed"
    else:
        record = contracts.read(folder / "run.json")
        if damage == "invalid":
            record["valid"] = False
        elif damage == "prompt":
            record["prompt"]["positive"] = "different"
        elif damage == "settings":
            record["settings"]["steps"] = 20
        elif damage == "schedule":
            record["adapter_applied_calls"] = []
        else:
            record["max_reserved_gib"] = 7.
        runner.write_trial_record(folder / "run.json", record)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.configure_trial(report, .5, True, tmp_path / "gpu-03")
    assert not (tmp_path / "gpu-03").exists()


@pytest.mark.parametrize("change", ["strength", "selection", "output"])
def test_preflight_lock_rejects_different_options(report, tmp_path, change):
    output = tmp_path / "gpu-03"
    configured = runner.configure_trial(report, .5, False, output)
    runner.save_preflight(configured)
    changed = dict(configured, options=dict(configured["options"]))
    if change == "strength":
        changed["options"]["sketch_strength"] = .9
        changed["strengths"] = [1.2, .9]
    elif change == "selection":
        changed["options"]["contour_only"] = True
    else:
        changed["options"]["output"] = str(tmp_path / "different")
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.require_saved_preflight(changed)


@pytest.mark.parametrize("strength", ["0.7", "nan", "inf", "-1"])
def test_cli_rejects_unsupported_strength_before_preflight(monkeypatch, strength):
    monkeypatch.setattr(runner, "preflight", lambda: pytest.fail("Preflight must not run"))
    with pytest.raises(SystemExit) as error:
        runner.main(["run", "--sketch-strength", strength])
    assert error.value.code == 2


def test_cli_defaults_keep_point_nine_and_all_four(trial_request, report, tmp_path, monkeypatch):
    inputs, settings = trial_request
    monkeypatch.setattr(runner, "DESIGN", tmp_path)
    monkeypatch.setattr(runner, "preflight", lambda: (inputs, settings, report))
    runner.main([])
    saved = contracts.read(tmp_path / "gpu-01_preflight.json")
    assert saved["strengths"] == [1.2, .9]
    assert saved["options"]["contour_only"] is False
    assert len(runner.cases(saved["options"]["contour_only"])) == 4


def test_changed_base_after_preflight_stops_before_output(trial_request, reused_baselines, tmp_path):
    inputs, settings = trial_request
    report, source = reused_baselines
    output = tmp_path / "gpu-03"
    configured = runner.configure_trial(report, .5, True, output)
    path = source / "BASE_209212003/run.json"
    record = contracts.read(path)
    record["new_field"] = "changed after validation"
    runner.write_trial_record(path, record)
    with pytest.raises(ValueError, match="since preflight"):
        runner.run_trial(output, inputs, settings, configured,
                         lambda *args: pytest.fail("No model loading"))
    assert not output.exists()


def test_selected_strength_reaches_backend(trial_request, tmp_path, monkeypatch):
    import sys
    _, settings = trial_request
    cuda = Cuda()
    cuda.reset_peak_memory_stats = lambda: None
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(runner, "verify_runtime_models", lambda *args: None)
    monkeypatch.setattr(runner, "verify_download", lambda *args: (tmp_path, {}))
    captured = []
    def backend(settings, root, strength):
        captured.append(strength)
        return FakeBackend(settings)
    monkeypatch.setattr(runner, "HeadContourBackend", backend)
    monkeypatch.setattr(runner, "MonitoredBackend", lambda backend, *args: backend)
    report = {"sketch_root": str(tmp_path), "options": {"sketch_strength": .5}}
    runner.create_backend(settings, True, report)
    assert captured == [.5]


@pytest.mark.parametrize("seeds", [None, [209212002, 209212004, 209212005, 209212006]])
def test_case_seed_selection_preserves_base_then_contour(seeds):
    selected = [209212003, 209212001] if seeds is None else seeds
    actual = runner.cases(seeds=seeds)
    assert [(c["seed"], c["contour"]) for c in actual] == [
        (seed, contour) for seed in selected for contour in (False, True)]
    for case in actual:
        if case["contour"]:
            assert case["expected_sha256"] is None
            assert case["sha_comparison"] == "not_applicable_contour"
        elif seeds is None:
            assert case["expected_sha256"] == contracts.BASE_SHAS[case["seed"]]
            assert case["sha_comparison"] == "reference_required"
        else:
            assert case["expected_sha256"] is None
            assert case["sha_comparison"] == "no_reference_available"


def test_new_seeds_eight_cpu_cases_keep_sha_and_options(trial_request, report, tmp_path):
    inputs, settings = trial_request
    output = tmp_path / "gpu-04"
    seeds = [209212002, 209212004, 209212005, 209212006]
    configured = runner.configure_trial(report, .5, False, output, seeds)
    runner.save_preflight(configured)
    runner.require_saved_preflight(configured)
    loaded = []
    def factory(settings, contour, actual_report):
        assert actual_report["options"]["seeds"] == seeds
        backend = FakeBackend(settings)
        backend.forward_calls = [1, 1] if contour else [1]
        loaded.append((contour, backend))
        return backend
    runner.run_trial(output, inputs, settings, configured, factory)
    assert [x[0] for x in loaded] == [False, True] * 4
    assert all(b.closed for _, b in loaded)
    state = contracts.read(output / "status.json")
    assert state["seeds"] == seeds
    assert state["strengths"] == [1.2, .5]
    assert state["status"] == "completed_pending_visual_review"
    assert len(state["completed"]) == 8
    saved_preflight = contracts.read(output / "preflight.json")
    assert saved_preflight["options"]["seeds"] == seeds
    assert saved_preflight["strengths"] == [1.2, .5]
    for case in runner.cases(seeds=seeds):
        folder = output / case["id"]
        record = contracts.read(folder / "run.json")
        assert record["valid"]
        assert record["raw_sha256"] == contracts.sha(folder / "raw.png")
        assert record["expected_sha256"] is None
        assert record["trial"]["case"]["sha_comparison"] == case["sha_comparison"]
        assert record["trial"]["seeds"] == seeds
        assert record["trial"]["options"] == state["options"]
        assert record["trial"]["strengths"] == ([1.2, .5] if case["contour"] else [1.2])
        assert record["adapter_applied_calls"] == list(range(11))
        assert contracts.canonical(record["prompt"]) == contracts.canonical(asdict(inputs.prompt))


@pytest.mark.parametrize("arguments", [
    ["--seeds"], ["--seeds", "209212001"], ["--seeds", "209212003"],
    ["--seeds", "209212007"], ["--seeds", "not-an-integer"],
    ["--seeds", "209212002", "209212002"],
    ["--seeds", "209212002", "--contour-only"]])
def test_cli_rejects_bad_seed_selection_before_preflight(arguments, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda: pytest.fail("No preflight for invalid options"))
    with pytest.raises(SystemExit) as error:
        runner.main(["run"] + arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("changed_seeds", [
    [209212004, 209212002, 209212005, 209212006], [209212002, 209212004]])
def test_seed_list_and_order_are_preflight_locked(report, tmp_path, changed_seeds):
    output = tmp_path / "gpu-04"
    original = runner.configure_trial(report, .5, False, output, runner.NEW_SEEDS)
    runner.save_preflight(original)
    changed = runner.configure_trial(report, .5, False, output, changed_seeds)
    with pytest.raises(ValueError, match="changed"):
        runner.require_saved_preflight(changed)


def test_cli_passes_selected_seed_plan_to_locked_run(trial_request, report, tmp_path, monkeypatch):
    inputs, settings = trial_request
    output = tmp_path / "gpu-04"
    args = ["--sketch-strength", "0.5", "--seeds", *map(str, runner.NEW_SEEDS), "--output", str(output)]
    monkeypatch.setattr(runner, "preflight", lambda: (inputs, settings, report))
    runner.main(["preflight"] + args)
    captured = []
    monkeypatch.setattr(runner, "run_trial", lambda *values: captured.append(values))
    runner.main(["run"] + args)
    assert len(captured) == 1
    dest, actual_inputs, actual_settings, configured = captured[0]
    assert dest == output and actual_inputs is inputs and actual_settings is settings
    assert configured["options"]["seeds"] == list(runner.NEW_SEEDS)
    assert configured["strengths"] == [1.2, .5]


def test_new_seed_failure_stops_without_retry(trial_request, report, tmp_path):
    inputs, settings = trial_request
    output = tmp_path / "gpu-04"
    configured = runner.configure_trial(report, .5, False, output, runner.NEW_SEEDS)
    attempted = []
    def factory(settings, contour, report):
        attempted.append(contour)
        raise RuntimeError("model load failed")
    with pytest.raises(RuntimeError, match="model load failed"):
        runner.run_trial(output, inputs, settings, configured, factory)
    assert attempted == [False]
    state = contracts.read(output / "status.json")
    assert state["status"] == "stopped" and state["completed"] == []
    assert state["seeds"] == list(runner.NEW_SEEDS)
    record = contracts.read(output / "BASE_209212002/run.json")
    assert not record["valid"]
    assert record["trial"]["case"]["sha_comparison"] == "no_reference_available"
    assert not (output / "CONTOUR_209212002").exists()


@pytest.fixture
def seed_maps(trial_request, report, tmp_path, monkeypatch):
    inputs, settings = trial_request
    monkeypatch.setattr(runner, "DESIGN", tmp_path)
    report.update(settings=asdict(settings), base_inputs=asdict(inputs), versions={"test": "cpu"},
                  applied_steps=list(range(11)), protected={str(Path(runner.__file__)): "old", "production.py": "same"})
    map_dir = tmp_path / "twopass/maps"
    map_dir.mkdir(parents=True)
    manifest, old_shas = {}, {}
    for index, seed in enumerate(runner.MAP_SEEDS):
        source = tmp_path / ("gpu-02" if seed in contracts.BASE_SHAS else "gpu-04")
        folder = source / f"BASE_{seed}"
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), (index, index, index)).save(folder / "raw.png")
        digest = contracts.sha(folder / "raw.png")
        if seed in contracts.BASE_SHAS:
            old_shas[seed] = digest
        runner.write_trial_record(folder / "run.json", {
            "completed": True, "valid": True, "seed": seed, "raw_sha256": digest,
            "settings": asdict(settings), "prompt": asdict(inputs.prompt), "validation_errors": [],
            "max_reserved_gib": 1., "adapter_applied_calls": list(range(11)), "trial": {"strengths": [1.2]}})
        pixels = np.zeros((1232, 736, 3), dtype=np.uint8)
        pixels[20:60, index + 10, :] = 255
        path = map_dir / f"sketch_{seed}.png"
        Image.fromarray(pixels).save(path)
        manifest[str(seed)] = {"base_raw": str(folder / "raw.png"), "base_raw_sha256": digest,
                               "sketch_sha256": contracts.sha(path)}
    for name in ("gpu-02", "gpu-04"):
        source = tmp_path / name
        runner.write_trial_record(source / "preflight.json", report)
        runner.write_trial_record(source / "status.json", {"completed": [p.name for p in source.glob("BASE_*")]})
    runner.write_trial_record(map_dir.parent / "maps.json", manifest)
    monkeypatch.setattr(runner, "BASE_SHAS", old_shas)
    report["protected"][str(Path(runner.__file__))] = "new"
    return report, map_dir


def test_seed_maps_six_cpu_cases_record_exact_selected_images(trial_request, seed_maps, tmp_path):
    inputs, settings = trial_request
    report, directory = seed_maps
    output = tmp_path / "gpu-05"
    configured = runner.configure_trial(report, .5, True, output, runner.MAP_SEEDS, directory)
    runner.save_preflight(configured)
    runner.require_saved_preflight(configured)
    loaded = []
    def factory(settings, contour, actual_report):
        assert contour
        class MapBackend(FakeBackend):
            forward_calls = [1, 1]
            def generate(self, inputs, images, seed, observation, cancelled):
                with runner.load_sketch_image(seed, actual_report["sketch_maps"]) as sketch:
                    loaded.append((seed, sketch.tobytes()))
                return super().generate(inputs, images, seed, observation, cancelled)
        return MapBackend(settings)
    runner.run_trial(output, inputs, settings, configured, factory)
    assert [seed for seed, _ in loaded] == list(runner.MAP_SEEDS)
    assert len({pixels for _, pixels in loaded}) == 6
    state = contracts.read(output / "status.json")
    assert state["sketch_mode"] == "seed_specific" and state["strengths"] == [1.2, .5]
    assert len(state["completed"]) == 6 and not list(output.glob("BASE_*"))
    assert state["sketch_maps"] == contracts.read(output / "preflight.json")["sketch_maps"]
    for seed in runner.MAP_SEEDS:
        record = contracts.read(output / f"CONTOUR_{seed}/run.json")
        item = state["sketch_maps"]["items"][str(seed)]
        assert record["trial"]["sketch_input"] == item
        assert record["trial"]["sketch_mode"] == "seed_specific"
        assert record["trial"]["control_image_sha256"] == ["pose-sha", contracts.sha(directory / f"sketch_{seed}.png")]
        assert record["trial"]["strengths"] == [1.2, .5]
        assert record["adapter_applied_calls"] == list(range(11))
        assert contracts.canonical(record["prompt"]) == contracts.canonical(asdict(inputs.prompt))


@pytest.mark.parametrize("damage", ["sha", "size", "grey", "color", "alpha", "missing", "base_sha", "base_invalid", "code"])
def test_seed_maps_reject_bad_inputs_before_generation(seed_maps, tmp_path, damage):
    report, directory = seed_maps
    seed = runner.MAP_SEEDS[0]
    path = directory / f"sketch_{seed}.png"
    manifest_path = directory.parent / "maps.json"
    manifest = contracts.read(manifest_path)
    if damage == "sha":
        manifest[str(seed)]["sketch_sha256"] = "0" * 64
    elif damage == "missing":
        path.unlink()
    elif damage == "base_sha":
        manifest[str(seed)]["base_raw_sha256"] = "0" * 64
    elif damage == "base_invalid":
        rp = tmp_path / f"gpu-02/BASE_{seed}/run.json"
        record = contracts.read(rp)
        record["valid"] = False
        runner.write_trial_record(rp, record)
    elif damage == "code":
        report["protected"]["production.py"] = "modified"
    else:
        size = (735, 1232) if damage == "size" else (736, 1232)
        mode = "RGBA" if damage == "alpha" else "RGB"
        color = (255, 0, 0) if damage == "color" else (128, 128, 128) if damage == "grey" else "black"
        Image.new(mode, size, color).save(path)
        manifest[str(seed)]["sketch_sha256"] = contracts.sha(path)
    runner.write_trial_record(manifest_path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.configure_trial(report, .5, True, tmp_path / "gpu-05", runner.MAP_SEEDS, directory)
    assert not (tmp_path / "gpu-05").exists()


@pytest.mark.parametrize("damage", ["manifest", "map", "mode"])
def test_seed_map_lock_rejects_changes(seed_maps, tmp_path, damage):
    report, directory = seed_maps
    output = tmp_path / "gpu-05"
    configured = runner.configure_trial(report, .5, True, output, runner.MAP_SEEDS, directory)
    runner.save_preflight(configured)
    if damage == "mode":
        configured["options"]["sketch_mode"] = "head_only"
        with pytest.raises(ValueError, match="changed"):
            runner.require_saved_preflight(configured)
        return
    manifest_path = directory.parent / "maps.json"
    manifest = contracts.read(manifest_path)
    if damage == "map":
        seed = runner.MAP_SEEDS[0]
        path = directory / f"sketch_{seed}.png"
        Image.new("RGB", (736, 1232), "white").save(path)
        manifest[str(seed)]["sketch_sha256"] = contracts.sha(path)
    else:
        manifest[str(runner.MAP_SEEDS[0])]["note"] = "changed"
    runner.write_trial_record(manifest_path, manifest)
    fresh = runner.configure_trial(report, .5, True, output, runner.MAP_SEEDS, directory)
    with pytest.raises(ValueError, match="changed"):
        runner.require_saved_preflight(fresh)


@pytest.mark.parametrize("args", [[], ["--contour-only"], ["--sketch-strength", "0.5"],
    ["--contour-only", "--sketch-strength", "0.5", "--seeds", "209212007"],
    ["--contour-only", "--sketch-strength", "0.5", "--seeds", "209212001", "209212001"]])
def test_bad_map_cli_options_stop_before_preflight(monkeypatch, args):
    monkeypatch.setattr(runner, "preflight", lambda: pytest.fail("Invalid options must stop first"))
    with pytest.raises(SystemExit) as error:
        runner.main(["run", "--sketch-map-dir", "maps", *args])
    assert error.value.code == 2


def test_sketch_change_between_cases_stops_before_next_model(trial_request, seed_maps, tmp_path):
    inputs, settings = trial_request
    report, directory = seed_maps
    output = tmp_path / "gpu-05"
    configured = runner.configure_trial(report, .5, True, output, runner.MAP_SEEDS, directory)
    loaded = []
    def factory(settings, contour, report):
        loaded.append(contour)
        Image.new("RGB", (736, 1232), "white").save(directory / f"sketch_{runner.MAP_SEEDS[1]}.png")
        backend = FakeBackend(settings)
        backend.forward_calls = [1, 1]
        return backend
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner.run_trial(output, inputs, settings, configured, factory)
    state = contracts.read(output / "status.json")
    assert loaded == [True] and state["status"] == "stopped"
    assert state["completed"] == [f"CONTOUR_{runner.MAP_SEEDS[0]}"]
    assert (output / state["completed"][0] / "raw.png").exists()
    failed = contracts.read(output / f"CONTOUR_{runner.MAP_SEEDS[1]}/run.json")
    assert not failed["valid"] and failed["trial"]["sketch_mode"] == "seed_specific"


def test_map_backend_generate_passes_seed_pixels_without_transform(trial_request, seed_maps, tmp_path, monkeypatch):
    inputs, settings = trial_request
    report, directory = seed_maps
    configured = runner.configure_trial(report, .5, True, tmp_path / "gpu-05", runner.MAP_SEEDS, directory)
    captured = []
    class Pipe:
        unet = SimpleNamespace(register_forward_hook=lambda *args, **kwargs: SimpleNamespace(remove=lambda: None))
        text_encoder = object()
        text_encoder_2 = SimpleNamespace(dtype="fake")
        _execution_device = "cpu"
        def set_ip_adapter_scale(self, value):
            assert value == .5
        def __call__(self, **kwargs):
            captured.append((kwargs["generator"], kwargs["image"][1].tobytes(), kwargs))
            return SimpleNamespace(images=[Image.new("RGB", (8, 8))])
    backend = runner.HeadContourBackend.__new__(runner.HeadContourBackend)
    backend.settings, backend.pipe = settings, Pipe()
    backend.sketch_maps, backend.sketch_strength = configured["sketch_maps"], .5
    backend.versions, backend.callback_mode = {}, "legacy_callback"
    backend.scheduler_type = SimpleNamespace(from_config=lambda config: SimpleNamespace(config={}))
    backend.torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None, max_memory_reserved=lambda: 0),
        Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed))
    monkeypatch.setattr(runner, "encode_prompt_plan", lambda *args: {})
    monkeypatch.setattr(runner, "pipeline_callback_kwargs", lambda *args: {})
    pose, face = object(), object()
    for seed in runner.MAP_SEEDS:
        backend.generate(inputs, (pose, face), seed, lambda *args: None, lambda: False)
    for seed, pixels, kwargs in captured:
        with Image.open(directory / f"sketch_{seed}.png") as expected:
            assert pixels == expected.tobytes()
        assert kwargs["image"][0] is pose and kwargs["ip_adapter_image"] is face
        assert kwargs["adapter_conditioning_scale"] == [1.2, .5]
        assert kwargs["num_inference_steps"] == 28
    assert len(captured) == 6
