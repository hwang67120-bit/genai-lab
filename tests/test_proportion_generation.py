"""CPU 제품 흐름 검사다. 실제 작업 조립과 대체 파이프라인·합성 픽셀을 사용한다."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from PIL import Image
import pytest

from genai_lab import onepass_generation as one
from genai_lab import proportion_inputs as contract
from genai_lab import proportion_generation as flow
from genai_lab.proportion_backend import ProportionBackend, StepGuard
from genai_lab.onepass_generation_settings import OnePassGenerationSettings
from test_onepass_generation import prompt, backend, FakePipeline, FakeTorch, FakeScheduler


@pytest.fixture
def case(tmp_path, monkeypatch):
    settings = OnePassGenerationSettings(model_root=tmp_path / "model", adapter_root=tmp_path / "pose",
                                         ip_root=tmp_path / "ip")
    for name in ("unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
        (settings.model_root / name).mkdir(parents=True)
    sketch = tmp_path / "sketch-model"
    config = dict(adapter_type="full_adapter_xl", channels=[1], downscale_factor=16, in_channels=3, num_res_blocks=2)
    files = {settings.model_root / "model_index.json": "{}", settings.adapter_root / "config.json": json.dumps(config),
             settings.adapter_root / "diffusion_pytorch_model.safetensors": "pose",
             settings.ip_root / settings.ip_subfolder / settings.ip_weight_name: "ip",
             settings.ip_root / settings.image_encoder_subfolder / "config.json": "{}",
             sketch / "config.json": json.dumps(config), sketch / "diffusion_pytorch_model.fp16.safetensors": "sketch",
             tmp_path / "isnet.onnx": "synthetic-isnet"}
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    monkeypatch.setattr(contract, "SKETCH_HASHES", {name: contract.sha(sketch / name) for name in contract.SKETCH_HASHES})
    normalized, control, face, mask_path = [tmp_path / name for name in ("normalized.png", "control.png", "face.png", "head.png")]
    for path in (normalized, control, face):
        Image.new("RGB", (736, 1232), (30, 80, 110)).save(path)
    mask = np.zeros((1232, 736), np.uint8)
    mask[40:220, 220:460] = 255
    Image.fromarray(mask).save(mask_path)
    head = contract.prepare_head_outline(normalized, control, mask_path, tmp_path / "head-preview", nose=(340, 170), neck=(340, 230))
    head = replace(head, confirmed=True)
    inputs = one.OnePassInputs(prompt(), control, face, contract.sha(control), contract.sha(face), .5, pose_mode="with_pose")
    monkeypatch.setattr(contract, "FOREGROUND_SHA", contract.sha(tmp_path / "isnet.onnx"))
    options = contract.ProportionOptions(True, head, sketch, tmp_path / "isnet.onnx", contract.sha(tmp_path / "isnet.onnx"))
    return inputs, settings, options


class Foreground:
    def __init__(self, options, events):
        self.events = events
        events.append("cpu-open")
    def alpha(self, image):
        self.events.append("alpha")
        result = np.zeros((image.height, image.width), np.uint8)
        result[40:1100, 220:460] = 255
        return result
    def close(self):
        self.events.append("cpu-close")


class TestTorch(FakeTorch):
    __test__ = False
    class cuda(FakeTorch.cuda):
        @staticmethod
        def max_memory_allocated():
            return 1000


class MultiPipeline(FakePipeline):
    def __call__(self, **kwargs):
        pose, sketch = kwargs.pop("image")
        assert kwargs["adapter_conditioning_scale"] == [1.2, .5]
        assert np.isin(np.asarray(sketch), [0, 255]).all()
        self.owner.forward_counts[:] = [1, 1]
        return super().__call__(image=pose, **kwargs)


def factories(case, events):
    inputs, settings, options = case
    def base_factory(settings):
        events.append("base-open")
        value = backend(settings)
        value.close = lambda: events.append("base-close")
        return value
    def contour_factory(settings, options, maps):
        events.append("contour-open")
        value = object.__new__(ProportionBackend)
        value.settings, value.maps = settings, maps
        value.models = contract.validate_models(settings, options)
        value.pipe = MultiPipeline()
        value.pipe.owner = value
        value.torch, value.scheduler_type = TestTorch, FakeScheduler
        value.callback_mode, value.versions = "callback_on_step_end", {"cpu": "fake"}
        value.forward_counts = [0, 0]
        value.close = lambda: events.append("contour-close")
        return value
    return dict(base_factory=base_factory, contour_factory=contour_factory,
                foreground_factory=lambda options: Foreground(options, events))


def test_disabled_uses_original_single_adapter_and_raw_bytes(case, tmp_path, monkeypatch):
    inputs, settings, options = case
    def forbidden(*args, **kwargs):
        pytest.fail("OFF must not call proportion path")
    monkeypatch.setattr(flow, "generate_proportion_batch", forbidden)
    request = one.OnePassRequest(inputs, (1, 2, 3, 4), settings)
    first = one.generate_onepass_request(request, tmp_path / "legacy", backend_factory=backend)
    second = one.generate_onepass_request(request, tmp_path / "off", backend_factory=backend,
                                          proportion=contract.ProportionOptions())
    assert [c.path.read_bytes() for c in first.candidates] == [c.path.read_bytes() for c in second.candidates]
    assert (first.directory / "request.json").read_bytes() == (second.directory / "request.json").read_bytes()
    assert not (second.directory / "preflight.json").exists()


def test_product_entry_dispatches_only_explicit_enabled_options(case, tmp_path, monkeypatch):
    inputs, settings, options = case
    captured = []
    def run(*args, **kwargs):
        captured.append((args, kwargs))
        return "two-pass"
    monkeypatch.setattr(flow, "generate_proportion_batch", run)
    request = one.OnePassRequest(inputs, (1, 2, 3, 4), settings)
    assert one.generate_onepass_request(request, tmp_path / "run", proportion=options) == "two-pass"
    assert captured[0][0][0] is inputs and captured[0][1]["options"] is options


def test_two_pass_records_actual_models_schedule_and_releases_before_next_stage(case, tmp_path):
    inputs, settings, options = case
    events, delivered = [], []
    batch = flow.generate_proportion_batch(inputs, (1, 2), tmp_path / "run", settings=settings, options=options,
                                           on_image=delivered.append, **factories(case, events))
    assert events == ["base-open", "base-close", "cpu-open", "alpha", "alpha", "cpu-close",
                      "contour-open", "contour-close", "cpu-open", "alpha", "alpha", "cpu-close"]
    assert len(delivered) == 2 and not batch.final_return_eligible
    assert batch.review_stage == "proportion_unreviewed"
    state = json.loads((batch.directory / "run.json").read_text(encoding="utf-8"))
    assert state["status"] == "review_pending" and state["products"] == [1, 2]
    assert state["adapter_strengths"] == [1.2, .5] and state["shared_adapter_steps"] == list(range(11))
    for candidate in batch.candidates:
        raw = json.loads((batch.directory / f"CONTOUR_{candidate.seed}/run.json").read_text(encoding="utf-8"))
        assert raw["adapter_forward_counts"] == [1, 1]
        assert len(raw["models"]["adapters"]) == 2
        assert raw["sketch"]["sha256"] == contract.sha(raw["sketch"]["path"])
        assert raw["adapter_applied_calls"] == list(range(11))
        assert [c["ip_scales"] for c in raw["unet_calls"]] == [[.5]] * 11 + [[.9]] * 17
        assert candidate.record["product_sha256"] == contract.sha(candidate.path)
        assert candidate.record["raw_sha256"] == contract.sha(candidate.record["raw_file"])
        with Image.open(candidate.path) as image:
            assert image.getpixel((0, 0)) == (255, 255, 255)
            assert image.getpixel((300, 300)) == (10, 20, 30)


@pytest.mark.parametrize("change", ["unconfirmed", "wrong_control", "tamper_contour", "wrong_settings", "no_pose", "model_tamper"])
def test_preflight_rejects_before_backend_or_output(case, tmp_path, change):
    inputs, settings, options = case
    if change == "unconfirmed":
        options = replace(options, head=replace(options.head, confirmed=False))
    elif change == "wrong_control":
        options = replace(options, head=replace(options.head, control_sha256="f" * 64))
    elif change == "tamper_contour":
        options.head.contour_file.write_bytes(b"changed")
    elif change == "wrong_settings":
        settings = replace(settings, adapter_steps=12)
    elif change == "no_pose":
        inputs = replace(inputs, pose_mode="without_pose", control_file=None, control_sha256=None, ip_early=0)
    elif change == "model_tamper":
        (options.sketch_root / "config.json").write_text("changed")
    with pytest.raises(ValueError):
        flow.generate_proportion_batch(inputs, (1,), tmp_path / "rejected", settings=settings, options=options,
                                        base_factory=lambda _: pytest.fail("GPU load"))
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("failure", ["empty_alpha", "cancel_cpu", "raw_sha", "map_sha", "postprocess"])
def test_failure_preserves_first_pass_and_no_success_callback(case, tmp_path, failure):
    inputs, settings, options = case
    events, sent = [], []
    providers = factories(case, events)
    cancelled = [False]
    class FaultyForeground(Foreground):
        def alpha(self, image):
            result = super().alpha(image)
            if failure == "empty_alpha":
                result[:] = 0
            if failure == "cancel_cpu":
                cancelled[0] = True
            if failure == "postprocess" and "contour-close" in events:
                raise RuntimeError("postprocess failed")
            return result
    providers["foreground_factory"] = lambda opt: FaultyForeground(opt, events)
    expected = {"expected_base": {1: "0" * 64}} if failure == "raw_sha" else {}
    if failure == "map_sha":
        expected = {"expected_sketch": {1: "0" * 64}}
    with pytest.raises((ValueError, one.OnePassGenerationError, RuntimeError)):
        flow.generate_proportion_batch(inputs, (1,), tmp_path / "failed", settings=settings, options=options,
                                        on_image=sent.append, cancelled=lambda: cancelled[0], **providers, **expected)
    assert not sent
    assert (tmp_path / "failed/BASE_1/raw.png").exists()
    state = json.loads((tmp_path / "failed/run.json").read_text(encoding="utf-8"))
    assert state["status"] == ("cancelled" if failure == "cancel_cpu" else "failed")
    assert "base-close" in events
    if failure != "raw_sha":
        assert "cpu-close" in events
    if failure == "postprocess":
        assert (tmp_path / "failed/CONTOUR_1/raw.png").exists() and "contour-close" in events


def test_pixel_rules_keep_original_head_remove_old_head_and_round_white_composite():
    mask = np.zeros((80, 60), np.uint8)
    mask[5:25, 15:45] = 255
    contour = contract.draw_head_contour(mask)
    alpha = np.zeros_like(mask)
    alpha[15:70, 22:38] = 255
    sketch = contract.make_sketch(alpha, mask, contour)
    assert np.isin(sketch, [0, 255]).all()
    assert np.array_equal(sketch[10:20, 20:40], contour[10:20, 20:40])
    assert sketch[50, 22, 0] == 255
    rgb = np.array([[[10, 20, 30], [11, 21, 31], [12, 22, 32]]], np.uint8)
    alpha = np.array([[255, 0, 128]], np.uint8)
    result = contract.white_background(rgb, alpha)
    assert result.tolist() == [[[10, 20, 30], [255, 255, 255], [133, 138, 143]]]


def test_step_guard_stops_immediately_on_memory_or_schedule():
    settings = OnePassGenerationSettings()
    observation = one.UNetObservation()
    module = SimpleNamespace(attn_processors={"ip": SimpleNamespace(scale=[.5])})
    guard = StepGuard(observation, settings, SimpleNamespace(ip_early=.5),
                      SimpleNamespace(max_memory_reserved=lambda: 7 * 2**30))
    with pytest.raises(one.OnePassGenerationError, match="6.5"):
        guard(module, (), {"down_intrablock_additional_residuals": []}, None)
    assert len(observation.calls) == 1



def test_second_pass_failure_keeps_base_and_releases_pipeline(case, tmp_path):
    inputs, settings, options = case
    events = []
    providers = factories(case, events)
    create = providers["contour_factory"]
    def faulty(*args):
        value = create(*args)
        value.generate = lambda *a: (_ for _ in ()).throw(RuntimeError("second pass failed"))
        return value
    providers["contour_factory"] = faulty
    with pytest.raises(RuntimeError, match="second pass failed"):
        flow.generate_proportion_batch(inputs, (1,), tmp_path / "failed", settings=settings, options=options, **providers)
    assert events[-1] == "contour-close"
    record = json.loads((tmp_path / "failed/CONTOUR_1/run.json").read_text(encoding="utf-8"))
    assert len(record["models"]["adapters"]) == 2 and not record["valid"]
    assert (tmp_path / "failed/BASE_1/raw.png").exists()
    assert not (tmp_path / "failed/seed-1/product.png").exists()


def test_bad_adapter_counts_reject_but_preserve_second_raw(case, tmp_path):
    inputs, settings, options = case
    events = []
    providers = factories(case, events)
    create = providers["contour_factory"]
    def faulty(*args):
        value = create(*args)
        generate = value.generate
        def wrong_counts(*a):
            image, timing = generate(*a)
            timing["adapter_forward_counts"] = [1, 0]
            return image, timing
        value.generate = wrong_counts
        return value
    providers["contour_factory"] = faulty
    with pytest.raises(one.OnePassGenerationError, match="각각 1회"):
        flow.generate_proportion_batch(inputs, (1,), tmp_path / "failed", settings=settings, options=options, **providers)
    assert (tmp_path / "failed/CONTOUR_1/raw.png").exists()
    record = json.loads((tmp_path / "failed/CONTOUR_1/run.json").read_text(encoding="utf-8"))
    assert record["completed"] and not record["valid"]


def test_replay_cli_default_is_cpu_preflight_and_serializes_paths(case, tmp_path, monkeypatch):
    from scripts import verify_proportion_product as replay
    monkeypatch.setattr(replay, "replay_inputs", lambda _: case)
    monkeypatch.setattr(replay, "replay_expectations", lambda: ({}, {}, {}))
    monkeypatch.setattr(replay, "generate_proportion_batch", lambda *a, **kw: pytest.fail("GPU action"))
    target = tmp_path / "preflight"
    replay.main(["--foreground-model", str(case[2].foreground_model), "--output", str(target)])
    result = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    assert result["status"] == "validated_no_generation"
    assert result["options"]["head"]["contour_file"] == str(case[2].head.contour_file)


def test_cancel_before_start_does_not_load_or_write(case, tmp_path):
    inputs, settings, options = case
    with pytest.raises(one.OnePassCancelled):
        flow.generate_proportion_batch(inputs, (1,), tmp_path / "cancelled", settings=settings,
                                        options=options, cancelled=lambda: True,
                                        base_factory=lambda _: pytest.fail("loaded"))
    assert not (tmp_path / "cancelled").exists()



def test_second_pass_legacy_callback_matches_ip_schedule(case, tmp_path):
    from test_onepass_generation import FakeLegacyPipeline
    inputs, settings, options = case
    events = []
    maps = {1: {"path": str(options.head.contour_file), "sha256": options.head.contour_sha256}}
    value = factories(case, events)["contour_factory"](settings, options, maps)
    class LegacyMultiPipeline(FakeLegacyPipeline):
        def __call__(self, **kwargs):
            pose, sketch = kwargs.pop("image")
            assert kwargs["adapter_conditioning_scale"] == [1.2, .5]
            assert kwargs["callback_steps"] == 1
            value.forward_counts[:] = [1, 1]
            return super().__call__(image=pose, **kwargs)
    value.pipe = LegacyMultiPipeline()
    value.callback_mode = "legacy_callback"
    result = one.generate_onepass_image(value, inputs, 1, tmp_path / "legacy-contour", settings=settings)
    assert result.record["adapter_forward_counts"] == [1, 1]
    assert value.pipe.history == [[.5] * 11 + [.9] * 17]
    assert result.record["callback_mode"] == "legacy_callback"
    assert not value.pipe.unet._forward_hooks


def test_head_preview_is_never_automatically_confirmed(case, tmp_path):
    inputs, settings, options = case
    head = contract.prepare_head_outline(options.head.normalized_file, inputs.control_file, options.head.mask_file,
                                         tmp_path / "another-preview", nose=options.head.nose, neck=options.head.neck)
    assert head.confirmed is False
    assert (tmp_path / "another-preview/preview.png").exists()
    assert head.contour_sha256 == options.head.contour_sha256


def test_second_raw_sha_mismatch_never_becomes_a_product(case, tmp_path):
    inputs, settings, options = case
    events, sent = [], []
    with pytest.raises(one.OnePassGenerationError, match="SHA"):
        flow.generate_proportion_batch(inputs, (1,), tmp_path / "failed", settings=settings, options=options,
                                        expected_raw={1: "0" * 64}, on_image=sent.append, **factories(case, events))
    assert not sent and events[-1] == "contour-close"
    assert (tmp_path / "failed/CONTOUR_1/raw.png").exists()
    assert not (tmp_path / "failed/seed-1/product.png").exists()
