"""CPU-only stage 4 contracts. No model loads, CUDA, or generated model images."""
import builtins
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from genai_lab import onepass_generation as generation
from genai_lab.onepass_generation_settings import OnePassGenerationSettings, scheduler_config
from genai_lab.onepass_prompt import OnePassPrompt, EncoderChunkPlan, PromptChunk


def prompt(count=1):
    def chunks(offset):
        return tuple(PromptChunk("ids", None, (offset + i,) * 77) for i in range(count))
    return OnePassPrompt("positive", "negative", tuple(
        EncoderChunkPlan(80 if count > 1 else 1, 1, chunks(10), chunks(20))
        for _ in range(2)
    ), {})


@pytest.fixture
def settings():
    return OnePassGenerationSettings(width=16, height=24)


@pytest.fixture
def inputs(tmp_path, settings):
    paths = [tmp_path / "control.png", tmp_path / "face.png"]
    Image.new("RGB", (settings.width, settings.height), (1, 2, 3)).save(paths[0])
    Image.new("RGB", (8, 8), (4, 5, 6)).save(paths[1])
    return generation.OnePassInputs(
        prompt(), *paths, *(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths), pose_mode="with_pose"
    )


class FakeEncoder:
    def __init__(self, size, projection=False):
        self.size, self.projection = size, projection

    def __call__(self, ids, *, output_hidden_states):
        assert output_hidden_states
        assert not torch.is_grad_enabled()
        hidden = ids.unsqueeze(-1).repeat(1, 1, self.size).float()
        pooled = ids[:, :1].float() if self.projection else hidden
        return EncoderOutput(pooled, (hidden * 0, hidden, hidden * 99))


class EncoderOutput:
    def __init__(self, first, hidden_states):
        self.first, self.hidden_states = first, hidden_states

    def __getitem__(self, key):
        assert key == 0
        return self.first


@pytest.mark.parametrize("count", [1, 2, 3])
def test_chunk_embeddings_use_penultimate_states_and_first_te2_pool(count):
    result = generation.encode_prompt_plan(
        prompt(count), (FakeEncoder(2), FakeEncoder(3, True)), "cpu", torch.float32
    )
    assert result["prompt_embeds"].shape == (1, 77 * count, 5)
    assert result["negative_prompt_embeds"].shape == (1, 77 * count, 5)
    assert result["pooled_prompt_embeds"].item() == 10
    assert result["negative_pooled_prompt_embeds"].item() == 20
    for i in range(count):
        assert torch.all(result["prompt_embeds"][:, 77*i:77*(i+1)] == 10 + i)


class FakeUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_processors = {"ip": SimpleNamespace(scale=[0.0])}

    def forward(self, *, down_intrablock_additional_residuals):
        if down_intrablock_additional_residuals is not None:
            down_intrablock_additional_residuals.clear()  # Diffusers consumes the list.
        return 7


class FakePipeline:
    def __init__(self):
        self.unet = FakeUNet()
        self._execution_device = "cpu"
        self.text_encoder = FakeEncoder(2)
        self.text_encoder_2 = FakeEncoder(3, True)
        self.text_encoder_2.dtype = torch.float32
        self.history = []
        self.scale_sets = []
        self.schedulers = []
        self.images = []

    def set_ip_adapter_scale(self, value):
        self.scale_sets.append(value)
        self.unet.attn_processors["ip"].scale = [value]

    def __call__(
        self, *, callback_on_step_end, ip_adapter_image, adapter_conditioning_factor,
        adapter_conditioning_scale, prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds, **kwargs,
    ):
        self.schedulers.append(self.scheduler)
        self.images.append((kwargs["image"].copy(), ip_adapter_image.copy()))
        steps = kwargs["num_inference_steps"]
        values = []
        for i in range(steps):
            values.append(self.unet.attn_processors["ip"].scale[0])
            residuals = [object()] if i < int(steps * adapter_conditioning_factor) else None
            assert self.unet(down_intrablock_additional_residuals=residuals) == 7
            payload = {"latents": object()}
            assert callback_on_step_end(self, i, None, payload) is payload
        self.history.append(values)
        return SimpleNamespace(images=[Image.new("RGB", kwargs["image"].size, (10, 20, 30))])


class FakeLegacyPipeline(FakePipeline):
    def __call__(
        self, *, callback, callback_steps, ip_adapter_image, adapter_conditioning_factor,
        adapter_conditioning_scale, prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds, **kwargs,
    ):
        assert callback_steps == 1
        assert "callback_on_step_end" not in kwargs
        def at_step_end(_pipeline, index, timestep, payload):
            assert callback(index, timestep, payload["latents"]) is None
            return payload
        return super().__call__(
            callback_on_step_end=at_step_end, ip_adapter_image=ip_adapter_image,
            adapter_conditioning_factor=adapter_conditioning_factor,
            adapter_conditioning_scale=adapter_conditioning_scale,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, **kwargs,
        )


class FakeTorch:
    # Allows the real backend orchestration to run without ever calling CUDA.
    class cuda:
        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def reset_peak_memory_stats():
            pass

        @staticmethod
        def max_memory_reserved():
            return 1234

    @staticmethod
    def Generator(*, device):
        assert device == "cuda"
        return SimpleNamespace(manual_seed=lambda seed: SimpleNamespace(seed=seed))


class FakeScheduler:
    @staticmethod
    def from_config(config):
        assert config == scheduler_config()
        return SimpleNamespace(config=dict(config))


def backend(settings, pipeline_type=FakePipeline):
    value = object.__new__(generation.DiffusersOnePassBackend)
    value.settings = settings
    value.torch = FakeTorch
    value.pipe = pipeline_type()
    value.callback_mode = generation.check_pipeline_support(pipeline_type)
    value.scheduler_type = FakeScheduler
    value.versions = {"torch": "fake-cpu", "diffusers": "fake-cpu"}
    value.closed = False

    def close():
        value.closed = True
    value.close = close
    return value


@pytest.mark.parametrize("pipeline_type", [FakePipeline, FakeLegacyPipeline])
@pytest.mark.parametrize("early", [0.0, 0.5])
def test_real_backend_resets_each_seed_and_observes_actual_schedule(
    inputs, settings, tmp_path, early, pipeline_type
):
    value = backend(settings, pipeline_type)
    inputs = replace(inputs, ip_early=early)
    request = generation.OnePassRequest(inputs, (100, 101, 102, 103), settings)
    delivered = []
    batch = generation.generate_onepass_request(
        request, tmp_path / "batch", backend_factory=lambda _: value, on_image=delivered.append
    )
    assert len(delivered) == 4 and delivered == list(batch.candidates)
    assert value.closed
    assert len({id(s) for s in value.pipe.schedulers}) == 4
    assert value.pipe.scale_sets == [early, 0.9] * 4
    assert value.pipe.history == [[early] * 11 + [0.9] * 17] * 4
    assert len(value.pipe.unet._forward_hooks) == 0
    assert not batch.final_return_eligible
    for candidate in batch.candidates:
        record = json.loads(candidate.record_path.read_text(encoding="utf-8"))
        assert record["valid"] and record["completed"]
        assert record["callback_mode"] == value.callback_mode
        assert not record["gates_executed"]
        assert record["adapter_applied_calls"] == list(range(11))
        assert len(record["unet_calls"]) == 28
        assert record["raw_sha256"] == hashlib.sha256(candidate.path.read_bytes()).hexdigest()
        assert record["inputs"]["face_sha256"] == inputs.face_sha256
    # RGB channels and pixel data arrive unchanged.
    assert value.pipe.images[0][0].getpixel((0, 0)) == (1, 2, 3)


def test_observer_is_post_forward_read_only():
    module = FakeUNet()
    observation = generation.UNetObservation()
    handle = module.register_forward_hook(observation, with_kwargs=True)
    residuals = [object()]
    assert module(down_intrablock_additional_residuals=residuals) == 7
    handle.remove()
    assert residuals == []
    assert observation.calls == [{"index": 0, "ip_scales": [0.0], "adapter_applied": True}]


def test_callback_cancels_without_switching():
    pipe = FakePipeline()
    with pytest.raises(generation.OnePassCancelled):
        generation.step_end_callback(OnePassGenerationSettings(), lambda: True)(pipe, 10, 0, {})
    assert pipe.scale_sets == []


@pytest.mark.parametrize("failure", ["sha", "memory", "schedule"])
def test_failure_preserves_raw_and_record(inputs, settings, tmp_path, failure):
    value = backend(settings)
    expected = None
    if failure == "sha":
        expected = "0" * 64
    elif failure == "memory":
        settings = replace(settings, max_reserved_gib=1e-9)
        value.settings = settings
    else:
        value.pipe.set_ip_adapter_scale = lambda _: None
    with pytest.raises(generation.OnePassGenerationError):
        generation.generate_onepass_image(
            value, inputs, 100, tmp_path / "case", settings=settings, expected_sha256=expected
        )
    assert (tmp_path / "case/raw.png").exists()
    record = json.loads((tmp_path / "case/run.json").read_text(encoding="utf-8"))
    assert record["completed"] and not record["valid"]
    assert record["validation_errors"]
    assert len(record["unet_calls"]) == 28


def test_batch_stops_without_retry_after_invalid_schedule(inputs, settings, tmp_path):
    value = backend(settings)
    value.pipe.set_ip_adapter_scale = lambda _: None
    with pytest.raises(generation.OnePassGenerationError):
        generation.generate_onepass_request(
            generation.OnePassRequest(inputs, (1, 2, 3, 4), settings),
            tmp_path / "batch", backend_factory=lambda _: value,
            on_image=lambda _: pytest.fail("Invalid image must not be delivered."),
        )
    assert len(value.pipe.history) == 1
    assert value.closed
    assert not (tmp_path / "batch/seed-2").exists()


def test_request_cancellation_stops_before_next_seed(inputs, settings, tmp_path):
    value = backend(settings)
    delivered = []
    generation_error = generation.OnePassCancelled
    with pytest.raises(generation_error):
        generation.generate_onepass_request(
            generation.OnePassRequest(inputs, (1, 2, 3, 4), settings),
            tmp_path / "batch", backend_factory=lambda _: value,
            on_image=delivered.append, cancelled=lambda: bool(delivered),
        )
    assert len(delivered) == 1 and value.closed


def test_no_retry_or_overwrite(inputs, settings, tmp_path):
    value = backend(settings)
    directory = tmp_path / "existing"
    directory.mkdir()
    sentinel = directory / "raw.png"
    sentinel.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        generation.generate_onepass_image(value, inputs, 1, directory, settings=settings)
    assert sentinel.read_bytes() == b"preserved"
    assert value.pipe.history == []


@pytest.mark.parametrize("bad", [(), (1,), (1, 1, 2, 3), (1, 2, 3, 4, 5), (1, 2, 3, -1)])
def test_exactly_four_unique_seeds(inputs, bad):
    with pytest.raises(ValueError):
        generation.OnePassRequest(inputs, bad)


def test_input_digest_mismatch_prevents_loading(inputs, settings, tmp_path):
    request = generation.OnePassRequest(replace(inputs, face_sha256="0"*64), (1, 2, 3, 4), settings)
    with pytest.raises(generation.OnePassGenerationError, match="SHA-256"):
        generation.generate_onepass_request(
            request, tmp_path / "batch", backend_factory=lambda _: pytest.fail("No load")
        )


def test_control_image_not_resized(inputs, settings):
    with pytest.raises(generation.OnePassGenerationError, match="크기"):
        generation.read_inputs(inputs, replace(settings, width=24))


def test_missing_model_paths_fail_before_library_import(tmp_path, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(("diffusers", "transformers")):
            pytest.fail("Missing paths must stop before model libraries.")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(FileNotFoundError, match="다운로드하지 않음"):
        generation.DiffusersOnePassBackend(OnePassGenerationSettings(model_root=tmp_path / "missing"))


def test_incomplete_callback_contract_is_rejected():
    class Legacy:
        def __call__(self, callback=None, **kwargs):
            raise AssertionError("Must not run.")
    with pytest.raises(generation.OnePassGenerationError, match="callback_on_step_end"):
        generation.check_pipeline_support(Legacy)
    assert generation.check_pipeline_support(FakePipeline) == "callback_on_step_end"
    assert generation.check_pipeline_support(FakeLegacyPipeline) == "legacy_callback"


@pytest.mark.parametrize("steps,n", [(28, 11), (30, 12), (28, 28)])
def test_factor_integer_boundary(steps, n):
    settings = OnePassGenerationSettings(steps=steps, adapter_steps=n)
    assert int(steps * settings.adapter_factor) == n


def test_corrupt_chunk_plan_is_rejected():
    plan = prompt()
    plan = replace(plan, encoders=(plan.encoders[0], replace(plan.encoders[1], negative=())))
    with pytest.raises(ValueError, match="청크 수"):
        generation.validate_prompt(plan)


def make_orchestrator(tmp_path, legacy_generate):
    from genai_lab.generation_orchestrator import GenerationOrchestrator
    approval = SimpleNamespace(fingerprint="a" * 64)
    inputs = SimpleNamespace(approved_generation=approval)
    instance = GenerationOrchestrator(
        {"refinement_execution": {"enabled": True, "mode": "sdxl_local"}},
        SimpleNamespace(seed=1), tmp_path, None,
        prepare_visual_inputs_fn=lambda *_: inputs,
        generate_visual_batch_fn=legacy_generate,
        approve_reference_run_fn=lambda *_: None,
        require_reference_run_fn=lambda *_: approval,
        save_replay_bundle_fn=lambda *_: tmp_path / "replay",
    )
    instance.restore_approved_inputs(inputs)
    return instance, inputs


def test_option_off_does_not_import_new_modules(tmp_path, monkeypatch):
    from genai_lab.generation_orchestrator import GenerationPhase
    original = builtins.__import__
    calls = []
    batch = SimpleNamespace(directory=None, candidates=[])
    orchestrator, inputs = make_orchestrator(
        tmp_path, lambda *a, **kw: calls.append(kw) or batch
    )
    def guarded(name, *args, **kwargs):
        if name.startswith("genai_lab.onepass"):
            pytest.fail("OFF must not even request a new module import.")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert orchestrator.generate_base_candidates(object(), inputs) is batch
    assert len(calls) == 1 and "onepass" not in calls[0]
    assert orchestrator.phase is GenerationPhase.BASE_COMPLETED


def test_opt_in_never_dispatches_legacy_refinement(tmp_path, inputs, settings, monkeypatch):
    from genai_lab.generation_orchestrator import GenerationPhase, GenerationOrchestrationError
    orchestrator, approved = make_orchestrator(
        tmp_path, lambda *a, **kw: pytest.fail("No legacy generation")
    )
    batch = generation.OnePassBatch(tmp_path / "raw", ())
    dispatched = []
    def fake(request, directory, **kwargs):
        dispatched.append(request)
        return batch
    monkeypatch.setattr(generation, "generate_onepass_request", fake)
    request = generation.OnePassRequest(inputs, (1, 2, 3, 4), settings)
    assert orchestrator.generate_base_candidates(None, approved, onepass=request) is batch
    assert dispatched == [request]
    assert orchestrator.phase is GenerationPhase.ONEPASS_RAW_COMPLETED
    assert orchestrator.run_context.record["final_return_eligible"] is False
    with pytest.raises(GenerationOrchestrationError):
        orchestrator.select_base_candidate(batch, 0)
    orchestrator.close_request(reason="done")
    assert orchestrator.phase is GenerationPhase.ONEPASS_RAW_COMPLETED


def test_opt_in_requires_existing_input_approval(tmp_path, inputs, settings):
    from genai_lab.generation_orchestrator import GenerationOrchestrationError
    orchestrator, _ = make_orchestrator(tmp_path, lambda *a, **kw: None)
    with pytest.raises(GenerationOrchestrationError, match="서로 다릅니다"):
        orchestrator.generate_base_candidates(
            None, object(), onepass=generation.OnePassRequest(inputs, (1, 2, 3, 4), settings)
        )


def test_option_cannot_mix_preloaded_legacy_pipeline(tmp_path):
    from genai_lab.generation_orchestrator import GenerationOrchestrationError
    orchestrator, inputs = make_orchestrator(tmp_path, lambda *a, **kw: None)
    with pytest.raises(GenerationOrchestrationError, match="자체 파이프라인"):
        orchestrator.generate_base_candidates(object(), inputs, onepass=object())

def test_unsupported_pipeline_stops_before_model_loads(monkeypatch, settings):
    import sys
    class Legacy:
        def __call__(self, callback=None, **kwargs):
            pass
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pytest.fail("Unsupported pipeline must not load any weights.")
    monkeypatch.setattr(generation, "validate_local_models", lambda _: None)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        StableDiffusionXLAdapterPipeline=Legacy
    ))
    with pytest.raises(generation.OnePassGenerationError, match="callback_on_step_end"):
        generation.DiffusersOnePassBackend(settings)


def test_backend_exception_removes_observation_hook(inputs, settings, tmp_path, monkeypatch):
    value = backend(settings)
    def fail(*args, **kwargs):
        raise RuntimeError("inference interrupted")
    monkeypatch.setattr(generation, "encode_prompt_plan", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        generation.generate_onepass_image(value, inputs, 1, tmp_path / "case", settings=settings)
    assert not value.pipe.unet._forward_hooks
    record = json.loads((tmp_path / "case/run.json").read_text(encoding="utf-8"))
    assert not record["completed"] and record["error_type"] == "RuntimeError"


def test_oom_stops_batch_without_retry(inputs, settings, tmp_path):
    value = backend(settings)
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated OOM; no CUDA call")
    value.generate = fail
    with pytest.raises(torch.OutOfMemoryError):
        generation.generate_onepass_request(
            generation.OnePassRequest(inputs, (1, 2, 3, 4), settings), tmp_path / "batch",
            backend_factory=lambda _: value,
        )
    assert value.closed and not (tmp_path / "batch/seed-2").exists()
    record = json.loads((tmp_path / "batch/request.json").read_text(encoding="utf-8"))
    assert record["status"] == "stopped"


@pytest.mark.parametrize("pipeline_type", [FakePipeline, FakeLegacyPipeline])
def test_local_only_loading_flags_and_fp16(monkeypatch, settings, pipeline_type):
    import sys
    calls = []
    class LocalAdapter:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("adapter", path, kwargs))
            return object()
    class LocalEncoder:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("encoder", path, kwargs))
            return object()
    class LocalPipeline(pipeline_type):
        model_cpu_offload_seq = "original-class-sequence"
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("pipeline", path, kwargs))
            return cls()
        def load_ip_adapter(self, path, **kwargs):
            calls.append(("ip", path, kwargs))
        def enable_model_cpu_offload(self):
            assert self.model_cpu_offload_seq == (
                "text_encoder->text_encoder_2->image_encoder->adapter->unet->vae"
            )
            calls.append(("offload",))
    monkeypatch.setattr(generation, "validate_local_models", lambda _: None)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        StableDiffusionXLAdapterPipeline=LocalPipeline, T2IAdapter=LocalAdapter,
        EulerAncestralDiscreteScheduler=FakeScheduler, __version__="mock",
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        CLIPVisionModelWithProjection=LocalEncoder
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True), float16="mock-fp16", __version__="mock"
    ))
    loaded = generation.DiffusersOnePassBackend(settings)
    assert loaded.pipe.scale_sets == [0.0]
    assert LocalPipeline.model_cpu_offload_seq == "original-class-sequence"
    assert loaded.callback_mode == generation.check_pipeline_support(pipeline_type)
    assert settings.max_reserved_gib == 6.5
    assert calls[-1] == ("offload",)
    assert [call[0] for call in calls[:-1]] == ["adapter", "encoder", "pipeline", "ip"]
    for call in calls[:-1]:
        assert call[2]["local_files_only"] is True
    for call in calls[:3]:
        assert call[2]["torch_dtype"] == "mock-fp16"
    assert calls[3][2]["image_encoder_folder"] is None


@pytest.mark.parametrize("mode", ["callback_on_step_end", "legacy_callback"])
def test_both_callback_modes_share_cancellation_and_leave_latents_unchanged(mode):
    pipe = FakePipeline()
    latents = torch.tensor([12.0])
    callbacks = generation.pipeline_callback_kwargs(
        pipe, OnePassGenerationSettings(), lambda: True, mode
    )
    with pytest.raises(generation.OnePassCancelled):
        if mode == "legacy_callback":
            assert callbacks["callback_steps"] == 1
            callbacks["callback"](10, None, latents)
        else:
            callbacks["callback_on_step_end"](pipe, 10, None, {"latents": latents})
    assert pipe.scale_sets == []
    assert latents.item() == 12.0


def test_legacy_callback_preserves_latents_and_transitions_after_step_ten():
    pipe = FakeLegacyPipeline()
    cb = generation.pipeline_callback_kwargs(
        pipe, OnePassGenerationSettings(), lambda: False, "legacy_callback"
    )["callback"]
    latents = torch.tensor([12.0])
    cb(9, None, latents)
    assert pipe.scale_sets == []
    assert cb(10, None, latents) is None
    assert pipe.scale_sets == [0.9]
    cb(11, None, latents)
    assert pipe.scale_sets == [0.9]
    assert latents.item() == 12.0


def test_modern_callback_is_preferred_if_both_signatures_exist():
    class Both:
        def __call__(
            self, callback_on_step_end=None, callback=None, callback_steps=1,
            ip_adapter_image=None, adapter_conditioning_factor=1.0,
            adapter_conditioning_scale=1.0, prompt_embeds=None,
            negative_prompt_embeds=None, pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
        ):
            pass
    assert generation.check_pipeline_support(Both) == "callback_on_step_end"


@pytest.mark.parametrize('pipeline_type',[FakePipeline,FakeLegacyPipeline])
def test_without_pose_request_observes_zero_adapter_and_n0_ip(inputs,settings,tmp_path,pipeline_type):
    from genai_lab.onepass_pose import InputDecision
    decision=InputDecision('reject',('K2',),(),('proceed_without_pose','choose_other_image'))
    inputs=replace(inputs,control_file=None,control_sha256=None,pose_mode='without_pose',
                   input_decision=decision,user_choice='proceed_without_pose')
    value=backend(settings,pipeline_type)
    batch=generation.generate_onepass_request(generation.OnePassRequest(inputs,(1,2,3,4),settings),
                    tmp_path/'n0',backend_factory=lambda _:value)
    assert value.pipe.history==[[0.0]*11+[.9]*17]*4
    for candidate in batch.candidates:
        record=candidate.record
        assert record['adapter_applied_calls']==[] and record['valid']
        assert record['pose_mode']=='without_pose' and record['adapter_factor']==0
        assert record['inputs']['control_file'] is None and record['inputs']['control_sha256'] is None
        assert record['input_decision']['rejects']==('K2',)
        assert record['user_choice']=='proceed_without_pose'
        assert record['adapter_input']=='black_placeholder_no_pose'
    assert value.pipe.images[0][0].getextrema()==((0,0),(0,0),(0,0))


@pytest.mark.parametrize('changes',[
    {'control_file':None},{'control_sha256':None},
    {'pose_mode':'without_pose'},{'pose_mode':'invalid'},
    {'pose_mode':'without_pose','control_file':None,'control_sha256':None,'ip_early':.5},
])
def test_pose_mode_contract_rejects_invalid_inputs(inputs,changes):
    with pytest.raises(ValueError):
        replace(inputs,**changes)


def test_n0i_diagnostic_override_is_explicit_and_recorded(inputs,settings,tmp_path):
    inputs=replace(inputs,control_file=None,control_sha256=None,pose_mode='without_pose',
                   ip_early=.5,diagnostic_ip_override=True)
    result=generation.generate_onepass_image(backend(settings),inputs,10,tmp_path/'n0i',settings=settings)
    assert result.record['diagnostic_ip_override']
    assert [c['ip_scales'] for c in result.record['unet_calls']]==[[.5]]*11+[[.9]]*17
    assert result.record['adapter_applied_calls']==[]


def test_without_pose_unexpected_residuals_fail_validation(settings):
    calls=[{'ip_scales':[0.0 if i<11 else .9],'adapter_applied':i==0} for i in range(28)]
    assert generation.schedule_errors(calls,settings,0.0,'without_pose')


def test_without_pose_builder_uses_existing_assembler_without_pose_tags(inputs,tmp_path):
    from PySide6.QtCore import QSettings
    from genai_lab.character_preferences import save_character_gender
    from genai_lab.onepass_gender import prepare_onepass_gender
    from genai_lab.onepass_pose import InputDecision
    from genai_lab.onepass_prompt import CharacterTagGroups,assemble_onepass_prompt
    from genai_lab.onepass_prompt_settings import OnePassPromptSettings
    class Tokenizer:
        bos_token_id=1000; eos_token_id=1001; pad_token_id=1001
        def __call__(self,text,**kwargs):
            return {'input_ids':[sum(map(ord,w)) for w in text.split()]}
    tokens=(Tokenizer(),Tokenizer())
    store=QSettings(str(tmp_path/'temporary.ini'),QSettings.Format.IniFormat)
    key=tmp_path/'key-only.png'
    save_character_gender(key,'male',store)
    cfg=OnePassPromptSettings()
    gender=prepare_onepass_gender(key,('blue hair',),cfg.negative_template,settings=store)
    groups=CharacterTagGroups(appearance=('blue hair',))
    decision=InputDecision('warn_user_choice',(),('GUESS',),
                           ('proceed_with_pose','proceed_without_pose','choose_other_image'))
    kwargs=dict(decision=decision,gender=gender,groups=groups,garment_tags=('shorts',),
                pose_tags=('kneeling','from above'),slim=False,tokenizers=tokens,
                face_file=inputs.face_file,face_sha256=inputs.face_sha256,
                control_file=inputs.control_file,control_sha256=inputs.control_sha256,ip_early=.5)
    result=generation.prepare_onepass_inputs(choice='proceed_without_pose',**kwargs)
    expected=assemble_onepass_prompt(gender,groups,('shorts',),(),slim=False,tokenizers=tokens)
    assert result.prompt==expected and result.ip_early==0
    assert result.control_file is None and result.pose_mode=='without_pose'
    with_pose=generation.prepare_onepass_inputs(choice='proceed_with_pose',**kwargs)
    assert with_pose.prompt==assemble_onepass_prompt(gender,groups,('shorts',),kwargs['pose_tags'],
                                                    slim=False,tokenizers=tokens)
    assert with_pose.ip_early==.5 and with_pose.control_file==inputs.control_file
    with pytest.raises(ValueError):
        generation.prepare_onepass_inputs(choice='choose_other_image',**kwargs)
    with pytest.raises(TypeError):
        generation.prepare_onepass_inputs(**kwargs)  # No default choice.


def test_pose_mode_has_no_implicit_default(inputs):
    with pytest.raises(TypeError,match='pose_mode'):
        generation.OnePassInputs(inputs.prompt,inputs.control_file,inputs.face_file,
                                 inputs.control_sha256,inputs.face_sha256)


def test_diagnostic_ip_cannot_override_product_choice(inputs):
    from genai_lab.onepass_pose import InputDecision
    decision=InputDecision('pass',(),(),('proceed_with_pose','proceed_without_pose','choose_other_image'))
    with pytest.raises(ValueError,match='제품 사용자 선택'):
        replace(inputs,pose_mode='without_pose',control_file=None,control_sha256=None,
                input_decision=decision,user_choice='proceed_without_pose',ip_early=.5,diagnostic_ip_override=True)


def test_rejected_input_cannot_select_pose_even_with_inconsistent_choices(inputs):
    from genai_lab.onepass_pose import InputDecision
    decision=InputDecision('reject',('K2',),(),('proceed_with_pose','proceed_without_pose','choose_other_image'))
    with pytest.raises(ValueError,match='사용자 선택'):
        replace(inputs,input_decision=decision,user_choice='proceed_with_pose')


def test_without_pose_does_not_allow_missing_face_reference(inputs):
    with pytest.raises(ValueError,match='얼굴 참조'):
        replace(inputs,pose_mode='without_pose',control_file=None,control_sha256=None,face_file=None)
