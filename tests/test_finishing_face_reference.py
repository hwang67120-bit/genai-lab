"""CPU 대체 구현으로 모델 일치·입력 보존·호출별 얼굴 참조 전달을 검증한다."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from PIL import Image
from genai_lab import finishing_backend as backend_module
from genai_lab.finishing_backend import FinishingBackend
from genai_lab.finishing_reference import checked_reference, observe_face_reference
from genai_lab.studio_generation import StudioRuntime
from genai_lab import studio_finishing as finishing
from genai_lab.onepass_generation import OnePassGenerationError
from genai_lab.proportion_inputs import sha, json_value
from genai_lab.qwen_record_io import write_json
from test_studio_finishing import batch_at, Backend
from test_onepass_generation import prompt
from scripts import verify_finishing_face_reference as runner


def reference_batch(tmp_path):
    batch=batch_at(tmp_path/"generation")
    runtime=StudioRuntime(finishing=True,finishing_face_reference=True)
    face=tmp_path/"face.png"
    Image.new("RGB",(64,64),(21,44,72)).save(face)
    for candidate in batch.candidates:
        record=json.loads(candidate.record_path.read_text(encoding="utf-8"))
        record.update(inputs={"face_file":str(face),"face_sha256":sha(face)},
                      settings=json_value(asdict(runtime.generation)),models=runtime.generation.model_record())
        write_json(candidate.record_path,record)
    return batch,runtime,face


def test_flag_default_false_and_explicit_runtime_json(tmp_path,monkeypatch):
    assert StudioRuntime().finishing_face_reference is False
    config=tmp_path/"runtime.json"
    write_json(config,{"finishing_face_reference":True})
    monkeypatch.setenv("GENAI_STUDIO_RUNTIME",str(config))
    assert StudioRuntime.from_environment().finishing_face_reference is True
    with pytest.raises(ValueError):StudioRuntime(finishing_face_reference="true")


@pytest.mark.parametrize("target",["face","revision","weight","scale"])
def test_invalid_reference_is_rejected_before_backend_load(tmp_path,target):
    batch,runtime,face=reference_batch(tmp_path)
    candidate=batch.candidates[0]
    record=json.loads(candidate.record_path.read_text(encoding="utf-8"))
    if target=="face":face.write_bytes(b"changed")
    elif target=="scale":record["settings"]["ip_scale"]=.5
    else:record["models"]["ip_adapter"][target]="changed"
    write_json(candidate.record_path,record)
    with pytest.raises(ValueError):
        finishing.finish_batch(batch,prompt(),runtime,backend_factory=lambda *a,**k:pytest.fail("load forbidden"),detector=lambda *_:[])
    assert json.loads((batch.directory/"finishing-status.json").read_text(encoding="utf-8"))["status"]=="failed"


def test_enabled_batch_records_both_stages_and_preserves_raw(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    original=batch.candidates[0].path.read_bytes()
    class FaceBackend(Backend):
        def set_face_reference(self,reference):self.reference=reference
        def refine(self,image,seed,cancelled):
            with self.reference.open_image() as ref:assert ref.getpixel((0,0))==(21,44,72)
            result,metrics=super().refine(image,seed,cancelled)
            return result,{**metrics,"ip_applied_calls":9,"ip_scale":.9}
    backend=FaceBackend(None)
    def factory(settings,*,face_reference):assert face_reference is True;return backend
    finishing.finish_batch(batch,prompt(),runtime,backend_factory=factory,
        detector=lambda *_:[([40,20,140,160],"head",.9)])
    record=finishing.read_finishing(batch.candidates[0])
    assert record["face_reference"]=="on" and record["face_sha256"]==sha(face)
    assert record["ip_scale"]==.9 and record["contract"]["ip_adapter"]=="loaded"
    assert all(record["stages"][stage]["ip_applied_calls"]==9 for stage in ("hires","face"))
    assert batch.candidates[0].path.read_bytes()==original and backend.closed


@pytest.mark.parametrize("enabled",[False,True])
def test_backend_loads_exact_local_ip_only_when_enabled(monkeypatch,enabled):
    seen={}
    class Pipe:
        @staticmethod
        def from_pretrained(path,**kwargs):seen["pipeline"]=(path,kwargs);return Pipe()
        def load_ip_adapter(self,path,**kwargs):seen["ip"]=(path,kwargs)
        def set_ip_adapter_scale(self,value):seen["scale"]=value
        def enable_model_cpu_offload(self):seen["offload"]=getattr(self,"model_cpu_offload_seq",None)
        def enable_vae_tiling(self):seen["tiling"]=True
    class Encoder:
        @staticmethod
        def from_pretrained(path,**kwargs):seen["encoder"]=(path,kwargs);return "encoder"
    monkeypatch.setattr(backend_module,"validate_local_models",lambda _:None)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(float16="fp16",cuda=SimpleNamespace(is_available=lambda:True)))
    monkeypatch.setitem(sys.modules,"diffusers",SimpleNamespace(StableDiffusionXLImg2ImgPipeline=Pipe,EulerAncestralDiscreteScheduler=object))
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(CLIPVisionModelWithProjection=Encoder))
    monkeypatch.setattr(backend_module,"configure_offload",lambda pipe,torch,record:seen.update(group_offload=True))
    settings=StudioRuntime().generation
    backend=FinishingBackend(settings,face_reference=enabled)
    assert seen["pipeline"][1]["local_files_only"] and seen["tiling"]
    if enabled:
        assert seen["encoder"]==(str(settings.ip_root/settings.image_encoder_subfolder),{"torch_dtype":"fp16","local_files_only":True})
        assert seen["ip"]==(str(settings.ip_root),{"subfolder":settings.ip_subfolder,"weight_name":settings.ip_weight_name,
                                              "image_encoder_folder":None,"local_files_only":True})
        assert seen["scale"]==.9 and seen["group_offload"] and "offload" not in seen
    else:
        assert "ip" not in seen and "encoder" not in seen and "image_encoder" not in seen["pipeline"][1]
        assert "offload" in seen and "group_offload" not in seen


def fake_backend(reference,*,wrong_scale=False):
    calls={"images":[],"removed":0}
    class Hook:
        def remove(self):calls["removed"]+=1
    import torch
    from genai_lab.finishing_memory import FaceEmbeddingCache, embedding_record
    class UNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_processors={"ip":SimpleNamespace(scale=[.9])}
        def forward(self, **kwargs): return None
        def register_forward_hook(self,*args,**kwargs):
            actual=super().register_forward_hook(*args,**kwargs)
            def remove():actual.remove();calls["removed"]+=1
            return SimpleNamespace(remove=remove)
        def register_forward_pre_hook(self,*args,**kwargs):
            actual=super().register_forward_pre_hook(*args,**kwargs)
            def remove():actual.remove();calls["removed"]+=1
            return SimpleNamespace(remove=remove)
    class Pipe:
        unet=UNet()
        def set_ip_adapter_scale(self,scale):assert scale==.9;self.unet.attn_processors["ip"].scale=[.5 if wrong_scale else scale]
        def __call__(self,*,callback_on_step_end,**kwargs):
            assert "ip_adapter_image" not in kwargs
            embeddings=kwargs["ip_adapter_image_embeds"]
            calls["images"].append(embeddings[0].numpy().tobytes())
            assert kwargs["strength"]==.35 and kwargs["num_inference_steps"]==28 and kwargs["guidance_scale"]==5.5
            for index in range(9):
                self.unet(added_cond_kwargs={"image_embeds":embeddings})
                callback_on_step_end(self,index,None,{})
            return SimpleNamespace(images=[kwargs["image"].copy()])
    class Generator:
        def __init__(self,device):assert device=="cuda"
        def manual_seed(self,seed):calls["seed"]=seed;return self
    backend=object.__new__(FinishingBackend)
    backend.pipe=Pipe();backend.embeds={};backend.face_reference_enabled=True;backend.face_reference=reference
    tensors=(torch.zeros((2,1,2,3),dtype=torch.float16),)
    backend.face_cache=FaceEmbeddingCache(tensors,{"face_sha256":reference.sha256,"embeddings":embedding_record(tensors),
        "attention":{"mode":"sdpa_efficient_only"}},SimpleNamespace(remove=lambda:None))
    backend.scheduler_type=SimpleNamespace(from_config=lambda config:SimpleNamespace(config=config))
    backend.torch=SimpleNamespace(Generator=Generator,cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None,
        synchronize=lambda:None,max_memory_reserved=lambda:123,max_memory_allocated=lambda:100))
    return backend,calls


def test_both_refinement_sizes_receive_identical_face_and_nine_scales(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    backend,calls=fake_backend(checked_reference(batch,runtime.generation))
    for size in ((1104,1848),(1024,1024)):
        with Image.new("RGB",size) as image:
            result,metrics=backend.refine(image,37,lambda:False);result.close()
        assert metrics["ip_applied_calls"]==9 and metrics["unet_calls"]==9
        assert all(c["ip_scales"]==[.9] for c in metrics["ip_observations"])
    assert calls["images"][0]==calls["images"][1] and calls["seed"]==37 and calls["removed"]==8


def test_changed_face_after_preflight_never_reaches_pipeline(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    backend,calls=fake_backend(checked_reference(batch,runtime.generation))
    face.write_bytes(b"changed")
    with Image.new("RGB",(16,16)) as image, pytest.raises(ValueError,match="SHA"):
        backend.refine(image,1,lambda:False)
    assert calls["images"]==[] and calls["removed"]==1


def test_wrong_actual_scale_stops_and_records_observation(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    backend,calls=fake_backend(checked_reference(batch,runtime.generation),wrong_scale=True)
    with Image.new("RGB",(16,16)) as image,pytest.raises(ValueError,match="강도"):
        backend.refine(image,1,lambda:False)
    assert calls["removed"]==4 and backend.last_refine["unet_calls"]==0


def test_missing_image_embeddings_is_rejected():
    module=SimpleNamespace(attn_processors={"ip":SimpleNamespace(scale=[.9])})
    with pytest.raises(ValueError,match="임베딩"):observe_face_reference(module,(),{},[])


def test_runner_is_cpu_by_default_and_restores_exact_token_ids(monkeypatch,capsys):
    monkeypatch.setattr(runner,"preflight",lambda *args:{"cases":[{"seeds":[1,2,3,4]}]*4})
    monkeypatch.setattr(runner,"run_replay",lambda *args:pytest.fail("GPU forbidden"))
    assert runner.main([])==0 and json.loads(capsys.readouterr().out)["gpu_executed"] is False
    saved=json_value(asdict(prompt()))
    assert json_value(asdict(runner.restore_prompt(saved)))==saved


def test_copied_face_is_allowed_only_with_original_sha(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    copied=tmp_path/"copied.png";copied.write_bytes(face.read_bytes())
    ref=checked_reference(batch,runtime.generation,face_file=copied)
    assert ref.path==copied and ref.sha256==sha(face)
    copied.write_bytes(b"wrong")
    with pytest.raises(ValueError):checked_reference(batch,runtime.generation,face_file=copied)



def test_copy_keeps_raw_request_and_face_bytes_and_detects_source_change(tmp_path):
    source=tmp_path/"source"
    batch,runtime,face=reference_batch(source)
    candidate=batch.candidates[0]
    names=[face.relative_to(source),candidate.path.relative_to(source),candidate.record_path.relative_to(source)]
    case={"name":"run-test", "seeds":[1],"files":{p.as_posix():sha(source/p) for p in names}}
    copied,prompt_copy=runner.copy_case(source,tmp_path/"copied",case)
    for name,digest in case["files"].items():
        assert sha(tmp_path/"copied"/name)==digest and sha(source/name)==digest
    assert prompt_copy==prompt()
    assert checked_reference(copied,runtime.generation,face_file=tmp_path/"copied/face.png").sha256==sha(face)
    with pytest.raises(FileExistsError):runner.copy_case(source,tmp_path/"copied",case)
    face.write_bytes(b"changed")
    with pytest.raises(ValueError,match="바뀌었습니다"):
        runner.copy_case(source,tmp_path/"new-copy",case)


def test_ip_memory_over_limit_stops_on_first_call_and_retains_metrics(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    backend,calls=fake_backend(checked_reference(batch,runtime.generation))
    backend.torch.cuda.max_memory_reserved=lambda:7*2**30 if calls["images"] else 0
    with Image.new("RGB",(16,16)) as image,pytest.raises(OnePassGenerationError,match="6.5"):
        backend.refine(image,1,lambda:False)
    assert backend.last_refine["unet_calls"]==1 and backend.last_refine["ip_applied_calls"]==1
    assert backend.last_refine["max_reserved_bytes"]==7*2**30 and calls["removed"]==4



def test_replay_runs_only_finishing_and_records_original_quality_setting(tmp_path,monkeypatch):
    source=tmp_path/"source";directory=source/"run-aaaaaaaa"
    batch,runtime,face=reference_batch(directory)
    inputs=directory/"inputs";inputs.mkdir()
    (inputs/"face.png").write_bytes(face.read_bytes())
    candidate=batch.candidates[0]
    record=json.loads(candidate.record_path.read_text(encoding="utf-8"))
    record["prompt"]["rules"]["quality_format"]="diagnostic_2b"
    write_json(candidate.record_path,record)
    names=["inputs/face.png","generation/seed-1/raw.png","generation/seed-1/run.json"]
    case={"name":"run-aaaaaaaa","seeds":[1],"files":{name:sha(directory/name) for name in names}}
    write_json(source/"summary.json",[])
    report={"source":str(source),"summary_sha256":sha(source/"summary.json"),"cases":[case]}
    events=[]
    def finish(copied,prompt,active,**kwargs):
        assert active.quality_tags is True and active.finishing_face_reference is True
        assert kwargs["face_file"]==tmp_path/"out/run-aaaaaaaa/inputs/face.png"
        assert copied.candidates[0].path.read_bytes()==candidate.path.read_bytes()
        events.append("finish")
    monkeypatch.setattr(runner,"finish_batch",finish)
    monkeypatch.setattr(runner,"prepare_backgrounds",lambda *a,**k:events.append("background"))
    monkeypatch.setattr(runner,"read_finishing",lambda _: {"status":"completed"})
    monkeypatch.setattr(runner,"read_background",lambda _: {"status":"completed"})
    import genai_lab.onepass_generation as generation
    monkeypatch.setattr(generation,"generate_onepass_request",lambda *a,**k:pytest.fail("No new generation"))
    state=runner.run_replay(report,tmp_path/"out",runtime)
    assert events==["finish","background"] and state["status"]=="completed"
    assert state["new_generation"] is False
    assert all(sha(directory/name)==digest for name,digest in case["files"].items())
