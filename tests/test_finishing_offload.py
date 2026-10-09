"""GPU·다운로드 없이 설치된 Diffusers 블록 훅과 Accelerate 수명을 검사한다."""
from types import SimpleNamespace
import json
import pytest
import torch
from diffusers import UNet2DConditionModel, DiffusionPipeline
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0, AttnProcessor2_0
from diffusers.models.embeddings import ImageProjection, MultiIPAdapterImageProjection
from genai_lab.finishing_offload import configure_offload, group_inventory
from genai_lab import finishing_backend as backend_module
from genai_lab.finishing_backend import FinishingBackend
from genai_lab import studio_finishing as finishing
from test_finishing_face_reference import reference_batch, fake_backend
from genai_lab.finishing_reference import checked_reference
from test_onepass_generation import prompt


@pytest.fixture
def small_pipe():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    unet=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,
        down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","CrossAttnUpBlock2D"),block_out_channels=(32,64),
        layers_per_block=1,cross_attention_dim=8,attention_head_dim=4,norm_num_groups=8)
    processors={}
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name]=AttnProcessor2_0();continue
        width=64 if name.startswith("mid_block") else (32,64)[int(name.split(".")[1])] if name.startswith("down_blocks") else (64,32)[int(name.split(".")[1])]
        processors[name]=IPAdapterAttnProcessor2_0(hidden_size=width,cross_attention_dim=8,scale=.9)
    unet.set_attn_processor(processors)
    unet.encoder_hid_proj=MultiIPAdapterImageProjection([ImageProjection(image_embed_dim=8,cross_attention_dim=8,num_image_text_embeds=4)])
    unet.register_to_config(encoder_hid_dim_type="ip_image_proj")
    models={name:torch.nn.Linear(2,2) for name in ("text_encoder","text_encoder_2","image_encoder","vae")}
    pipe=SimpleNamespace(unet=unet,**models)
    pipe.components={"unet":unet,**models}
    try:yield pipe
    finally:torch.set_num_threads(threads)


def forward(unet):
    return unet(torch.ones(2,4,8,8),torch.tensor(1),encoder_hidden_states=torch.ones(2,3,8),
                added_cond_kwargs={"image_embeds":[torch.ones(2,1,8)]}).sample


def test_real_block_hooks_include_ip_weights_and_preserve_cpu_outputs(small_pipe,monkeypatch):
    with torch.inference_mode():expected=forward(small_pipe.unet)
    before={name:value.clone() for name,value in small_pipe.unet.state_dict().items()}
    record={};owner=configure_offload(small_pipe,torch,record,device="cpu")
    assert record["num_blocks_per_group"]==1 and not record["use_stream"]
    assert record["ip_resident_bytes"]==0 and record["ip_parameter_bytes"]>0
    assert any(row["group"].startswith("down_blocks.") for row in record["ip_membership"])
    assert any(row["group"]=="unet" for row in record["ip_membership"] if row["parameter"].startswith("encoder_hid_proj"))
    assert not any(hasattr(module,"_hf_hook") for module in small_pipe.unet.modules())
    assert all(hasattr(getattr(small_pipe,name),"_hf_hook") for name in ("vae","text_encoder","text_encoder_2","image_encoder"))
    monkeypatch.setattr(small_pipe,"enable_model_cpu_offload",lambda **_:pytest.fail("Cannot reinstall whole UNet offload"),raising=False)
    try:
        for _ in range(2):
            owner.begin_stage()
            with torch.inference_mode():actual=forward(small_pipe.unet)
            assert torch.equal(actual,expected)
            assert owner.observations and all(row["calls"]==1 and row["device"]=="cpu" for row in owner.observations.values())
            DiffusionPipeline.maybe_free_model_hooks(small_pipe)
            owner.release()
        assert all(torch.equal(before[name],value) for name,value in small_pipe.unet.state_dict().items())
        json.dumps(record)
    finally:owner.close()
    assert not small_pipe.unet._forward_pre_hooks
    assert all(not hasattr(getattr(small_pipe,name),"_hf_hook") for name in ("vae","text_encoder","text_encoder_2","image_encoder"))


def test_vae_released_before_first_block_and_all_groups_released_on_error(small_pipe,monkeypatch):
    owner=configure_offload(small_pipe,torch,{},device="cpu");events=[]
    real_release=owner.release_models
    def release_models():
        events.append("release_models")
        real_release()
    monkeypatch.setattr(owner,"release_models",release_models)
    first=small_pipe.unet.down_blocks[0]
    def fail(*_):events.append("block");raise RuntimeError("block failed")
    hook=first.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError,match="block failed"),torch.inference_mode():forward(small_pipe.unet)
        assert events==["release_models","block"]
        for group,_,name in owner.groups:
            real=group.offload_
            monkeypatch.setattr(group,"offload_",lambda real=real,name=name:(events.append(name),real()))
        owner.release()
        assert len(events)==3+len(owner.groups)
    finally:hook.remove();owner.close()


def test_failed_setup_records_error_without_retry(small_pipe,monkeypatch):
    calls=[];record={}
    def fail(**kwargs):calls.append(kwargs);raise RuntimeError("offload failed")
    monkeypatch.setattr(small_pipe.unet,"enable_group_offload",fail)
    with pytest.raises(RuntimeError,match="offload failed"):configure_offload(small_pipe,torch,record,device="cpu")
    assert len(calls)==1 and calls[0]["offload_type"]=="block_level"
    assert record["status"]=="failed" and record["error"]=="offload failed" and not record["retry"]


def test_ungrouped_ip_weight_is_rejected(small_pipe):
    owner=configure_offload(small_pipe,torch,{},device="cpu")
    small_pipe.unet.encoder_hid_proj.extra=torch.nn.Linear(2,2)
    # 지원하지 않는 라이브러리 구조를 재현하려고 해당 투영층을 소유 그룹 목록에서 제거한다.
    root=next(group for group,_,name in owner.groups if name=="unet")
    projection_ids={id(p) for p in small_pipe.unet.encoder_hid_proj.parameters()}
    root.modules=[m for m in root.modules if m is not small_pipe.unet.encoder_hid_proj]
    root.parameters=[p for p in root.parameters if id(p) not in projection_ids]
    try:
        with pytest.raises(ValueError,match="소속"):group_inventory(small_pipe.unet)
    finally:owner.close()


def test_batch_retains_offload_setup_failure_and_original_raw(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    raw=batch.candidates[0].path.read_bytes()
    def fail(*a,**kw):
        error=RuntimeError("offload setup failed")
        error.finishing_offload={"status":"failed","unet":"block_level","retry":False}
        raise error
    with pytest.raises(RuntimeError,match="offload setup failed"):
        finishing.finish_batch(batch,prompt(),runtime,backend_factory=fail,detector=lambda *_:[])
    record=json.loads((batch.directory/"finishing-status.json").read_text())
    assert record["status"]=="failed" and record["offload"]["status"]=="failed"
    assert batch.candidates[0].path.read_bytes()==raw and record["completed_seeds"]==[]


@pytest.mark.parametrize("fail",[False,True])
def test_refine_records_offload_and_releases_on_success_and_failure(tmp_path,fail):
    from PIL import Image
    batch,runtime,_=reference_batch(tmp_path)
    backend,calls=fake_backend(checked_reference(batch,runtime.generation),wrong_scale=fail)
    events=[]
    backend.offload_record={"unet":"block_level","use_stream":False}
    backend.offload=SimpleNamespace(begin_stage=lambda:events.append("begin"),release=lambda:events.append("release"),
                                    verify_stage=lambda calls:None, observations={"ip":{"calls":1,"device":"cuda:0"}})
    with Image.new("RGB",(16,16)) as source:
        if fail:
            with pytest.raises(ValueError):backend.refine(source,1,lambda:False)
        else:
            result,record=backend.refine(source,1,lambda:False);result.close()
            assert record["offload"]==backend.offload_record and record["ip_device_observations"]
    assert events==["begin","release"]
    assert backend.last_refine["offload"]==backend.offload_record
