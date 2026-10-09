"""CPU-only cache equivalence, encoder release, SDPA scope and restoration tests."""
from types import SimpleNamespace
import pytest
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from genai_lab import finishing_backend as backend_module
from genai_lab.finishing_backend import FinishingBackend
from genai_lab import finishing_memory as memory
from genai_lab.finishing_reference import checked_reference
from test_finishing_face_reference import reference_batch, fake_backend


def cpu_pipeline(events, fail=False):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1,dtype=torch.float16))
        def forward(self, image, output_hidden_states):
            events.append("encode")
            if fail:raise RuntimeError("encoder failed")
            values=image.mean().expand(1,2,3)
            return SimpleNamespace(hidden_states=(values,values,values))
        def to(self,device):events.append("encoder_"+str(device));return super().to(device)
    class IPAdapterAttnProcessor2_0:pass
    class Pipe:
        prepare_ip_adapter_image_embeds=StableDiffusionXLImg2ImgPipeline.prepare_ip_adapter_image_embeds
        encode_image=StableDiffusionXLImg2ImgPipeline.encode_image
        image_encoder=Encoder()
        feature_extractor=staticmethod(lambda image,return_tensors:SimpleNamespace(pixel_values=torch.ones((1,3,4,4))))
        unet=SimpleNamespace(encoder_hid_proj=SimpleNamespace(image_projection_layers=[object()]),
                            attn_processors={"ip":IPAdapterAttnProcessor2_0()})
    return Pipe()


def cpu_torch(events):
    return SimpleNamespace(inference_mode=torch.inference_mode,cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,max_memory_reserved=lambda:123,max_memory_allocated=lambda:100,
        empty_cache=lambda:events.append("empty_cache"),memory_reserved=lambda:3,memory_allocated=lambda:1))


def test_cached_cfg_bytes_match_installed_pipeline_and_encoder_is_released(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    reference=checked_reference(batch,runtime.generation)
    events=[];pipe=cpu_pipeline(events)
    with reference.open_image() as image,torch.inference_mode():
        original=pipe.prepare_ip_adapter_image_embeds(image,None,"cpu",1,True)
    events.clear();record={}
    cache=memory.prepare_face_cache(pipe,reference,cpu_torch(events),lambda:None,record,device="cpu")
    assert record["embeddings"]==memory.embedding_record(original)
    assert events==["encode","encode","encoder_cpu","empty_cache"]
    assert record["encoder_forward_calls"]==2 and record["preparation_calls"]==1
    assert record["cfg_order"]==["negative","positive"]
    assert record["max_reserved_bytes"]==123 and record["reserved_after_offload_bytes"]==3
    assert cache.tensors[0][0].count_nonzero()==0 and cache.tensors[0][1].min()==1
    for _ in range(2):
        delivered=pipe.prepare_ip_adapter_image_embeds(None,list(cache.tensors),"cpu",1,True)
        cache.verify_delivered(delivered)
    assert events.count("encode")==2
    with pytest.raises(RuntimeError,match="다시 호출"):
        pipe.image_encoder(torch.ones((1,3,4,4)),output_hidden_states=True)
    cache.close()
    assert not pipe.image_encoder._forward_pre_hooks and cache.tensors==()


def test_backend_prepares_cache_only_once(tmp_path,monkeypatch):
    batch,runtime,face=reference_batch(tmp_path)
    events=[];backend=object.__new__(FinishingBackend)
    backend.pipe=cpu_pipeline(events);backend.torch=cpu_torch(events)
    backend.face_reference=checked_reference(batch,runtime.generation);backend.face_cache=None
    real=memory.prepare_face_cache
    monkeypatch.setattr(backend_module,"prepare_face_cache",lambda *a:real(*a,device="cpu"))
    backend.prepare_face_embeddings(lambda:False)
    first=backend.face_cache
    backend.prepare_face_embeddings(lambda:False)
    assert backend.face_cache is first and events.count("encode")==2
    first.close()


def test_cache_or_delivered_tensor_changes_are_rejected(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    ref=checked_reference(batch,runtime.generation)
    tensors=(torch.zeros((2,1,2,3),dtype=torch.float16),)
    cache=memory.FaceEmbeddingCache(tensors,{"face_sha256":ref.sha256,"embeddings":memory.embedding_record(tensors)},None)
    with pytest.raises(ValueError,match="전달된"):
        cache.verify_delivered([torch.ones_like(tensors[0])])
    tensors[0][0,0,0,0]=1
    with pytest.raises(ValueError,match="변경"):
        cache.verify(ref)


def test_encoder_error_still_offloads_and_records_failure(tmp_path):
    batch,runtime,face=reference_batch(tmp_path)
    events=[];pipe=cpu_pipeline(events,fail=True);record={}
    with pytest.raises(RuntimeError,match="encoder failed"):
        memory.prepare_face_cache(pipe,checked_reference(batch,runtime.generation),cpu_torch(events),lambda:None,record,device="cpu")
    assert record["status"]=="failed" and events[-2:]==["encoder_cpu","empty_cache"]
    assert not pipe.image_encoder._forward_hooks and not pipe.image_encoder._forward_pre_hooks


def sdpa_flags():
    return tuple(f() for f in (torch.backends.cuda.flash_sdp_enabled,torch.backends.cuda.mem_efficient_sdp_enabled,
                              torch.backends.cuda.math_sdp_enabled,torch.backends.cuda.cudnn_sdp_enabled))


@pytest.mark.parametrize("fail",[False,True])
def test_sdpa_scope_only_covers_unet_and_restores_flags_even_after_error(fail):
    original=sdpa_flags()
    class UNet(torch.nn.Module):
        def forward(self):
            assert sdpa_flags()==(False,True,False,False)
            if fail:raise RuntimeError("forward error")
    module=UNet()
    with memory.unet_attention(module):
        assert sdpa_flags()==original  # VAE before denoising retains its policy.
        if fail:
            with pytest.raises(RuntimeError):module()
        else:module()
        assert sdpa_flags()==original  # VAE after denoising retains its policy.
    assert sdpa_flags()==original and not module._forward_hooks and not module._forward_pre_hooks


def test_unknown_attention_processor_is_not_silently_replaced():
    pipe=SimpleNamespace(unet=SimpleNamespace(attn_processors={"unknown":object()}))
    with pytest.raises(ValueError,match="processor"):memory.attention_policy(pipe)


def test_off_prepare_never_builds_face_cache(monkeypatch):
    class Encoder:
        def to(self,device):return self
    backend=object.__new__(FinishingBackend)
    backend.face_reference_enabled=False;backend.torch=cpu_torch([]);backend.torch.float16=torch.float16
    backend.pipe=SimpleNamespace(text_encoder=Encoder(),text_encoder_2=Encoder())
    monkeypatch.setattr(backend_module,"encode_prompt_plan",lambda *a:{"test":"embeddings"})
    monkeypatch.setattr(backend,"prepare_face_embeddings",lambda *a:pytest.fail("OFF must not precompute"))
    backend.prepare(None,lambda:False)
    assert backend.embeds=={"test":"embeddings"}


def test_refine_records_same_cached_tensor_for_both_sizes(tmp_path):
    from PIL import Image
    batch,runtime,face=reference_batch(tmp_path)
    backend,calls=fake_backend(checked_reference(batch,runtime.generation))
    hashes=[]
    for size in ((1104,1848),(1024,1024)):
        with Image.new("RGB",size) as source:
            result,record=backend.refine(source,1,lambda:False);result.close()
            hashes.append(record["ip_embeddings"])
    assert hashes[0]==hashes[1]==backend.face_cache.record["embeddings"]
