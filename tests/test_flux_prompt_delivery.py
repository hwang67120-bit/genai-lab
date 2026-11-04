"""FLUX 긴 지시의 텍스트 인코더 → 생성 모델 연결을 CPU 대역으로 검증한다."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler

from genai_lab.flux_prompt_delivery import build_flux_condition, plan_flux_prompt, validate_text_condition


class WordTokenizer:
    """채팅 경계 토큰을 포함하며, 인코더 호출 방식과 동일한 텐서를 제공한다."""
    def __init__(self):
        self.vocabulary = {}
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "<user> " + messages[0]["content"] + " <assistant>"
    def __call__(self, text, truncation=False, padding=None, max_length=None, return_tensors=None):
        words = text.split()
        ids = [self.vocabulary.setdefault(word, len(self.vocabulary)+1) for word in words]
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            missing = max_length-len(ids)
            ids += [0] * missing
            mask += [0] * missing
        result = dict(input_ids=ids, attention_mask=mask)
        return {key: torch.tensor([value]) for key,value in result.items()} if return_tensors else result
    def decode(self, ids):
        table = {value:key for key,value in self.vocabulary.items()}
        return " ".join(table.get(value, "<pad>") for value in ids)


class RecordingEncoder:
    dtype = torch.float32
    device = torch.device("cpu")
    def __init__(self):
        self.calls = []
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        # 각 원본 토큰이 끝까지 보존됐는지 추적할 수 있는 작은 가짜 특징.
        ids = kwargs["input_ids"].float().unsqueeze(-1).repeat(1,1,2)
        return SimpleNamespace(hidden_states=tuple(ids+index for index in range(28)))


class TransformerReached(Exception):
    pass


class RecordingTransformer:
    dtype = torch.float32
    config = SimpleNamespace(joint_attention_dim=6, in_channels=4)
    def cache_context(self, *args):
        return nullcontext()
    def __call__(self, **kwargs):
        self.received = kwargs
        raise TransformerReached()


class NativeRoutingPipe:
    """설치된 실제 encode_prompt와 __call__을 사용하고 가중치 계산 부분만 대역으로 교체한다."""
    _get_qwen3_prompt_embeds = staticmethod(Flux2KleinPipeline._get_qwen3_prompt_embeds)
    _prepare_text_ids = staticmethod(Flux2KleinPipeline._prepare_text_ids)
    encode_prompt = Flux2KleinPipeline.encode_prompt
    check_inputs = Flux2KleinPipeline.check_inputs
    __call__ = Flux2KleinPipeline.__call__
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    _execution_device = torch.device("cpu")
    config = SimpleNamespace(is_distilled=True)
    vae_scale_factor = 8
    default_sample_size = 1
    do_classifier_free_guidance = False
    interrupt = False
    attention_kwargs = None
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or WordTokenizer()
        self.text_encoder = RecordingEncoder()
        self.transformer = RecordingTransformer()
        self.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    def prepare_latents(self, **kwargs):
        return torch.zeros((1,1,4)), torch.zeros((1,1,4))
    def progress_bar(self, **kwargs):
        return nullcontext(SimpleNamespace(update=lambda: None))


def words(total):
    # 사용자/assistant 경계까지 포함한 토큰 수.
    return " ".join("word"+str(i) for i in range(total-2))


def test_short_prompt_keeps_existing_arguments_and_does_not_preencode():
    pipe = NativeRoutingPipe()
    text = words(256)
    args, record = build_flux_condition(pipe, text)
    assert args == dict(prompt=text, max_sequence_length=256)
    assert record["mode"] == "prompt" and not pipe.text_encoder.calls


@pytest.mark.parametrize("total", [257, 316, 512])
def test_long_prompt_reaches_real_transformer_call_without_truncation(total):
    pipe = NativeRoutingPipe()
    text = words(total)
    args, record = build_flux_condition(pipe, text)
    assert "prompt" not in args
    assert args["max_sequence_length"] == total
    assert record["embedding_shape"] == [1,total,6]
    assert not record["memory_validated"] and not record["image_quality_validated"]
    assert len(pipe.text_encoder.calls) == 1
    encoder_ids = pipe.text_encoder.calls[0]["input_ids"]
    assert encoder_ids.shape == (1,total)
    assert pipe.text_encoder.calls[0]["attention_mask"].sum().item() == total
    with pytest.raises(TransformerReached):
        pipe(width=16,height=16,num_inference_steps=1,guidance_scale=1.0,**args)
    received = pipe.transformer.received
    # 실제 파이프라인이 조건을 다시 잘라 인코딩하지 않고 생성 모델까지 보낸다.
    assert len(pipe.text_encoder.calls) == 1
    assert received["encoder_hidden_states"].shape == (1,total,6)
    assert torch.equal(received["encoder_hidden_states"][0,:,0],encoder_ids[0]+9)
    assert torch.equal(received["txt_ids"][0,:,3],torch.arange(total))
    assert record["continuous_text_positions"]


def test_overflow_is_failure_before_encoder_and_does_not_drop_remainder():
    pipe = NativeRoutingPipe()
    text = words(513)
    plan = plan_flux_prompt(pipe.tokenizer,text)
    assert plan["mode"] == "blocked" and plan["token_count"] == 513
    with pytest.raises(ValueError,match="513/512"):
        build_flux_condition(pipe,text)
    assert not pipe.text_encoder.calls


@pytest.mark.parametrize("damage", ["shape", "nan", "position", "integer"])
def test_invalid_encoded_condition_is_rejected(damage):
    embeds = torch.ones((1,300,6))
    positions = torch.zeros((1,300,4))
    positions[0,:,3] = torch.arange(300)
    if damage == "shape": embeds = embeds[:,:299]
    elif damage == "nan": embeds[0,-1,0] = float("nan")
    elif damage == "position": positions[0,-1,3] = 0
    elif damage == "integer": embeds = embeds.long()
    with pytest.raises(ValueError):
        validate_text_condition(embeds,positions,300,6)


def test_each_prompt_is_encoded_separately_no_stale_condition_cache():
    pipe = NativeRoutingPipe()
    a,_ = build_flux_condition(pipe,words(300)+" blue_eyes")
    b,_ = build_flux_condition(pipe,words(300)+" red_eyes")
    assert len(pipe.text_encoder.calls) == 2
    assert not torch.equal(a["prompt_embeds"],b["prompt_embeds"])


def test_face_img2img_supplies_built_condition_to_pipeline(monkeypatch):
    from genai_lab import flux_img2img, flux_prompt_delivery
    pipe=SimpleNamespace(
        _execution_device=torch.device("cpu"),
        image_processor=SimpleNamespace(preprocess=lambda *args,**kw: torch.zeros((1,3,16,16))),
        vae=SimpleNamespace(dtype=torch.float32),
        scheduler=SimpleNamespace(time_shift=lambda mu, scale, value: value),
        _encode_vae_image=lambda *args,**kw: torch.zeros((1,1,1,1)))
    calls=[]
    class CallablePipe:
        def __getattr__(self,key): return getattr(pipe,key)
        def __call__(self,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=["result"])
    marker=torch.ones((1,300,6))
    monkeypatch.setattr(flux_prompt_delivery,"build_flux_condition",
                        lambda *a: (dict(prompt_embeds=marker,max_sequence_length=300),{}))
    result=flux_img2img.redraw(CallablePipe(),SimpleNamespace(size=(16,16)),[],"prompt",.5,42)
    assert result=="result" and "prompt" not in calls[0]
    assert calls[0]["prompt_embeds"] is marker and calls[0]["max_sequence_length"]==300
