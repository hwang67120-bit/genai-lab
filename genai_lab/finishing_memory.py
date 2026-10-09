"""One verified CFG face embedding cache and scoped, IP-preserving SDPA policy."""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import time
from genai_lab.proportion_inputs import require


def embedding_record(tensors):
    """Hash contiguous tensor bytes without changing dtype or CFG ordering."""
    digest = hashlib.sha256()
    records = []
    for tensor in tensors:
        data = tensor.detach().cpu().contiguous().numpy().tobytes()
        digest.update(data)
        records.append({"sha256":hashlib.sha256(data).hexdigest(), "dtype":str(tensor.dtype),
                        "shape":list(tensor.shape), "bytes":len(data)})
    return {"sha256":digest.hexdigest(), "tensors":records}


def efficient_attention():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    # Keep the existing processors/IP weights; disallow the full-matrix math fallback.
    return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)


@contextmanager
def unet_attention(unet):
    """Apply the efficient kernel only to UNet; preserve VAE and encoder behavior."""
    active = [None]
    def enter(*_):
        require(active[0] is None, "중첩된 UNet attention 실행입니다.")
        active[0] = efficient_attention()
        active[0].__enter__()
    def leave(*_):
        if active[0] is not None:
            context, active[0] = active[0], None
            context.__exit__(None,None,None)
    before = unet.register_forward_pre_hook(enter)
    try:
        after = unet.register_forward_hook(leave, always_call=True)
        try:
            yield
        finally:
            leave()
            after.remove()
    finally:
        before.remove()


def attention_policy(pipe):
    names = sorted({type(p).__name__ for p in pipe.unet.attn_processors.values()})
    require("IPAdapterAttnProcessor2_0" in names and
            set(names) <= {"AttnProcessor2_0", "IPAdapterAttnProcessor2_0"},
            "확인된 SDPA/IP attention processor 구성이 아닙니다.")
    return {"mode":"sdpa_efficient_only", "scope":"unet_only", "allowed_backends":["EFFICIENT_ATTENTION"],
            "math_fallback":False, "processors":names, "attention_slicing":False}


@dataclass
class FaceEmbeddingCache:
    tensors: tuple
    record: dict
    encoder_guard: object

    def verify(self, reference):
        require(reference.sha256 == self.record["face_sha256"], "캐시와 얼굴 SHA가 다릅니다.")
        with reference.open_image():
            pass
        require(embedding_record(self.tensors) == self.record["embeddings"], "캐시된 얼굴 임베딩이 변경됐습니다.")

    def verify_delivered(self, tensors):
        require(embedding_record(tensors) == self.record["embeddings"], "UNet에 전달된 얼굴 임베딩이 다릅니다.")

    def close(self):
        self.encoder_guard.remove()
        self.tensors = ()


def prepare_face_cache(pipe, reference, torch, guard, record, *, device="cuda"):
    """Compute positive and CFG-negative once; offload encoder before any img2img."""
    started = time.perf_counter()
    encoder = pipe.image_encoder
    calls = []
    hook = None
    record.update(status="started", face_sha256=reference.sha256, preparation_calls=1,
                  cfg_order=["negative","positive"])
    torch.cuda.reset_peak_memory_stats()
    try:
        record["attention"] = attention_policy(pipe)
        hook = encoder.register_forward_hook(lambda *_:calls.append(len(calls)))
        guard()
        with reference.open_image() as image, torch.inference_mode():
            embeddings = pipe.prepare_ip_adapter_image_embeds(image, None, device, 1, True)
            # Plus-face uses penultimate hidden states and a separately encoded zero image.
            require(len(embeddings) == 1 and embeddings[0].ndim == 4 and
                    embeddings[0].shape[:2] == (2,1) and str(embeddings[0].dtype) == "torch.float16",
                    "CFG 얼굴 임베딩의 개수·shape·dtype이 다릅니다.")
            cached = tuple(t.detach().cpu().contiguous() for t in embeddings)
            del embeddings
        require(len(calls) == 2, "얼굴/CFG 음성 인코더 호출 수가 2회가 아닙니다.")
        guard()
        record["embeddings"] = embedding_record(cached)
    except BaseException as error:
        record.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if hook is not None: hook.remove()
        record.update(encoder_forward_calls=len(calls), seconds=time.perf_counter()-started,
                      max_reserved_bytes=torch.cuda.max_memory_reserved(),
                      max_allocated_bytes=torch.cuda.max_memory_allocated())
        encoder.to("cpu")
        torch.cuda.empty_cache()
        record.update(encoder_after_device="cpu", reserved_after_offload_bytes=torch.cuda.memory_reserved(),
                      allocated_after_offload_bytes=torch.cuda.memory_allocated())
    require(all(t.device.type == "cpu" for t in (*encoder.parameters(), *encoder.buffers())),
            "얼굴 인코더가 CPU로 내려가지 않았습니다.")
    def forbid_forward(*_):
        raise RuntimeError("사전 계산 뒤 얼굴 인코더를 다시 호출할 수 없습니다.")
    encoder_guard = encoder.register_forward_pre_hook(forbid_forward)
    record["status"] = "completed"
    return FaceEmbeddingCache(cached,record,encoder_guard)
