"""FLUX 전체 문장의 조건 전달. 256 이하 기존 경로, 그 이상은 전체 문맥 임베딩으로 연결한다."""
import hashlib

from genai_lab.style_redraw_quality import audit_flux_prompt

# 설치된 FLUX.2-klein 라이브러리의 기본 길이 안으로 제한한다.
# VRAM 상한과 별개이며, 긴 조건의 실제 GPU 메모리/품질은 별도 검증 대상이다.
MAX_PROMPT_TOKENS = 512
BASE_SEQUENCE_LENGTH = 256


def plan_flux_prompt(tokenizer, prompt, base_length=BASE_SEQUENCE_LENGTH):
    """채팅 템플릿을 포함해 한 번에 전달할 길이를 결정한다. 문장을 자르거나 분할하지 않는다."""
    if not isinstance(base_length, int) or isinstance(base_length, bool) or not 1 <= base_length <= MAX_PROMPT_TOKENS:
        raise ValueError("FLUX 기본 입력 길이가 허용 범위를 벗어났습니다.")
    audit = audit_flux_prompt(tokenizer, prompt, MAX_PROMPT_TOKENS)
    blocked = audit["truncated"]
    count = audit["token_count"]
    audit.update(
        policy="flux_whole_context_embeddings_v1", base_length=base_length,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        encoding_length=None if blocked else max(base_length, count),
        mode="blocked" if blocked else "prompt" if count <= base_length else "prompt_embeds",
        exceeds_short_window=count > base_length,
        chunked=False, memory_validated=False, image_quality_validated=False)
    return audit


def validate_text_condition(embeddings, text_ids, sequence_length, expected_width):
    """생성기에 넘길 조건의 길이·차원·유한성·연속 위치를 확인한다."""
    import torch
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
        raise ValueError("FLUX 텍스트 조건은 3차원 텐서여야 합니다.")
    if tuple(embeddings.shape) != (1, sequence_length, expected_width):
        raise ValueError("FLUX 텍스트 조건의 길이 또는 특징 차원이 맞지 않습니다.")
    if not embeddings.is_floating_point() or not torch.isfinite(embeddings).all().item():
        raise ValueError("FLUX 텍스트 조건에 유효하지 않은 값이 있습니다.")
    if not isinstance(text_ids, torch.Tensor) or tuple(text_ids.shape) != (1, sequence_length, 4):
        raise ValueError("FLUX 텍스트 위치 정보의 크기가 맞지 않습니다.")
    expected = torch.zeros_like(text_ids)
    expected[0, :, 3] = torch.arange(sequence_length, device=text_ids.device, dtype=text_ids.dtype)
    if not torch.equal(text_ids, expected):
        raise ValueError("FLUX 텍스트 위치가 문장 전체에서 연속적이지 않습니다.")


def build_flux_condition(pipe, prompt, base_length=BASE_SEQUENCE_LENGTH):
    """생성 호출용 인자와 기록을 반환한다. 긴 문장의 인코딩 결과는 prompt_embeds로 직접 전달한다."""
    plan = plan_flux_prompt(pipe.tokenizer, prompt, base_length)
    if plan["truncated"]:
        raise ValueError(f"생성 지시가 입력 한도를 넘었습니다({plan['token_count']}/{plan['limit']}). "
                         "필수 조건을 버리지 않고 중단합니다.")
    if plan["mode"] == "prompt":
        return dict(prompt=prompt, max_sequence_length=base_length), plan
    import torch
    with torch.inference_mode():
        embeddings, text_ids = pipe.encode_prompt(
            prompt=prompt, device=pipe._execution_device, num_images_per_prompt=1,
            max_sequence_length=plan["encoding_length"])
    width = pipe.transformer.config.joint_attention_dim
    validate_text_condition(embeddings, text_ids, plan["encoding_length"], width)
    plan.update(embedding_shape=list(embeddings.shape), embedding_dtype=str(embeddings.dtype),
                text_ids_shape=list(text_ids.shape), destination="transformer.encoder_hidden_states",
                continuous_text_positions=True)
    # prompt와 prompt_embeds를 함께 넘기지 않는다. 파이프라인이 이 조건에서 위치 ID를 다시 만든다.
    return dict(prompt_embeds=embeddings, max_sequence_length=plan["encoding_length"]), plan
