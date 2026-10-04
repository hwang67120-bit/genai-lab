"""Approval-gated, phrase-preserving SDXL dual text-encoder chunks."""
from __future__ import annotations

import hashlib
import json

from scripts.generation_inputs import _token_count, _tokenizer_limit

POLICY_VERSION = "sdxl_dual_encoder_phrase_chunks_v1"


def resolve_long_prompt_settings(raw):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("긴 프롬프트 설정은 객체여야 합니다.")
    enabled = bool(raw.get("enabled", False))
    maximum = raw.get("maximum_chunks", 2)
    if type(maximum) is not int or not 1 <= maximum <= 2:
        raise ValueError("긴 프롬프트 청크는 최대 2개만 허용합니다.")
    return {
        "enabled": enabled,
        "maximum_chunks": maximum,
        "policy_version": POLICY_VERSION,
    }


def plan_phrase_chunks(text, tokenizers, settings):
    """Split only at comma phrase boundaries and verify both SDXL tokenizers."""
    settings = resolve_long_prompt_settings(settings)
    tokenizers = tuple(tokenizers)
    if not tokenizers or any(tokenizer is None for tokenizer in tokenizers):
        raise ValueError("SDXL의 두 실제 토크나이저가 모두 필요합니다.")
    phrases = tuple(part.strip() for part in text.split(",") if part.strip())
    chunks = []
    current = []
    limits = tuple(_tokenizer_limit(tokenizer) for tokenizer in tokenizers)
    for phrase in phrases:
        proposed = ", ".join((*current, phrase))
        counts = tuple(_token_count(tokenizer, proposed) for tokenizer in tokenizers)
        if all(count <= limit for count, limit in zip(counts, limits)):
            current.append(phrase)
            continue
        if not current:
            raise ValueError(f"하나의 조건 구문이 CLIP 한도를 초과합니다: {phrase}")
        chunks.append(", ".join(current))
        current = [phrase]
    if current or not chunks:
        chunks.append(", ".join(current))
    if len(chunks) > settings["maximum_chunks"]:
        raise ValueError(
            f"필수 조건이 승인된 최대 {settings['maximum_chunks']}청크를 초과합니다.")
    records = [{
        "index": index + 1,
        "text": chunk,
        "token_counts": [
            _token_count(tokenizer, chunk) for tokenizer in tokenizers],
    } for index, chunk in enumerate(chunks)]
    payload = {
        "policy_version": settings["policy_version"],
        "chunk_count": len(records),
        "chunks": records,
        "tokenizer_limits": list(limits),
        "phrase_boundaries_preserved": True,
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return payload


def _encode_chunks(pipeline, plan):
    import torch

    tokenizers = (pipeline.tokenizer, pipeline.tokenizer_2)
    encoders = (pipeline.text_encoder, pipeline.text_encoder_2)
    per_chunk, pooled = [], None
    for chunk_index, chunk in enumerate(plan["chunks"]):
        hidden_parts = []
        for encoder_index, (tokenizer, encoder) in enumerate(
                zip(tokenizers, encoders)):
            encoded = tokenizer(
                chunk["text"], padding="max_length",
                max_length=_tokenizer_limit(tokenizer), truncation=False,
                return_tensors="pt", verbose=False)
            input_ids = encoded.input_ids.to(pipeline._execution_device)
            attention_mask = None
            if getattr(encoder.config, "use_attention_mask", False):
                attention_mask = encoded.attention_mask.to(
                    pipeline._execution_device)
            output = encoder(
                input_ids, attention_mask=attention_mask,
                output_hidden_states=True)
            hidden_parts.append(output.hidden_states[-2])
            if encoder_index == 1 and chunk_index == 0:
                pooled = output[0]
        per_chunk.append(torch.cat(hidden_parts, dim=-1))
    if pooled is None:
        raise ValueError("SDXL pooled prompt embedding을 만들지 못했습니다.")
    return torch.cat(per_chunk, dim=1), pooled


def build_long_prompt_embeddings(pipeline, positive_plan, negative_text):
    """Encode approved chunks; pad the negative side with explicit empty chunks."""
    import torch

    settings = {
        "enabled": True,
        "maximum_chunks": positive_plan["chunk_count"],
    }
    negative_plan = plan_phrase_chunks(
        negative_text, (pipeline.tokenizer, pipeline.tokenizer_2), settings)
    while negative_plan["chunk_count"] < positive_plan["chunk_count"]:
        negative_plan["chunks"].append({
            "index": negative_plan["chunk_count"] + 1,
            "text": "",
            "token_counts": [
                _token_count(tokenizer, "")
                for tokenizer in (pipeline.tokenizer, pipeline.tokenizer_2)],
        })
        negative_plan["chunk_count"] += 1
    with torch.inference_mode():
        prompt_embeds, pooled = _encode_chunks(pipeline, positive_plan)
        negative_embeds, negative_pooled = _encode_chunks(
            pipeline, negative_plan)
    if prompt_embeds.shape != negative_embeds.shape:
        raise ValueError("양성부정 프롬프트 임베딩 shape가 다릅니다.")
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": negative_embeds,
        "negative_pooled_prompt_embeds": negative_pooled,
    }, {
        "policy_version": POLICY_VERSION,
        "positive_plan_fingerprint": positive_plan["fingerprint"],
        "positive_shape": list(prompt_embeds.shape),
        "negative_shape": list(negative_embeds.shape),
        "negative_padding_policy": "encode_empty_phrase_to_match_sequence",
    }
