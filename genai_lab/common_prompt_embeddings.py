"""Encode an approved SDXL prompt once and reuse immutable tensor inputs."""

import hashlib


def _text_digest(prompt, negative_prompt):
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(negative_prompt.encode("utf-8"))
    return digest.hexdigest()


def encode_common_prompt_embeddings(pipeline, prompt, negative_prompt):
    import torch

    if not hasattr(pipeline, "encode_prompt"):
        return None, {
            "version": "common_prompt_embeddings_v1",
            "status": "unsupported",
        }
    with torch.inference_mode():
        encoded = pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=getattr(pipeline, "_execution_device", None),
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
    if not isinstance(encoded, (tuple, list)) or len(encoded) != 4:
        raise ValueError("SDXL 공통 프롬프트 임베딩 반환 형식이 올바르지 않습니다.")
    names = (
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    )
    condition = dict(zip(names, encoded))
    if any(not isinstance(value, torch.Tensor) or not value.numel()
           or not torch.isfinite(value).all().item()
           for value in condition.values()):
        raise ValueError("SDXL 공통 프롬프트 임베딩에 유효하지 않은 값이 있습니다.")
    return condition, {
        "version": "common_prompt_embeddings_v1",
        "status": "encoded",
        "text_sha256": _text_digest(prompt, negative_prompt),
        "shapes": {
            name: list(value.shape) for name, value in condition.items()
        },
        "candidate_policy": "shared_read_only",
    }


def validate_common_prompt_embeddings(condition, record, prompt, negative_prompt):
    import torch

    if record.get("status") != "encoded":
        raise ValueError("공통 프롬프트 임베딩 완료 기록이 없습니다.")
    if record.get("text_sha256") != _text_digest(prompt, negative_prompt):
        raise ValueError("공통 프롬프트 임베딩과 승인 프롬프트가 다릅니다.")
    if not isinstance(condition, dict):
        raise ValueError("공통 프롬프트 임베딩이 없습니다.")
    for name, shape in record.get("shapes", {}).items():
        value = condition.get(name)
        if (not isinstance(value, torch.Tensor)
                or list(value.shape) != shape
                or not torch.isfinite(value).all().item()):
            raise ValueError(f"공통 프롬프트 임베딩이 변경됐습니다: {name}")
