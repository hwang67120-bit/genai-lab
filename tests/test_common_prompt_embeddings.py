from types import SimpleNamespace

import pytest
import torch

from genai_lab.common_prompt_embeddings import (
    encode_common_prompt_embeddings,
    validate_common_prompt_embeddings,
)


def test_common_prompt_embeddings_are_encoded_once_and_bound_to_text():
    calls = []

    def encode_prompt(**kwargs):
        calls.append(kwargs)
        return tuple(torch.ones(1, 2, index + 1) for index in range(4))

    pipeline = SimpleNamespace(
        encode_prompt=encode_prompt,
        _execution_device="cpu",
    )
    condition, record = encode_common_prompt_embeddings(
        pipeline, "1boy, blue hair", "1girl")
    validate_common_prompt_embeddings(
        condition, record, "1boy, blue hair", "1girl")
    assert len(calls) == 1
    assert record["candidate_policy"] == "shared_read_only"


def test_common_prompt_embeddings_reject_different_approved_text():
    pipeline = SimpleNamespace(
        encode_prompt=lambda **kwargs: tuple(
            torch.ones(1, 2, 2) for _ in range(4)),
        _execution_device="cpu",
    )
    condition, record = encode_common_prompt_embeddings(
        pipeline, "approved", "negative")
    with pytest.raises(ValueError, match="승인 프롬프트"):
        validate_common_prompt_embeddings(
            condition, record, "changed", "negative")
