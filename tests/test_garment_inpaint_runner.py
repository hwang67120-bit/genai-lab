from types import SimpleNamespace

import pytest

from scripts.garment_inpaint_runner import (
    prepare_prompt_for_clip,
    validate_ip_adapter_dimensions,
)


class FakeTokenizer:
    model_max_length = 6

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [0, *range(len(text.split())), 1]}


def test_prompt_is_truncated_at_comma_boundary_for_both_clip_tokenizers():
    prompt, record = prepare_prompt_for_clip(
        "one two, three, four five, six",
        (FakeTokenizer(), FakeTokenizer()),
    )

    assert prompt == "one two, three"
    assert record["truncated"] is True
    assert record["original_token_counts"] == (8, 8)
    assert record["effective_token_counts"] == (5, 5)
    assert record["tokenizer_limits"] == (6, 6)


def _pipeline(encoder_size: int, adapter_size: int):
    projection = SimpleNamespace(
        proj_in=SimpleNamespace(in_features=adapter_size)
    )
    return SimpleNamespace(
        image_encoder=SimpleNamespace(
            config=SimpleNamespace(hidden_size=encoder_size)
        ),
        unet=SimpleNamespace(
            encoder_hid_proj=SimpleNamespace(
                image_projection_layers=[projection]
            )
        ),
    )


def test_ip_adapter_dimensions_accept_matching_vit_h_encoder():
    assert validate_ip_adapter_dimensions(
        _pipeline(1280, 1280),
        "ip-adapter-plus_sdxl_vit-h.safetensors",
    ) == {
        "image_encoder_hidden_size": 1280,
        "adapter_projection_input_size": 1280,
    }


def test_ip_adapter_dimensions_reject_big_g_for_vit_h_adapter():
    with pytest.raises(RuntimeError, match="인코더=1664, 어댑터=1280"):
        validate_ip_adapter_dimensions(
            _pipeline(1664, 1280),
            "ip-adapter-plus_sdxl_vit-h.safetensors",
        )
