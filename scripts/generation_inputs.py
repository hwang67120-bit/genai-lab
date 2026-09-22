"""Lightweight resolution and CLIP prompt helpers shared by generation entrypoints."""

def resolve_inference_size(source_size, width=None):
    """원본은 유지하고 연산 캔버스만 8배수로 정한다. 비율 오차는 반올림 범위다."""
    if width is None or width == source_size[0]:
        return source_size
    if not isinstance(width, int) or not 256 <= width <= 2048 or width % 8:
        raise ValueError("생성 가로 크기는 256~2048 사이 8의 배수여야 합니다.")
    height = max(8, round(source_size[1] * width / source_size[0] / 8) * 8)
    if height > 4096:
        raise ValueError("생성 세로 크기는 4096px 이하여야 합니다.")
    return width, height

def _token_count(tokenizer, text: str) -> int:
    token_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)

def _tokenizer_limit(tokenizer) -> int:
    configured = int(getattr(tokenizer, "model_max_length", 77))
    return configured if 2 <= configured < 100_000 else 77

def prepare_prompt_for_clip(
    text: str,
    tokenizers: tuple,
) -> tuple[str, dict[str, object]]:
    """쉼표 단위 의미를 보존하며 모든 CLIP 토크나이저 한도에 맞춘다."""
    usable_tokenizers = tuple(item for item in tokenizers if item is not None)
    if not usable_tokenizers:
        raise RuntimeError("SDXL CLIP 토크나이저를 찾을 수 없습니다.")
    segments = tuple(part.strip() for part in text.split(",") if part.strip())
    retained: list[str] = []
    for segment in segments:
        candidate = ", ".join((*retained, segment))
        if all(
            _token_count(tokenizer, candidate) <= _tokenizer_limit(tokenizer)
            for tokenizer in usable_tokenizers
        ):
            retained.append(segment)
        else:
            break
    effective = ", ".join(retained)
    if text.strip() and not effective:
        raise RuntimeError(
            "프롬프트 첫 구문이 CLIP 최대 토큰 길이를 초과합니다."
        )
    limits = tuple(_tokenizer_limit(item) for item in usable_tokenizers)
    original_counts = tuple(_token_count(item, text) for item in usable_tokenizers)
    effective_counts = tuple(
        _token_count(item, effective) for item in usable_tokenizers
    )
    return effective, {
        "original": text,
        "effective": effective,
        "truncated": effective != text.strip(),
        "tokenizer_limits": limits,
        "original_token_counts": original_counts,
        "effective_token_counts": effective_counts,
        "retained_segment_count": len(retained),
        "source_segment_count": len(segments),
    }
