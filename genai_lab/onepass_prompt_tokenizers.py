"""Explicit local tokenizer I/O, separate from pure prompt assembly."""
from pathlib import Path
from genai_lab.onepass_prompt_settings import OnePassPromptSettings


def load_onepass_tokenizers(settings=OnePassPromptSettings()):
    from transformers import CLIPTokenizer
    root=Path(settings.tokenizer_root)
    paths=tuple(root/name for name in ('tokenizer','tokenizer_2'))
    if not all(p.is_dir() for p in paths):
        raise FileNotFoundError('설정된 로컬 SDXL 토크나이저가 없습니다. 자동 다운로드하지 않습니다.')
    return tuple(CLIPTokenizer.from_pretrained(str(p),local_files_only=True) for p in paths)
