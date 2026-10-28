"""로컬 See-through 분할 전용 수명. 추론 뒤 모델을 내려 다시 그리기와 겹치지 않는다."""
from pathlib import Path
import sys
import gc
import numpy as np
from genai_lab.proportion_inputs import sha, require
from genai_lab.head_paste_rules import TAGS, validate_parts

CHECKPOINT_SHA = "dda82b916f6c340bc5a6c3a077ff1214f9906906abe3707b1076d55b508f253f"


def semantic_source_file(repo):
    """기존 도구의 common 배치와 평면 배치를 확인한다. 누락 시 다운로드하지 않는다."""
    root = Path(repo)
    for path in (root / "common/modules/semanticsam.py", root / "modules/semanticsam.py"):
        if path.is_file():
            return path
    raise ValueError("로컬 머리 분할 도구가 없습니다.")


class HeadSegmenter:
    def __init__(self, repo, checkpoint):
        import torch
        require(sha(checkpoint) == CHECKPOINT_SHA, "머리 분할 모델 SHA가 다릅니다.")
        repo = Path(repo)
        semantic_source_file(repo)
        sys.path.insert(0, str(repo / "common"))
        sys.path.insert(0, str(repo))
        from modules.semanticsam import SemanticSam
        torch.cuda.reset_peak_memory_stats()
        self.model = SemanticSam(class_num=19)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model = self.model.to("cuda").eval()
        from genai_lab.head_paste_redraw import memory_guard
        memory_guard(torch)

    def parse(self, image):
        import torch
        from genai_lab.head_paste_redraw import memory_guard
        with torch.inference_mode():
            parts = self.model.inference(image)[0]
        parts = parts.cpu().numpy() if hasattr(parts, "cpu") else np.asarray(parts)
        result = {name: parts[index] > 0 for index, name in enumerate(TAGS)}
        validate_parts(result)
        memory_guard(torch)
        return result

    def metrics(self):
        import torch
        return dict(max_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    checkpoint_sha256=CHECKPOINT_SHA)

    def close(self):
        import torch
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


def save_parts(path, parts):
    validate_parts(parts)
    np.savez_compressed(path, **parts)


def load_parts(path):
    with np.load(path, allow_pickle=False) as data:
        parts = {name: data[name].copy() for name in data.files}
    validate_parts(parts)
    return parts
