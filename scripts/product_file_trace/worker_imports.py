"""작업 프로세스 불러오기 점검(CPU, D-170): Qwen·머리 분할 작업 프로세스가 실행 중에 늦게 불러오는 모듈까지 불러와 위치를 남긴다.

모델은 불러오지 않는다(GPU 없음). 생성·분할이 실제로 되는지는 확인하지 않는다 — 패키지 목록을 만들기 위한 '불러오기' 근거다.
사용: <python> worker_imports.py <qwen|segment> <출력 json>
"""
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def qwen():
    # qwen_pose_worker.generate()가 함수 안에서 불러오는 것과 같은 이름들
    import torch  # noqa: F401
    from PIL import Image  # noqa: F401
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler  # noqa: F401
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE  # noqa: F401
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig  # noqa: F401
    GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    for name in ("genai_lab.qwen_pose_worker", "genai_lab.outfit_edit", "genai_lab.generation_preview", "genai_lab.qwen_free_generation",
                 "genai_lab.qwen_outfit_edit", "genai_lab.qwen_accelerators", "genai_lab.style_loras", "genai_lab.style_pass"):
        importlib.import_module(name)
    from genai_lab.qwen_pose_settings import check_runtime_versions
    return dict(version_contract=check_runtime_versions())


def segment():
    # outfit_edit_worker parts → head_paste_semantics.HeadSegmenter가 불러오는 see-through 코드(모델은 올리지 않음)
    from genai_lab.studio_generation import StudioRuntime
    from genai_lab.head_paste_semantics import semantic_source_file
    importlib.import_module("genai_lab.outfit_edit_worker")
    repo = Path(StudioRuntime.from_environment().head_semantic_repo)
    semantic_source_file(repo)
    sys.path.insert(0, str(repo / "common"))
    sys.path.insert(0, str(repo))
    from modules.semanticsam import SemanticSam  # noqa: F401
    return dict(see_through=str(repo))


def main():
    role, out = sys.argv[1], Path(sys.argv[2])
    record = dict(role=role, executable=sys.executable)
    try:
        record["result"] = dict(qwen=qwen, segment=segment)[role]()
        record["ok"] = True
    except Exception as error:  # noqa: BLE001
        record.update(ok=False, error=f"{type(error).__name__}: {str(error)[:300]}")
    record["modules"] = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items()) if getattr(m, "__file__", None)}
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(role, "ok" if record["ok"] else record["error"], len(record["modules"]), "modules")


if __name__ == "__main__":
    main()
