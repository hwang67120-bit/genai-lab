"""모델 폴더(GENAI_MODEL_HOME) 배치 점검(CPU, D-171): 제품 단계가 그 폴더만으로 준비되는지, 빠지면 어떤 안내가 나오는지 기록한다.

사용(GPU 숨김, 관찰 도구와 함께 쓰면 연 파일도 남는다):
  set GENAI_MODEL_HOME=<모델 폴더>
  python model_home_check.py <출력 json> [--sha]     (--sha: Qwen 명세 27개 + GGUF 전체 SHA 대조 — 약 28GB 읽기)
단계: 실행 설정 → Qwen 설정 확인 → 기준 분석(의상 없음) → 입력 분석(의상 있음) → (선택) Qwen 모델 파일 SHA.
"""
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DOWNLOADS = Path.home() / "Downloads"
CHAR = DOWNLOADS / "참조 이미지" / "86a72742408d9b4d04f5fb23fb0b93bd.jpg"
GARM = DOWNLOADS / "참조 의상" / "4b750036a001e45a39973c408855e970.jpg"


def step(record, name, fn):
    start = time.time()
    try:
        value = fn()
        record[name] = dict(ok=True, seconds=round(time.time() - start, 1), value=value)
    except Exception as error:  # noqa: BLE001 — 사용자에게 보일 오류 문장을 그대로 남긴다
        record[name] = dict(ok=False, seconds=round(time.time() - start, 1), error_type=type(error).__name__,
                            message=str(error)[:600], tb=traceback.format_exc()[-800:])
    print(name, "ok" if record[name]["ok"] else f"실패: {record[name]['message'][:160]}", flush=True)
    return record[name]


def main(argv):
    out, full_sha = Path(argv[0]), "--sha" in argv
    record = dict(model_home=os.environ.get("GENAI_MODEL_HOME"), gpu_visible=None)
    import torch
    record["gpu_visible"] = torch.cuda.is_available()
    from genai_lab import studio_free_generation, studio_generation
    from genai_lab.qwen_outfit_edit import load_qwen_settings
    runtime = step(record, "runtime", lambda: {k: str(v) for k, v in vars(studio_generation.StudioRuntime.from_environment()).items()
                                               if isinstance(v, Path)})
    rt = studio_generation.StudioRuntime.from_environment() if runtime["ok"] else None
    settings = step(record, "qwen_settings", lambda: {k: v for k, v in vars(load_qwen_settings(rt.qwen_settings)).items()
                                                       if k in ("python_executable", "model_root", "gguf_file", "artifact_manifest")})
    if rt is not None:
        def base():
            result = studio_free_generation.analyze_base(CHAR, rt)
            return dict(tag_count=len(result.get("character_scores") or {}))
        step(record, "analyze_base", base)
        work = Path(tempfile.mkdtemp(prefix="model-home-check-"))
        step(record, "analyze_inputs", lambda: sorted(studio_generation.analyze_inputs(CHAR, GARM, work / "inputs", rt, require_face=False))[:12])
    if full_sha and settings["ok"]:
        from genai_lab.qwen_pose_settings import validate_model_files
        step(record, "qwen_model_sha", lambda: validate_model_files(load_qwen_settings(rt.qwen_settings)))
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
