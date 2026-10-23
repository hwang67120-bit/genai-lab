"""제품 흐름의 CPU 단계를 실제로 돌려 열리는 파일을 기록한다(2026-10-23 패키징 조사). Qwen 생성(GPU)은 기존 결과 그림으로 대신한다.

단계: A 입력 분석(의상 있음) · B 기준 분석(의상 없음) · C 옷 바꾸기 파이프라인(가짜 Qwen) · D 자세·장면 파이프라인(가짜 Qwen).
결과 폴더는 인자로 받은 임시 폴더에만 쓴다. 공유 캐시·실행 환경은 읽기만 한다.
"""
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(r"Z:\genai-lab")
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])
CHAR = Path(r"C:\Users\user\Downloads\참조 이미지\86a72742408d9b4d04f5fb23fb0b93bd.jpg")
GARM = Path(r"C:\Users\user\Downloads\참조 의상\4b750036a001e45a39973c408855e970.jpg")
B2 = ROOT / "outputs/product-run-20261022/runs/b5cb43fef5bf4277a15f16db4fd23644"
FEATURES = {"hair_length": ["short_hair"], "hair_style": None, "hair_color": ["black_hair"], "eye_color": None,
            "head_accessories": None, "skin": None}


def mark(stage):
    import os
    with open(os.environ["GENAI_OPEN_TRACE"], "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stage=stage, t=time.time()), ensure_ascii=False) + "\n")


def fake_qwen(raw):
    def run(request, directory, workflow, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True)
        shutil.copy(raw, directory / "raw.png")
        (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        return directory / "raw.png"
    return run


def main():
    from genai_lab import studio_generation, studio_free_generation
    from genai_lab.qwen_outfit_edit import load_qwen_settings
    runtime = studio_generation.StudioRuntime.from_environment()
    settings = load_qwen_settings(runtime.qwen_settings)
    status = {}
    stages = [
        ("A_analyze_inputs", lambda: studio_generation.analyze_inputs(CHAR, GARM, OUT / "A" / "inputs", runtime, require_face=False)),
        ("B_analyze_base", lambda: studio_free_generation.analyze_base(CHAR, runtime)),
    ]
    def outfit():
        import genai_lab.qwen_outfit_edit as q
        q.run_outfit_edit = fake_qwen(B2 / "outfit-edit/qwen/raw.png")
        from genai_lab.outfit_edit_pipeline import run_outfit_pipeline
        return run_outfit_pipeline(CHAR, GARM, settings=settings, head_accessories="keep", runtime=runtime, directory=OUT / "C",
                                   gender="female", body_parts=[], character_features=FEATURES)
    def free():
        import genai_lab.free_generation_pipeline as p
        p.run_free_generation_edit = fake_qwen(B2 / "free-generation/qwen/raw.png")
        return p.run_free_generation(CHAR, FEATURES, "", base_kind="original", gender="female", body_parts=[], confirmed=True,
                                     adult_confirmed=True, settings=settings, runtime=runtime, directory=OUT / "D", selection_seed=42,
                                     automatic_edit_base=False, confirmation_source="automatic_product")
    stages += [("C_outfit_pipeline", outfit), ("D_free_generation", free)]
    for name, fn in stages:
        mark(name)
        start = time.time()
        try:
            result = fn()
            status[name] = dict(state="ok", seconds=round(time.time() - start, 1),
                                result=str(result.get("status") if isinstance(result, dict) else type(result).__name__))
        except Exception as error:  # noqa: BLE001 — 단계 실패도 관찰 결과로 남긴다
            status[name] = dict(state="failed", seconds=round(time.time() - start, 1), error=str(error)[:300],
                                tb=traceback.format_exc()[-1500:])
        print(name, status[name]["state"], status[name]["seconds"], flush=True)
    mark("end")
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
