"""제품 실행 환경 점검(D-148). 환경을 바꾸거나 복구한 뒤 GUI를 쓰기 전에 돌린다. GPU 생성은 하지 않는다.

CPU 시험은 함수 안에서 늦게 불러오는 패키지(예: imgutils — 2026-10-21 GUI 편집 경로가 이것 없이 멈춤)를 지나가지 않는다.
여기서는 제품 경로가 실제로 쓰는 패키지·작업 프로세스 환경·모델 파일을 직접 불러와 본다.
사용: python scripts/check_runtime_env.py [--sha]   (--sha: 큰 모델 파일 SHA까지 대조, 수 분)
종료 코드: 필수 항목 실패가 있으면 1.
"""
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (모듈, 쓰는 곳, 필수 여부)
MAIN_IMPORTS = [
    ("PySide6.QtWidgets", "작업실 GUI", True), ("numpy", "전체", True), ("PIL", "전체", True), ("cv2", "분할·보정", True),
    ("torch", "생성·분할", True), ("diffusers", "Qwen 생성", True), ("transformers", "Qwen 문장 처리", True),
    ("accelerate", "Qwen 생성", True), ("gguf", "GGUF 변환기", True), ("bitsandbytes", "문장 처리 양자화", True),
    ("peft", "가속 LoRA(미리보기)", True), ("onnxruntime", "태거·머리 검출", True), ("imgutils.detect", "편집 캐릭터 분석·마무리·동일성 측정", True),
    ("scipy", "분석", True), ("sklearn", "분석", True), ("pandas", "imgutils", True), ("huggingface_hub", "모델 캐시", True),
    ("psutil", "작업 프로세스 감시", True), ("yaml", "설정", True),
    ("ultralytics", "예전 SDXL 경로(detail·reference_face)", False), ("spandrel", "예전 업스케일 경로", False),
    ("controlnet_aux", "예전 장면 참조 경로", False),
]


def check_import(name):
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", "")
    except Exception as error:   # noqa: BLE001 — 점검 도구는 모든 실패를 보고한다
        return False, f"{type(error).__name__}: {error}"[:160]


def check_interpreter(python, code):
    if not Path(python).is_file():
        return False, f"없음: {python}"
    run = subprocess.run([str(python), "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    return run.returncode == 0, (run.stdout.strip() or run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "")[:160]


def main(full_sha=False):
    from genai_lab.qwen_accelerators import check_file
    from genai_lab.qwen_pose_settings import QwenPoseSettings, validate_model_files
    from genai_lab.studio_generation import StudioRuntime
    runtime = StudioRuntime.from_environment()
    settings = QwenPoseSettings.load(ROOT / "configs/qwen_edit_settings.json")
    rows = []
    for name, where, required in MAIN_IMPORTS:
        ok, detail = check_import(name)
        rows.append(("패키지", name, where, required, ok, detail))
    ok, detail = check_interpreter(runtime.pose_python, "import easy_dwpose, onnxruntime; print('easy_dwpose ok')")
    rows.append(("작업 환경", "관절 분석(pose_python)", str(runtime.pose_python), True, ok, detail))
    ok, detail = check_interpreter(settings.python_executable, "import torch, diffusers, gguf; print(torch.__version__, diffusers.__version__, torch.cuda.is_available())")
    rows.append(("작업 환경", "Qwen 작업 프로세스", settings.python_executable, True, ok, detail))
    try:
        import torch
        cuda = torch.cuda.is_available()
        rows.append(("GPU", "CUDA", "생성", True, cuda, torch.cuda.get_device_name(0) if cuda else "CUDA 없음"))
    except Exception as error:  # noqa: BLE001
        rows.append(("GPU", "CUDA", "생성", True, False, str(error)[:160]))
    files = [("Qwen 모델 폴더", Path(settings.model_root)), ("GGUF 변환기", Path(settings.gguf_file)),
             ("관절 모델(DWPose)", Path(runtime.pose_models)), ("모델 캐시(WD 태거 등)", Path(runtime.model_cache)),
             ("머리 검출 캐시", Path(runtime.head_cache)), ("머리 분할(see-through)", Path(runtime.head_semantic_repo)),
             ("머리 분할 체크포인트", Path(runtime.head_semantic_checkpoint))]
    for name, path in files:
        rows.append(("모델 파일", name, str(path), True, path.exists(), "" if path.exists() else "없음"))
    try:
        rows.append(("모델 파일", "가속 LoRA(미리보기)", str(check_file("lightning8", settings.model_root)), False, True, "크기 일치"))
    except ValueError as error:
        rows.append(("모델 파일", "가속 LoRA(미리보기)", "", False, False, str(error)[:160]))
    if full_sha:
        try:
            validate_model_files(settings)
            rows.append(("모델 SHA", "Qwen·GGUF 명세 대조", "", True, True, "일치"))
        except Exception as error:  # noqa: BLE001
            rows.append(("모델 SHA", "Qwen·GGUF 명세 대조", "", True, False, str(error)[:160]))
    failed_required = [r for r in rows if r[3] and not r[4]]
    sys.stdout.reconfigure(encoding="utf-8")
    for kind, name, where, required, ok, detail in rows:
        mark = "OK " if ok else ("실패" if required else "없음(선택)")
        print(f"[{mark}] {kind} · {name} — {where} {('· ' + detail) if detail else ''}")
    print(f"\n필수 실패 {len(failed_required)}건" + ("" if failed_required else " — 제품 실행 환경 정상"))
    report = [dict(kind=k, name=n, where=w, required=r, ok=o, detail=d) for k, n, w, r, o, d in rows]
    (ROOT / "outputs" / "runtime-env-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main(full_sha="--sha" in sys.argv))
