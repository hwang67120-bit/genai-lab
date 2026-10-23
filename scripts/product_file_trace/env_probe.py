"""실행 환경 점검(CPU, D-169): 이 Python이 실제로 어떤 기반 Python·경로·.pth를 쓰는지, 역할별 모듈을 불러오는지와 그 위치를 기록한다.

사용: <python> -I env_probe.py <역할> <출력 json>   (-I: PYTHONPATH·사용자 site 영향 없이 환경 자체만 본다)
역할: gui(작업실·분석·후처리) · pose(관절 분석 하위 프로세스) · qwen(Qwen 작업 프로세스 — 불러오기만, 생성 안 함)
GPU는 쓰지 않는다. Qwen 생성 호환은 이 점검으로 확인하지 않는다(GPU 확인 몫).
"""
import importlib
import json
import os
import site
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES = {
    "gui": ["numpy", "PIL", "cv2", "PySide6.QtWidgets", "torch", "onnxruntime", "imgutils.detect", "huggingface_hub", "scipy", "sklearn"],
    "pose": ["numpy", "PIL", "cv2", "onnxruntime", "easy_dwpose.body_estimation", "huggingface_hub"],
    "qwen": ["numpy", "PIL", "torch", "diffusers", "transformers", "bitsandbytes", "gguf", "accelerate", "safetensors",
             "diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus"],
}
REPO_MODULES = {
    "gui": ["genai_lab.studio_product", "genai_lab.free_generation_pipeline", "genai_lab.outfit_edit_pipeline"],
    "pose": [],
    "qwen": ["genai_lab.qwen_pose_worker"],
}


def env_of(path):
    p = str(path).lower()
    for name in ("qwen-test-venv", "catvton-venv", "venv"):
        if f"\\genai-cache\\{name}\\" in p or f"/genai-cache/{name}/" in p or f"\\model\\genai-cache\\{name}\\" in p:
            return name
    if p.startswith(str(Path(sys.base_prefix)).lower()):
        return "base_python"
    if p.startswith(str(ROOT).lower()) or "genai-lab" in p:
        return "repo"
    return "other"


def main():
    role, out = sys.argv[1], Path(sys.argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    purelib = Path(sysconfig.get_paths()["purelib"])
    pth = {p.name: p.read_text(encoding="utf-8", errors="replace").strip()[:500] for p in purelib.glob("*.pth")}
    record = dict(role=role, executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix, version=sys.version,
                  flags=dict(isolated=sys.flags.isolated, no_user_site=sys.flags.no_user_site),
                  stdlib=sysconfig.get_paths()["stdlib"], purelib=str(purelib), user_site_enabled=site.ENABLE_USER_SITE,
                  path=[dict(entry=p, env=env_of(p)) for p in sys.path], pth_files=pth, imports={})
    for name in ROLES[role] + REPO_MODULES[role]:
        if name.startswith("genai_lab") and str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))           # 실제 하위 프로세스는 스크립트가 저장소 루트를 경로에 넣거나 cwd=저장소로 -m 실행
        try:
            module = importlib.import_module(name)
            origin = getattr(module, "__file__", None) or ""
            version = getattr(sys.modules.get(name.split(".")[0]), "__version__", None)
            record["imports"][name] = dict(ok=True, origin=origin, env=env_of(origin), version=str(version) if version else None)
        except Exception as error:  # noqa: BLE001 — 불러오기 실패도 결과
            record["imports"][name] = dict(ok=False, error=f"{type(error).__name__}: {str(error)[:200]}")
    if role == "qwen":
        try:
            from genai_lab.qwen_pose_settings import check_runtime_versions
            record["qwen_version_contract"] = dict(ok=True, versions=check_runtime_versions())
        except Exception as error:  # noqa: BLE001
            record["qwen_version_contract"] = dict(ok=False, error=str(error)[:300])
        try:
            import torch
            record["cuda_visible_in_probe"] = torch.cuda.is_available()
        except Exception as error:  # noqa: BLE001
            record["cuda_visible_in_probe"] = f"error: {error}"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [k for k, v in record["imports"].items() if not v["ok"]]
    print(role, sys.executable, "base:", sys.base_prefix, "| import 실패:", bad or "없음")


if __name__ == "__main__":
    main()
