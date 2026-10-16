"""D: 고장 임시 복구 2단계: 모델 다시 받기 + 예전 기록 SHA 대조(사용자 승인 "진행", 2026-10-16).

받는 곳은 가상 D:(subst D: C:\\genai-temp-d) 아래 기존 경로 그대로. 예전 SHA가 기록된 파일은 대조하고, 다르면 멈춘다.
인증서는 REQUESTS_CA_BUNDLE/SSL_CERT_FILE(Windows 신뢰 저장소 PEM)로 검증한다.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path("D:/genai-cache")
OUT = Path(__file__).resolve().parent
report = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def save():
    (OUT / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def check(name, path, expected):
    actual = sha(path)
    ok = expected is None or actual == expected
    report[name] = dict(path=str(path), size=Path(path).stat().st_size, sha256=actual, expected=expected,
                        status="match" if expected and ok else ("no_record" if expected is None else "MISMATCH"))
    save()
    print(name, report[name]["status"], flush=True)
    if not ok:
        sys.exit(f"SHA 불일치: {name}")


steps = sys.argv[1:] or ["qwen", "gguf", "l2d", "wd", "dwpose"]
if "qwen" in steps:
    manifest = json.loads((ROOT / "configs/qwen_model_files.json").read_text(encoding="utf-8"))["files"]
    target = CACHE / "models/Qwen-Image-Edit-2511"
    snapshot_download("Qwen/Qwen-Image-Edit-2511", revision="6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
                      local_dir=str(target), allow_patterns=list(manifest))
    for rel, expected in manifest.items():
        check("qwen/" + rel, target / rel, expected)
if "gguf" in steps:
    path = hf_hub_download("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_K_M.gguf",
                           revision="0d33d9692b4b26212297240d87b0d4719aa4fd06", local_dir=str(CACHE / "models/Qwen-Image-Edit-2511-GGUF"))
    settings = json.loads((ROOT / "configs/qwen_edit_settings.json").read_text(encoding="utf-8"))
    check("gguf", path, settings["gguf_sha256"])
if "l2d" in steps:
    path = hf_hub_download("24yearsold/l2d_sam_iter2", "checkpoint-18000.pt", revision="0b0608310fb7a89e32ecd7397147a249540cf1e7",
                           cache_dir=str(CACHE / "huggingface"))
    check("l2d_sam", path, "dda82b916f6c340bc5a6c3a077ff1214f9906906abe3707b1076d55b508f253f")
if "wd" in steps:
    for name in ("model.onnx", "selected_tags.csv"):
        path = hf_hub_download("SmilingWolf/wd-vit-tagger-v3", name, cache_dir=str(CACHE / "huggingface"))
        check("wd/" + name, path, None)
if "dwpose" in steps:
    target = CACHE / "huggingface/easy-dwpose/checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        path = hf_hub_download("yzd-v/DWPose", name, cache_dir=str(CACHE / "huggingface"))
        shutil.copyfile(path, target / name)
        check("dwpose/" + name, target / name, None)
print("MODELS DONE", flush=True)
