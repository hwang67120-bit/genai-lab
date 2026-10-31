"""FLUX.2-klein-base-4B 받기(D-229, 사용자 10/31 승인): Apache-2.0, 공식 저장소, 익명(사용자 HF 토큰을 읽지 않음).
transformer와 작은 설정 파일만 받고, SHA가 같은 큰 파일(문장 인코더 2개·토크나이저·VAE)은 지금 klein 폴더에서 하드링크로 잇는다.
사용: python download_klein_base.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
from genai_lab.child_env import isolate_hf_credentials
isolate_hf_credentials()
from run import configure_system_certificates
configure_system_certificates()
from huggingface_hub import HfApi, snapshot_download

REPO = "black-forest-labs/FLUX.2-klein-base-4B"
TARGET = Path(r"D:\genai-cache\models\FLUX.2-klein-base-4B")
SOURCE = Path(r"D:\genai-cache\models\FLUX.2-klein-4B-e7b7dc27")
SHARED = ("text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors",
          "tokenizer/tokenizer.json", "vae/diffusion_pytorch_model.safetensors")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    info = HfApi().model_info(REPO, files_metadata=True, token=False)
    remote = {s.rfilename: s for s in info.siblings}
    path = snapshot_download(REPO, revision=info.sha, local_dir=str(TARGET), token=False,
                             ignore_patterns=list(SHARED) + ["*.jpg", "flux-2-klein-base-4b.safetensors"])
    print("downloaded to", path, flush=True)
    for name in SHARED:
        target = TARGET / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(SOURCE / name, target)
        assert sha(target) == remote[name].lfs.sha256, name
    weight = TARGET / "transformer/diffusion_pytorch_model.safetensors"
    ok = sha(weight) == remote["transformer/diffusion_pytorch_model.safetensors"].lfs.sha256
    record = dict(repo=REPO, revision=info.sha, license="Apache-2.0", transformer_sha256_ok=ok,
                  shared_from=str(SOURCE), shared=list(SHARED))
    (TARGET / "download-record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
