#!/usr/bin/env bash
# S33: Qwen3-14B(Apache-2.0, 29.5GB, 사용자 토큰 없이)를 받아 WD 태그 종류 표를 만든다. 컨테이너 디스크만, /workspace는 건드리지 않음.
set -euo pipefail
REV=40c069824f4251a91eefaf281ebe4c544efd3e18
export PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_XET=1
pip uninstall -y -q torchaudio || true
pip install -q torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers==4.57.6 accelerate==1.15.0 safetensors huggingface_hub==0.36.2
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-14B", revision="$REV", local_dir="/root/qwen3-14b", token=False)
PY
echo "$REV" > /root/qwen3-14b/revision.txt
rm -rf /root/qwen3-14b/.cache
python3 /root/tag_kinds_pod_s33.py /root/qwen3-14b /root/selected_tags.csv /root/tag-kinds-qwen3-14b.json
echo S33_DONE
