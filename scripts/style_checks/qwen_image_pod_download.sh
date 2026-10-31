#!/usr/bin/env bash
# Qwen-Image-2512 받기(D-233). /dev/shm은 noexec라 venv를 못 둔다 → 기본 Python(이미 이 PC와 같은 버전) 사용, /dev/shm에는 데이터만.
# transformer(40.9GB) → /root/qi/model(디스크), 나머지(문장 인코더 16.6GB 등) → /dev/shm/qi/rest, 링크로 한 폴더처럼.
set -euo pipefail
REV=25468b98e3276ca6700de15c6628e51b7de54a26
mkdir -p /root/qi/model /dev/shm/qi
export HF_HUB_DISABLE_XET=1
/usr/bin/python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen-Image-2512", revision="$REV", local_dir="/dev/shm/qi/rest", token=False, ignore_patterns=["transformer/*"])
snapshot_download("Qwen/Qwen-Image-2512", revision="$REV", local_dir="/root/qi/model", token=False, allow_patterns=["transformer/*"])
PY
for item in /dev/shm/qi/rest/*; do ln -sfn "$item" "/root/qi/model/$(basename "$item")"; done
echo "$REV" > /root/qi/model-revision.txt
rm -rf /root/qi/model/.cache /dev/shm/qi/rest/.cache
df -h / /dev/shm | tail -2
ls -la /root/qi/model
echo DOWNLOAD_DONE
