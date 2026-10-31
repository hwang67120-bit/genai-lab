#!/usr/bin/env bash
# 그림체 자료 1단계(D-233): Qwen-Image-2512(Apache-2.0, 20B) 문장→그림을 클라우드에서. /workspace는 건드리지 않는다.
# 컨테이너 디스크가 50GB라 나눠 둔다: transformer(40.9GB)는 디스크 /root/qi/model, 문장 인코더·VAE·토크나이저와 venv는 메모리(/dev/shm, 55GB).
# 사용자 토큰 없이 공개 모델을 받는다. diffusers는 이 PC와 같은 0.38.0. pip 캐시를 남기지 않는다.
set -euo pipefail
REV=25468b98e3276ca6700de15c6628e51b7de54a26
mkdir -p /root/qi/model /dev/shm/qi
python3 -m venv /dev/shm/qi/venv
. /dev/shm/qi/venv/bin/activate
export PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_XET=1
pip install -q --upgrade pip
pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q diffusers==0.38.0 transformers==4.57.6 accelerate==1.15.0 safetensors huggingface_hub==0.36.2 pillow sentencepiece
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen-Image-2512", revision="$REV", local_dir="/root/qi/model", token=False, allow_patterns=["transformer/*"])
snapshot_download("Qwen/Qwen-Image-2512", revision="$REV", local_dir="/dev/shm/qi/rest", token=False, ignore_patterns=["transformer/*"])
PY
for item in /dev/shm/qi/rest/*; do ln -sfn "$item" "/root/qi/model/$(basename "$item")"; done
echo "$REV" > /root/qi/model-revision.txt
rm -rf /root/qi/model/.cache /dev/shm/qi/rest/.cache
df -h / /dev/shm | tail -2
python -c "import torch,diffusers;print('versions',torch.__version__,diffusers.__version__,torch.cuda.get_device_name(0))"
echo SETUP_DONE
