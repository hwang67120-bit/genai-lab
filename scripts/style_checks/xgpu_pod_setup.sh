#!/usr/bin/env bash
# 다른 GPU 일관성 시험(D-231) Pod 준비: 이 PC와 같은 패키지 버전 + 같은 FLUX.2-klein-4B 리비전. 컨테이너 디스크(/root/xgpu)만 쓴다.
# /workspace(예전 작업 볼륨)는 건드리지 않는다. 사용자 토큰 없이 공개 모델을 받는다.
set -euo pipefail
mkdir -p /root/xgpu && cd /root/xgpu
python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q diffusers==0.38.0 transformers==4.57.6 bitsandbytes==0.50.2 accelerate==1.15.0 "safetensors==0.9.0rc1" \
    numpy==2.2.6 pillow==12.3.0 huggingface_hub==0.36.2 tokenizers==0.22.2 scipy
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("black-forest-labs/FLUX.2-klein-4B", revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
                      local_dir="/root/xgpu/flux", token=False, ignore_patterns=["*.jpg", "*.png", "flux-2-klein-4b.safetensors"])
print("flux", p)
PY
python -c "import torch,diffusers,transformers,bitsandbytes;print('versions',torch.__version__,diffusers.__version__,transformers.__version__,bitsandbytes.__version__, torch.cuda.get_device_name(0))"
echo SETUP_DONE
