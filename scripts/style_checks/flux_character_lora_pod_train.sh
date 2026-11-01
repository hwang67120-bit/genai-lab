#!/usr/bin/env bash
# S34 캐릭터 기억 LoRA 학습(클라우드 A100): 자료 /root/char-data(character_lora_data_s34.py: 원본 변형 + 자동 선별 그림 + metadata.jsonl).
# 그림체 LoRA(S26)와 같은 공식 스크립트·기본판(klein-base-4B a3b4f484), 짧게(600단계, 200마다 저장), rank 16. 크기 묶음 3개(세로·정사각·가로 — 원본 자르기 때문).
set -euo pipefail
BASE_REV=a3b4f4849157f664bdbc776fd7453c2783562f4d
STEPS=${STEPS:-600}
TRIGGER=${TRIGGER:-"f1chr character."}
export PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_XET=1
pip uninstall -y -q torchaudio || true
pip install -q torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q diffusers==0.38.0 transformers==4.57.6 accelerate==1.15.0 safetensors huggingface_hub==0.36.2 pillow sentencepiece peft==0.21.2 datasets tensorboard
curl -sSfL -o /root/train_dreambooth_lora_flux2_klein.py \
  https://raw.githubusercontent.com/huggingface/diffusers/v0.38.0/examples/dreambooth/train_dreambooth_lora_flux2_klein.py
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("black-forest-labs/FLUX.2-klein-base-4B", revision="$BASE_REV", local_dir="/root/klein-base", token=False)
PY
rm -rf /root/klein-base/.cache /root/char-lora
date
accelerate launch --mixed_precision=bf16 --num_processes=1 /root/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path=/root/klein-base \
  --dataset_name=/root/char-data --image_column=image --caption_column=text \
  --instance_prompt="$TRIGGER" \
  --output_dir=/root/char-lora --mixed_precision=bf16 \
  --aspect_ratio_buckets="1360,1024;1024,1024;832,1248" \
  --gradient_checkpointing --cache_latents \
  --train_batch_size=1 --gradient_accumulation_steps=1 --guidance_scale=1 \
  --rank=16 --lora_alpha=16 --optimizer=adamW --learning_rate=1e-4 \
  --lr_scheduler=constant --lr_warmup_steps=50 \
  --max_train_steps=$STEPS --checkpointing_steps=200 --seed=0 --report_to=tensorboard
date
ls -la /root/char-lora
echo CHAR_TRAIN_DONE
