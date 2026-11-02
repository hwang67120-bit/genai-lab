#!/usr/bin/env bash
# S26 그림체 LoRA 학습(클라우드 A100). 자료는 /root/s26-data(이 PC에서 flux_style_lora_s26.py로 만든 그림 + metadata.jsonl).
# 기본판 FLUX.2-klein-base-4B(Apache-2.0, a3b4f484, 사용자 토큰 없이)로 학습 → 앱의 빠른 판(e7b7dc27, 4단계)에 붙여 이 PC에서 확인.
# diffusers v0.38.0 공식 예제 스크립트(이 PC·Pod와 같은 버전). A100은 fp8 학습(8.9 이상) 불가 → bf16. 크기는 앱 결과와 같은 1024x1360.
# 컨테이너 디스크 50GB: 그림 생성이 끝난 Qwen-Image 모델(/root/qi, /dev/shm/qi)은 지운다(Pod 안의 임시 사본, 이 PC 파일 아님).
set -euo pipefail
BASE_REV=a3b4f4849157f664bdbc776fd7453c2783562f4d
STEPS=${STEPS:-2000}
RANK=${RANK:-32}
LR=${LR:-1e-4}
DATA=${DATA:-/root/s26-data}
OUT=${OUT:-/root/s26-lora}
BUCKETS=${BUCKETS:-1360,1024}   # 2판: 잘라낸 옷·머리카락·얼굴 그림이 섞여 세로·정사각·가로 묶음
# 2판(S43): 같은 스크립트를 자료·출력·rank·학습률만 바꿔 두 번 돌린다(DATA=/root/s43-data OUT=/root/s43-lora-a …).
export PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_XET=1
rm -rf /root/qi /dev/shm/qi /root/klein-base/.cache
pip install -q diffusers==0.38.0 transformers==4.57.6 accelerate==1.15.0 peft==0.21.2 datasets tensorboard sentencepiece
pip uninstall -y -q torchaudio   # Pod 기본 이미지의 옛 torchaudio가 torch 2.9.1과 맞지 않아 transformers 가져오기에서 멈춤(학습엔 안 씀)
curl -sSfL -o /root/train_dreambooth_lora_flux2_klein.py \
  https://raw.githubusercontent.com/huggingface/diffusers/v0.38.0/examples/dreambooth/train_dreambooth_lora_flux2_klein.py
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("black-forest-labs/FLUX.2-klein-base-4B", revision="$BASE_REV", local_dir="/root/klein-base", token=False)
PY
rm -rf /root/klein-base/.cache
df -h / | tail -1
date
accelerate launch --mixed_precision=bf16 --num_processes=1 /root/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path=/root/klein-base \
  --dataset_name=$DATA --image_column=image --caption_column=text \
  --instance_prompt="gnlbq4 style illustration." \
  --output_dir=$OUT \
  --mixed_precision=bf16 \
  --aspect_ratio_buckets="$BUCKETS" \
  --random_flip \
  --gradient_checkpointing \
  --cache_latents \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --guidance_scale=1 \
  --rank=$RANK --lora_alpha=$RANK \
  --optimizer=adamW \
  --learning_rate=$LR \
  --lr_scheduler=constant --lr_warmup_steps=100 \
  --max_train_steps=$STEPS \
  --checkpointing_steps=250 \
  --seed=0 \
  --report_to=tensorboard
date
ls -la $OUT
echo TRAIN_DONE
