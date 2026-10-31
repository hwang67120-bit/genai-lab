#!/usr/bin/env bash
# S26 그림체 LoRA 학습(클라우드 A100). 자료는 /root/s26-data(이 PC에서 flux_style_lora_s26.py로 만든 그림 + metadata.jsonl).
# 기본판 FLUX.2-klein-base-4B(Apache-2.0, a3b4f484, 사용자 토큰 없이)로 학습 → 앱의 빠른 판(e7b7dc27, 4단계)에 붙여 이 PC에서 확인.
# diffusers v0.38.0 공식 예제 스크립트(이 PC·Pod와 같은 버전). A100은 fp8 학습(8.9 이상) 불가 → bf16. 크기는 앱 결과와 같은 1024x1360.
# 컨테이너 디스크 50GB: 그림 생성이 끝난 Qwen-Image 모델(/root/qi, /dev/shm/qi)은 지운다(Pod 안의 임시 사본, 이 PC 파일 아님).
set -euo pipefail
BASE_REV=a3b4f4849157f664bdbc776fd7453c2783562f4d
STEPS=${STEPS:-2000}
export PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_XET=1
rm -rf /root/qi /dev/shm/qi
pip install -q peft==0.21.2 datasets tensorboard
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
  --dataset_name=/root/s26-data --image_column=image --caption_column=text \
  --instance_prompt="gnlbq4 style illustration." \
  --output_dir=/root/s26-lora \
  --mixed_precision=bf16 \
  --aspect_ratio_buckets="1360,1024" \
  --random_flip \
  --gradient_checkpointing \
  --cache_latents \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --guidance_scale=1 \
  --rank=32 --lora_alpha=32 \
  --optimizer=adamW \
  --learning_rate=1e-4 \
  --lr_scheduler=constant --lr_warmup_steps=100 \
  --max_train_steps=$STEPS \
  --checkpointing_steps=250 \
  --seed=0 \
  --report_to=tensorboard
date
ls -la /root/s26-lora
echo TRAIN_DONE
