# 로컬 FLUX GPU 실행 검증

> 이 실행은 runtime_smoke입니다. 모델 로딩, GPU 추론과 PNG 저장만
> 확인합니다. 제품 파이프라인의 최종 결과 증거는 GUI가 저장한 승인
> 번들을 scripts/run_product_generation.py로 실행한 기록입니다.

이 절차는 FLUX.2 Klein이 외부 이미지 생성 API가 아니라 로컬 모델 파일과
RTX GPU에서 실행되는지 확인합니다.

## 실행

Windows 탐색기에서 다음 파일을 실행합니다.

    \\192.168.0.109\win_g\genai-lab\scripts\run_local_gpu_verification.cmd

또는 PowerShell에서 실행합니다.

    & "\\192.168.0.109\win_g\genai-lab\scripts\run_local_gpu_verification.ps1" -Prompt "현재 참조 이미지에 맞는 캐릭터와 의상 지시문"

실행기는 HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, DIFFUSERS_OFFLINE,
HF_HUB_DISABLE_TELEMETRY를 1로 설정하고 --local-files-only를 전달합니다.
로컬 캐시에 모델이 없으면 외부 API로 대체하지 않고 실패합니다.

각 실행은 outputs\local-gpu-verification-YYYYMMDD-HHMMSS를 만들며 환경,
GPU 사용량, 실제 인수, 종료 코드, 결과 PNG와 JSON을 기록합니다.

공식 자료:

- FLUX.2 Klein: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Diffusers bitsandbytes: https://huggingface.co/docs/diffusers/quantization/bitsandbytes
