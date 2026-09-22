# Qwen-Image 논문 검토와 GPU 회귀 결과

## 논문에서 확인한 일관성 메커니즘

Qwen-Image Technical Report는 편집 일관성을 위해 T2I와 TI2I 외에 I2I
재구성 학습을 포함한다. 원본 이미지를 Qwen2.5-VL과 VAE에 각각 넣어 의미
표현과 재구성 표현을 함께 사용한다. 이 구조는 기존 Animagine 또는
IP-Adapter에 프롬프트를 추가하는 방식과 다르므로 독립 편집 엔진으로
실험했다.

- 논문: https://arxiv.org/abs/2508.02324
- 모델: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- 공식 Diffusers 파이프라인:
  https://huggingface.co/docs/diffusers/api/pipelines/qwenimage
- 공식 저메모리 실행 예제:
  https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2509.py

## 로컬 구현

scripts/qwen_image_edit_smoke.py는 Qwen-Image-Edit-2509를 로컬 CUDA에서
실행한다. RTX 4060 8GB에 맞춰 DiffSynth-Studio의 레이어·디스크 오프로딩을
사용한다. 이미지 병합, hard paste, 마스크 합성은 사용하지 않는다.

## 2026-09-19 GPU 결과

동일한 승인 Base, 격리된 의상 보드, 기존 Seed 1767824957을 사용했다.

- 해상도: 320×512
- 단계: 20
- 추론 시간: 1,820.05초
- 전체 시간: 1,900.64초
- 실행 상태: 완료
- 결과 SHA-256:
  a2b99fb99e187ec0e5665814151e0e9eb3b64a8d4c1ead4e0cc01b36cccec255
- 결과:
  outputs/qwen-product-contract-smoke/qwen-image-edit-smoke.png
- 측정:
  outputs/qwen-product-contract-smoke/qwen-image-edit-smoke.json

결과는 승인 Base의 얼굴, 헤어, 동물귀, 꼬리, 자세, 전신 구도를 보존하지
못했다. 격리된 의상판의 몸통을 편집 대상처럼 확대 생성했다. 따라서 생성
프로세스는 정상 완료됐지만 제품 계약에는 실패했다.

## 적용 결정

Qwen 논문의 메커니즘과 로컬 실행기는 연구 증거로 유지한다. Qwen을 GUI와
자동 fallback 순서에서는 제외한다. 제품 경로는 다시 아래 순서로 고정한다.

    승인된 Animagine Base
    → FLUX.2 Klein
    → 실패 시 승인 Base와 Seed 고정
    → 모두 실패하면 승인 Base·Seed 고정

Qwen을 다시 자동 경로에 넣으려면 별도 평가에서 얼굴, 헤어, 동물귀, 꼬리,
자세, 구도, 의상 게이트를 모두 통과해야 한다. 생성 성공만으로 승격하지
않는다.
