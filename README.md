# GenAI Lab

캐릭터 참조 이미지와 의상 참조 이미지를 받아, 기준 캐릭터가 새로운 의상을 입은 이미지를 생성하는 Windows용 로컬 이미지 생성 애플리케이션입니다.

주요 사용자는 다음과 같습니다.

- 좋아하는 캐릭터의 팬아트를 만들고 싶은 사용자
- 자신이 만든 캐릭터에게 새로운 의상을 입히고 싶은 사용자
- 캐릭터의 얼굴·체형·헤어 특징을 유지하면서 의상만 바꾼 후보를 비교하고 싶은 사용자

## 현재 제품 흐름

```text
캐릭터 참조 이미지 + 의상 참조 이미지
→ 두 이미지의 영역 마스크 분석
→ 마스크 영역별 태그와 프롬프트 추출
→ 사용자가 입력과 추출 조건 확인
→ 두 참조 이미지와 승인된 태그·프롬프트로 새 이미지 생성
→ 기술적 손상 검사와 유사도 진단
→ 사용자가 결과 승인 또는 재생성
→ 승인한 이미지와 실행 기록 저장
```

마스크는 참조 이미지에서 캐릭터·얼굴·헤어·선택적 동물 특징·의상 정보를 분리하기 위해 사용합니다. 서로 다른 이미지의 픽셀을 잘라 붙이는 합성에는 사용하지 않습니다.

현재 활성 제품 계약과 단계별 책임은 [FLOW.md](docs/FLOW.md)에만 정의합니다. 다른 문서의 과거 실험, 실패 원인, 측정 결과는 활성 계약이 아니라 개발 기록입니다.

## 실행

PowerShell에서 다음 명령으로 GUI를 실행합니다.

```powershell
& "D:\genai-cache\venv\Scripts\python.exe" "\\192.168.0.109\win_g\genai-lab\gui_main.py"
```

제품 GUI·CLI·GPU 검증은 모두 `genai_lab/generation_orchestrator.py`의 `GenerationOrchestrator`를 통해 같은 생성 서비스를 호출해야 합니다. 제품 CLI 진입점은 `scripts/run_product_generation.py`입니다.

루트의 `run.py`는 설정 검사 함수와 과거 프롬프트 파일 배치 생성을 제공하는 실험용 도구입니다. 승인 fingerprint, 요청별 `GenerationRunContext`, 8단계 비교 증거를 만들지 않으므로 제품 생성 결과나 제품 GPU 검증에 사용하지 않습니다.

## 문서

- [활성 제품 계약과 전체 흐름](docs/FLOW.md)
- [데이터 객체 정의](docs/DATA_MODELS.md)
- [프로젝트 구조](docs/STRUCTURE.md)
- [결정 기록](docs/DECISIONS.md)
- [문제 해결 기록](docs/TROUBLESHOOTING.md)
- [수치와 증거 작성 규칙](docs/DOCUMENTATION_RULES.md)
- [참조 일관성 실패와 Anchor 전환 기록](docs/REFERENCE_CONSISTENCY_POSTMORTEM.md)

## 구현 원칙

- 이전 요청의 캐릭터·성별·동물귀·꼬리·의상 정보는 새 요청에 상속하지 않습니다.
- 캐릭터와 의상 태그는 각각의 참조 이미지에서 독립적으로 추출합니다.
- 캐릭터에 존재하지 않는 동물귀·꼬리 같은 특징을 공통 기본값으로 강제하지 않습니다.
- 생성 결과의 유사도는 비교와 미세조정 방향을 위한 진단값으로 공개합니다.
- 실제 이미지 손상만 자동 차단하고, 미적 유사성과 최종 사용 가능 여부는 사용자가 판단합니다.
- 승인되지 않은 이미지는 미세조정 데이터로 사용하지 않습니다.
- 생성 방법을 바꿀 때는 같은 입력·Seed로 비교하고 결과와 실행 조건을 함께 남깁니다.

## 주요 근거

- [Diffusers IP-Adapter](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)
- [Diffusers Image-to-Image](https://huggingface.co/docs/diffusers/using-diffusers/img2img)
- [Diffusers Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)
- [FLUX.2 Klein 모델 카드](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [CLIP 논문](https://arxiv.org/abs/2103.00020)