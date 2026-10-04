# GenAI Lab

캐릭터 참조 이미지와 의상 참조 이미지를 받아, 기준 캐릭터가 새로운 의상을 입은 이미지를 생성하는 Windows용 로컬 이미지 생성 애플리케이션입니다.

주요 사용자는 다음과 같습니다.

- 좋아하는 캐릭터의 팬아트를 만들고 싶은 사용자
- 자신이 만든 캐릭터에게 새로운 의상을 입히고 싶은 사용자
- 캐릭터의 얼굴·체형·헤어 특징을 유지하면서 의상만 바꾼 후보를 비교하고 싶은 사용자

## 현재 작업실 GUI 흐름

코드를 읽으려면 [작업실 코드 흐름 지도](docs/CODE_FLOW.md)에서 시작하세요.
중심 파일은 [studio_controller.py](genai_lab/studio_controller.py)이며,
사용자 작업 순서대로 입력 준비 → 확인 → 생성 → 선택 → 승인 → 저장을 읽을 수 있습니다.

```text
캐릭터 참조 이미지 + 의상 참조 이미지
→ CPU 분석: 얼굴 크롭·외형·체형·의상 태그 준비
→ 사용자가 얼굴·성별·의상 설명 확인
→ 승인 정보로 자세 없이 이미지 4장 생성
→ 사용자가 결과를 비교하고 캐릭터·의상·노출 확인
→ 승인한 이미지와 검토 기록 저장
```

분석에서 얻는 캐릭터 골격은 얼굴 크롭·체형 분석용입니다.
현재 작업실 생성 요청에는 자세 제어를 넣지 않습니다.
저장 이미지의 Qwen 자세 편집은 별도 사용자 작업이며 자동으로 이어지지 않습니다.

제품 계약은 [FLOW.md](docs/FLOW.md), 현재 구현의 호출 관계는 [CODE_FLOW.md](docs/CODE_FLOW.md)로 구분합니다.
아래 구현 지도는 계약을 변경하거나 과거 시험 결과를 새로 검증한 기록이 아닙니다.

## 실행

PowerShell에서 다음 명령으로 GUI를 실행합니다.

```powershell
& "D:\genai-cache\venv\Scripts\python.exe" "\\192.168.0.109\win_g\genai-lab\gui_main.py"
```

현재 작업실 GUI는 `StudioController → studio_generation → onepass_generation`으로 실행됩니다. 기존 `GenerationOrchestrator`와 `scripts/run_product_generation.py` CLI는 별도 경로이므로, 작업실의 동작 확인에는 작업실 경로의 테스트를 사용합니다.

루트의 `run.py`는 설정 검사 함수와 과거 프롬프트 파일 배치 생성을 제공하는 실험용 도구입니다. 승인 fingerprint, 요청별 `GenerationRunContext`, 8단계 비교 증거를 만들지 않으므로 제품 생성 결과나 제품 GPU 검증에 사용하지 않습니다.

## 문서

- [활성 제품 계약과 전체 흐름](docs/FLOW.md)
- [데이터 객체 정의](docs/DATA_MODELS.md)
- [작업실 코드 읽기 순서와 실제 연결](docs/CODE_FLOW.md)
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