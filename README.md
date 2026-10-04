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
→ 사용자가 얼굴·성별·의상 설명 확인 (꼬리 외형 문구는 선택)
→ 승인 정보로 자세 없이 이미지 4장 생성
→ 사용자가 결과를 비교하고 캐릭터·의상·노출 확인 (해당 시 꼬리·귀도 확인)
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

## 프로젝트 구조

2026-10-04 현재 코드 기준입니다. 전체 파일 목록 대신 **현재 작업실에서 읽을 파일을 실행 순서로** 묶었습니다. 아래는 역할 구분이며 실제 폴더를 옮긴 것은 아닙니다.

```text
genai-lab/
├─ gui_main.py                         # 앱 시작·입력 선택·작업실/별도 편집 진입
├─ genai_lab/
│  ├─ studio_ui.py                     # 작업실 화면 배치
│  ├─ studio_controller.py             # 입력 → 확인 → 생성 → 검토 → 저장 연결
│  ├─ studio_generation.py             # 분석 프로세스·요청 조립·승인/저장 관리
│  ├─ studio_analysis.py               # CPU 분석: 얼굴 크롭·캐릭터/의상 태그
│  │
│  ├─ onepass_character.py             # 얼굴·머리 크롭, 외형·체형 분석 규칙
│  ├─ onepass_gender.py                # 사용자가 확인한 성별 조건
│  ├─ onepass_prompt.py                # 프롬프트·청크 계획·확인된 꼬리/귀 문구
│  ├─ onepass_garment_vocabulary.py     # 의상 명사·피복·주머니 어휘
│  ├─ reference_tag_policy.py          # 태그 정규화·입력 문구 제한
│  ├─ onepass_prompt_tokenizers.py     # 로컬 토크나이저 연결
│  ├─ onepass_generation.py            # SDXL 생성·단계별 조건·실행 기록
│  ├─ onepass_pose.py                  # 자세 입력 검사·정책·제어 이미지 준비
│  ├─ onepass_input_backends.py        # 검출·분할 도구 연결
│  ├─ onepass_*_settings.py            # 입력·프롬프트·생성 설정
│  │
│  ├─ qwen_pose_gui.py                 # 저장 이미지의 별도 자세 편집 화면
│  ├─ qwen_preservation*.py            # 보존 항목 분석·사용자 확인 자료
│  ├─ qwen_pose_prompt.py              # 확인된 보존 항목으로 편집 지시문 조립
│  ├─ qwen_pose_edit.py                # 별도 프로세스 실행·취소·결과 처리
│  ├─ qwen_pose_worker.py              # Qwen 환경에서 모델 실행
│  ├─ qwen_pose_review.py              # 편집 결과 검토 기록
│  ├─ qwen_pose_settings.py            # Qwen 실행 환경·모델 설정
│  └─ qwen_record_io.py                # 프로세스 사이 실행 기록 파일 처리
├─ configs/                            # 기존 생성 설정·공유 어휘 자료
├─ scripts/                            # 분석 실행기·검증·기존 CLI 도구
├─ tests/                              # 단위·GUI 흐름·가짜 백엔드 회귀 테스트
│  └─ fixtures/                        # 재현 검증용 입력·기대 결과
├─ docs/                               # 제품 계약·코드 흐름·결정·백로그
├─ inputs/                             # 로컬 입력 자료·기존 배치 요청
├─ outputs/                            # 실행 기록·시험 코드·측정·생성 산출물
└─ run.py                              # 기존 프롬프트 배치 실험 도구
```

### 어디부터 읽으면 되나요?

| 확인하려는 기능 | 먼저 읽을 파일 | 이어서 읽을 파일 |
|---|---|---|
| 버튼을 누른 뒤 전체 순서 | [studio_controller.py](genai_lab/studio_controller.py)의 `StudioController.start` | [studio_generation.py](genai_lab/studio_generation.py) |
| 화면 배치·사용자 확인창 | [studio_ui.py](genai_lab/studio_ui.py) | [studio_controller.py](genai_lab/studio_controller.py)의 `confirm_inputs`, `confirm_result` |
| 캐릭터·의상에서 정보 추출 | [studio_analysis.py](genai_lab/studio_analysis.py) | [onepass_character.py](genai_lab/onepass_character.py), [clothing_analysis.py](genai_lab/clothing_analysis.py) |
| 성별·의상·꼬리 설명의 전달 | [onepass_prompt.py](genai_lab/onepass_prompt.py) | [onepass_gender.py](genai_lab/onepass_gender.py), [reference_tag_policy.py](genai_lab/reference_tag_policy.py) |
| 실제 이미지 생성과 실행 기록 | [onepass_generation.py](genai_lab/onepass_generation.py) | [onepass_generation_settings.py](genai_lab/onepass_generation_settings.py) |
| 후보 선택·승인·저장 | [studio_generation.py](genai_lab/studio_generation.py)의 `StudioResults` | [test_studio_generation.py](tests/test_studio_generation.py) |
| 저장 이미지의 Qwen 자세 편집 | [qwen_pose_gui.py](genai_lab/qwen_pose_gui.py) | [qwen_pose_edit.py](genai_lab/qwen_pose_edit.py) → [qwen_pose_worker.py](genai_lab/qwen_pose_worker.py) |

작업실 실행 환경 경로는 `studio_generation.py`의 `StudioRuntime`에서 관리하고, `GENAI_STUDIO_RUNTIME`이 가리키는 JSON으로 바꿀 수 있습니다. 1회 생성의 설정은 `onepass_input_settings.py`, `onepass_prompt_settings.py`, `onepass_generation_settings.py`에 나뉘어 있습니다. `configs/animagine.yaml`만 바꾼다고 작업실의 모든 설정이 바뀌지는 않습니다.

### 현재 연결된 경로와 남아 있는 경로

- **현재 이미지 만들기:** `gui_main.start_generation → StudioController → studio_generation → onepass_generation`. 캐릭터·의상을 함께 생성하며, 현재 GUI 요청은 자세 미적용입니다. 꼬리·귀가 검출되면 결과 확인 항목이 추가됩니다.
- **별도 자세 편집:** `open_qwen_pose_editor → QwenPoseDialog → qwen_pose_edit → qwen_pose_worker`. 현재는 완성 이미지와 준비된 골격 PNG를 직접 선택합니다. 자세 사진의 자동 골격 확인 화면까지 연결됐다는 뜻은 아닙니다.
- **이전 Base·의상 정밀화 경로:** `start_legacy_generation`, `generation_orchestrator.py`, `generator.py` 등은 남아 있지만 현재 만들기 버튼의 기본 경로가 아닙니다. 기존 CLI와 시험 도구도 작업실과 구분합니다.
- **시험·검증 자료:** `outputs/`의 코드는 제품에 자동 연결되지 않습니다. 다만 현재 머리 검출기 캐시 기본 경로(`StudioRuntime.head_cache`)가 이 폴더 아래를 참조하므로, 단순 임시 폴더로 보고 통째로 지우면 안 됩니다.

자세한 호출 순서는 [CODE_FLOW.md](docs/CODE_FLOW.md)를 보세요. [STRUCTURE.md](docs/STRUCTURE.md)에는 이전 경로와 과거 목표 구조도 남아 있으므로, 현재 작업실의 진입점은 위 구조도와 코드 흐름 지도를 기준으로 읽습니다.

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