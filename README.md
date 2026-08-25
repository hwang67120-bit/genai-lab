# GenAI Lab

캐릭터·의상·자세 참조 이미지를 바탕으로 기존 디자인을 유지하며 제한된 영역만 합성하고, 사용자가 결과를 승인하는 Windows 설치형 이미지 앱 프로젝트입니다. 최종 앱은 서버 없이 사용자의 PC에서 실행합니다.

## 개발 방식

- 기간은 짧게 잡되 결과 이미지의 품질 기준은 낮추지 않습니다.
- 검증된 오픈소스와 공식 기능이 있으면 직접 다시 만들지 않고 먼저 활용합니다.
- 바이브 코딩은 AI가 제시한 코드를 그대로 붙이는 방식이 아니라, 계획을 먼저 세우고 빠르게 구현·검증하는 도구로 사용합니다.
- 추가한 코드와 오픈소스는 목적, 입력, 처리, 출력, 실패 조건과 확인 방법을 한글로 설명할 수 있어야 합니다.
- 실제 오류를 해결하거나 중요한 판단을 확정하면, AI의 평가를 덧붙이지 않고 대화에서 확인된 내용만 `docs/TROUBLESHOOTING.md`에 이어서 기록합니다.
- 기능을 합의하거나 구현한 작업은 같은 작업 안에서 메인 `README.md`의 현재 단계, 실행 흐름과 남은 일을 함께 최신화합니다.
- 공식 문서, 모델 문서와 라이선스를 확인하지 않은 도구는 프로젝트에 추가하지 않습니다.
- 품질 문제가 실제로 확인되기 전에는 LoRA, ControlNet 또는 새로운 기능을 미리 추가하지 않습니다.
- 현재 MVP는 캐릭터 디자인 유지, 일반 의상 참조 합성, 자세 참조 적용과 사용자 승인 저장까지 포함합니다. 자유 생성은 자세 단계 완료 이후 별도 모드로 검토합니다.

## 먼저 알아둘 점

- 실행 환경: Windows, Python 3.10.6, NVIDIA RTX 4060 8GB
- 비교 모델: Stable Diffusion 1.5, 512×512
- 실제 작업 모델: Animagine XL 3.1, 전신 시험 해상도 768×1344
- 처리 방식: 후보 3장을 한 장씩 순서대로 생성
- 제외 범위: Java, 백엔드, 웹 서버, DB, API, 로그인, 개인정보 및 보안 기능, 텍스트 생성
- 현재 단계: 캐릭터 디자인 유지와 얼굴·손 부분 보정은 연결됐습니다. CatVTON 의상 합성은 실행되지만 일반 의상 이미지를 자동으로 추출·분석하지 않아 잘못된 전신 마스크가 생길 수 있으므로, 의상 추출·디자인 분석·사전 마스크 승인이 다음 구현 대상입니다.

전문 용어는 한글 뜻을 먼저 적습니다. 예를 들어 **그림체 학습용 추가 가중치(LoRA)**는 큰 모델 전체를 다시 학습하지 않고 그림체에 필요한 작은 부분만 학습하는 방법입니다.

## 전체 흐름

```text
캐릭터 기준 이미지 + 화면 범위
                ↓
참조 화질 확인과 필요한 보정 승인
                ↓
Animagine XL + 참조 특징 전달 장치(IP-Adapter)
                ↓
캐릭터 기본 후보
                ↓
[다음 구현] 일반 의상 이미지 입력
                ↓
의상 위치 추출 → 디자인 분석 → 사용자 승인
                ↓
캐릭터 신체 비교 → 변경 마스크 사전 검사
                ↓
CatVTON 제한 합성
                ↓
얼굴·손 제한 영역 보정
                ↓
사용자가 전후 결과 판단
        ├─ 승인 → 저장 여부 선택
        └─ 거절 → 저장하지 않고 재생성
```

현재 GUI는 기준 이미지 화질 확인, 필요한 확대 복원 승인, 화면 범위 선택, 선택 의상 이미지와 종류 입력, 의상 합성 전후 선택, 얼굴·손 보정 전후 선택, 후보 승인과 저장 승인을 지원합니다. 현재 의상 이미지는 추출 없이 CatVTON에 전달되므로 사람이나 배경이 포함된 일반 이미지는 의상 영역이 잘못 잡힐 수 있습니다. 이 문제가 해결되기 전까지 의상 결과와 마스크는 시험 결과이며 사용자의 승인이 필요합니다.

### 확정된 개발 순서

1. 캐릭터 디자인 유지와 승인 전 저장 금지를 유지합니다.
2. 일반 의상 이미지에서 의상 위치를 추출하고 디자인을 분석합니다.
3. 추출 의상과 적용 예정 마스크를 사용자가 승인한 뒤 CatVTON을 실행합니다.
4. 의상 단계가 안정되면 자세 참조 적용을 구현합니다.
5. 자세 단계까지 완료된 이후에만 자유 생성 모드를 별도 흐름으로 검토합니다.

현재 프로젝트는 텍스트로 새로운 디자인을 만드는 자유 생성기가 아니라 참조 기반 제한 합성 앱입니다. 보이지 않는 부분을 연결하는 최소 추정만 허용하고, 캐릭터·의상·자세 디자인의 임의 변경은 허용하지 않습니다.

향후 외부 LLM 또는 멀티모달 API를 연결해도 분석 후보만 반환하게 하며, Python 마스크 검사와 사용자 승인을 우회할 수 없습니다. 외부 API가 신규 생성을 요청하거나 거절되면 현재 합성 가드레일을 풀지 않고 지원하지 않는 요청 또는 제공자 실패 상태로 처리합니다.

사용자 판단 흐름은 [SCENARIOS.md](docs/SCENARIOS.md), 자세한 입력과 출력 흐름은 [FLOW.md](docs/FLOW.md), 블록 사이의 데이터는 [DATA_MODELS.md](docs/DATA_MODELS.md), 각 파일이 필요한 이유는 [STRUCTURE.md](docs/STRUCTURE.md)를 먼저 읽습니다. 프로젝트를 진행하며 막힌 점과 내가 이해하게 된 내용은 [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)에 기록합니다.

## 프로젝트 구조

```text
genai-lab/
├─ README.md
├─ run.py
├─ genai_lab/
│  ├─ model.py              # 모델과 GPU
│  ├─ reference.py          # 참조 화질 검사와 승인 전 확대 복원
│  ├─ detail.py             # 얼굴·손 탐지와 제한된 부분 보정
│  ├─ style.py              # 참조 그림
│  ├─ clothing.py           # 의상 합성 허용 영역과 신체 보호 검사
│  ├─ generator.py          # 한 장씩 생성
│  └─ result.py             # 파일과 실행 기록
├─ scripts/
│  └─ catvton_runner.py     # CatVTON 별도 환경을 한 번 실행하는 연결점
├─ requirements.txt
├─ configs/
│  └─ base.yaml
├─ inputs/
│  ├─ reference/
│  │  └─ README.md
│  └─ prompts.csv
├─ outputs/
│  └─ README.md
├─ docs/
│  ├─ PROJECT_SCOPE.md
│  ├─ SCENARIOS.md
│  ├─ ROADMAP.md
│  ├─ STRUCTURE.md
│  ├─ FLOW.md
│  ├─ DECISIONS.md
│  ├─ DATA_MODELS.md
│  ├─ TROUBLESHOOTING.md
│  └─ GLOSSARY.md
└─ tests/
   └─ test_run.py
```

## 설치

PowerShell에서 프로젝트 폴더로 이동한 뒤 가상 환경을 만듭니다.

```powershell
python -m venv D:\genai-cache\venv
D:\genai-cache\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

PyTorch 설치 명령은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)의 Windows CUDA 12.8 조합을 따릅니다. Diffusers는 Windows에서 Python 3.8~3.11을 지원하며 가상 환경 사용을 권장합니다. [Diffusers 공식 설치 안내](https://huggingface.co/docs/diffusers/main/installation)

현재 Windows Python은 자체 인증서 묶음이 오래되어 모델 저장소 연결이 실패할 수 있습니다. 프로그램은 인증서 검사를 끄지 않고 `truststore`로 Windows 인증서 저장소를 사용합니다. [truststore 공식 PyPI 안내](https://pypi.org/project/truststore/)

## 실행 전 확인

```powershell
python run.py --config configs/base.yaml --check-only
```

이 명령은 모델을 내려받지 않고 Python, GPU, 설정과 입력 파일을 검사합니다.

## 기준 이미지 한 장 생성

`configs/base.yaml`에서 다음 값을 유지합니다.

```yaml
style:
  enabled: false
generation:
  limit: 1
```

실행합니다.

```powershell
python run.py --config configs/base.yaml
```

첫 실행에는 모델을 `D:\genai-cache\huggingface`로 내려받으므로 시간이 걸립니다. 결과 폴더에는 PNG 한 장과 `result.json`이 생깁니다.

## 참조 그림 적용

1. `inputs/reference/style.png`에 참조 그림을 넣습니다.
2. `configs/animagine.yaml`에서 `style.enabled`를 `true`로 바꿉니다.
3. 기능 확인은 `generation.limit: 1`, 후보 비교는 `generation.limit: 3`으로 실행합니다.

참조 그림 특징 전달 장치(IP-Adapter)는 텍스트와 함께 이미지를 조건으로 사용할 수 있습니다. 실제 작업 모델은 `h94/IP-Adapter`의 SDXL용 `ip-adapter_sdxl.bin`을 사용합니다. [Diffusers IP-Adapter 안내](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)

## Animagine XL 한 장 시험

기존 SD 1.5 설정은 비교용으로 유지합니다. Animagine XL과 참조 그림 장치로 한 장을 먼저 생성합니다.

```powershell
& "D:\genai-cache\venv\Scripts\python.exe" .\run.py --config .\configs\animagine.yaml
```

현재 전신 시험 설정은 768×1344, 28단계, 한 장이며 RTX 4060 8GB를 위해 사용하지 않는 모델 부분을 RAM으로 옮깁니다. 기존 IP-Adapter 768×768 시험은 98.707초, 최대 GPU 메모리 약 5.75GB로 완료되었습니다. 새 세부 보정 흐름의 실제 시간과 품질은 사용자 수동 시험으로 확인합니다.

## Windows 화면 실행

```powershell
& "D:\genai-cache\venv\Scripts\python.exe" .\gui_main.py
```

현재 화면의 필수 입력은 캐릭터 기준 이미지 한 장이며, 의상 참조 이미지와 의상 종류는 선택 입력입니다. 화질이 부족하면 확대 복원 결과를 자동 채택하지 않고 원본과 비교해 승인해야 생성할 수 있습니다. 의상 참조를 사용하면 의상 적용 전·허용 마스크·적용 후 이미지를 비교해 선택합니다. 후보 생성 뒤 얼굴·손 보정에 성공하면 보정 전후도 선택하며, 최종 저장 승인 전에는 PNG와 JSON을 만들지 않습니다.

첫 실행에는 공식 Real-ESRGAN 확대 복원 가중치와 얼굴·손 탐지 가중치를 내려받을 수 있습니다. 확대 복원 모델은 `D:\genai-cache\models`, Hugging Face 모델은 설정된 캐시에 보관됩니다. 부분 보정에 실패하면 전체 생성을 실패시키지 않고 보정 전 후보를 유지하며 로그에 원인과 복구 행동을 남깁니다.

설치형 앱을 다른 사람에게 배포하기 전에는 Ultralytics 라이선스 조건에 맞는 공개 방식, 상용 라이선스 또는 탐지기 교체 중 하나를 확정해야 합니다. 로컬 시험 구현이 프로젝트 전체 라이선스를 자동으로 결정하지는 않습니다.

PyTorch 2의 기본 효율적인 주의 처리를 사용하므로 별도의 `enable_attention_slicing()`을 적용하지 않습니다. [Diffusers 메모리 처리 안내](https://huggingface.co/docs/diffusers/api/pipelines/overview)

## 후보 3장 생성

한 장 생성과 참조 그림 적용이 모두 확인된 뒤 `generation.limit: 3`으로 실행합니다. 기본 설정도 3장입니다.

### CatVTON 별도 환경

CatVTON은 현재 앱과 라이브러리를 섞지 않고 `D:\genai-cache\catvton-venv`와 `D:\genai-cache\tools\CatVTON`에서 별도로 실행합니다. 2026-08-25에 RTX 4060으로 공식 상의 예제의 합성, 의상 마스크, 신체 보호 마스크와 보호 합성까지 확인했습니다. 해당 환경이 없으면 의상 합성만 건너뛰고 기본 후보를 유지합니다.

공식 README의 Python 3.9와 최신 개발판 Diffusers 요구 조건은 현재 서로 충돌했습니다. Windows 시험 환경은 Python 3.10.6, PyTorch 2.5.1+cu121과 Diffusers 0.35.2로 고정했습니다. 공식 코드는 1024×768에서 8GB 미만 GPU 메모리를 목표로 하지만, 이 프로젝트는 RTX 4060 8GB의 여유를 위해 576×1024로 시작합니다. 재현 명령과 이유는 [CATVTON_SETUP.md](docs/CATVTON_SETUP.md)에 기록했습니다.

CatVTON 코드와 가중치는 비상업용 조건(CC BY-NC-SA 4.0)이므로 현재 로컬 연구·시험에만 사용합니다. 설치형 앱을 배포하거나 유료로 제공하기 전에는 CatVTON을 교체하거나 별도 허가를 받아야 합니다.

시험에서 일반 의상 이미지를 추출하지 않고 `전신 의상`으로 전달했을 때 몸통·다리·꼬리까지 넓게 잡힌 마스크 위에 회색 천 형태가 합성됐습니다. 파일이 JPG인 것이 직접 원인은 아니며, 원인은 의상 자동 추출·디자인 분석 부재와 애니 캐릭터 신체 분리 오류입니다. 다음 구현은 의상 위치 탐색, 의상 픽셀 추출, 디자인 분석, 캐릭터 신체 비교와 마스크 사전 승인을 CatVTON 앞에 추가합니다.

- [CatVTON 공식 저장소와 설치 안내](https://github.com/Zheng-Chong/CatVTON)
- [CatVTON 공식 실행 예제](https://github.com/Zheng-Chong/CatVTON/blob/main/app.py)
```powershell
python run.py --config configs/base.yaml
```

중간에 실패하면 오류 메시지에 표시된 결과 폴더를 사용하여 이어서 실행합니다.

```powershell
python run.py --config configs/base.yaml --resume outputs\20260816-223000
```

이미 존재하는 `요청번호_시드.png`는 건너뛰고 빠진 이미지만 생성합니다.

## 결과 선택과 다시 생성

생성된 3장을 확인해 마음에 드는 이미지가 있으면 해당 실행 폴더를 보관합니다. 모두 마음에 들지 않으면 `outputs` 아래에서 방금 만든 실행 폴더 이름을 확인한 뒤 그 폴더만 삭제합니다. 그다음 `prompts.csv`의 요청 문장이나 시드를 고치고 다시 실행합니다.

프로그램은 결과를 자동으로 삭제하지 않습니다. 잘못된 폴더 삭제를 막기 위해 사용자가 확인한 실행 폴더만 직접 정리합니다.

## 테스트

```powershell
python -m pytest -q
```

자동 테스트는 모델을 다운로드하지 않고 설정 검사, CSV 읽기, 파일명, 이어서 실행과 한글 오류 메시지를 확인합니다. 실제 GPU와 이미지 품질은 별도의 수동 확인 항목입니다.

## 완료 기준

- 생성 후보에서 기준 캐릭터의 얼굴, 머리, 귀, 꼬리와 색상 특징이 유지됩니다.
- 일반 의상 이미지에서 의상만 추출하고 디자인 분석 결과를 생성 전에 확인할 수 있습니다.
- 의상 변경 마스크가 얼굴, 피부, 손, 발, 꼬리와 장신구를 침범하면 CatVTON 실행 전에 중단합니다.
- 사용자 승인 전에는 생성 후보와 중간 결과를 자동 저장하지 않습니다.
- 의상 합성이 안정된 뒤 자세 참조를 적용해도 캐릭터와 의상 디자인이 유지됩니다.
- 승인 이미지의 참조 파일, 시드, 모델, 설정과 실행 시간을 추적할 수 있습니다.

## 공식 참고 문서

- [PyTorch Windows 설치](https://pytorch.org/get-started/locally/)
- [Diffusers 설치](https://huggingface.co/docs/diffusers/main/installation)
- [Diffusers Stable Diffusion 추론](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation)
- [Diffusers IP-Adapter](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)
- [Diffusers 재현 가능한 시드](https://huggingface.co/docs/diffusers/main/using-diffusers/reusing_seeds)

### 화면 범위 선택

GUI에서 다음 화면 범위를 선택할 수 있습니다.

- 전신: 576×896 세로 화면, 머리부터 발끝과 양발이 보이도록 요청
- 상반신: 768×768 화면, 허리 위와 얼굴이 보이도록 요청
- 얼굴 중심: 768×768 화면, 머리와 어깨 중심으로 요청

전신 선택은 잘림 방지 문구를 강하게 적용하지만 생성형 모델 특성상 100% 보장은 아닙니다. 전신 결과는 머리와 발끝이 모두 화면 안에 있는지 확인하고, 계속 실패할 때만 자세·관절 인식(DWPose) 기반 자동 검사를 다음 단계로 추가합니다.
