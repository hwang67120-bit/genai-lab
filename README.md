# GenAI Lab

참조 그림의 색감, 선화, 명암과 질감을 반영한 후보 아이콘 3장을 빠르게 만들고 직접 선택하는 Windows 설치형 앱 프로젝트입니다. 최종 앱은 서버 없이 사용자의 PC에서 실행합니다.

## 개발 방식

- 기간은 짧게 잡되 결과 이미지의 품질 기준은 낮추지 않습니다.
- 검증된 오픈소스와 공식 기능이 있으면 직접 다시 만들지 않고 먼저 활용합니다.
- 바이브 코딩은 AI가 제시한 코드를 그대로 붙이는 방식이 아니라, 계획을 먼저 세우고 빠르게 구현·검증하는 도구로 사용합니다.
- 추가한 코드와 오픈소스는 목적, 입력, 처리, 출력, 실패 조건과 확인 방법을 한글로 설명할 수 있어야 합니다.
- 공식 문서, 모델 문서와 라이선스를 확인하지 않은 도구는 프로젝트에 추가하지 않습니다.
- 품질 문제가 실제로 확인되기 전에는 LoRA, ControlNet 또는 새로운 기능을 미리 추가하지 않습니다.
- 1차 출시는 작업시간 20~35시간을 목표로 하며, 참조 그림 입력·후보 3장 생성·선택 저장·Windows 설치만 포함합니다.

## 먼저 알아둘 점

- 실행 환경: Windows, Python 3.10.6, NVIDIA RTX 4060 8GB
- 비교 모델: Stable Diffusion 1.5, 512×512
- 실제 작업 모델: Animagine XL 3.1, 시험 해상도 768×768
- 처리 방식: 후보 3장을 한 장씩 순서대로 생성
- 제외 범위: Java, 백엔드, 웹 서버, DB, API, 로그인, 개인정보 및 보안 기능, 텍스트 생성
- 현재 단계: 기준 이미지 생성과 한 장 참조 그림 적용 준비

전문 용어는 한글 뜻을 먼저 적습니다. 예를 들어 **그림체 학습용 추가 가중치(LoRA)**는 큰 모델 전체를 다시 학습하지 않고 그림체에 필요한 작은 부분만 학습하는 방법입니다.

## 전체 흐름

```text
참조 그림 + 3개 생성 요청 + 설정
                 ↓
         Animagine XL 3.1
                 ↓
참조 그림 특징 전달 장치(IP-Adapter)
                 ↓
        한 장씩 순서대로 생성
                 ↓
PNG 이미지 3장 + 실행 기록(result.json)
                 ↓
       사용자가 한 장 선택·저장
```

현재는 명령으로 생성 기능을 검증합니다. 참조 그림 적용과 후보 3장 생성이 안정된 뒤 같은 Python 생성 모듈에 Windows 화면과 설치 기능을 연결합니다. 별도의 Java 프로그램이나 서버는 만들지 않습니다.

자세한 입력과 출력 흐름은 [FLOW.md](docs/FLOW.md), 각 파일이 필요한 이유는 [STRUCTURE.md](docs/STRUCTURE.md)를 먼저 읽습니다.

## 프로젝트 구조

```text
genai-lab/
├─ README.md
├─ run.py
├─ genai_lab/
│  ├─ model.py              # 모델과 GPU
│  ├─ style.py              # 참조 그림
│  ├─ generator.py          # 한 장씩 생성
│  └─ result.py             # 파일과 실행 기록
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
│  ├─ ROADMAP.md
│  ├─ STRUCTURE.md
│  ├─ FLOW.md
│  ├─ DECISIONS.md
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

시험 설정은 768×768, 20단계, 한 장이며 RTX 4060 8GB를 위해 사용하지 않는 모델 부분을 RAM으로 옮깁니다. IP-Adapter 적용 시험은 98.707초, 최대 GPU 메모리 약 5.75GB로 완료되었습니다. 결과의 실행 시간과 최대 GPU 메모리는 `result.json`에 기록됩니다.

## Windows 화면 실행

```powershell
& "D:\genai-cache\venv\Scripts\python.exe" .\gui_main.py
```

현재 화면에서 실제 사용하는 입력은 캐릭터 기준 이미지 한 장입니다. 의상과 자세는 아직 생성 기능에 연결되지 않았으므로 버튼이 비활성화되어 있습니다. 생성 작업은 화면과 분리된 작업 흐름에서 실행되며 오류가 나면 상세 내용과 `result.json`을 남깁니다.

PyTorch 2의 기본 효율적인 주의 처리를 사용하므로 별도의 `enable_attention_slicing()`을 적용하지 않습니다. [Diffusers 메모리 처리 안내](https://huggingface.co/docs/diffusers/api/pipelines/overview)

## 후보 3장 생성

한 장 생성과 참조 그림 적용이 모두 확인된 뒤 `generation.limit: 3`으로 실행합니다. 기본 설정도 3장입니다.

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

- 512×512 PNG 후보 3장이 오류 없이 생성됩니다.
- 후보 3장 중 최소 1장을 사용자가 채택합니다.
- 참조 그림과 다른 장면을 만들면서 그림체가 유지됩니다.
- 각 이미지의 요청 문장, 제외 문장, 시드, 모델과 실행 시간을 `result.json`에서 확인할 수 있습니다.
- 같은 환경과 시드로 결과를 다시 만들 수 있습니다.

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
