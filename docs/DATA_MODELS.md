# 데이터 객체 정의

이 문서는 GUI 후보 생성 흐름에서 블록 사이로 전달하는 데이터의 이름과 필드를 정의합니다. 구현할 때 이름 없는 `dict`, `list[dict]`와 `Any` 대신 이 계약을 사용합니다.

## 이름 규칙

객체 이름은 다음 순서를 사용합니다.

```text
[업무 대상] + [세부 의미] + [객체 종류]
```

현재 업무 대상은 캐릭터(Character), 세부 의미는 생성(Generation)입니다.

| 객체 이름 | 한글 뜻 | 객체 종류가 뜻하는 단계 |
|---|---|---|
| `CharacterGenerationInput` | 캐릭터 생성 입력 | 사용자가 화면에서 직접 선택한 원본 값 |
| `CharacterGenerationSettings` | 캐릭터 생성 설정 | YAML 검사를 통과해 이름이 붙은 실행 설정 |
| `CharacterGenerationRequest` | 캐릭터 생성 실행 요청 | 검사와 가공이 끝나 모델이 바로 사용할 값 |
| `CharacterGenerationCandidate` | 캐릭터 생성 후보 | 모델 실행은 성공했지만 사용자가 아직 승인하지 않은 결과 |

`UserGenerationRequest`, `PreparedGenerationRequest`와 `GenerationCandidate`는 더 이상 목표 이름으로 사용하지 않습니다.

## 공통 화면 범위 종류

**캐릭터 화면 범위 종류(CharacterFramingType)**는 화면 선택값을 문자열로 흩어놓지 않기 위한 이름 있는 종류입니다.

| 값 | 한글 뜻 |
|---|---|
| `FULL_BODY` | 전신 |
| `UPPER_BODY` | 상반신 |
| `FACE` | 얼굴 중심 |

코드에서는 `CharacterFramingType`만 전달합니다. `full_body`, `full`과 한글 표시 이름을 여러 함수에서 직접 비교하지 않습니다. 화면 표시와 파일명 변환은 각 경계에서 한 번만 수행합니다.

## 1. CharacterGenerationInput

### 한글 뜻

사용자가 GUI에서 직접 선택한 캐릭터 생성 입력입니다. 아직 크기, 생성 문장이나 시드가 정해지지 않은 원본 값입니다.

### 생성 위치

`gui_main.py`의 사용자 입력 블록에서 생성합니다.

### 포함 필드

| 필드 | 타입 | 한글 뜻 | 값이 오는 곳 |
|---|---|---|---|
| `reference_image_path` | `Path` | 사용자가 선택한 기준 이미지 경로 | 파일 선택 화면 |
| `framing_type` | `CharacterFramingType` | 전신·상반신·얼굴 중심 중 하나 | 화면 범위 선택 |

### 포함하지 않는 값

- 가로와 세로
- 생성 문장과 제외 문장
- 시드
- 모델 이름과 실행 설정
- 이미지 파일 데이터
- 승인과 저장 상태

### 처리와 저장

AI 모델을 호출하지 않습니다. 파일로 저장하지 않고 데이터 가공 블록에만 전달합니다.

## 2. CharacterGenerationSettings

### 한글 뜻

YAML 파일에서 읽고 기존 설정 검사를 통과한 캐릭터 생성 설정입니다. 설정 `dict`가 데이터 가공 함수 안으로 퍼지지 않도록 필요한 값에 이름과 타입을 붙입니다.

### 생성 위치

`gui_main.py`에서 YAML 설정 검사가 끝난 뒤 생성합니다.

### 포함 필드

| 필드 | 타입 | 한글 뜻 |
|---|---|---|
| `model_id` | `str` | 사용할 기본 이미지 모델 이름 |
| `reference_adapter_id` | `str` | 기준 이미지 특징 전달 모델과 파일 이름 |
| `inference_steps` | `int` | 모델이 이미지를 다듬는 반복 횟수 |
| `guidance_scale` | `float` | 생성 문장을 따르는 강도 |
| `reference_image_strength` | `float` | 기준 이미지 특징을 반영하는 강도 |
| `default_negative_prompt` | `str` | 모든 생성에서 피할 기본 표현 |

AI 모델을 호출하지 않으며 파일로 따로 저장하지 않습니다. `CharacterGenerationInput`과 함께 데이터 가공 블록에 전달합니다.

## 3. CharacterGenerationRequest

### 한글 뜻

입력 검사와 데이터 가공이 모두 끝나 AI 모델이 추가 계산 없이 바로 실행할 수 있는 캐릭터 생성 요청입니다.

### 생성 위치

`genai_lab/request.py`의 데이터 가공 블록에서 생성합니다.

### 포함 필드

| 필드 | 타입 | 한글 뜻 | 값이 만들어지는 곳 |
|---|---|---|---|
| `reference_image` | `PIL.Image.Image` | RGB 형식으로 읽은 기준 이미지 | 기준 이미지 경로 검사 후 읽기 |
| `reference_image_name` | `str` | 결과 기록에 사용할 기준 이미지 파일명 | 기준 이미지 경로 |
| `framing_type` | `CharacterFramingType` | 선택한 화면 범위 | 사용자 입력 |
| `width` | `int` | 생성할 이미지 가로 크기 | 화면 범위 규칙 |
| `height` | `int` | 생성할 이미지 세로 크기 | 화면 범위 규칙 |
| `prompt` | `str` | 모델에 전달할 생성 문장 | 기본 문장과 화면 범위 규칙 |
| `negative_prompt` | `str` | 생성에서 피할 표현 | 기본 제외 문장과 화면 범위 규칙 |
| `seed` | `int` | 같은 생성 조건을 다시 시작할 숫자 | 후보를 준비할 때 한 번 생성 |
| `candidate_number` | `int` | 현재 실행의 몇 번째 후보인지 나타내는 1~3 | 재생성 횟수 |
| `inference_steps` | `int` | 모델이 이미지를 다듬는 반복 횟수 | 설정 파일 |
| `guidance_scale` | `float` | 생성 문장을 따르는 강도 | 설정 파일 |
| `reference_image_strength` | `float` | 기준 이미지 특징을 반영하는 강도 | 설정 파일의 IP-Adapter scale |
| `model_id` | `str` | 사용할 기본 이미지 모델 이름 | 설정 파일 |
| `reference_adapter_id` | `str` | 사용할 기준 이미지 특징 전달 파일 이름 | 설정 파일 |

### 가공 규칙

- 기준 이미지가 실제 파일인지 확인합니다.
- 이미지를 열 수 있는지 확인하고 RGB 형식으로 바꿉니다.
- 화면 범위를 가로·세로와 생성 문장으로 한 번만 변환합니다.
- 가로·세로가 허용 범위인지 확인합니다.
- 후보 번호가 1~3인지 확인합니다.
- 시드는 후보마다 한 번 만들고 이후 바꾸지 않습니다.
- 기준 이미지 특징 반영 강도가 허용 범위인지 확인합니다.

시드는 무작위로 만들지만 AI 판단은 아닙니다. 생성된 시드를 요청에 기록하므로 같은 요청을 다시 실행할 수 있습니다.

### 포함하지 않는 값

- 생성된 후보 이미지
- 실행 시간과 GPU 사용량
- 사용자 승인·거절
- 저장 상태와 저장 경로

### 처리와 저장

규칙으로만 생성하며 AI 모델을 호출하지 않습니다. 객체 자체는 저장하지 않고 AI 모델 실행 블록으로 전달합니다.

## 4. CharacterGenerationCandidate

### 한글 뜻

AI 모델 실행에 성공하여 만들어진 캐릭터 이미지 후보입니다. 사용자가 승인하기 전의 임시 결과입니다.

### 생성 위치

`genai_lab/generator.py`의 AI 모델 실행 블록에서 생성할 예정입니다.

### 포함 필드

| 필드 | 타입 | 한글 뜻 | 값이 만들어지는 곳 |
|---|---|---|---|
| `generation_request` | `CharacterGenerationRequest` | 이 후보를 만든 정확한 실행 요청 | AI 모델 실행 입력 |
| `image` | `PIL.Image.Image` | 시스템 메모리에 있는 생성 이미지 | AI 모델 실행 결과 |
| `generated_at` | `datetime` | 후보 생성이 끝난 시각 | 모델 실행 종료 시점 |
| `elapsed_seconds` | `float` | 후보 한 장 생성에 걸린 초 | 모델 실행 시간 측정 |
| `peak_gpu_memory_bytes` | 정수 또는 `None` | 실행 중 확인한 최대 GPU 메모리, 확인할 수 없으면 없음 | GPU 실행 기록 |

`generation_request`를 포함하므로 시드, 크기, 생성 문장과 모델 정보를 후보에 다시 복사하지 않습니다.

### 포함하지 않는 값

- 사용자 승인·거절 상태
- 저장 대기·완료·실패·거절 상태
- 저장 경로와 최종 파일명
- 다음 후보의 시드
- 실패 오류

모델 실행에 실패하면 후보 객체를 만들지 않습니다. 실패 원인은 한글 오류로 결과 출력 블록에 전달합니다.

### 처리와 저장

AI 모델 실행으로 만들어집니다. 처음에는 이미지와 객체 모두 시스템 메모리에만 존재합니다. 사용자가 승인하고 저장까지 허용한 경우에만 결과 출력 블록이 이 객체의 값을 이용해 PNG와 JSON을 만듭니다.

## 객체 전달 순서

```text
CharacterGenerationInput + CharacterGenerationSettings
        ↓ 규칙 기반 검사와 가공
CharacterGenerationRequest
        ↓ AI 모델 실행
CharacterGenerationCandidate
        ↓ 사용자 판단
승인 또는 거절
```

## 결정 방식과 AI 실행 경계

| 작업 | 처리 방식 |
|---|---|
| 파일 존재 여부 검사 | 규칙 |
| 화면 범위에서 크기 결정 | 규칙 |
| 생성 문장과 제외 문장 조립 | 규칙 |
| 시드 생성과 후보 번호 검사 | 규칙 |
| 기준 이미지 RGB 변환 | 규칙 |
| 후보 이미지 생성 | AI 모델 |
| 캐릭터 유사성 승인·거절 | 사용자 |
| PNG와 JSON 저장 | 규칙 |

이 프로젝트에는 LLM과 외부 API 호출이 없습니다. Animagine XL과 IP-Adapter는 로컬 PC에서만 실행합니다.

## 외부 전달과 저장 여부

흐름 객체와 설정 객체는 모두 Python 프로그램 내부에서만 사용합니다. Java, 서버나 외부 API에 전달하지 않습니다.

- `CharacterGenerationInput`: 저장하지 않음
- `CharacterGenerationSettings`: 저장하지 않음
- `CharacterGenerationRequest`: 저장하지 않음
- `CharacterGenerationCandidate`: 승인 전 저장하지 않음
- 승인 후: 후보 이미지와 재현에 필요한 요청 필드만 PNG와 JSON으로 저장

## 구현 형태

이 객체들은 외부 API의 요청·응답이 아니라 로컬 Python 프로그램 내부에서 전달하는 값입니다. dataclass를 사용하고 모든 필드에 타입을 적습니다.

- 이름 없는 `dict`, `list[dict]`와 `Any`를 사용하지 않습니다.
- 객체를 만든 뒤 필드를 임의로 추가하지 않습니다.
- 사용자 승인과 저장 상태는 모델 실행 객체에 섞지 않고 결과 출력 단계에서 별도로 정의합니다.
- 외부 요청 검증용 Pydantic 모델은 서버나 외부 API가 없으므로 현재 추가하지 않습니다.

## 구현 전 확인 위치

구현할 때 객체가 처음 등장하는 위치에는 다음 내용을 짧은 주석으로 설명합니다.

- 한글 뜻
- 포함하는 값
- 어디에서 생성되는가
- 규칙 또는 AI 중 무엇으로 처리되는가
- 저장되는가
- 다음 단계에서 어디에 사용되는가

같은 객체가 다시 등장할 때는 설명을 반복하지 않습니다.
## 5. 참조 화질과 부분 보정 객체

### ReferenceImageQualityReport

기준 이미지의 가로·세로, 짧은 변, 선명도 점수와 화질 상태를 담는 규칙 검사 결과입니다. AI 모델을 호출하지 않고 저장하지 않습니다.

### ReferenceImageEnhancementCandidate

원본 이미지, 확대 복원 이미지, 화질 검사 결과와 사용한 복원 모델 이름을 담는 사용자 확인 전 후보입니다. 보정본은 아직 생성 기준으로 확정된 값이 아닙니다.

### ApprovedReferenceImage

사용자가 생성에 사용하도록 확정한 RGB 이미지와 원본 파일명, 확대 복원 적용 여부를 담습니다. 원본과 보정본 중 무엇을 선택했는지 명확히 구분하며 객체 자체는 저장하지 않습니다.

### CharacterDetailDetection

YOLO가 생성 후보에서 찾은 얼굴 또는 손의 종류, 위치와 신뢰도를 담는 임시 탐지 결과입니다. 탐지는 AI 처리지만 캐릭터가 정상이라는 확정 결과는 아닙니다.

### CharacterDetailCorrectionResult

보정 전후 이미지, 탐지한 얼굴·손 수, 보정·거절 영역 수, 마스크 밖 변경 픽셀 수와 확인 경고를 담습니다. 파일로 자동 저장하지 않고 GUI의 전후 비교에 전달합니다.

결정 경계는 다음과 같습니다.

- 화질 기준 계산과 마스크 면적 검사는 결정론적 규칙입니다.
- 확대 복원, 얼굴·손 탐지와 부분 재생성은 로컬 AI 모델 처리입니다.
- 어떤 참조 이미지와 어떤 생성 후보를 채택할지는 사용자 판단입니다.
- 사용자 승인 전 원본, 보정본과 탐지 결과는 모두 임시 데이터입니다.


## 6. 의상 참조와 합성 보호 객체

### ClothingReferenceInput

사용자가 GUI에서 고른 의상 참조 경로와 의상 종류를 담습니다. 현재 실행 가능한 종류는 상의, 하의, 드레스와 전신 의상입니다.

### CharacterClothingTryOnRequest

기본 캐릭터 이미지, 의상 참조 이미지와 의상 종류를 의상 합성기에 전달합니다. 합성기는 임시 이미지만 반환하며 저장하지 않습니다.

### CharacterTryOnProtectionPlan

다음 세 마스크를 포함합니다.

- clothing_change_mask: 실제로 의상을 바꿀 수 있는 영역
- identity_protection_mask: 얼굴, 머리, 손, 발, 노출 피부와 배경을 포함한 변경 금지 영역
- boundary_blend_mask: 의상 허용 영역 안에서만 경계를 자연스럽게 섞는 영역

사람·의상 분리 모델이 만든 의상 마스크에서 신체 보호 마스크를 빼서 규칙으로 만듭니다.

### CharacterTryOnVerification

의상 허용 영역 밖에서 달라진 픽셀 수와 통과 여부를 담습니다. 한 픽셀이라도 달라지면 실패입니다.

### CharacterClothingTryOnCandidate

보호 합성과 픽셀 검사를 통과한 사용자 승인 전 후보입니다. 자동 저장하지 않으며 GUI의 적용 전·마스크·적용 후 비교 화면으로 전달합니다.

### CatVTONLocalSettings

별도 Python 실행 파일, 공식 CatVTON 저장소, 임시 폴더, 모델 캐시, 처리 크기와 제한 시간을 담습니다. 현재 Animagine 환경과 라이브러리를 섞지 않고 별도 프로세스를 실행하는 설정입니다.

### CatVTONClothingTryOnResult

보호 검사를 통과한 의상 후보와 사용자에게 보여줄 실제 변경 허용 마스크를 담습니다. 사용자 승인 전 임시 데이터입니다.

### CharacterIdentityProtectionMask

CatVTON의 SCHP 사람 영역 분리 결과에서 얼굴, 머리카락, 노출된 팔·다리, 손·발과 장신구를 합쳐 만든 변경 금지 마스크입니다. CatVTON 별도 프로세스의 임시 폴더에만 존재합니다.

### 현재 AI 경계

- 의상·신체 마스크 예측: CatVTON의 SCHP와 DensePose
- 의상 합성 원본 생성: 별도 Python 환경의 CatVTON
- 허용 영역 계산, 제한 합성, 보호 영역 검사: 결정론적 Python 규칙
- 최종 채택: 사용자 승인

현재 구현은 GUI 입력, CatVTON 별도 실행, 보호 합성과 사용자 비교까지 연결했습니다. 의상 세부 태그 자동 추출(WD14·JoyCaption)과 자세 참조는 포함하지 않았습니다.
