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

보호 검사를 통과한 의상 후보, 사용자에게 보여줄 실제 변경 허용 마스크와 실행 증명 수치를 담습니다. 사용자 승인 전 임시 데이터입니다.

### CharacterAgnosticApprovedInput

GUI에서 사용자가 승인한 Human-Agnostic 이미지 1개, 변경 마스크 1개, CatVTON 의상 종류, 승인 마스크 픽셀 수와 Preflight SHA-256 4개를 담습니다. 해시는 처리 Person, 이분화 마스크, blur 이후 model_mask와 padding 이후 의상 입력을 구분합니다. GUI가 이미지 복사본 2개를 작업 스레드에 넘기며 작업 성공·실패와 관계없이 스레드 종료 시 닫습니다.

### CatVTONExecutionMetadata

별도 실행기가 실제 사용한 입력을 증명하는 기존 수치와 모델 입력 SHA-256 4개를 담습니다.

- 마스크 출처: `user_approved`
- AutoMasker 실행 횟수: `0회`
- 승인 이미지 크기: 가로·세로 픽셀
- 승인 마스크 픽셀 수
- CatVTON 처리 크기로 변환한 뒤의 마스크 픽셀 수
- 안전 검사 실행 여부: 로컬 시험에서는 `false`
- 인물 입력 출처: `generated_candidate`
- 인물 입력 크기: 가로·세로 픽셀
- 승인 의상 추출 원본 크기: 가로·세로 픽셀
- 투명 여백 제거 후 조건 이미지 크기: 가로·세로 픽셀
- 의상 알파 픽셀 수와 조건 이미지 내부 점유율
- 처리 Person·이분화 마스크·model_mask·처리 의상 SHA-256 4개

출처가 다르거나 AutoMasker 실행 횟수가 1회 이상이거나 입력 크기·픽셀 수·SHA-256 중 1개라도 승인 기록과 다르면 결과를 거절합니다.
안전 검사 실행 여부도 요청 설정과 다르면 결과를 거절합니다.

### CatVTONClothingConditionImage

사용자가 승인한 RGBA 의상 추출본은 원본 증거로 유지하고, CatVTON에 전달할 복사본만 알파 픽셀 경계 상자로 자릅니다. 원본 크기, 잘라낸 좌표 4개, 조건 이미지 크기, 알파 픽셀 수와 점유율을 포함합니다. 알파 픽셀이 `0개`이면 GPU 호출 전에 중단합니다.

### CharacterIdentityProtectionMask

사용자 승인 변경 마스크를 반전해 만든 변경 금지 마스크입니다. CatVTON 별도 프로세스의 임시 폴더에만 존재하며 별도 SCHP·DensePose 예측을 다시 실행하지 않습니다.

### 현재 AI 경계

- 의상·신체 마스크 예측: 사용자 승인 전 신체 비교 단계의 SCHP와 DensePose
- 의상 합성 원본 생성: 별도 Python 환경의 CatVTON
- CatVTON 내부 AutoMasker: 실행 0회
- 승인 마스크 동일성·제한 합성·보호 영역 검사: 결정론적 Python 규칙
- 최종 채택: 사용자 승인

현재 구현은 GUI 입력, CatVTON 별도 실행, 보호 합성과 사용자 비교까지 연결했습니다. 의상 세부 태그 자동 추출(WD14·JoyCaption)과 자세 참조는 포함하지 않았습니다.

### HumanAgnosticImageCandidate

기존 의상 교체 허용 영역을 RGB `(127, 127, 127)`로 중립화한 사용자 승인 전 임시 이미지입니다.

- 입력: 승인 전 캐릭터 원본, 보호 영역 차감 후 마스크, CatVTON 원본 마스크
- 포함: 중립화 이미지 1개, 중립화 픽셀 수·비율, 원본 마스크 포함률, 마스크 밖 변경 픽셀 수
- 결정론적 처리: 마스크 내부 픽셀만 중립 RGB로 교체하며 LLM과 이미지 생성 모델 호출은 0회
- 실패: 이미지 크기가 다르거나 중립화 대상이 0픽셀이면 다음 단계로 전달하지 않음
- 저장: 파일로 자동 저장하지 않고 GUI 검토 중 메모리에만 유지

### CatVTONInputSnapshot

CatVTON 전처리 전에 GUI가 공개하는 입력 자료입니다. 기준 캐릭터 원본, 실제 변경 마스크 원천, 추출된 참조 의상, 변경 금지 보호 영역과 캐릭터 외곽 허용 범위 총 5개 이미지를 소유합니다. 캐릭터·변경·보호·외곽 좌표는 완전히 같아야 하며, 참조 의상은 원본 비율과 크기를 유지하므로 같은 좌표를 요구하지 않습니다. 변경 마스크는 1픽셀 이상, 보호 영역 침범과 캐릭터 외곽 밖 침범은 각각 0픽셀이어야 합니다. 기준 캐릭터·변경 마스크·참조 의상 SHA-256 3개와 판정 사유를 포함합니다.

### CatVTONPreflightCandidate

CatVTON 공식 전처리를 실행하되 모델 다운로드와 GPU 추론은 하지 않는 승인 전 자료입니다. 처리 Person, 이분화 마스크, `blur_factor=9` 직후 원본 마스크, 1~127 약한 침범, 128~255 강한 침범, 금지 영역 제한 후 실제 model_mask, padding 의상과 최종 보호·외곽 침범 총 9개 이미지를 소유합니다. 입력 SHA-256 4개, 이분화·model_mask 픽셀 수, 약한·강한 침범, 제거 픽셀과 최종 침범 수를 포함합니다. 약한·강한 침범은 금지 영역에서 0으로 제거하고 보정 전 `WARNING` 수치로 남깁니다. 최종 model_mask의 보호 영역 또는 외곽 밖 침범이 `1픽셀 이상`일 때만 `BLOCK`으로 승인할 수 없습니다.

### GuardResult와 GuardDecision

`GuardResult`는 한 검사의 코드, 단계, `PASS·WARNING·BLOCK`, 측정값, 기준값, 단위, 보정 후 값, 한글 근거와 복구 행동을 담는 결정론적 결과입니다. 보정 전 문제를 자동으로 제거해 최종 침범이 0픽셀이면 `WARNING`, 최종 입력에 문제가 남으면 `BLOCK`입니다.

`GuardDecision`은 모든 `GuardResult`와 심각도별 목록을 소유합니다. `BLOCK=0개`일 때만 `approval_enabled=True`입니다. 결과가 0개면 승인 여부를 계산하지 않고 예외를 발생시킵니다. GUI는 개별 `passed` 값을 직접 `and`로 연결하지 않고 이 결정 하나만 사용합니다.

### CharacterForegroundMaskCandidate

`isnet-anime`이 같은 생성 후보에서 추출한 사용자 승인 전 캐릭터 전체 외곽입니다. L 마스크 1개, 기준 128 이상인 외곽 픽셀 수·전체 비율, 모델 ID와 CPU 추론 시간을 포함합니다. 파일로 자동 저장하지 않고 GUI 검토 중 메모리에만 유지합니다.

### CharacterClothingMaskRefinement

SCHP 의상 마스크 닫기·5~15px 팽창, `isnet-anime` 전체 외곽 15px 팽창과 신체 보호 차감을 끝낸 임시 결과입니다. 원시 의상, 닫힌 의상, 팽창 의상, 원본 외곽, 팽창 외곽, 최종 변경 영역과 보호 마스크 총 7개를 소유합니다. 외곽 밖에서 거절한 픽셀 수를 포함하며 최종 변경 영역의 외곽 밖 픽셀은 0px여야 합니다.

### ConfirmedCharacterBodyComparison

사용자가 전처리 전 입력 5개, 기존 마스크 가공 12개와 CatVTON Preflight 9개를 합친 26개 중간 결과를 확인한 뒤 승인한 신체·의상 마스크 분석 결과입니다.

- 포함: 정확한 Human-Agnostic 이미지 1개, 원본 좌표 변경 마스크 1개, 처리 좌표 최종 model_mask 1개, 마스크·중립화 수치
- 외곽 검사: 캐릭터 외곽 밖 SCHP 오탐 픽셀 수·비율
- 기존 의상 제거: 탐지 픽셀 수, 잔여 픽셀 수와 제거율
- 임시 값: 사용자가 의상 종류나 캐릭터 이미지를 바꾸면 이미지 2개를 즉시 닫고 승인을 무효화
- 현재 외부 전달: 복사본을 GUI 작업 스레드와 CatVTON 별도 프로세스에 전달
- 실행 전 검사: 크기, 의상 종류, 마스크 픽셀 수가 모두 승인 기록과 일치해야 함
- Preflight 증거: 실제 모델 입력 SHA-256 4개, 보호 영역·외곽 밖 침범 각각 0픽셀

현재 구현은 같은 생성 후보에서 만든 승인 마스크를 CatVTON 입력까지 연결합니다. Human-Agnostic 이미지는 검토 자료로 사용하며 실행기는 공식 구조대로 `generated_candidate` 원본과 마스크를 받습니다. 승인한 Preflight와 실제 실행 입력 SHA-256 4개가 다르면 모델 다운로드 전에 중단합니다. 실제 사용자 이미지 Preflight와 GPU 재검증은 각각 0회입니다.

### OriginalClothingRemovalVerification

기존 의상 원천은 GUI의 사용자 선택 SAM2 승인 마스크이며, 호환 호출에서는 AutoMasker 후보입니다. 전체를 외곽 밖과 외곽 안 검증 대상으로 나누고 보호 겹침은 검증 대상의 부분집합으로 유지합니다. 보호와 겹친 픽셀을 분모에서 빼지 않습니다. 포함·잔여·충돌 수와 위치, 외곽 밖 수·비율을 공개합니다. 상태는 covered, incomplete, needs_review, not_evaluable이며 검증 대상 0px의 removal_percent는 None입니다. 이는 의상 의미 인식이나 생성 성공을 검증하는 자료형이 아닌 선택 영역 포함 검사입니다.

### ApprovedTargetMasks

생성된 기준 캐릭터 RGB의 SHA-256과 같은 캔버스의 이진 clothing_mask·special_protection_mask를 소유합니다. 사용자 승인 때 크기·빈 교체 마스크·역할 충돌을 검사하고 실행 직전 기준 픽셀 해시를 재검사합니다. copy()는 이미지 소유권을 분리하며 close()는 두 마스크를 해제합니다.

### NeutralResidualDiagnostic

Inpaint 시작 이미지의 중립색 인접 픽셀 중 승인 영역 안에 있는 평가 대상 수, 출력에서도 중립색 인접인 의심 수·비율·마스크를 가집니다. 평가 대상 0px이면 비율은 None입니다. 실제 회색 옷과 미생성을 구분할 수 없는 경고 전용이며 승인 조건에는 포함하지 않습니다. GarmentInpaintReviewCandidate가 이를 소유하고 함께 해제합니다.

### PoseReferenceReviewCandidate

사용자가 선택했지만 아직 승인하지 않은 자세 참조 원본입니다. RGB 이미지, PNG·JPEG 형식, 가로·세로, 전체 픽셀 수, 가로/세로 비율과 파일 크기를 포함합니다. 최소 변 `64px` 미만 또는 전체 `40,000,000px` 초과 입력은 거절합니다. AI 호출과 파일 저장은 각각 `0회·0개`입니다.

### PoseReferenceApprovedInput

사용자가 미리보기와 수치를 확인한 뒤 승인한 독립 이미지 복사본입니다. GUI 메모리에만 유지하며 선택 해제·교체·앱 종료 때 닫습니다. DWPose Worker에 전달하는 입력이며 Worker가 별도 복사본을 소유합니다.

### PoseJointCoordinateCandidate

DWPose가 추정한 몸 관절 18개 각각의 이름, 원본 픽셀 좌표 `x·y`, 신뢰도 `0.0~1.0`, 기준 통과 여부와 모델 추정 여부를 포함합니다. 사용자가 승인하기 전 확정 자세가 아닙니다.

### PoseEstimationReviewCandidate

자세 원본, 원본 위 관절 확인본, 표준 OpenPose 뼈대 지도, 관절 좌표 18개, 탐지·누락 수, 기준 `30.0%`, 모델 ID와 처리 시간을 포함합니다. 실제 GUI DWPose 실행은 아직 `0회`입니다.

### PoseEstimationApprovedInput

사용자가 3개 이미지를 확인한 뒤 승인한 표준 OpenPose 지도 복사본과 관절 좌표입니다. GUI와 생성 Worker가 서로 다른 이미지 복사본을 소유하며 Worker 종료 때 생성용 복사본을 닫습니다. ControlNet 전달 코드는 연결됐고 실제 GPU 전달과 자동 저장은 각각 `0회·0개`입니다.

### PoseQualityDecision

DWPose 승인 전·저장 자세 재사용 전 공통 품질 판정입니다. 탐지 관절 수와 초기 기준 `8/18개`, 어깨·골반·무릎·발목 필수 그룹 통과 수 `0~4개`, ControlNet 지도에서 검은색이 아닌 픽셀 수와 모든 거절 사유를 포함합니다. 각 필수 그룹은 좌우 관절 중 `1개 이상` 탐지돼야 하며 유효 뼈대 픽셀은 `1px 이상`이어야 합니다. 이 수치는 실제 성공·실패 표본으로 보정되기 전 초기 정책값입니다.

### SavedApprovedPose

가장 최근 정상 추출·사용자 승인을 통과한 자세 `1개`입니다. 저장 단위는 `source_preview.png`, `control_map.png`, `metadata.json` 총 `3개`이며 기본 ID는 `last-approved`, 스키마 버전은 `1`입니다. JSON에는 UTC 저장 시각, 크기, 관절 18개, 탐지·누락 수, 신뢰도 기준, 모델 ID, 품질 수치와 ControlNet 지도 픽셀 SHA-256을 기록합니다. 불러올 때 크기·관절 수·합계·현재 품질 기준·SHA-256 중 `1개`라도 다르면 폴백을 거절합니다. GUI는 독립 이미지 복사본을 받은 뒤 디스크 로드 객체를 닫습니다.

### PoseControlPreparedInput

승인 OpenPose 지도를 생성 해상도로 바꾼 ControlNet 전용 임시 입력입니다. 원본·목표 가로와 세로, 확대 비율, 좌·상·우·하 검은 여백 픽셀, 검은색이 아닌 뼈대 픽셀 수를 포함합니다. 원본 비율을 유지하고 자르기는 `0px`이며, 뼈대 픽셀이 `0px`이면 GPU 호출 전에 중단합니다. 모델 호출 뒤 즉시 닫고 파일로 저장하지 않습니다.

### CharacterGenerationCandidate 자세 실행 기록

후보에는 자세 제어 상태, ControlNet 모델 ID, 강도, 시작 비율과 종료 비율을 기록합니다. 현재 초기값은 `xinsir/controlnet-openpose-sdxl-1.0`, `0.65`, `0.00`, `0.80`이며 자세 입력이 없으면 상태는 `not_requested`, 나머지 4개 값은 `null`입니다.

### GenerationWorkflowContext

GUI 자동 실행 중에만 존재하는 임시 상태입니다. 캐릭터 필수 경로 1개, 선택 의상 경로 최대 1개, 선택 자세 경로 최대 1개, 현재 단계, 실패 단계, 재시도 횟수와 실행 여부를 포함합니다. 이미지 픽셀과 AI 모델은 소유하지 않으며 파일로 저장하지 않습니다.

현재 단계 이름은 과거 진단 자산까지 포함해 13개를 유지하지만 사용자 활성 진행률은 고정 8단계입니다. `REFERENCE_PREPARING=1/8`, `CLOTHING_MASKING=2/8`, `CLOTHING_ANALYZING=3/8`, `POSE_ESTIMATING=4/8`, `BASE_GENERATING=5/8`, `BODY_MASKING=6/8`, `CLOTHING_COMPOSITING=7/8`, `FINAL_REVIEW·COMPLETED=8/8`입니다. `GARMENT_GEOMETRY`와 `GARMENT_LINEART`는 수동 진단 상태로만 6/8에 매핑하며 자동 생성 흐름에서는 진입하지 않습니다. 실패하면 별도 `FAILED` 상태와 실제 실패 단계 번호를 함께 보존합니다.

## 7. 의상 전처리와 GUI 상태 객체

2026-08-27에 확정한 의상 전처리 계약입니다. 전체 전처리는 5단계이며 현재 1/5 입력 정규화, 2/5 의상 영역 최대 8개 선택과 3/5 영역별 SAM2 후보 선택·합치기까지 코드로 구현했습니다. 단일 영역 SAM2는 사용자 확인 1회이며 새 복수 영역 SAM2 실제 추론은 0회입니다. WD Tagger와 CatVTON 호출도 이번 검증에서는 각각 0회입니다.

### ClothingSourceInput

사용자가 선택한 정규화 전 파일 경로 1개를 담습니다. 저장하지 않고 JPEG·PNG 입력 검사에만 사용합니다.

### NormalizedClothingSource

다음 9개 값을 포함합니다.

- 정규화된 RGB 이미지
- 원본 파일명
- 실제 파일 형식
- 원본 색상 형식
- 파일 크기(바이트)
- 원본 가로 픽셀
- 원본 세로 픽셀
- 정규화 후 가로 픽셀
- 정규화 후 세로 픽셀

EXIF 방향 보정, 투명 PNG의 흰 배경 합성과 RGB 변환을 Python 규칙으로 처리합니다. AI 호출은 0회이며 이미지 해제 책임은 호출자에게 있습니다.

### ClothingRegionCandidate

Grounding DINO 또는 사용자가 정한 위치 좌표 4개, 이름과 0.0~1.0 자동 신뢰도를 담는 사용자 확인 전 후보입니다. 수동 후보는 신뢰도를 만들지 않기 위해 측정값에서 측정 불가로 기록합니다.


### ClothingDetectionSettings

자동 탐지는 IDEA-Research/grounding-dino-tiny 모델 1개를 사용합니다. 초기 운용 기준은 상자 신뢰도 0.30, 글자 연결 신뢰도 0.25, 최소 면적 2.0%, 최대 면적 95.0%입니다. 기본 추론 장치는 CPU이며 모델 파일은 D:/genai-cache/huggingface에 저장합니다. 이 수치는 실제 승인·거절 자료로 보정하기 전의 초기 기준입니다.

### ClothingRegionDetectionResult

면적 기준을 통과한 후보 목록, 최상위 후보 1개, 처리 시간(초), 수동 선택 필요 여부와 한글 사유를 담습니다. 유효 후보가 0개이거나 모델 실행이 실패하면 GUI가 수동 사각형 선택으로 전환합니다.

### ClothingRegionMeasurement

다음 5개 값을 기록하며 임의 종합점수는 만들지 않습니다.

- 탐지 방법: grounding_dino 또는 manual
- 자동 탐지 신뢰도: 0.0%~100.0%, 수동 선택은 측정 불가
- 이미지 점유율: 0.0%~100.0%
- 영역 가로 픽셀
- 영역 세로 픽셀
### ClothingExtractionCandidate

원본 픽셀 추출이 반환하는 투명 RGBA 이미지, 0~255 알파 마스크, 자동 잘라보기 좌표, 경계·공백·RGB 보존 수치를 담습니다. 사용자가 흰 와이셔츠 1건에서 추출을 확인했으며 체감 품질 약 99%는 자동 측정값이 아닙니다.

### ClothingDesignSummary

사용자가 승인한 WD14 태그와 확인 불가 항목을 담습니다. 주요 RGB 색상 자동 확정은 아직 하지 않으며 현재 단색 시험 의상 WD14 실행은 1회, 실제 사용자 의상 실행은 0회입니다.

### ClothingReviewCandidate

입력 이미지, 추출 후보, 디자인 분석과 경고 문구를 GUI에 전달하는 사용자 승인 전 임시 객체입니다. 현재 GUI 전달 횟수는 0회입니다.

### ConfirmedClothingReference

사용자가 승인한 의상 이미지, 마스크, 종류와 디자인 분석을 담습니다. CatVTON 실행 입력으로 변환될 예정이며 현재 생성 횟수는 0회입니다.

### ClothingPreparationProgress

비동기 Worker가 GUI에 전달할 다음 8개 값을 포함합니다.

- job_id
- stage
- stage_index
- total_stage_count
- elapsed_seconds
- message_ko
- can_cancel
- can_use_manual_region

현재 상태 종류는 총 11개입니다.

1. IDLE
2. NORMALIZING
3. DETECTING
4. WAITING_REGION_SELECTION
5. EXTRACTING
6. ANALYZING
7. WAITING_APPROVAL
8. CONFIRMED
9. FAILED
10. CANCELLING
11. CANCELLED

진행률 퍼센트는 사용하지 않고 현재 단계와 전체 5단계를 표시합니다. GUI는 의상 탐지 작업과 SAM2 작업을 종류별로 동시에 최대 1개 실행합니다. 위치 탐지는 상태 문구 2단계(정규화 1/2, 자동 탐지 2/2)를 전달하고, 위치 승인 뒤 SAM2 후보 생성 상태를 추가로 전달합니다.

### ClothingPreparationFailure

작업 ID, 실패 단계, 오류 코드, 한글 원인, 복구 행동, 경과 시간과 기술 세부 내용을 담습니다. 사용자 화면에는 한글 원인과 행동을 표시하고 Traceback은 로그에만 기록합니다.

### ClothingSourceValidationError

파일 없음, 파일 아님, 0바이트, 미지원 형식, 해석 실패, 위험한 픽셀 수와 잘못된 크기를 7개 오류 코드로 구분합니다.

### 자원 생명주기

정규화 함수는 파일을 읽은 뒤 원본 파일을 닫고 독립된 RGB 이미지와 ICC 색상 프로필을 반환합니다. GUI는 영역 취소 시 정규화 이미지를 닫고, 영역 승인 시 SAM2 검토가 끝날 때까지 메모리에 유지합니다. 사용자가 영역별 후보 1개를 선택하면 각 영역에서 선택하지 않은 후보 최대 2개를 닫습니다. 픽셀 추출 승인 뒤에는 선택 마스크 최대 8개, 합친 마스크 1개, 투명 추출본 1개와 공백 처리 마스크 1개만 유지합니다. 의상 취소 또는 새 의상 선택 시 이 이미지를 모두 닫습니다. Grounding DINO와 SAM2는 실행 뒤 모델 참조를 제거하고 CUDA 캐시 해제를 요청합니다. 프로젝트 출력 파일 생성 수는 0개입니다.

현재 의상 전처리·분석 단위 테스트는 16개 통과, 0개 실패, Windows 최종 pytest 실행 시간 12.96초입니다. Windows py_compile 구문 검사와 GUI import 검사 1회를 통과했습니다. 단색 시험 의상 WD14 CPU 추론은 1회·3.735초이며 실제 사용자 의상 WD14와 CatVTON 호출은 각각 0회입니다.


## 8. 의상 추출 감사와 사용자 승인 객체

D-025, D-028과 D-029에서 확정한 사용자 승인 전 데이터 계약입니다. SAM2 후보 생성·GUI 선택, 원본 픽셀 추출, WD14 분석과 태그별 사용자 포함·제외까지 코드로 구현했습니다. 이번 검증의 실제 사용자 의상 WD14와 CatVTON 호출은 각각 0회입니다.

### ClothingDetectionReviewCandidate

정규화 원본, 위치 좌표 4개, 자동 신뢰도 0.0%~100.0%, 영역 가로·세로 픽셀, 이미지 점유율 0.0%~100.0%와 처리 시간(초)을 포함합니다. 저장하지 않는 임시 객체입니다.

### ClothingMaskReviewCandidate

SAM2 마스크 후보 번호 1~3, 흑백 마스크, 모델 점수 0.0%~100.0%, 선택 픽셀 수, 사각형 내부 점유율, 분리 영역 수와 전체 이미지 경계 접촉 픽셀 수를 포함합니다. `ClothingMaskRegionCandidateGroup`은 영역 번호, 승인 좌표와 후보 최대 3개를 묶습니다. `ClothingMaskExtractionResult`는 모델 ID, 체크포인트 원본 구성 종류, 실제 실행 구성 종류, 입력 크기, 영역 묶음 최대 8개와 전체 처리 시간(초)을 보관합니다. 현재 조합은 `sam2_video→sam2`이며 GUI에도 같은 값을 표시합니다. 사용자가 각 영역에서 후보 1개를 선택하면 영역당 나머지 최대 2개를 해제합니다.

### ClothingCombinedMaskCandidate

영역별로 선택한 마스크 1개~8개를 픽셀별 최대 알파값으로 합친 사용자 최종 승인 전 객체입니다. 0~255 알파 마스크, 원본 영역 수, 알파 128 이상 픽셀 수, 분리 영역 수와 경계 접촉 픽셀 수를 포함합니다. 입력 마스크의 원본 픽셀은 변경하지 않으며 결과 파일 생성 수는 0개입니다.

### ClothingExtractionCandidate

투명 RGBA 추출본, 공백 처리 후 알파 마스크, 자동 잘라보기 좌표 4개, 알파 픽셀 수, 반투명 경계 픽셀 수, 내부 공백 수, 복원 공백 수·픽셀 수, 보류 공백 수, 원본 RGB 변경 픽셀 수와 보존율을 포함합니다. 완료 기준은 변경 픽셀 0개와 보존율 100.000%입니다. 변경 픽셀이 1개 이상이면 실패합니다.

### ClothingDesignAnalysisResult

WD14 모델 ID, CPU 실행 장치, 원본 입력 크기, 모델 입력 448×448, 기준 35.0%, 전체·일반·제외 라벨 수, 태그 후보 최대 30개와 처리 시간(초)을 포함합니다. 각 `ClothingDesignTagCandidate`는 원본 태그 이름, 화면 표시 이름과 0.0%~100.0% 점수를 담습니다. 사용자 체크 전에는 확정값이 아닙니다.

### ClothingExtractionReviewCandidate

입력 원본, 위치 사각형, 선택 마스크, 투명 배경 추출본, 흰 배경 확인본, 픽셀 보존 측정과 의상 인식 결과를 포함합니다. GUI 선택지는 승인, 마스크 다시 선택과 의상 취소 3개입니다. 객체와 이미지는 저장하지 않습니다.

### ConfirmedClothingExtraction

사용자가 승인한 의상 RGB 이미지, 알파 마스크, 확정 의상 종류와 분석 태그를 포함합니다. 이 객체가 만들어지기 전 CatVTON 호출은 0회여야 합니다.

### 저장과 해제

승인 전 프로젝트 출력 파일 수는 0개입니다. 승인하면 ConfirmedClothingExtraction만 다음 CatVTON 요청으로 전달합니다. 거절하면 중간 마스크 후보 최대 3개, 투명 배경 추출본 1개와 흰 배경 확인본 1개를 메모리에서 해제합니다.

## 9. CatVTON 효과 측정 객체

### TryOnEffectMetricsResult

CatVTON이 프로세스를 정상 종료했는지가 아니라 실제 픽셀 효과가 남았는지를 다음 9개 값으로 기록합니다.

1. `raw_changed_inside_model_mask`: 처리 좌표의 model_mask를 원본 좌표에 최근접 투영한 뒤 원시 출력이 기준 후보와 달라진 픽셀 수
2. `final_changed_inside_approved_mask`: 최종 보호 합성이 승인 변경 영역 안에서 기준 후보와 달라진 픽셀 수
3. `discarded_by_protection_pixels`: 승인 영역 안에서 원시 출력은 달랐지만 최종 결과가 기준 후보와 같아진 픽셀 수
4. `mean_rgb_l1_inside`: 승인 영역 안 RGB 절대 차이 합의 픽셀 평균, 범위 0.0000~765.0000
5. `mask_leakage_pixels`: 승인 영역 밖 최종 변경 픽셀 수, 기준 0픽셀
6. `no_effect`: 최종 승인 영역 안 변경이 0픽셀이면 `true`
7. `clip_similarity`: 현재 `None`, 보정 표본 수집용 예약 필드
8. `dinov2_distance`: 현재 `None`, 보정 표본 수집용 예약 필드
9. `color_histogram_corr`: 현재 `None`, 보정 표본 수집용 예약 필드

CLIP·DINOv2·색상 히스토그램 임계값은 성공·실패 표본을 각각 30건 이상 확보하기 전에는 승인 차단에 사용하지 않습니다. CatVTON 원시 출력과 4배 차이맵은 사용자 검토 중에만 메모리에 유지하며 적용본 승인, 원본 선택, 후보 폐기와 앱 종료 경로에서 닫습니다.

## 10. TPS 의상 워핑 객체

### GarmentTpsWarpRequest

같은 좌표 크기의 RGBA 의상 캔버스, `N×2` 의상 원본 XY, `N×2` 캐릭터 목표 XY, 캔버스 가로·세로와 0 이상의 TPS 정규화 값을 포함합니다. 대응점은 양쪽 각각 최소 5개이며 개수 일치, 유한값, 중복 0개, 캔버스 밖 0개와 볼록 껍질 면적 0 초과를 만족해야 합니다.

### GarmentTpsWarpResult

워핑된 RGBA, 0~255 소프트 알파 마스크, 대응점 수, 원본·워핑 알파 픽셀 수와 두 픽셀 수의 차이를 포함합니다. 결과 이미지는 자동 저장하지 않으며 사용자가 검토하거나 다음 PoC 단계가 끝나면 `close()`로 2개 모두 해제합니다.

## 11. 의상 마스크 기하 좌표 객체

### GarmentMaskLandmarkSettings

기본값은 알파 임계값 128, 수직 비율 `(0.10, 0.50, 0.90)`, 탐색 밴드 비율 0.03과 최소 연결요소 면적 16px입니다. 수직 비율은 0과 1 사이의 오름차순 값 3개여야 하며 탐색 밴드는 0 초과 0.50 이하입니다.

### GarmentComponentLandmarks

유효 의상 조각 1개의 정렬 순번, 전경 픽셀 수, `(x, y, width, height)` 외접 영역, `upper_left/right → middle_left/right → lower_left/right` 순서의 `6×2` XY, 실제 선택된 Y행 3개와 기하 폭 점수 0.0~1.0을 포함합니다. 좌표 출처는 신체 관절 추론이 아니라 승인 마스크 경계입니다.

### GarmentMaskLandmarkResult

캔버스 크기, 조각별 좌표, 임계값 이상 원본·유지 전경 픽셀, 제거 노이즈 픽셀, 원본·유지·제거 연결요소 수와 `mask_geometry_v1` 방식을 포함합니다. 유효 조각이 정확히 1개일 때만 `single_component_tps_points()`가 좌표 복사본을 반환하며 0개 또는 2개 이상이면 TPS 자동 연결을 차단합니다.

## 12. 캐릭터 TPS 목표 좌표 객체

### CharacterTargetLandmarkSettings

기본값은 필수 DWPose 관절 신뢰도 0.30, 승인 변경 마스크 알파 임계값 128, 목표 행 탐색 비율 0.05와 최대 탐색 반경 64px입니다. 실제 탐색 반경은 `min(64px, 목표 높이×0.05)`이며 최소 1px입니다.

### CharacterTargetLandmarkResult

`upper_left/right → middle_left/right → lower_left/right` 순서의 `6×2` TPS 목표 XY, 행 3개의 의미 출처, 실제 선택 Y행, 목표 캔버스 크기, 승인 마스크 픽셀·외접 영역, 사용 관절 이름·최소/평균 신뢰도, 탐색 반경과 수평 구간 겹침 점수 0.0~1.0을 포함합니다. `dwpose_mask_intersection_v1`은 DWPose로 Y축 의미와 중심을 정하고 최종 좌우 픽셀은 승인 변경 마스크 안에서만 선택했다는 뜻입니다.

상의는 어깨·몸통 중간·골반 3행, 하의는 골반·무릎·발목 3행, 드레스·전신 의상은 어깨·골반·승인 마스크 외접 영역 하단 90% 3행을 사용합니다. 필수 관절 누락·중복·신뢰도 미달, 캔버스 불일치, 빈 마스크, 중복 목표점, 볼록 껍질 면적 0과 승인 영역 밖 목표점은 TPS 전에 차단합니다.

## 13. 복수 의상 조각 대응 제안 객체

### GarmentComponentMatchingSettings

전신 의상 분류 기본값은 상체 종료 0.45, 신발 시작 0.78, 전신 조각 최소 세로 점유율 0.55, 좌우 미확정 신발 X구간 0.45~0.55와 최소 규칙 적합도 0.35입니다. 모든 값은 전체 유효 의상 조각 외접 영역을 기준으로 0.0~1.0 정규화합니다.

### GarmentComponentMatchProposal

원본 조각 인덱스, 목표 슬롯, 배정 근거, 원본 외접 영역, 정규화 중심 XY·세로 점유율, 전경 픽셀 수, 규칙 적합도 0.0~1.0, 모호성 여부와 검토 사유를 포함합니다. 목표 슬롯은 `upper_body`, `lower_body`, `full_body`, `image_left_foot`, `image_right_foot`, `footwear_pair` 6종입니다. `rule_fit_score`는 학습 모델의 정확도나 확률이 아닙니다.

### GarmentComponentMatchResult

카테고리, 조각별 제안, 전체 외접 영역, 원본 조각·제안·모호 조각·공유 슬롯 조각 수와 `category_geometry_slots_v1` 방식을 포함합니다. 같은 목표 슬롯을 공유하는 조각에는 `shared_target_slot`, 좌우가 불명확한 신발에는 `footwear_left_right_unresolved`를 기록합니다. 모든 결과는 `requires_user_approval=True`, `automatic_warp_allowed=False`이며 승인 GUI가 구현되기 전에는 TPS 입력이 될 수 없습니다.

## 14. 조각별 TPS 검토·승인 객체

### GarmentWarpReviewSettings

기본값은 승인 마스크 임계값 128, 하드 알파 임계값 128, 원본 소프트 알파 여백 2px, TPS 정규화 0.0과 최대 조각 8개입니다. 참조 캔버스는 자르지 않고 목표 캔버스에 비율 유지 배치하며 반올림된 실제 가로·세로 비율과 좌·상·우·하 여백을 기록합니다.

### GarmentComponentWarpPreview

원본 조각 인덱스, 목표 슬롯, 목표 캔버스로 변환한 원본 6점, 슬롯별 목표 6점, 워핑 RGBA·소프트 알파, 원본·워핑 알파 픽셀, 증감과 조각 분리 때 제외한 다른 조각 하드 알파 픽셀을 포함합니다. 각 조각은 알파 128 이상 연결요소의 외접 영역·면적을 원본 좌표 자료와 1:1 대조한 뒤 2px 팽창 영역만 사용합니다.

### GarmentWarpReviewCandidate

조각별 미리보기, 제한 전 통합 RGBA, 승인 영역 밖 알파, 보호 후 RGBA·알파, 기준 캐릭터 위 오버레이와 승인 마스크를 소유합니다. 조각 수, 모호·공유 슬롯 수, 조각 간 하드 알파 겹침, 원본 분리에서 제외한 다른 조각 하드 알파, 승인 영역 밖 소프트·하드 알파, 제거 픽셀, 보호 후 밖 픽셀, 처리 시간과 자동 저장 0개를 기록합니다. `close()`는 조각당 2개와 통합 이미지 6개를 모두 닫습니다.

### GarmentWarpApprovedInput

`approve_garment_tps_warp_review()`를 명시적으로 호출할 때만 보호 후 RGBA·알파 복사본을 만듭니다. 보호 후 승인 영역 밖 알파는 정확히 0픽셀이고 검토 단계 자동 저장은 0개여야 합니다. 승인 객체는 Lineart·Inpaint 연결 전까지 메모리에만 존재하며 `close()`로 이미지 2개를 해제합니다.

## 15. 의상 Lineart 검토·승인 객체

### GarmentLineartSettings

기본값은 알파 임계값 128, Canny 하한·상한 50·150, Gaussian 커널 3×3, 외곽선 반경 1px, 내부 영역 침식 반경 1px와 승인 최소 선 1px입니다. Gaussian 커널은 1 이상의 홀수, Canny 값은 `0 <= lower < upper <= 255`여야 합니다.

### GarmentLineartReviewCandidate

흰 배경 의상, 외곽선, 내부 디테일, 통합선, ControlNet RGB와 적색 선 오버레이 6개를 메모리에 보유합니다. 캔버스 크기, 의상 알파 픽셀, TPS RGBA·별도 알파 불일치 픽셀, 원시 외곽·내부·중복·전체 선 픽셀, 승인 영역 밖 원시·보호 후 선 픽셀, 최종 선 픽셀, 의상 알파 대비 선 밀도 0.0%~100.0%, 처리 시간과 자동 저장 0개를 기록합니다.

TPS RGBA와 별도 알파는 독립 소유 이미지 전체를 비교하며 불일치 1px 이상이면 후보 생성을 중단합니다. 투명 RGB는 흰색 배경에 알파 합성한 뒤 Canny에 전달합니다. 원시 선의 승인 영역 밖 픽셀은 삭제 전 수치로 공개하고 최종 선은 승인 변경 영역과 교집합을 적용해 밖 픽셀 0개로 제한합니다. 선 밀도는 실제 성공·실패 표본 임계값이 확정되기 전까지 차단에 사용하지 않습니다.

### GarmentLineartApprovedInput

`approve_garment_lineart_review()`를 명시적으로 호출할 때만 ControlNet RGB와 통합 선 마스크 복사본을 만듭니다. 최종 선 1px 이상, 승인 영역 밖 최종 선 0px와 자동 저장 0개가 승인 조건입니다. `close()`는 이미지 2개를 해제합니다.

## 16. 2D 의상 Inpaint 검토·승인 객체

### GarmentInpaintSettings

별도 Python·실행기·임시·캐시·영구 벤치마크 경로, 실제 SDXL Inpaint 모델·fp16 변형, IP-Adapter Plus 모델 ID와 `models/image_encoder` ViT-H 경로를 포함합니다. 초기 수치는 strength 0.90, 28단계, CFG 5.5, IP-Adapter 0.80, mask crop 64px, 마스크 임계값 128, 1024px 참조 보드, 바깥 여백 32px, 셀 여백 16px, 최소 조각 16px, 최대 조각 8개, 제한 시간 1,800초와 float16입니다. Step 5 해제 전·후 할당 VRAM과 해제 후 예약 VRAM 3개를 MiB로 전달하며 이 값은 품질 성공률이 아니라 GPU 비교 시험값입니다.

### GarmentReferenceBoard

승인 의상 RGBA의 알파 128 이상 픽셀을 OpenCV `connectedComponentsWithStats()`로 분리합니다. 면적이 가장 큰 조각은 보드 위쪽 62%, 나머지 최대 7개는 아래쪽 최대 3열 격자에 비율 유지 배치합니다. 흰 배경 RGB 보드, 원본·유지·제외 조각 수, 원본·유지 픽셀, 보드 점유 픽셀과 조각별 원본·보드 외접 영역을 포함합니다. 원본 의상 픽셀은 변경하지 않습니다.

### GarmentInpaintReviewCandidate

기준 캐릭터, 승인 Human-Agnostic 시작 이미지, 승인 마스크, 흰 배경 원본 의상, IP-Adapter Plus 실제 참조 보드, 원시 출력, 보호 출력과 4배 RGB 차이맵 8개를 메모리에 보유합니다. 마스크 하드·소프트 픽셀, Human-Agnostic 변경 픽셀, 원시·보호 결과의 기준 캐릭터 대비 마스크 안팎 변경 픽셀, 원시 출력의 Human-Agnostic 대비 마스크 안 변경 픽셀, 승인 영역 RGB L1 평균, 참조 보드 조각·점유 픽셀, 영구 벤치마크 경로·실제 파일 수, 처리 시간과 사용자 승인 결과 자동 저장 0개를 기록합니다. 성공 벤치마크 스키마는 4이며 모델 입력 PNG 3개·감사 PNG 1개·원시 A PNG 1개·보호 PNG 1개·메타데이터 JSON 1개·실행 프롬프트 JSON 1개·stdout/stderr 2개와 진행 JSONL 최대 1개로 10~11개입니다.

기준 캐릭터·Human-Agnostic 이미지·승인 마스크는 같은 캔버스여야 합니다. Human-Agnostic 이미지는 승인 마스크 밖 변경 0px, 안 변경 1px 이상이어야 합니다. 원시 결과는 승인 마스크로 기준 캐릭터와 다시 합성하며 마스크 알파 0 위치의 최종 변경은 0px여야 합니다.

### GarmentInpaintApprovedInput

보호 후 마스크 밖 변경 0px, 기준 캐릭터 대비 마스크 안 변경 1px 이상, Human-Agnostic 대비 원시 출력의 마스크 안 변경 1px 이상과 자동 저장 0개일 때 명시적 승인 함수가 이미지 복사본을 만듭니다. 실제 의상 유사도 임계값은 사용자 성공·실패 표본이 없으므로 아직 적용하지 않습니다.

### GarmentInpaintProgress

단계, 단계 경과 초, 전체 경과 초, 실제 Diffusers 콜백 횟수, 설정 단계와 메시지를 포함합니다. 단계는 6종만 허용하고 시간은 0 이상의 유한값, 콜백은 0 이상, 설정 단계는 1 이상이어야 합니다. 실제 콜백 횟수가 설정 단계보다 크면 잘못된 이벤트로 분류합니다.

### GarmentInpaintExecutionMetrics

파이프라인 로딩, IP-Adapter 로딩, Diffusion, 출력 저장, 실행기 전체와 부모 전체 시간 6개를 초 단위로 포함합니다. 유효 이벤트, 잘못된 이벤트와 Heartbeat 횟수도 포함합니다. 잘못된 JSONL 1줄은 GPU 생성을 중단하지 않고 잘못된 이벤트 수만 1 증가시킵니다.

## 17. 자세 결과 관측 정책 설정

### pose_result_policy

임시 설정 객체이며 `mode`, `target_sample_count`, `block_on_pose_mismatch`, `switch_to_text_to_image`, `use_identity_crop` 5개 필드를 가집니다. 현재 허용값은 각각 `observe_only`, `3`, `false`, `false`, `false`입니다. 이 객체는 이미지를 만들거나 좌표를 변환하지 않고 실행 정책과 로그만 제어합니다. 허용값과 다른 설정은 GPU 호출 전에 구성 오류로 중단합니다.
