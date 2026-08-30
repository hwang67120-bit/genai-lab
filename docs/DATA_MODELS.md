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

GUI에서 사용자가 승인한 Human-Agnostic 이미지 1개, 변경 마스크 1개, CatVTON 의상 종류와 승인 마스크 픽셀 수를 담습니다. GUI가 복사본 2개를 작업 스레드에 넘기며 작업 성공·실패와 관계없이 스레드 종료 시 두 이미지를 닫습니다.

### CatVTONExecutionMetadata

별도 실행기가 실제 사용한 입력을 증명하는 10개 값을 담습니다.

- 마스크 출처: `user_approved`
- AutoMasker 실행 횟수: `0회`
- 승인 이미지 크기: 가로·세로 픽셀
- 승인 마스크 픽셀 수
- CatVTON 처리 크기로 변환한 뒤의 마스크 픽셀 수
- 안전 검사 실행 여부: 로컬 시험에서는 `false`
- 인물 입력 출처: `generated_candidate`
- 인물 입력 크기: 가로·세로 픽셀

출처가 다르거나 AutoMasker 실행 횟수가 1회 이상이거나 입력 크기·픽셀 수가 승인 기록과 다르면 결과를 거절합니다.
안전 검사 실행 여부도 요청 설정과 다르면 결과를 거절합니다.

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

### ConfirmedCharacterBodyComparison

사용자가 10개 중간 결과를 확인한 뒤 승인한 신체 비교 결과입니다.

- 포함: 정확한 Human-Agnostic 이미지 1개, 정확한 변경 마스크 1개, 관절·마스크·중립화 수치
- 기존 의상 제거: 탐지 픽셀 수, 잔여 픽셀 수와 제거율
- 임시 값: 사용자가 의상 종류나 캐릭터 이미지를 바꾸면 이미지 2개를 즉시 닫고 승인을 무효화
- 현재 외부 전달: 복사본을 GUI 작업 스레드와 CatVTON 별도 프로세스에 전달
- 실행 전 검사: 크기, 의상 종류, 마스크 픽셀 수가 모두 승인 기록과 일치해야 함

현재 구현은 같은 생성 후보에서 만든 승인 마스크를 CatVTON 입력까지 연결합니다. Human-Agnostic 이미지는 검토 자료로만 사용하며 실행기는 `generated_candidate` 원본을 받습니다. 실제 GPU 재검증은 0회입니다.

### OriginalClothingRemovalVerification

SCHP가 탐지한 기존 의상 중 승인 변경 마스크 밖에 남은 위치를 수치와 흑백 마스크로 나타냅니다. 탐지·포함·잔여 픽셀 수, 제거율과 통과 여부를 포함합니다. 통과 조건은 제거율 `100.000%`, 잔여 `0픽셀`이며 실패하면 CatVTON 호출은 `0회`입니다.
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
