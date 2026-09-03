# GenAI Lab

캐릭터·의상·자세 참조 이미지를 바탕으로 기존 디자인을 유지하며 제한된 영역만 합성하고, 사용자가 결과를 승인하는 Windows 설치형 이미지 앱 프로젝트입니다. 최종 앱은 서버 없이 사용자의 PC에서 실행합니다.

## 개발 방식

### 기준 캐릭터의 기존 의상 교체 영역 확인

6/8 신체·Human-Agnostic 검토 전에 생성된 기준 캐릭터에서 **교체할 기존 의상**과 **꼬리·귀 등 특수 보호**를 나누어 선택합니다. 기존 SAM2 후보 UI를 재사용하며 신발·다리 의상도 바꾸려면 교체 대상으로 포함하세요. 보호할 특수 부위가 없다면 명시적으로 '특수 보호 없음'을 선택합니다. 두 역할이 겹치면 재선택하고, 취소 시 '실패 단계 다시 시도'로 재개할 수 있습니다.

AutoMasker 후보는 적용 마스크와 구별해 표시합니다. Inpaint 검토의 9번은 중립색 잔여 의심 위치로, 실제 회색 옷도 포함할 수 있어 경고만 합니다. 이번 수정 후 실제 GPU 품질은 아직 검증하지 않았습니다.

- 기간은 짧게 잡되 결과 이미지의 품질 기준은 낮추지 않습니다.
- 검증된 오픈소스와 공식 기능이 있으면 직접 다시 만들지 않고 먼저 활용합니다.
- 바이브 코딩은 AI가 제시한 코드를 그대로 붙이는 방식이 아니라, 계획을 먼저 세우고 빠르게 구현·검증하는 도구로 사용합니다.
- 추가한 코드와 오픈소스는 목적, 입력, 처리, 출력, 실패 조건과 확인 방법을 한글로 설명할 수 있어야 합니다.
- 실제 오류를 해결하거나 중요한 판단을 확정하면, AI의 평가를 덧붙이지 않고 대화에서 확인된 내용만 `docs/TROUBLESHOOTING.md`에 이어서 기록합니다.
- 기능을 합의하거나 구현한 작업은 같은 작업 안에서 메인 `README.md`의 현재 단계, 실행 흐름과 남은 일을 함께 최신화합니다.
- 모든 설명과 문서는 측정 가능한 값을 단위와 함께 기록하고 근거 위치를 연결합니다. 측정하지 않은 값은 추정하지 않고 `미측정`으로 표시하며 자세한 형식은 `docs/DOCUMENTATION_RULES.md`를 따릅니다.
- 모든 코딩 작업은 공식 기능·오픈소스 우선 확인, 직접 작성 범위 제한, 시나리오형 상위 흐름과 구현 전 사용자 승인을 규정한 [구현 계약](docs/IMPLEMENTATION_CONTRACT.md)을 따릅니다.
- 공식 문서, 모델 문서와 라이선스를 확인하지 않은 도구는 프로젝트에 추가하지 않습니다.
- 품질 문제가 실제로 확인되기 전에는 LoRA, ControlNet 또는 새로운 기능을 미리 추가하지 않습니다.
- 현재 MVP는 캐릭터 디자인 유지, 일반 의상 참조 합성, 자세 참조 적용과 사용자 승인 저장까지 포함합니다. 자유 생성은 자세 단계 완료 이후 별도 모드로 검토합니다.

## 먼저 알아둘 점

- 실행 환경: Windows, Python 3.10.6, NVIDIA RTX 4060 8GB
- 비교 모델: Stable Diffusion 1.5, 512×512
- 실제 작업 모델: Animagine XL 3.1, 전신 시험 해상도 768×1344
- 처리 방식: 후보 3장을 한 장씩 순서대로 생성
- 제외 범위: Java, 백엔드, 웹 서버, DB, API, 로그인, 개인정보 및 보안 기능, 텍스트 생성
- 현재 단계: 기준 후보 1장을 먼저 생성한 뒤 같은 후보에서 SCHP·DensePose 신체·의상 마스크와 `isnet-anime` 캐릭터 전체 외곽을 분석합니다. 원시 의상 마스크는 반경 1~3px 닫기 연산 뒤 긴 변의 1.0%를 기준으로 5~15px 팽창하고, 캐릭터 외곽은 15px 팽창합니다. 최종 변경 영역은 `팽창한 의상 ∩ 팽창한 캐릭터 외곽 - 얼굴·피부·손·발 보호 영역`입니다. SCHP 의상 픽셀은 `외곽 밖 오탐·외곽 안 보호 겹침·외곽 안 제거 검증 대상` 3개로 분류합니다. 사용자는 전처리 전 입력 5개와 가공·진단 21개를 합친 26개 화면을 `A. 입력`, `B. 마스크 가공·실제 모델 입력·오류 진단`으로 나눠 확인합니다. 실제 마스크 원천은 입력 2의 `safe_change_mask` 1개입니다. 처리 크기 변환·이분화·blur=9 뒤 금지 영역에 생긴 1~127 약한 침범과 128~255 강한 침범은 모두 최종 model_mask에서 0으로 제거하고 `WARNING` 수치로 남깁니다. 최종 model_mask의 보호 영역 또는 외곽 밖 침범이 1픽셀 이상일 때만 `BLOCK`으로 승인을 막습니다. CatVTON 실행기는 사용자가 확인한 최종 model_mask를 직접 읽으며 다시 blur하지 않습니다. 승인한 Person·이분화 마스크·model_mask·의상 입력 SHA-256 4개 중 1개라도 다르면 모델 다운로드와 GPU 추론 전에 중단합니다.
- 투명 입력 규칙: 승인 추출본 유무와 관계없이 CatVTON 의상 입력의 RGBA·LA·투명 P 형식은 흰색 `(255, 255, 255)` 배경에 합성한 RGB로 변환합니다. 2026-08-30 집중 검사에서 승인 경로와 예비 경로 2개가 통과하고 0개가 실패했으며 실행 시간은 2.28초, GPU 호출은 0회였습니다.
- 자세 참조 입력: PNG·JPEG 1개를 선택해 원본, `가로×세로`, 전체 픽셀, 가로/세로 비율과 파일 크기를 확인하고 승인 또는 거절할 수 있습니다. 최소 변은 `64px`, 전체 상한은 `40,000,000px`입니다. 승인 뒤 DWPose가 CPU에서 관절 18개를 추출하고 자세 원본·관절 겹쳐보기·표준 OpenPose 뼈대 지도 3개를 다시 사용자에게 공개합니다. 초기 품질 기준은 탐지 관절 `8/18개` 이상과 어깨·골반·무릎·발목 그룹 각각 좌우 중 `1개 이상`, 뼈대 유효 픽셀 `1px 이상`입니다. 통과 자세는 `D:\genai-cache\genai-lab\approved-poses\last-approved`의 PNG 2개·JSON 1개로 저장합니다. 이후 DWPose 실행 실패·`600초` 초과·품질 미달이면 SHA-256을 검증한 마지막 승인 자세 3화면을 공개하고 사용자가 재승인한 경우에만 계속합니다. 이번 변경의 실제 GUI 폴백 실행·실제 저장은 각각 `0회·0건`입니다. ControlNet 초기값은 강도 `0.65`, 적용 구간 `0.00~0.80`, 이미지 변경 강도 `0.35`입니다.
- 자동 실행: 캐릭터·의상·자세 경로를 먼저 등록한 뒤 `전체 이미지 생성 시작` 버튼을 1회 누릅니다. 참조 준비, 의상 추출·분석, 자세 추출, 기준 후보 생성, 후보 신체/Human-Agnostic 승인, 2D Inpaint와 최종 검토를 활성 8단계로 순차 실행합니다. 별도 `관절 추출 시작`과 `캐릭터 신체 비교 시작` 버튼은 숨겼습니다. TPS·Garment Lineart 모듈은 진단 자산으로만 보존하고 자동 생성 입력과 승인 단계에서는 제외합니다.
- 로컬 시험 정책: CatVTON의 안전 검사는 `safety_check_enabled: false`로 비활성화합니다. 외부 API 호출 정책이 아니라 로컬 CatVTON 결과를 `NSFW.jpg`로 교체하는 기능만 끕니다. 실행 로그와 별도 실행 기록에 실제 값 `false`가 남아야 하며 설정과 기록이 다르면 결과를 거절합니다.

전문 용어는 한글 뜻을 먼저 적습니다. 예를 들어 **그림체 학습용 추가 가중치(LoRA)**는 큰 모델 전체를 다시 학습하지 않고 그림체에 필요한 작은 부분만 학습하는 방법입니다.

## 전체 흐름

활성 Inpaint에 전달하는 의상은 승인 원본을 직접 변경하지 않습니다. 모델 입력용 복사본만 흰 배경 RGB로 만들고, 추출 원본 크기·알파 픽셀 수·알파 점유율을 로그에 남깁니다. CatVTON 입력 전처리는 진단·비교 경로로 보존합니다.

```text
캐릭터·의상·자세 이미지 경로 등록
                ↓ 전체 생성 버튼 1회
[1/8] 캐릭터 참조 화질 확인·필요한 보정 승인
                ↓ 자동
[2/8] 의상 영역 탐지·SAM2 마스크 → 사용자 승인
                ↓ 자동
[3/8] 원본 의상 픽셀 추출·WD14 디자인 분석 → 사용자 승인
                ↓ 자동
[4/8] DWPose 관절 18개·표준 OpenPose 지도 → 사용자 승인
                ↓ 자동
[5/8] Animagine XL + IP-Adapter + 자세 ControlNet 기준 후보 생성
                ↓ 자동
[6/8] 같은 후보의 신체·기존 의상 마스크·Human-Agnostic → 사용자 승인
                ↓ 자동
[7/8] Human-Agnostic + 승인 마스크 + 추출 의상 → Animagine Inpaint + IP-Adapter
                ↓
[8/8] 얼굴·손 제한 영역 보정 → 최종 사용자 판단
        ├─ 승인 → 저장 여부 선택
        └─ 거절 → 저장하지 않고 재생성
```

현재 GUI는 파일 선택 시 AI 호출 `0회`로 경로만 등록하고, 전체 생성 버튼 1회 뒤 활성 8단계를 자동 실행합니다. 의상 위치, SAM2 마스크, 원본 픽셀 추출, WD14 분석, 선택 자세 DWPose, 기준 후보 생성, 같은 후보 신체/Human-Agnostic 승인, 2D Inpaint 순서입니다. Inpaint 검토는 기준 캐릭터·Human-Agnostic 시작 이미지·승인 마스크·승인 의상 원본·IP-Adapter Plus 실제 참조 보드·원시 출력·보호 출력·차이맵 `8개`를 공개합니다. 검토 내용은 스크롤되며 승인·거절 버튼은 창 아래에 고정됩니다. 동시에 열 수 있는 승인 창은 1개이고 승인 창이 열린 동안 자동 재개는 0회, 닫힌 뒤 예약 재개는 1회입니다.

기존 의상 전처리·분석 단위 테스트는 16개 통과, 0개 실패, Windows 실행 시간 12.96초였습니다. 2026-08-28 신체 비교·CatVTON 마스크 집중 검사는 15개 통과, 0개 실패, 공유 폴더 실행 시간 1.25초였습니다. 2026-08-29 승인 입력 연결 뒤 의상 계약·신체 비교 집중 검사는 21개 통과, 0개 실패, 1.51초였고 변경 Python 파일 5개의 구문 검사를 통과했습니다. 사용자는 흰 와이셔츠 1건에서 마스킹·원본 의상 추출·WD14 키워드 추출을 직접 확인했습니다. 마스킹 체감 품질 약 99%는 자동 정확도가 아니라 사용자 육안 평가 1건입니다. 새 승인 입력으로 실제 GPU CatVTON을 실행한 횟수는 아직 0회이므로 단위 테스트 결과를 실제 합성 품질 증명으로 기록하지 않습니다.

2026-08-30 안전 검사를 끈 실제 GPU CatVTON 실행 1회는 전체 329.5초, 마스크 밖 변경 0픽셀, AutoMasker 0회였습니다. 결과에는 회색 중립화 영역이 남고 새 의상이 반영되지 않아 품질 실패 1건으로 기록합니다. 원인은 Human-Agnostic 이미지를 CatVTON에 전달해 공식 파이프라인 내부 마스킹과 겹친 이중 마스킹 및 생성 후보와 승인 마스크의 좌표 불일치였습니다. 수정 뒤 집중 검사는 27개 통과, 0개 실패, 1.74초이며 실제 GPU 재검증은 아직 0회입니다.

2026-08-30 의상 조건 입력 보완 뒤 의상 계약 테스트는 18개 통과, 0개 실패, 1.56초였습니다. 승인 RGBA 추출본의 알파 경계 상자를 사용하므로 모델 입력의 불필요한 투명 여백은 0픽셀 폭으로 줄어듭니다. 실제 GPU 재검증은 아직 0회이므로 의상 재현 품질이 개선됐다고 기록하지 않습니다.

2026-08-31 `isnet-anime` 외곽 제한 구현 뒤 신체 비교 테스트는 13개 통과, 0개 실패, 1.18초였고 변경 Python 파일 4개의 구문 검사와 별도 runner 도움말 실행 1회를 통과했습니다. 실제 `isnet-anime` 추론과 GPU CatVTON 재검증은 각각 0회이므로 꼬리 보호와 의상 합성 품질이 개선됐다고 기록하지 않습니다.

2026-08-31 기존 의상 잔여 검사 기준 통일 뒤 신체 비교 테스트는 14개 통과, 0개 실패, 2.23초였습니다. 합성 입력 1건에서 외곽 밖 오탐 1픽셀과 외곽 안 의상 1픽셀을 분리해 오탐률 50.000%, 외곽 안 잔여 0픽셀, 제거율 100.000%를 확인했습니다. 실제 사용자 이미지 재검증과 GPU CatVTON 호출은 각각 0회입니다.

2026-08-31 CatVTON Preflight 연결 뒤 합성 입력 1건을 256×384로 변환해 이분화 마스크 8,385픽셀, blur=9 이후 model_mask 18,701픽셀, 보호 영역 침범 0픽셀, 외곽 밖 침범 0픽셀을 확인했습니다. 잘못된 SHA-256 4개를 넣은 실행 1건은 모델 다운로드와 GPU 추론 전에 실패했습니다. 현재 Windows Python 2개와 CatVTON Python에는 pytest가 없어 단위 테스트 실행 수는 0개이며, 변경 Python 파일 7개의 구문 검사와 실행기 도움말 2개는 통과했습니다. 실제 사용자 이미지 Preflight와 GPU CatVTON 재검증은 각각 0회입니다.

2026-08-31 CatVTON 전처리 전 입력 공개를 추가해 입력 5개와 가공·진단 18개를 합친 23개 화면으로 분리했습니다. 합성 입력 집중 검사 3건은 정상 입력 1건 통과, 보호·외곽 침범 입력 1건 차단, 좌표 불일치 입력 1건 예외로 모두 통과했으며 실패 0개, 실행 시간 0.003초였습니다. 변경 Python 파일 3개의 구문 검사와 diff 검사도 통과했습니다. 새 GUI 화면의 실제 사용자 이미지 확인과 GPU 호출은 각각 0회입니다.

2026-08-31 blur 침범 분류 뒤 실제 Diffusers `blur_factor=9` 합성 입력 1건은 약한 침범 13,800픽셀, 강한 침범 0픽셀, 금지 영역 제거 13,800픽셀, 최종 model_mask 177,338픽셀, 보호·외곽 침범 각각 0픽셀로 통과했습니다. 잘못된 SHA-256 4개를 넣은 실제 실행기 검사 1건은 `model_mask` 불일치를 포함해 실패했고 출력 파일 0개, 가짜 모델 ID 접근 0회였습니다. 승인 model_mask 해시와 입력 좌표 집중 테스트는 5개 통과, 0개 실패, 0.004초였고 Python 파일 8개의 구문 검사와 GUI import를 통과했습니다. GPU 추론은 0회입니다.

2026-08-31 Human-Agnostic 공통 가드레일 판정을 추가했습니다. 각 검사는 `PASS`, `WARNING`, `BLOCK` 중 하나와 코드·측정값·기준값·보정 후 값·복구 행동을 반환합니다. `128×128px` 합성 마스크 내부의 `8×8px` 보호 영역에 Diffusers `blur_factor=9`를 적용했을 때 보정 전 강한 침범은 `64px`, 금지 영역 제한 후 최종 침범은 `0px`이었습니다. 이 사례는 `WARNING 1개`, `BLOCK 0개`, 승인 가능 `YES`로 판정합니다. 가드레일 테스트 8개와 Preflight 테스트 5개는 총 13개 통과, 0개 실패였습니다. 실제 사용자 이미지 GUI 재검증과 CatVTON GPU 추론은 각각 0회입니다.

### 확정된 의상 추출 검토 정책

의상 추출은 다음 7개 화면을 순서대로 공개하며 중간 단계를 숨기지 않습니다. 승인 뒤 CatVTON 로그에는 추출 원본 크기, 모델 조건 이미지 크기, 알파 픽셀 수와 조건 이미지 내부 알파 점유율을 각각 수치로 남깁니다.

1. 정규화된 원본
2. Grounding DINO 위치 사각형
3. 의상 위치 최대 8개와 영역별 SAM2 흑백 마스크 후보 최대 3개
4. 영역별 선택 마스크를 합친 전체 마스크
5. 원본 픽셀을 유지한 투명 배경 의상
6. WD14 의상 종류·색상·재질·장식 분석
7. 전체 추출 자료 사용자 승인

추출은 원본 RGB를 다시 그리지 않고 알파 채널만 바꿉니다. 마스크 내부 변경 픽셀은 0개, 원본 픽셀 보존율은 100.000%여야 하며 변경 픽셀이 1개 이상이면 실패 처리합니다. 공백 복원도 RGB를 새로 칠하지 않고 원본 픽셀의 알파만 복원합니다. 흰 의상 규칙이 실제 구멍을 의상으로 오인할 수 있으므로 공백 처리 전후 마스크와 추출본을 사용자가 반드시 다시 승인합니다. 사용자 승인 전 CatVTON 실행은 0회, 결과 폴더 저장은 0개입니다. 거절하면 중간 이미지를 메모리에서 해제합니다.

### 확정된 개발 순서

1. 캐릭터 디자인 유지와 승인 전 저장 금지를 유지합니다.
2. 일반 의상 이미지에서 의상 위치를 추출하고 디자인을 분석합니다.
3. 추출 의상과 적용 예정 마스크를 사용자가 승인한 뒤 CatVTON을 실행합니다.
4. 의상 단계가 안정되면 자세 참조 적용을 구현합니다.
   - 완료: 자세 이미지 선택, 규칙 검사, 미리보기와 사용자 승인
   - 완료: DWPose CPU 관절 18개 추출, 원본 겹쳐보기, 표준 OpenPose 지도와 사용자 승인
   - 코드 완료: 승인 OpenPose 지도를 SDXL ControlNet 입력으로 전달
   - 실제 확인 필요: 같은 시드 1장으로 자세 반영·캐릭터 유지·최대 VRAM 측정
5. 자세 단계까지 완료된 이후에만 자유 생성 모드를 별도 흐름으로 검토합니다.

현재 프로젝트는 텍스트로 새로운 디자인을 만드는 자유 생성기가 아니라 참조 기반 제한 합성 앱입니다. 보이지 않는 부분을 연결하는 최소 추정만 허용하고, 캐릭터·의상·자세 디자인의 임의 변경은 허용하지 않습니다.

향후 외부 LLM 또는 멀티모달 API를 연결해도 분석 후보만 반환하게 하며, Python 마스크 검사와 사용자 승인을 우회할 수 없습니다. 외부 API가 신규 생성을 요청하거나 거절되면 현재 합성 가드레일을 풀지 않고 지원하지 않는 요청 또는 제공자 실패 상태로 처리합니다.

사용자 판단 흐름은 [SCENARIOS.md](docs/SCENARIOS.md), 자세한 입력과 출력 흐름은 [FLOW.md](docs/FLOW.md), 블록 사이의 데이터는 [DATA_MODELS.md](docs/DATA_MODELS.md), 수치와 증거 작성 방식은 [DOCUMENTATION_RULES.md](docs/DOCUMENTATION_RULES.md), 각 파일이 필요한 이유는 [STRUCTURE.md](docs/STRUCTURE.md)를 먼저 읽습니다. 프로젝트를 진행하며 막힌 점과 내가 이해하게 된 내용은 [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)에 기록합니다.

## 프로젝트 구조

```text
genai-lab/
├─ README.md
├─ run.py
├─ genai_lab/
│  ├─ model.py              # 모델과 GPU
│  ├─ reference.py          # 참조 화질 검사와 승인 전 확대 복원
│  ├─ catvton_preflight.py  # 실제 CatVTON 전처리 입력·해시·침범 검토 계약
│  ├─ image_digest.py       # 이미지 모드·크기·픽셀 SHA-256 계산
│  ├─ detail.py             # 얼굴·손 탐지와 제한된 부분 보정
│  ├─ style.py              # 참조 그림
│  ├─ clothing.py           # 의상 합성 허용 영역과 신체 보호 검사
│  ├─ garment_warp.py       # 공통 좌표 의상 RGB·알파 TPS 워핑 PoC
│  ├─ garment_landmarks.py  # 의상 조각별 상·중·하단 좌우 6점 추출
│  ├─ character_target_landmarks.py # DWPose·승인 마스크 TPS 목표 6점
│  ├─ pose_fallback.py      # 마지막 승인 자세 저장·품질·SHA-256 폴백
│  ├─ garment_component_matching.py # 복수 의상 조각 목표 슬롯 제안
│  ├─ garment_warp_review.py # 조각별 TPS·보호 전후 승인 미리보기
│  ├─ garment_lineart.py    # TPS 의상 외곽선·내부 디테일 조건 검토
│  ├─ garment_inpaint.py    # 2D Inpaint 실행·보호 합성·승인 후보
│  ├─ clothing_reference.py # 의상 입력 정규화와 추출·분석 준비 계약
│  ├─ generator.py          # 한 장씩 생성
│  └─ result.py             # 파일과 실행 기록
├─ scripts/
│  └─ catvton_runner.py     # CatVTON 별도 환경을 한 번 실행하는 연결점
│  └─ catvton_preflight_runner.py # 공식 전처리만 실행하고 GPU 추론은 하지 않음
│  └─ garment_inpaint_runner.py # Animagine Inpaint·IP-Adapter 별도 실행
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
│  ├─ DOCUMENTATION_RULES.md
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

DWPose 관절 18개 분석은 사용자가 승인한 자세 참조 이미지에만 같은 CatVTON 가상환경의 CPU로 1회 실행합니다. 2026-08-28에 `easy-dwpose 1.0.2`, `onnxruntime 1.20.1`, `loguru 0.7.3` 설치와 import를 확인했습니다. 최초 실제 실행 1회에는 `RedHash/DWPose`의 사람 탐지 ONNX와 자세 추정 ONNX를 `D:\genai-cache\huggingface`에 내려받습니다. 다음 3개 패키지가 없으면 자세 참조 단계에서 한글 오류를 표시하고 기준 후보 생성으로 넘어가지 않습니다.

```powershell
& "D:\genai-cache\catvton-venv\Scripts\python.exe" -m pip install --no-deps `
  easy-dwpose==1.0.2 onnxruntime==1.20.1 loguru==0.7.3
```

캐릭터 전체 외곽은 같은 환경의 CPU에서 `skytnt/anime-seg`의 `isnetis.onnx`를 ONNX Runtime으로 1회 직접 실행해 만듭니다. 입력은 1024×1024에 비율 유지로 맞추고 자르기 0px, 검은 여백을 사용합니다. 출력에서는 같은 여백을 제거하고 기준 후보 크기로 복원합니다. 최초 실행 때 176MB ONNX 파일을 `D:\genai-cache\huggingface`에 내려받으며 추가 rembg 패키지는 사용하지 않습니다.

이 마스크는 귀·꼬리를 의미별로 분류하지 않고 캐릭터의 보이는 전체 외곽만 제한합니다. 따라서 몸에서 15px보다 멀리 떨어진 장식이나 분리된 꼬리 조각은 차단될 수 있으며, 12개 미리보기의 원본 외곽과 팽창 외곽을 사용자가 승인해야 합니다. [anime-segmentation 저장소](https://github.com/SkyTNT/anime-segmentation), [공식 ONNX 모델](https://huggingface.co/skytnt/anime-seg/blob/main/isnetis.onnx)

DWPose 좌표는 승인 자세의 OpenPose 지도를 만드는 용도로만 사용합니다. 기준 후보의 기존 의상과 보호 영역은 SCHP·DensePose로 분석하며 DWPose를 다시 실행하지 않습니다. [easy-dwpose 저장소](https://github.com/reallyigor/easy_dwpose), [OpenCV 닫기 연산](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)

저장 자세 폴백은 임의 자세를 생성하거나 무작위로 고르지 않습니다. 정상 추출과 사용자 승인을 모두 통과한 가장 최근 자세 `1개`만 교체 저장하며, 불러올 때 지도 크기·관절 `18개`·탐지/누락 합계 `18개`·현재 품질 기준·픽셀 SHA-256을 다시 확인합니다. 요청 자세 실패 원본, 저장 자세 원본, 저장 뼈대 지도 `3개`를 비교한 뒤 사용자가 거절하면 4/8 단계에서 중단합니다. 사용자의 정상 자세 승인 거절은 모델 추출 실패가 아니므로 자동 폴백하지 않습니다. [DWPose](https://github.com/IDEA-Research/DWPose), [Python `os.replace`](https://docs.python.org/3/library/os.html#os.replace)

CatVTON 코드와 가중치는 비상업용 조건(CC BY-NC-SA 4.0)이므로 현재 로컬 연구·시험에만 사용합니다. 설치형 앱을 배포하거나 유료로 제공하기 전에는 CatVTON을 교체하거나 별도 허가를 받아야 합니다.

과거 시험에서 회색 Human-Agnostic 이미지를 CatVTON 인물 입력으로 전달해 회색 영역이 결과에 남았습니다. 현재는 기준 후보 원본, 승인 마스크와 투명 의상 추출본만 전달하며 Human-Agnostic 이미지는 검토 화면에만 사용합니다. 12개 미리보기에서 캐릭터 외곽 안 기존 의상 잔여 0픽셀과 검증 대상 제거율 100.000%를 통과해야 실행하며, 실행기 반환 마스크가 승인 마스크와 1픽셀이라도 다르면 실패합니다.

CatVTON 프로세스 종료 코드와 의상 영역 밖 변경 0픽셀만으로 합성 성공을 판정하지 않습니다. 비교창은 의상 적용 전, CatVTON 원시 출력, 의상 변경 허용 영역, 최종 보호 합성과 RGB 차이를 4배 밝힌 차이맵 5개를 공개합니다. `TryOnEffectMetricsResult`는 원시 model_mask 안 변경 픽셀, 최종 승인 영역 안 변경 픽셀, 보호 합성에서 제거된 픽셀, 승인 영역 RGB L1 평균과 영역 밖 변경 픽셀을 기록합니다. 최종 승인 영역 안 변경이 정확히 0픽셀이면 `completed`가 아닌 `no_effect`이며 적용본 승인 버튼을 비활성화하고 원본 후보 선택은 유지합니다. CLIP·DINOv2·색상 히스토그램은 성공·실패 표본을 각각 30건 이상 수집해 임계값을 보정하기 전까지 차단 조건으로 사용하지 않습니다.

참조 의상과 캐릭터 신체의 좌표 대응이 없는 문제를 분리 검증하기 위해 OpenCV TPS 독립 PoC를 추가했습니다. `GarmentTpsWarpRequest`는 같은 캔버스의 의상 RGBA, 의상 원본점과 캐릭터 목표점을 각각 최소 5개 받습니다. RGB는 알파 선곱 상태로, 0~255 소프트 알파는 별도로 같은 TPS에 통과시켜 투명 경계의 검정색 번짐을 막습니다. 좌표 개수 불일치, 중복, NaN·무한대, 캔버스 밖 좌표, 일직선 배치와 알파 0픽셀 입력은 AI·GPU 호출 전에 중단합니다.

`extract_garment_mask_landmarks()`는 승인 알파를 128 기준으로 기하 분석하고, 16px 미만 연결요소를 노이즈로 분리한 뒤 각 유효 조각의 높이 10%·50%·90% 주변 ±3% 탐색 밴드에서 가장 넓은 행의 좌우 6점을 반환합니다. 이 6점은 어깨·허리 관절이 아니라 `mask_geometry_v1` 경계 좌표입니다. 재킷과 신발처럼 유효 조각이 2개 이상이면 조각별 좌표를 공개하지만 단일 TPS 자동 연결은 차단합니다. 현재 GUI·CatVTON·Inpaint 연결, 캐릭터 목표점 산출과 실제 사용자 의상 좌표 검증은 각각 0회입니다.

`extract_character_target_landmarks()`는 승인 DWPose 좌표를 기존 `PoseControlPreparedInput`의 확대 비율·좌우/상단 여백으로 생성 캔버스에 투영하고, 승인 변경 마스크 알파 128 이상 영역에서 TPS 목표 6점을 찾습니다. 상의는 어깨·몸통 중간·골반, 하의는 골반·무릎·발목, 드레스·전신 의상은 어깨·골반·승인 마스크 하단 90%를 Y축 기준으로 사용합니다. 각 행은 최대 `min(64px, 높이의 5%)`까지만 검색하고 실제 X축은 승인 마스크의 가장 가까운 연속 구간 경계로 제한합니다. 필수 관절 신뢰도는 각각 0.30 이상이며 최종 6점은 모두 승인 영역 안이어야 합니다. 장갑·신발 단독은 아직 지원하지 않습니다.

`propose_garment_component_matches()`는 추출된 복수 의상 조각을 상체·하체·전신·이미지 기준 왼발·오른발·좌우 미확정 신발 슬롯으로 분류합니다. 상의·하의·드레스는 사용자가 선택한 카테고리를 우선하며, 전신 의상은 전체 외접 영역 기준 상단 0.00~0.45, 하단 0.45~0.78, 신발 0.78~1.00과 세로 점유율 0.55 이상 전신 조각 규칙을 사용합니다. 이 값은 모델 정확도가 아닌 `category_geometry_slots_v1` 규칙 적합도입니다. 같은 슬롯에 2개 이상 배정되거나 신발 중심 X가 0.45~0.55이면 모호성 사유를 기록하며, 모든 제안은 사용자 승인 전 `automatic_warp_allowed=False`입니다.

`create_garment_tps_warp_review()`는 참조 RGBA를 생성 캔버스에 비율 유지 배치하고, 조각별 원본 6점과 슬롯별 목표 6점으로 기존 TPS를 실행합니다. 목표 세로 구간은 상체 `(0.00, 0.25, 0.50)`, 하체 `(0.50, 0.75, 1.00)`, 전신 `(0.00, 0.50, 1.00)`, 신발 `(0.78, 0.89, 1.00)`입니다. 왼발·오른발은 이미지 X축 중앙에서 나눕니다. 조각 분리는 외접 사각형이 아니라 알파 128 이상 연결요소 라벨을 대조하고 2px 팽창하되, 다른 조각 하드 알파는 제거 전 픽셀 수를 기록한 뒤 제외합니다.

검토 후보는 조각별 워핑, 마스크 제한 전 통합 RGBA, 승인 영역 밖 알파, 보호 후 RGBA·알파, 기준 캐릭터 위 기하 합성 미리보기와 승인 마스크를 메모리에만 보관합니다. 승인 영역 밖 알파를 1~127 소프트와 128~255 하드로 나눠 기록하고 보호 후 밖 픽셀은 0개여야 승인 복사본을 만들 수 있습니다.

`create_garment_lineart_review()`와 TPS 모듈은 기존 비교·진단 자산으로 보존합니다. 다만 실제 GPU 결과에서 TPS RGB 확대와 동일 TPS 기반 Lineart가 늘어난 사람 형상을 강화한 것이 확인되어 활성 자동 워크플로우와 2D Inpaint 입력에서는 모두 제외합니다. 두 모듈의 개별 테스트와 수동 진단 함수는 유지하지만 사용자 자동 승인 창은 호출하지 않습니다. [OpenCV Canny](https://docs.opencv.org/4.13.0/da/d22/tutorial_py_canny.html), [OpenCV Morphological Gradient](https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html)

`execute_garment_inpaint()`는 `GarmentGenerationEngine`을 통해 승인 Human-Agnostic 이미지와 승인 변경 마스크를 별도 프로세스에 전달합니다. 승인 의상 RGBA는 OpenCV 연결요소로 최대 8개를 분리하고 가장 큰 조각을 위쪽 주 영역, 나머지를 아래 격자에 배치한 1024×1024 흰 배경 참조 보드로 바꿉니다. 실행기는 실제 SDXL Inpaint 체크포인트와 IP-Adapter Plus를 조합하고 `models/image_encoder`의 ViT-H 인코더를 명시적으로 사용합니다. 인코더 hidden size와 Plus 투영층 입력 차원이 다르면 Diffusion 전에 중단하며, SDXL의 두 CLIP 토크나이저를 모두 검사한 실사용 프롬프트는 `prompt_execution.json`에 기록합니다. TPS RGB 직접 합성과 Garment ControlNet 호출은 각각 `0회`입니다. 초기값은 strength `0.90`, 28단계, CFG `5.5`, IP-Adapter `0.80`, mask crop `64px`, 제한 `1,800초`입니다. 원시 출력은 그대로 채택하지 않고 승인 마스크로 기준 캐릭터와 다시 합성하여 마스크 알파 0 위치의 변경을 `0px`로 만듭니다.

별도 실행기는 영구 `progress.jsonl` 1개에 `runner_started → pipeline_loading → ip_adapter_loading → diffusion_running → output_saving → completed` 6단계를 기록합니다. 부모는 1초 간격으로 읽고 새 이벤트가 없어도 5초마다 Heartbeat를 GUI에 전달합니다. Diffusers `callback_on_step_end`가 발생한 실제 횟수와 설정값 28단계를 분리해 표시하므로 실제 콜백 수가 28보다 작아도 거짓 백분율을 만들지 않습니다. 검토창은 파이프라인·IP-Adapter·Diffusion·출력 저장·실행기 전체·부모 전체 6개 시간, 유효·잘못된 이벤트와 Heartbeat 횟수, 벤치마크 경로와 실제 파일 수를 공개합니다. [Diffusers Pipeline Callback](https://huggingface.co/docs/diffusers/using-diffusers/callback)

7/8 직전에는 Step 5 Worker 종료를 100ms 간격으로 확인하고, GUI·Worker의 파이프라인 참조를 모두 제거한 뒤 `gc.collect()`, CUDA 동기화와 `empty_cache()`를 순서대로 실행합니다. 해제 전·후 할당 VRAM과 해제 후 예약 VRAM을 MiB로 GUI와 `metadata.json`에 기록합니다. 벤치마크 폴더는 `outputs/debug_benchmark/<시각>_seed<시드>`이며 모델 입력 `initial.png`, `mask.png`, `garment_board.png` 3개와 감사용 `garment_source.png` 1개, 각 SHA-256, 전체 프롬프트·시드·모델·추론·보드 배치 수치를 저장합니다. 성공하면 `raw_output_A.png`와 `protected_output.png`, 실패·1,800초 초과면 모든 입력·마지막 진행 상태·stdout·stderr를 보존합니다. 이 파일은 최적화 비교 자료이며 사용자 승인 결과 자동 저장 `0개` 계약과 구분합니다.

2026-09-02 Step 5 해제와 Step 9 영구 벤치마크 변경의 Inpaint·GUI 집중 테스트는 `28/28` 통과했고 실행 시간은 `10.61초`였습니다. 전체 회귀는 `124개+72개=196/196` 통과했고 실행 시간은 `17.32초+10.21초=27.53초`였습니다. 변경 후 실제 GPU Inpaint 실행은 `0회`이므로 1,800초 초과 해결 여부와 이미지 품질은 아직 미측정입니다.

검토 후보는 기준 캐릭터·Human-Agnostic 시작 이미지·승인 마스크·원본 의상·실제 참조 보드·원시 출력·보호 출력·4배 차이맵 `8개`와 안팎 변경 픽셀·RGB L1 평균·참조 보드 구성요소 수를 공개합니다. Human-Agnostic 이미지가 승인 마스크 밖을 `1px`이라도 바꾸거나 마스크 안 변경이 `0px`이면 GPU 실행 전에 중단합니다. 승인 영역 안 결과 변경 `0px` 또는 원시 출력과 Human-Agnostic 시작 이미지의 마스크 안 차이 `0px` 후보는 승인할 수 없습니다. 이전 Animagine legacy Inpaint 실행은 `232.6초`였지만 회색 영역 `91.881%`가 남아 품질 실패로 판정했습니다. 실제 Inpaint 체크포인트와 IP-Adapter Plus 변경 후 사용자 이미지 GPU 검증은 `0회`이므로 속도와 의상 재현 품질은 아직 미측정입니다. [Diffusers Inpainting](https://huggingface.co/docs/diffusers/using-diffusers/inpaint), [IP-Adapter](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter), [PyTorch CUDA 캐시](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html)

### 자세 결과 우선 관측 정책

자세와 의상 합성 결과를 먼저 확보하기 위해 `pose_result_policy.mode=observe_only`를 사용합니다. 기존 생성 입력은 Img2Img strength `0.35`, 자세 ControlNet scale `0.65`·적용 구간 `0.00~0.80`, 기준 이미지 IP-Adapter scale `0.80`으로 유지합니다. Text2Img 전환 `0회`, 정체성 crop 사용 `0회`, 생성 후 자세 불일치 차단 `0회`이며 새 레이아웃 제약도 추가하지 않습니다.

실제 사용자 결과 목표는 `3건`이고 현재 신규 GPU 검증은 `0/3건`입니다. 각 실행 로그의 `임시 자세 결과 정책` 단계에 모드, 목표 건수, 자세 불일치 차단, Text2Img 전환, 정체성 crop과 기존 생성 경로 유지 여부를 기록합니다. 승인 마스크 밖 최종 변경 `0px`, 승인 마스크 안 변경 `1px 이상`, 자동 저장 `0개`의 기존 결정론적 보호 계약은 그대로 유지합니다.

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
