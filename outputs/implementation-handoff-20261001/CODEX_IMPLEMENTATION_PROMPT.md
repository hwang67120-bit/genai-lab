# Codex 구현 프롬프트 — 1회 생성 후보 경로 (옵션, 기본 OFF)

작성: Claude, 2026-10-01. 이 프롬프트는 `outputs/handoff-20260930/02_implementation_prompt.md` 를 대체한다.
근거·결정 상태·파일 위치는 같은 폴더의 `README.md`(구현 인수인계)가 기준이다. 둘이 다르면 README 가 우선한다.

---

## 0. 먼저 읽을 것
1. `outputs/implementation-handoff-20261001/README.md` 전체 — 특히 2-3 해석상 주의점, 3 구현 후보 표, 4 결정 대기, 5 출시 차단, 6 검증.
2. `docs/FLOW.md`(현재 제품 계약), `docs/DECISIONS.md` D-074(후보 방향, 운영 미반영).
3. 시험 구현 참고(복사 금지, 구조만): `outputs/integration-20260929/{make_plan.py,run_int.py,gate.py}`, `outputs/e2e-20261001/{analyze.py,face.py,plan.py,post.py}`, `outputs/gender-contract-20261001/make_plan.py`.

읽은 뒤, 구현 시작 전에 아래를 보고하고 사용자 확인을 받는다.
- 작업 브랜치(현재 `codex/sdxl-local-garment-strength-20260923` 에 미푸시 커밋 2개가 있다 — 그 위에서 할지, 새 브랜치를 만들지)
- 추적 안 된 항목 `afe_load(p.read_text())`, `approved-bases/`, `inputs/training_candidates/` 처리(건드리지 말 것, 출처 확인만)
- 의존성: 얼굴 자동 크롭에 `dghs-imgutils`(0.19.0)가 운영 venv(D:/genai-cache/venv)에 없다. DWPose(`easy_dwpose`)도 운영 venv 에 없고 catvton-venv 에만 있다. **패키지 설치는 사용자 승인 후에만.** 승인 전에는 해당 단계를 인터페이스·테스트까지만 만들고 멈춘다.

## 1. 목표와 경계
- 목표: README 3절의 "이번 구현" 기능을 제품 코드에 **명시적 옵션(기본 OFF)** 으로 넣는다. 현재 운영 경로(Animagine Base → FLUX 정밀화, FLOW.md)는 그대로다.
- OFF 에서 기존 Base·Final 출력이 바이트 동일해야 한다(BACKLOG 66).
- 운영 기본값 전환, FLOW.md 계약 변경은 하지 않는다(별도 결정).
- 새 시험·파라미터 탐색·최적화는 하지 않는다. 시험 수치(동일성·붕괴 등)를 다시 재는 것은 완료 조건이 아니다.

## 2. 사용자 결정 대기 항목의 처리 원칙
결정이 안 난 값은 **설정 한 곳**에 두고, 아래 기본값으로 구현한다. 구현이 결정을 대신하지 않는다 — 보고서에 "결정 대기, 현재 기본값 X" 로 적는다.

| 항목 | 설정 기본값 | 다른 선택지 |
|---|---|---|
| 마른 체형 규칙 | `ps` (사용자가 마지막으로 명시 채택한 값) | `ns`, `off` |
| 하이앵글 처리 | `warn` (얼굴 방향 경고와 별도로 "동일성이 낮거나 몸이 무너질 수 있음" 안내, 진행/다른 이미지 선택) | `out_of_scope` |
| 노출 숨김 | `on`, 표시 문구는 "노출 의심 N장 제외" (안전 보장 표현 금지 — 단독 기준 미달) | `off` |
| 노출 게이트 의상 규칙(몸통 덮음·핵심 부품) | 의상 태그 확인 화면에서 **사용자가 확인한 값** 사용. 사용자가 확인하지 않으면 숨김 규칙 C·D 를 적용하지 않음(A·B 만) | 자동 생성(알려진 버그 경로, 쓰지 말 것) |
| 배경 분리(isnet) | `off` (채택 근거 없음) | `on` |
| 의상 태그 확인 단계 | 1회 경로에서도 표시, 건너뛰기 허용 | 필수 |

## 3. 단계 (단계마다 멈추고 보고, 커밋은 사용자 확인 후)

### 단계 1. 성별 계약 (출시 차단 — 가장 먼저)
- 1회 경로의 성별 출처는 `genai_lab/character_preferences.py:14 load_character_gender` 만. 태거의 1boy/1girl 은 후보 표시만.
- 긍정·부정 성별 처리는 `genai_lab/clothing_reference_generation.py` 의 규칙(:12 `BASE_GENDER_CONDITION_TAGS`, :105 `gender_guard`, 같은 성별 부정어 제거)을 **함수로 분리해 재사용**한다. 복사 금지.
- 부정 프롬프트 템플릿에는 성별어를 넣지 않는다(시험의 고정 "1boy" 금지).
- 테스트: 남성·여성·지정 안 함 세 경우에 1회 경로와 `prepare_design_reference_request` 의 긍정 첫 태그·부정 첫 태그가 같음.

### 단계 2. 입력 전처리 모듈
- 자세: DWPose → 몸 bbox 정규화(.20H 위/.10H 아래/.15H 옆, 736:1232, 회색 128) → 재검출 → 검사 K1~K6 + K7(무릎·발목 좌우 거리 모두 < 0.15T) + 추측 관절(팔·다리 신뢰도 0.30~0.50 이 2개 이상) + 얼굴 방향 태거 조건(looking_at_viewer < 0.35 또는 profile·from_side·from_behind·from_above >= 0.35) → 카드 색 반전 제어 이미지.
- 검사 처리 정책은 설정값: K1·K2 = `reject`, 나머지 = `warn`. 문턱값도 같은 설정에. 모듈은 판정값만 내고 처리는 정책이 정한다.
- 자세 태그: `outputs/pose-range-20260928/pose_tags.json` 의 allow 목록 + 주머니 규칙.
- 캐릭터: 태그 선택 규칙(성별 제외 — 단계 1), 마른 체형 판정(DWPose 어깨폭/몸통 < 0.48), 얼굴 자동 크롭 C6(애니 머리 검출 head_detect_v2.0_s 최대 상자 +10%, 팔꿈치·손목이 머리 주변이면 거부 + 안내) + 미리보기. imgutils 미승인 시 크롭은 수동 크롭 경로만.
- 재사용: `genai_lab/pose_estimation.py:104·255`, `scripts/pose_reference_runner.py:34`, `genai_lab/clothing_analysis.py:56 WdTagSession`, 검사 문턱 `outputs/pose-identity-20260927/skeleton-check/rule-values.json`.
- 검증: `outputs/e2e-20261001/analysis.json` 의 5 시나리오와 시연 3장(U1 경고, U2·U3 K2 거부)에서 같은 판정. 제어 이미지는 `outputs/pose-norm-20260928/control_*.png`, `outputs/pose-newimg-20260929/control_P2.png` 와 SHA 비교.

### 단계 3. 프롬프트 조립 모듈
- 순서: [성별 태그 + 보조 태그(단계 1)] + 의상 승인 태그 + 덮지 않는 부위 + 외형·체형·고정(+마른 체형 규칙 설정값) + 자세 태그(주머니 규칙) + 꼬리(solo, full body, white background, simple background, coherent anatomy, best quality).
- 의상 명사·덮지 않는 부위 목록은 시험마다 손으로 늘린 목록이다(`integration make_plan.py`, `e2e plan.py`) — 제품 어휘 표 하나로 통합.
- 부정: `outputs/bg-prevent-20260930/plan.json[0].negative` 에서 성별어를 뺀 템플릿 + 단계 1 의 성별 차단.
- 75토큰 초과 시 고정 청크 인코딩(`outputs/integration-20260929/run_int.py` 의 pieces/ids_for/encode_pcs 동작과 동일). 조립 결과·토큰 수·청크 수를 실행 기록에 남긴다.
- 검증: `outputs/integration-20260929/plan.json` 120개와 `outputs/final-int-20261001/plan.json` 120개의 긍정·부정 문자열을 같은 입력으로 재조립해 일치. 단, 성별 계약으로 바뀌는 부분은 `outputs/gender-contract-20261001/plan.json` 기준.

### 단계 4. 생성 모듈 (옵션, 기본 OFF)
- Animagine XL 3.1, T2I-Adapter openpose SDXL(D:/genai-cache/models/t2i-adapter-openpose-sdxl-1.0), ip-adapter-plus-face_sdxl_vit-h + ViT-H 인코더, 736×1232, 28단계, CFG 5.5, EulerAncestral(`outputs/pose-identity-20260927/exposure-v2/preflight-data.json` 의 `A.scheduler_config`), enable_model_cpu_offload.
- 골격 세기 1.2, 0~10단계. 얼굴 IP: 기본 0.0 → 11단계부터 0.9 / 얼굴 비정면(단계 2 태거 조건) 0.5 → 11단계부터 0.9.
- 요청당 seed 4장 순차, 장마다 스케줄러·IP scale·어댑터 상태 초기화, 한 장씩 콜백 전달.
- 시험의 UNet forward pre-hook 잔차 주입과 `scheduler.step` 래퍼는 관측 장치다 — 쓰지 말고 `StableDiffusionXLAdapterPipeline`(또는 같은 계약의 명시 함수)으로. IP 일정 전환은 콜백으로.
- 경로 상수(D:/genai-cache 는 G: NVMe 정션)는 설정 한 곳에서.
- 연결: `genai_lab/generation_orchestrator.py:249` 의 `reference_pose` 와 같은 방식의 명시적 요청 옵션. 옵션이 없으면 기존 경로 그대로.

### 단계 5. 결과 처리
- 노출 게이트 v4(A·B·C + D, WD 태거 0.35, 2절 원칙) → 숨김, "노출 의심 N장 제외" 표시.
- 각 장의 seed·게이트 결과·선택 여부를 실행 기록에 저장. "다른 seed 로 4장 더"는 seed 만 바꿔 재호출.
- 배경 분리는 설정 `off` 기본(on 이면 `scripts/body_comparison_runner.py:234` 재사용 + 흰 배경).

### 단계 6. GUI 연결
- 입력 확인: 겹친 골격 그림, 경고 사유, "진행 / 다른 이미지"(reject 정책 검사는 "다른 이미지"만), 범위 밖 안내, "캐릭터 유사도가 낮을 수 있음"(얼굴 방향), 하이앵글 안내(2절), 얼굴 크롭 미리보기.
- 의상 태그 확인: 기존 승인 화면을 1회 경로에 연결. 노출 게이트용 "몸통을 덮는 옷인가 / 핵심 부품" 확인 항목 추가(2절).
- 결과: 4장 순차 표시, 제외 수, 사용자 선택 → 기존 8·9단계(승인·저장)로.

## 4. 검증 (이 네 범주만)
1. **OFF 바이트 동일**: 옵션 OFF 에서 기존 Base·Final 이 구현 전과 바이트 동일.
2. **채택 구성 재현**: 제품 코드로 생성한 raw 가 시험 SHA 와 일치(같은 PC·라이브러리). 불일치면 원인부터 보고.
   | 사례 | 기대 raw SHA-256 앞 16자리 | 근거 |
   |---|---|---|
   | integration `original_ordinary-female_KNEEL_209210101` (초기 IP 0.0) | 55a58ebbbdaab330 | outputs/integration-20260929/cases/…/run.json |
   | final-int `F_original_ordinary-female_KNEEL_209210101` | 2a850ce9da4a66ba | outputs/final-int-20261001/cases/…/run.json |
   | final-int `F_original_raccoon_HIGH_209210101` (초기 IP 0.5, 성별 계약 적용 전 구성) | 51efedaa8de0c4cd | outputs/final-int-20261001/cases/…/run.json |
   | gender-contract `U_original_raccoon_KNEEL_209210101` (성별 계약 적용) | 3383557338b7a727 | outputs/gender-contract-20261001/cases/…/run.json |
   최대 reserved ≤ 6.5 GiB(시험 6.41~6.42), 장당 시간 기록(시험 약 22초).
   성별 계약 적용 전 구성(위 3번째)은 성별을 "지정 안 함"으로 둔 재현이다. 계약 적용 후 결과가 달라지는 것은 정상.
3. **입력·결과 처리 연결**: K1·K2 거부와 정책 변경 테스트, 경고 → 사용자 선택 흐름, 얼굴 크롭 C6 거부·미리보기, 성별 3종 테스트(단계 1), 노출 숨김 표시.
4. **기존 테스트·회귀**: 기존 `tests/` 전체 + 변경 모듈 단위 테스트.

## 5. 금지
- 사용 금지 파일: `143960292_19`(이름 무관 모두), `C:\Users\user\Downloads\참조 의상\0e0f64558c7c6f84b97b43b5b60e693e.jpg`(미성년 사진). 테스트·검증 입력에 넣지 않는다.
- 다운로드·패키지 설치는 사용자 승인 후.
- 의상 이미지 두 번째 IP-Adapter(8GB 에서 7.21 GiB·30배 느림), 깊이 어댑터, 리타게팅은 구현하지 않는다.
- 노출 게이트를 "안전 검증 완료"로 표시하지 않는다.
- 시험 결과 파일(outputs/ 의 잠근 criteria·labels·raw)을 수정하지 않는다.

## 6. 보고 형식 (단계마다)
- 바꾼 파일:줄, 추가한 설정 키와 기본값, 테스트 결과(명령과 출력 요약), 결정 대기 항목의 현재 기본값, 남은 위험.
- 커밋은 사용자 확인 후. 커밋 메시지에 단계 번호.
