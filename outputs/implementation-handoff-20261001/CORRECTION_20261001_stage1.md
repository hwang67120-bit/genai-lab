# Codex 전달 — 구현 1단계(성별 계약) 검토 결과와 검증 방법 정정

작성: Claude, 2026-10-01. 대상: `codex/onepass-implementation` 작업 트리의 미커밋 변경
(`genai_lab/clothing_reference_generation.py`, `genai_lab/gender_prompt.py`, `genai_lab/onepass_gender.py`, `tests/test_onepass_gender.py`).

## 1. 검토 결과: 1단계 승인 가능
- `prepare_design_reference_request` 의 성별 처리 이동은 동작 동일(성별 차단·같은 성별 부정어 제거·보조 태그). `BASE_GENDER_CONDITION_TAGS` 재노출로 `native_pipeline_contract.py` 호환 유지.
- 1회 경로 성별 출처가 `load_character_gender` 만이고, 미저장·잘못된 값은 태거로 대체하지 않고 오류. 태거 성별 태그는 후보 기록만. 성별어가 든 부정 템플릿 거부. 인수인계 요구와 일치.
- 테스트: 관련 23개 파일 394개 통과(`pytest -p no:cacheprovider`, PYTHONDONTWRITEBYTECODE=1).

## 2. 정정 — 구현 검증 방법 (인수인계 문서 오류)
`CODEX_IMPLEMENTATION_PROMPT.md` 4절 2번과 `README.md` 6절의 검증 사례 1~3번 SHA 는 **성별 계약 적용 전 프롬프트**로 생성됐다.
계약을 적용하면 프롬프트가 바뀌므로 그 SHA 와 일치할 수 없다. 또 실제 사용자 성별 저장소에는 raccoon(남성)만 있어 ordinary-female 은 1회 경로에서 오류가 난다.

재현 검증을 둘로 나눈다(두 문서 모두 이렇게 고쳐 두었다):
- **(a) 생성 모듈 재현**: 시험 plan 의 긍정·부정 **문자열을 그대로 생성 모듈에 주입**해 raw SHA 비교. 성별·프롬프트 조립을 거치지 않는다.
  | 사례 | 문자열 출처 | 기대 raw SHA-256 앞 16자리 |
  |---|---|---|
  | integration `original_ordinary-female_KNEEL_209210101` | outputs/integration-20260929/plan.json | 55a58ebbbdaab330 |
  | final-int `F_original_ordinary-female_KNEEL_209210101` | outputs/final-int-20261001/plan.json | 2a850ce9da4a66ba |
  | final-int `F_original_raccoon_HIGH_209210101` (초기 IP 0.5) | outputs/final-int-20261001/plan.json | 51efedaa8de0c4cd |
  | gender-contract `U_original_raccoon_KNEEL_209210101` | outputs/gender-contract-20261001/plan.json | 3383557338b7a727 |
- **(b) 성별·프롬프트 조립**: 문자열 일치로 비교.
  - 남성: `outputs/gender-contract-20261001/plan.json` 의 U_* 긍정·부정과 일치.
  - 여성·지정 안 함: 테스트용 QSettings 임시 저장소로 기대 문자열을 만들어 비교(실제 사용자 저장소는 쓰지 않는다).

## 3. 3단계(프롬프트 조립)에 넣을 테스트 제안
- 1회 경로 접두부는 `1boy, male focus, masculine silhouette, <외형…>` 이고, 기존 제품 경로는 `1boy, <외형…>, male focus, masculine silhouette` 이다. 1회 경로 순서가 gender-contract 시험과 같아 재현에는 맞다.
- 현재 테스트는 첫 태그만 비교한다. 3단계에서 **전체 긍정 문자열**을 gender-contract plan(U_*)과 비교하는 테스트를 추가할 것. 순서 차이는 의도된 것으로 docstring 에 적을 것.
- "지정 안 함"에서 제품은 승인된 성별 태그를 남기고 1회 경로는 태거 후보라 뺀다 — 의도된 차이(테스트에 이미 명시). 유지.

## 4. 커밋 전 확인
- 수정 파일 작업본이 CRLF 라 `git diff` 가 277줄 변경으로 보인다(실제 내용 29줄). 저장소 `core.autocrlf=true` 라 커밋 시 LF 로 정규화된다. `git add` 후 `git diff --cached --stat` 으로 변경 규모만 확인.

## 5. 이후 단계 선행 조건 (사용자 작업)
1회 경로는 성별 저장이 있어야 동작한다. 검증에 쓸 캐릭터는 사용자가 GUI 에서 성별을 저장해야 한다
(사용자 지정 2026-10-01: e2e S1(흰 단발 메이드)·S3(파란 머리 동물 귀) 남성, S2·S4·S5 여성, ordinary-female 여성, raccoon 남성 — 이미 저장됨).
저장 전에는 해당 캐릭터의 1회 경로 검증이 "성별 선택이 없습니다" 오류로 멈추는 것이 정상이다.
