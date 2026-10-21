# GPU 시험 요청 31 — 새 남성 M7, 8단계로 먼저 훑기 (승인 2026-10-21)

M3(옅은 분홍빛 금발 + 손에 가려진 머리)는 조정이 오래 걸려 중단(D-145). 새 성인 남성으로 같은 장면을 따로 본다. 운영 규칙 11절 7항(D-144)에 따라 **8단계 가속으로 먼저 훑고**, 통과한 줄만 40단계로 확인한다. 상태: **승인** — 사용자 "M7 먼저 진행해줘"(2026-10-21). 실행: **Codex**. 실행 코드: 이 문서 커밋 이후 HEAD.

## 입력

- `eval-new-male-31/inputs.json` — M7(흰 머리, 검은 티셔츠·회색 통바지), 원본 `eval-new-male-31/sources/M7-white.jpg`(736×1308), 성별 남성, 부위 없음, 특징표 사용자 승인(짧은 머리, white_hair, blue_eyes, 머리 장식·피부 없음, 머리 모양 모름).
- 두 번째 남성은 이미지 대기(분홍 머리 후보는 원작상 15세라 제외).

## 줄(3장)

| 줄 | 장면 |
|---|---|
| M7-S1 | standing in a sunny park with trees in the background |
| M7-S2 | sitting on a simple stool in a cozy cafe |
| M7-S3 | walking along a snowy street at night |

- 호출: `run_free_generation(..., base_kind="original", gender="male", body_parts=[], selection_seed=42, experiment=dict(lightning8=True, single_attempt=True))`. 생성 seed 42. 제품 기본값(없는 부위 부정 지시문, 머리색 자동 보정 문턱 7, 재시도 R3)은 그대로 — 단 single_attempt라 재시도 없음.
- 시간: 줄당 약 10분 → **약 30분**. 상태 파일 `batch-status.json`.

## 판정과 다음

- 판정 묶음: 원본 | 8단계 결과(+ 자동 보정본이 있으면). 이유는 고정 실패 분류표.
- 통과한 줄은 같은 seed로 40단계 확인(별도 짧은 실행, 사용자 승인 뒤). 실패는 기록만(고정 조합 — 낱말·문턱을 캐릭터에 맞춰 고치지 않는다).
