# 옷 바꾸기 판단 추적 기록 (D-112, 2026-10-13)

> 목적: "왜 그 문장을 넣었나"와 "결과에 무엇이 생기거나 사라졌나"에 추측이 아니라 기록으로 답한다.
> 실행 재현 기록(`qwen-inputs/request.json`, SHA)은 전부터 있었다. 이 문서의 기록은 **입력 근거 → 편집 의도 → 전달 조건 → 출력 관찰 → 판정**을 잇는다.

## 결론부터
- 실행 폴더마다 두 파일이 생긴다.
  - `qwen-inputs/conditions.json`: 지시문을 조건별로 나눈 목록. 조건마다 **출처**(policy 공통 규칙 / analysis 자동 분석 / user 사용자 확인)와 **채택 이유**, 넣지 않은 조건은 `excluded`에 이유. 입력 사실(`input_facts`)도 여기 있다.
  - `trace.json`: 위 내용 + 시도별 자동 검사(팔·머리 위치·머리 형태) + 판정 + 사람이 확인해야 할 관찰 항목.
- 두 질문을 섞지 않는다.
  - **우리 프로그램이 왜 그 문장을 넣었나** → `conditions[].source/reason`으로 답한다(코드·입력·규칙).
  - **생성 모델이 왜 그렇게 그렸나** → 기록하지 않는다. 보고서에서 관찰 근거와 함께 가설로만 적는다.

## 입력 사실 (`input_facts`)
- `gender`: 값(male/female/unspecified), `state`(confirmed/unspecified), 출처(`source`: user_confirmed / saved / flag / default), 미리 채운 값의 출처(`prefill`: saved / tags), 태그 점수.
- `body_parts.{ears,tail,wings}`: **present / absent / unknown** 세 가지. 확인하지 않았으면 unknown이다 — 검출하지 못했다는 이유로 absent라고 적지 않는다. 출처·미리 채운 값·태그 점수는 성별과 같다.

## 조건 (`conditions` / `excluded`)
| key | 출처 | 뜻 |
|---|---|---|
| replace_outfit | policy | 원래 옷 전부 교체(걸친 코트·넥타이·장갑 포함), 하나도 남기지 않음 |
| remove_props | policy (시험에서는 test_override로 제외) | 들고 있는 소품·무기 제거 |
| background | policy | 흰 배경 |
| garment_terms | analysis (제품에서는 제외) | 사용자가 확인한 옷 이름 목록 |
| gender | user (지정 안 함이면 제외) | 확인된 성별의 체형 유지 |
| head | policy(keep) / user(remove) | 얼굴·머리·머리 장식 유지 |
| body_parts | user(확인됨) / policy(확인 안 함 → 조건문) | 있는 부위만 유지, 없는 부위 추가 금지 |
| framing | policy | 자세·구도·크기 유지 |

조건 문장을 순서대로 이어 붙인 것이 실제 전달 지시문이며, `prompt_sha256`으로 `request.json`과 대조할 수 있다.

## 출력 관찰 (`output_observation`)
- `automatic`: 팔 검사(`limbs`, 구간별 연속 빗나감), 머리 위치(`alignment.decision`), 머리 형태(`hair_shape`). 시도마다 `attempts[].checks`에 남는다.
- `manual`: 귀·꼬리·날개 유무·개수, 기존 의상 잔존, 자세. **자동 검출이 없다.** 판단 주체는 사용자 승인(`user-review.json`의 appendages·garment 확인) 또는 검토 보고서다. 검출기를 넣기 전까지는 이 항목을 자동으로 "통과"로 적지 않는다.

## 판정 (`judgement`)
- `automatic`: composited 또는 거부 이유(머리 위치 / 머리 형태 / 팔).
- `certainty`: 자동 검사 결과는 확정, 생성 원인은 미확정.

## 예: 요청 15의 K (기록이 있었다면)
| 기록 | 내용 |
|---|---|
| 입력 사실 | body_parts: ears/tail/wings 모두 **unknown**(당시 확인 절차 없음) |
| 편집 의도 | 의상 교체, 고유 신체 특징 유지 |
| 조건 | body_parts: source=policy, 당시 규칙 "모든 입력에 귀·꼬리·날개 유지" |
| 전달 문장 | "Keep the character's animal ears, tail or tails and wings…" |
| 출력 관찰 | manual: 귀 2·꼬리 1·날개 2 추가(Codex 육안) |
| 판정 | 없는 부위 추가 → 품질 미달 |
| 원인 확실성 | 추가된 사실은 확인. 지시문이 원인 후보이지만 단독 원인으로 확정하지 않음 |

이 기록 덕분에 D-111의 수정(확인된 부위만, 없으면 추가 금지)이 어느 조건을 왜 바꿨는지 추적된다.

## 사람 판정 (`judgement.json`, D-114)
- 항목: `same_character`(yes/no), `outfit`(same/partial/different/accepted), `body_parts`(kept/changed/missing/added/not_applicable),
  `original_garment_remaining`(none/some), `pose`(kept/changed), `severe_defect`(none/present + `defect_note`). 묻지 않은 항목은 `not_judged` — 추측해서 채우지 않는다.
- 판정자 `judge`(user/codex/claude)와 기록자 `recorded_by`, 근거 `source`(보고서·비교 그림·줄 번호)를 함께 적는다.
- `quality_pass`: 머리 합성 + 같은 캐릭터 + 옷 same/partial/accepted + 기존 의상 잔존 none + 심한 이상 none. 핵심 항목에 not_judged가 있으면 `null`(불완전).
- 같은 실행에 여러 판정이 있으면 사용자 판정이 우선이고, `trace.json`의 `judgement.quality`에 최신 판정이 요약된다.
- 작업실: 결과 승인 체크(캐릭터·옷·노출·원래 옷 잔존 없음·꼬리/귀)가 `gui_approval` 판정으로, 거절은 `gui_reject`(항목 not_judged)로 기록된다.
- 시험 판정: `python -B scripts/record_judgement.py record RUN_DIR --judge user --same-character yes --outfit partial --garment-remaining some ...`,
  집계는 `summary RUN_DIR...`(합성 수·판정 수·품질 통과 수·불완전 수).

## 보관
- 결과 저장 시 `.review.json`에 `trace`·`conditions` 경로를 함께 적는다.
- 확인 실행기(`scripts/outfit_edit_gpu_check.py`)의 `summary.json`에도 `trace` 경로가 들어간다.
