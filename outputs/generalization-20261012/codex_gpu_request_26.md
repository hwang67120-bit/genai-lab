# GPU 시험 요청 26 — 자유 생성 1차 완료 판정용 고정 평가 세트(A) (승인 2026-10-16)

완료 기준 `docs/FREE_GENERATION_CONTRACT.md` 10절 A(D-133)를 한 번에 재는 고정 세트다. 상태: **승인** — 사용자 "특징표 그대로 승인, 요청 26 승인하고 푸쉬 해줘"(2026-10-16). 실행: **Codex**. 실행 코드는 D-134(피부 톤 문장) 커밋 이후 HEAD.

## 목적과 완료 판정

- 사람 판정으로 두 수치를 따로 잰다: **1장 통과율**(첫 시도 원 결과만)과 **선택 통과율**(그 줄에서 만든 후보 중 하나라도 통과: 재시도·머리색 보정 후보 포함).
- 완료 = 선택 통과율 **70% 이상**(21줄 중 15줄 이상) + 같은 실패 유형이 **2명 이상에서 반복되지 않음** + **처음 보는 캐릭터(M2·M3·M6·F5)에도 통과 줄 있음**.
- 모자라면 공통 개선 최대 2번(D-134 피부 수정이 1회차로 이미 쓰임 → **남은 개선 1번**), 같은 세트로 다시 잰다. 남는 실패 유형은 '지원 범위 밖' 경고로 돌리고 운영 모드로 넘어간다(10절 C).

## 캐릭터(7명: 여성 3·남성 4, 처음 보는 4명)

| 캐릭터 | 처음 봄 | 성별·부위 | 기준(원본) | 특징표 |
|---|---|---|---|---|
| N1 | 아니오 | 여성 · 꼬리·날개 | `free-r20/N1-1/inputs/canvas.png` | 요청 21 사용자 확정값 그대로 |
| N2 | 아니오 | 남성 · 귀·꼬리 | `free-r20/N2-1/inputs/canvas.png` | 요청 21 사용자 확정값 그대로 |
| K | 아니오 | 여성 · 없음 | `free-r20/K-1/inputs/canvas.png` | 요청 21 사용자 확정값 그대로 |
| M2 드레드 | 예 | 남성 · 없음 | `eval-smoke-26/sources/M2-dreads.jpg` | 아래 제안값, 사용자 승인 |
| M3 땋은 머리 | 예 | 남성 · 없음 | `eval-smoke-26/sources/M3-braid.jpg` | 아래 제안값, 사용자 승인 |
| M6 후드 | 예 | 남성 · 없음 | `eval-smoke-26/sources/M6-hoodie.jpg` | 아래 제안값, 사용자 승인 |
| F5 치어 | 예 | 여성 · 없음 | `eval-smoke-26/sources/F5-cheer.jpg` | 아래 제안값, 사용자 승인 |

- 제외: M1(앞만 가리는 천 의상, 노출 차단 — 사용자 제외), M5 강아지 소년(N2와 같은 분홍 동물 귀 남성이라 겹침, 예비).
- 기준은 원본. 승인된 의상 편집본 기준 평가는 이후 별도.
- 새 캐릭터 특징표 제안값(`eval-set-26/proposed-inputs.json`, 자동 태그 + Claude 눈 확인, 사용자 그대로 승인; `null` = 모름 → 지시·검사 안 함, `[]` = 없다고 확정):

| 항목 | M2 | M3 | M6 | F5 |
|---|---|---|---|---|
| 머리 길이 | 모름 | long_hair | short_hair | long_hair |
| 머리 모양 | dreadlocks | braid | 모름 | ponytail, ahoge |
| 머리색 | white_hair | blonde_hair | blue_hair | black_hair, multicolored_hair, streaked_hair |
| 눈 색 | 모름(머리에 가림) | blue_eyes | blue_eyes | brown_eyes |
| 머리 장식 | 없음 | 없음 | earrings | hair_ornament |
| 피부 | dark_skin(톤 유지 문장) | 없음 | 없음 | dark_skin(톤 유지 문장) |
| 고유 형태 메모 | — | — | — | red streak in the black hair |

## 연출(배경 있는 장면 4종, 모두 지원 자세)

| 연출 | 지원 자세 | 요청 문장 |
|---|---|---|
| S1 | standing | `standing in a sunny park with trees in the background` |
| S2 | sitting_stool | `sitting on a simple stool in a cozy cafe` |
| S3 | walking | `walking along a snowy street at night` |
| S4 | kneeling_one_knee | `kneeling on one knee on a beach at sunset` |

- CPU 사전 확인(Claude): 네 문장 모두 요청 검사 통과, 자세 분류가 의도대로, 가리는 특징 없음, 배경 요청으로 인식(흰 배경 강제 해제 + 장면 고유 색 문장 D-129).
- 발견: `in a flower field`·`in a garden`·`in a meadow`는 배경 요청으로 인식되지 않아 흰 배경 문장이 함께 붙는다(서로 충돌). 이번 세트는 인식되는 해변으로 바꿨다. 배경 낱말 목록은 D-135로 보강했다(네 장면 문장의 인식·지시문은 그대로).

## 줄(21줄: 캐릭터마다 연출 3종, 돌려 가며 하나씩 뺌)

| 캐릭터 | 연출 |
|---|---|
| N1 | S1 · S2 · S3 |
| N2 | S2 · S3 · S4 |
| K | S3 · S4 · S1 |
| M2 | S4 · S1 · S2 |
| M3 | S1 · S2 · S3 |
| M6 | S2 · S3 · S4 |
| F5 | S1 · S3 · S4 |

연출별 줄 수: S1 5 · S2 5 · S3 6 · S4 5.

## 실행

- 제품 함수 `run_free_generation` 그대로: 생성 seed 42, 확신 있는 위반만 seed 43 재시도, 골격 유도 꺼짐(D-132 기본값), `confirmed`·`adult_confirmed` True.
- 실행기 `codex-preflight-26/run_26.py`(요청 25 구조): 인자 없이 = CPU 사전 검사(입력 SHA, 특징표 확정 여부, 요청 검사·자세 분류·배경 인식, 코드·설정 잠금), `--run` = 21줄 순서대로, **이어서 실행 가능**(끝난 줄 건너뜀). 거부·실패는 기록하고 다음 줄로. UNC 경로로 실행.
- 시간: 한 줄 약 35분, 21줄 약 12시간(재시도 있으면 더). 밤새 한 번에 돌리거나 두 번(1~11줄, 12~21줄)으로 나눠도 된다.
- **환경 주의(D: 고장, 2026-10-16)**: 모델·venv는 `C:\genai-temp-d\genai-cache`에 임시 설치, `subst D: C:\genai-temp-d`로 기존 `D:\genai-cache` 경로 유지. 재부팅 뒤에는 subst를 다시 걸어야 한다. Avast HTTPS 가로채기 때문에 인증서 PEM(`C:\genai-temp-d\genai-cache\windows-roots.pem`)이 필요(모델은 이미 받아 둠, 실행 중 내려받기 없음).
- 배치 중 Claude는 genai_lab 코드를 수정·커밋하지 않는다.

## 기록

- 비교 그림: **캐릭터별 한 장(원본 | S? | S? | S?)**, 7장. 재시도·머리색 보정 후보가 있으면 그 줄 옆에 작게.
- 사람 판정(줄마다, 후보마다): 같은 캐릭터 · 의상(같음/일부/다름) · 연출 반영(예/일부/아니오) · 심한 이상(팔다리 개수·겹친 자세 포함) · 노출 이상 없음 · **완성도(good/lacking)** · 고유 형태 메모(있는 캐릭터만). 그리고 그 줄에서 고른 후보.
- 자동 기록: 조건별 관찰(not_violated·needs_review·violated), 재시도 이유, 머리색 보정 후보 유무, text encode·생성 메모리, 실제 지시문.
- 집계: 1장 통과율, 선택 통과율, 실패 유형별 캐릭터 수, 처음 본 캐릭터 통과 줄 수(`scripts/studio_judgement_summary.py`).

## 결과별 다음 결정

- 완료 조건 충족: 자유 생성 1차 완료, 운영 모드(판정 20건마다 공통 규칙 수정)로 전환. 다음 단계(그림체 반영·구조 결함 자동 검출) 논의.
- 선택 통과율 70% 미만이거나 같은 실패가 2명 이상 반복: 가장 많은 실패 유형 하나만 공통 개선(마지막 1회) → 같은 세트 재측정.
- 특정 연출만 반복 실패(예: 무릎 꿇기, 밤 장면): 그 연출을 '지원 범위 밖' 경고로 돌린다.
