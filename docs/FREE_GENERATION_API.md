# 자유 생성 호출 규칙 — 외부 프롬프트 작성기·API용 (D-133, 2026-10-16)

> 사람이나 LLM이 프롬프트를 써서 자유 생성을 부를 때 지켜야 할 입력·출력 규칙. 계약은 [FREE_GENERATION_CONTRACT.md](FREE_GENERATION_CONTRACT.md), 배포 방향은 [DEPLOYMENT_API_COMPATIBILITY.md](DEPLOYMENT_API_COMPATIBILITY.md)(D-110). API 서버·LLM 연결은 아직 만들지 않는다. 이 문서는 그때 바로 붙일 수 있게 규칙을 고정한다.

## 1. 호출

`genai_lab.free_generation_pipeline.run_free_generation(...)` — GUI 없이 부를 수 있는 하나의 공통 함수. GUI 확인 창 대신 결과를 **사람 검토 대기**로 돌려준다.

| 입력 | 형식 | 비고 |
|---|---|---|
| `base_path` | PNG/JPG 경로 | 기준 그림 한 장 |
| `base_kind` | `original` \| `edited` | `edited`면 같은 바이트의 `.review.json`(사용자 의상 편집 승인)이 있어야 함 |
| `character_features` | `{항목: [태그] \| None}` + 선택 `distinctive: [메모] \| None` | 항목 = hair_length·hair_style·hair_color·eye_color·head_accessories·skin. `[]` = 없다고 확정, `None` = 모름 |
| `request_text` | 문자열(2000자 이내) | 비우면 고정 자세 목록에서 `selection_seed`로 선택 |
| `gender` | `male` \| `female` \| `unspecified` | |
| `body_parts` | `ears`·`tail`·`wings` 목록 | 없으면 빈 목록 |
| `confirmed`, `adult_confirmed` | `True` | 성인 캐릭터·특징표·연출 범위 확인 |
| `selection_seed` | 정수(선택) | 재현용. 생성 seed는 42, 재시도 43 고정 |
| `pose_guidance` | `False`(기본) | 지원 자세 골격 유도(D-131/132, 기본 꺼짐) |
| `settings`, `runtime`, `directory` | 실행 환경 | `directory`는 새 폴더 |

## 2. 요청 문장 규칙(프롬프트 작성기가 지킬 것)

- **허용**: 자세·구도·시점·배경·장면·표정·간단한 연출. 예: `sitting on a simple stool`, `walking in a snowy street`, `standing with eyes closed`, `close-up portrait`.
- **거부**(아래 고정 코드): 외형 변경(머리색·눈 색·피부·꼬리·귀·날개·뿔·장식 바꾸기/없애기/추가, 머리 길이 바꾸기), 의상 바꾸기·입히기, 여러 캐릭터, 노출·아동 외형.
- **지원 자세 목록**(글로도 잘 됨, 골격 유도 대상): standing·sitting_stool·kneeling_one_knee·walking·arms_crossed. 목록 밖(눕기·엎드리기 등)은 생성은 하되 결과 보장 없음.
- **가리는 연출**: 눈 감기·뒷모습·손 가리기(뒷짐·주머니)·꼬리 숨기기는 그 특징을 지시문에서 '보이게 유지'하라고 하지 않는다(D-127/128). 결과 확인 체크에는 남는다.
- **장면 요청**이면 "장면 조명이 고유 색을 바꾸지 않게" 문장이 붙는다(D-129). 배경색에 머리색이 끌리면 결과에 머리색 보정 후보가 생긴다(D-130).
- 요청 문장 검사는 낱말 규칙이다. 통과가 의미 승인이나 안전 보장은 아니다(사람이 최종 확인).

## 3. 거부 코드

거부는 `FreeGenerationRejected`(= `ValueError`)로 올라오고 `code`·사람용 문장·`detail`을 가진다. 실행 폴더 `trace.json`에도 `status: "rejected"`, `rejection: {code, message, detail}`로 남는다.

| code | 뜻 |
|---|---|
| `unsafe` | 노출·아동 외형 등 금지된 요청 |
| `multiple_characters` | 여러 캐릭터(한 캐릭터만 지원) |
| `outfit_change` | 의상을 바꾸거나 입히는 요청(의상 교체는 편집 단계) |
| `appearance_change` | 외형(머리·눈·피부·부위·장식)을 바꾸는 요청 |
| `request_too_long` | 2000자 초과 |
| `invalid_input` | 기준 종류·성별·부위·seed·문자열 형식 오류 |
| `invalid_features` | 특징표 형식 오류(태그 항목·외형 태그·고유 형태 메모) |
| `confirmation_required` | 성인·특징표·연출 범위 확인 없음 |
| `approval_required` | 편집본 기준인데 사용자 의상 편집 승인 없음 |
| `unsafe_input` | 기준 그림에서 금지 태그 관찰 |

생성 중 실패(모델·메모리·계약 불일치)는 거부가 아니라 실패(`status: "failed"`)이며 자동 재실행하지 않는다.

## 4. 결과 형식 `free_generation_outcome/2`

| 키 | 뜻 |
|---|---|
| `schema` | `"free_generation_outcome/2"` — 키를 바꾸면 버전을 올린다(/2: `auto_selection`, 변형 `hair_auto`, D-137) |
| `status` | `awaiting_user_review` \| `blocked`(안전 태그 차단, 사용 불가) |
| `result`, `result_sha256` | 처음 보여 줄 결과(자동 위반이 적은 시도의 원 결과) |
| `candidates` | 고를 수 있는 결과 목록: `attempt`·`variant`(`raw` \| `hair_restored` \| `hair_auto`)·`seed`·`raw`(경로)·`sha256`·`violation_score`·`retry_items`·`blocked`·`automatic_features`·`garment_tags` |
| `selected_attempt` | 자동 위반이 적은 시도 번호 |
| `auto_selection` | 재검사를 통과한 머리색 자동 보정본 `{attempt, variant: "hair_auto", reason}` 또는 `null`(D-137). 있으면 처음 보여 주는 후보지만 **승인은 여전히 사람**이 한다. 원 결과는 `candidates`에 그대로 있다 |
| `condition_judgements` | 조건별 자동 관찰: `not_violated`·`needs_review`·`violated`·`unmeasurable`. **자동은 '충족'을 말하지 않는다** — 충족은 사람 판정만 |
| `automatic_features` | 특징 비교 요약(kept·changed·missing·added·unknown·review) |
| `distinctive_review` | 사람이 결과에서 하나씩 확인할 고유 형태 메모 |
| `quality_pass` | 항상 `null` — 품질 승인은 사람이 한다 |
| `trace`, `conditions` | 입력 사실·조건·실제 지시문·시도별 관찰 기록 경로 |

사람 승인: 같은 캐릭터·같은 의상·연출 반영·심한 이상(팔다리 개수 포함)/노출 없음·완성도, 그리고 고유 형태 메모. API에서는 이 검토를 호출자 쪽 사람이 해야 하며 결과를 자동 승인하지 않는다.
