# GPU 시험 요청 25 — 자세 골격 유도(D-131) 있음/없음 비교 (승인 2026-10-15)

설계 `docs/FREE_GENERATION_POSE_GUIDANCE.md`. 자유 생성이 글 지시만으로 자세를 정하면 구조가 틀린 그림(요청 23 N1-tail 다리 3개)이 나오고 문장 조정이 끝없이 늘어난다. 지원 자세 목록의 골격을 캐릭터 비율에 맞춰 image 3으로 주면 구조 오류가 줄어드는지 같은 seed로 비교한다. 상태: **승인** — 사용자 "요청 25 승인, 권고대로 진행해"(2026-10-15). 실행: **Codex**. 실행 코드는 D-131 커밋 이후 HEAD.

## 비교 목적

- 골격 있음/없음 두 장을 같은 기준·요청·selection_seed 42·생성 seed 42로 만든다. 지시문 차이는 `pose_skeleton` 문장 하나(`Follow the body pose shown by the colored skeleton in image 3, with exactly two arms and two legs.`)와 image 3뿐이다.
- 주로 볼 것: 다리·팔 개수와 겹침(사람 확인), 요청 자세 반영, 캐릭터·의상 유지가 골격 때문에 나빠지지 않는지.
- 위험: 10월 초 골격 입력 자세 편집은 불안정했다(D-077). 그림이 3장이 되어 text encode 메모리가 오를 수 있다(지금 최고 6.36GiB, 상한 6.5GiB) — 상한 초과는 즉시 보고, 자동 재실행 금지.

## 입력과 실행(6장)

기준·성별·부위·특징표·고유 형태 메모는 요청 21~24와 같다(`codex-preflight-21-rerun1/proposed-inputs.json`).

| 줄 | 기준 | 요청 | 지원 자세 | 장 |
|---|---|---|---|---|
| N1 | 원본 N1 | sitting on a simple stool with the hands on the knees | sitting_stool(3/4 시점, 허벅지 수평) | 골격 있음 / 없음 |
| N2 | 원본 N2 | kneeling on one knee with one hand resting on the raised knee | kneeling_one_knee(3/4 시점) | 골격 있음 / 없음 |
| K | 원본 K | walking toward the viewer with the arms swinging naturally | walking(정면) | 골격 있음 / 없음 |

- 실행기: `codex-preflight-25/run_25.py`(요청 23 실행기 구조). 인자 없이 = CPU 사전 검사(자세 분류, 입력 SHA, 코드·설정 잠금), `--run` = 6장 순서대로. 실제 요청마다 그림 수(골격 있음 3, 없음 2)·pose_key·seed·설정을 대조한다. UNC 경로로 실행.
- 골격 그림은 기준 그림 관절(DWPose, CPU)로 뼈 길이를 맞춰 코드가 그린다. 작업 프로세스가 기록된 관절로 다시 그려 같은지 대조한다. 사전 확인 그림: Claude가 N1·N2·K 골격을 미리 그려 봄(앉기·무릎 꿇기는 3/4 시점이라 정면보다 구분이 쉬움).
- 재시도는 제품 규칙대로(확신 있는 위반만 seed 43). 시간: 한 장 약 35분, 6장 약 3.5시간(재시도 있으면 더).
- 배치 중 Claude는 genai_lab 코드를 수정·커밋하지 않는다.

## 기록

- 비교 그림(기준별 한 장): 원본 | 골격 그림 | 골격 있음 결과 | 골격 없음 결과.
- 사람 기록(줄마다 두 장 각각): 팔·다리 개수와 겹침(정상/이상+내용), 요청 자세 반영(예/일부/아니오), 같은 캐릭터, 의상(같음/일부/다름), 심한 이상, 그리고 두 장 중 나은 쪽.
- 자동 기록: `pose.skeleton_match`(골격 있음만, 팔·다리 각도), text encode·생성 reserved, 실제 프롬프트 차이.

## 결과별 다음 결정

- 골격 있음이 구조 이상이 적고 캐릭터·의상 유지가 나빠지지 않음: 지원 자세에서 골격 유도를 기본으로 켜는 것을 사용자에게 제안(지금 기본은 꺼짐). 지원 자세 목록 넓히기 검토.
- 차이 없음: 골격 유도의 이득이 없으므로 끄고, 구조 오류는 사람 선택(장수 늘리기)로 다룬다.
- 골격 있음이 더 나쁨(캐릭터·의상 변형, 자세 경직): 끄고 기록. 10월 초 결과와 같은 결론.
- 메모리 상한 초과: 그림 3장 경로는 보류, 골격을 image 2와 합치는 방식 등은 별도 설계.
