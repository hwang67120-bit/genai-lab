# 요청35 그림체 LoRA 시험 기술보고

진행 완료: 12칸 모두 생성했고 사용자 품질 판정은 대기 중이다.

- 실행: 2026-10-23 05:21:16 ~ 2026-10-23 07:19:31 KST. 상태 파일 시각 기준 1시간 58분 15초.
- 지정 환경 점검 종료 코드 0, 필수 실패 0. 선택 controlnet_aux 없음은 필수 실패가 아니다.
- 완료 근거: batch-status finished, summary, 12개 Qwen run.final completed와 결과 파일, 정상 요약 stdout·빈 stderr, Windows Get-CimInstance에서 실행기·Qwen 프로세스 없음(07:23:43 KST). 실행기 자체 종료 코드는 별도 수집하지 않아 0이라고 추정 보고하지 않는다.
- 승인 HEAD 146d3c51b83e449e442363b1e2c1829ce9830c81. 코드·설정 267개, 요청서·cases, 입력 4개 SHA가 시작 전과 일치하고 git status도 같다.
- 추가 GPU·재시도·코드 수정·설정 변경·설치·다운로드 없음. 전체 모델 SHA를 새로 검사하지 않았다. 기존 모델 기록값과 스타일 어댑터 실제 검증 기록을 구분한다.

## 판정용 비교표

[비교표 sheet.png](sheet.png) · [구조화 근거](codex-evidence-35.json) · [오류 기록](codex-errors-35.txt)

4행의 열은 원본 | LoRA 없음 | 강도 0.6 | 강도 1.0이다. 실행기 원본은 한국어 글꼴이 깨져 sheet-runner-original.png로 보존했다. GPU 종료 후 CPU와 기존 맑은 고딕으로 제목·상태를 표시한 sheet.png를 작성했다. AI 재생성이나 결과 보정은 하지 않았다. 12칸 모두 실제 outcome.result인 qwen/raw.png를 표시했다.

## 칸별 기술 결과

모든 칸은 awaiting_user_review이다. 생성 완료는 품질 합격이 아니다. 총시간은 실행기 측정값을 초 단위 반올림한 값이며 작업 프로세스 시간은 worker.seconds다. 후자는 순수 GPU 연산 시간과 같지 않다.

| 캐릭터 | 강도 | 상태 | 총시간 | 작업 프로세스 | 최고 reserved GiB | 실제 어댑터 |
|---|---:|---|---:|---:|---:|---|
| N1 | 없음 | 생성 완료·미판정 | 11분 5초 | 316.676초 | 6.371 | lightning8 |
| N1 | 0.6 | 생성 완료·미판정 | 11분 6초 | 322.682초 | 6.375 | lightning8 + style |
| N1 | 1.0 | 생성 완료·미판정 | 9분 36초 | 320.840초 | 6.375 | lightning8 + style |
| K | 없음 | 생성 완료·미판정 | 9분 22초 | 306.491초 | 6.373 | lightning8 |
| K | 0.6 | 생성 완료·미판정 | 9분 40초 | 317.095초 | 6.373 | lightning8 + style |
| K | 1.0 | 생성 완료·미판정 | 9분 37초 | 317.230초 | 6.373 | lightning8 + style |
| M6 | 없음 | 생성 완료·미판정 | 9분 30초 | 307.145초 | 6.365 | lightning8 |
| M6 | 0.6 | 생성 완료·미판정 | 9분 34초 | 322.119초 | 6.369 | lightning8 + style |
| M6 | 1.0 | 생성 완료·미판정 | 9분 49초 | 330.219초 | 6.369 | lightning8 + style |
| M7 | 없음 | 생성 완료·미판정 | 9분 29초 | 310.619초 | 6.361 | lightning8 |
| M7 | 0.6 | 생성 완료·미판정 | 9분 42초 | 321.824초 | 6.365 | lightning8 + style |
| M7 | 1.0 | 생성 완료·미판정 | 9분 44초 | 320.751초 | 6.365 | lightning8 + style |

- 전체 최고 PyTorch reserved는 6.375GiB로 6.5GiB 이하다. dedicated/shared 전체 GPU 메모리 측정은 불가능했다. 모든 memory.json, run.final, progress의 reserved_bytes·stage_peak_reserved_gib를 확인했고 limit 설정값을 실측으로 세지 않았다. memory_stopped 없음.
- 실제 call_arguments는 12칸 모두 8단계, CFG 1.0, seed42, 콜백 index0~7, 시도1회다. 요청 설정 스냅샷의 40/CFG4는 실제 실행값이 아니다. 실제 accelerator 및 call_arguments에서 8/CFG1을 확인했다. CFG1에서는 부정 지시문을 사용하지 않는다는 실행기 기록도 있다.
- 각 캐릭터의 원본·머리 참조 SHA, 장면 원문, seed는 세 강도에서 같다. positive 문장은 스타일 trigger를 제외하면 같고 0.6·1.0에는 Transform into anime.가 앞에 붙는다.
- 미리보기는 12작업×8콜백, preview_error/preview_errors 없음. 최종 preview-007.png와 최근005·006·007 파일 존재. 단계별 쓰기 성공 근거이며 중간8장을 모두 보존했다는 뜻은 아니다. 잠재값의 선형 색 근사로 최종 이미지와 다르다.
- 실행·SHA·어댑터 로딩·적용·미리보기 오류 기록은 없다. 오류 전문을 잘라 보고한 항목도 없다. 각 원본 worker.log 및 run.final/progress를 보존했다.

## 스타일 어댑터 실제 기록

- 스타일8칸은 name=anime_borrowed, adapter 이름 style, strength=0.6 또는1.0, adapters=[lightning8, style]. 승인 코드의 가중치 [1.0, strength]와 실제 adapters·strength 기록이 일치한다.
- 스타일 SHA-256: dc84f7662ed724ebfb9b8c18967eff0eabd5eaf17c2d206e2e09062db9ff8df9. worker가 불러오기 직전 SHA를 대조하고 기록했다.
- 저장소 prithivMLmods/Qwen-Image-Edit-2511-Anime, revision977f8c2ed22c8581c8750370bfce34058b253598, 파일 Qwen-Image-Edit-2511-Anime-2000.safetensors.
- 대조군4칸은 스타일 없음. 해당 trace에 style_trial 필드가 없어도 batch와 본 보고에서 trial: style_lora_borrowed 대조군으로 격리했다. trace를 수정하지 않았다.
- 시험 전용이며 품질 기준선·운영 집계·D-118 판정에 넣지 않는다. 요청서의 Apache-2.0 표기는 별도 웹 재검증하지 않았다. 학습 자료 미공개로 허가 자료 학습이나 제품 채택을 증명하지 못한다.

## 직접 열람 관찰 — 사용자 판정 아님

### N1

- 없음: 은회색 긴 양갈래·분홍 눈·뿔·날개·꼬리가 보인다. 카페와 얼굴의 주황·분홍 빛이 강하고 원본 V 손 모양이 남아 있다.
- 0.6: 대조군보다 배경 대비와 주황빛이 줄고 선·면이 정돈되어 보인다. 은회색 머리·분홍 눈·뿔·날개·꼬리와 V 손 모양이 보인다.
- 1.0: 0.6과 캐릭터 큰 윤곽이 비슷하고 배경 테이블·조명·반짝이 위치가 다르다. 은회색 머리·분홍 눈·뿔·날개·꼬리가 보인다.

### K

- 없음: 검은 단발·흰 줄무늬·파랑과 빨강의 눈색 차이가 보인다. 온실의 청록색과 꽃 색 대비가 강하고 걷는 구도다.
- 0.6: 배경 채도가 낮아지고 선·명암이 단순해 보인다. 양쪽 눈이 파란색으로 보여 원본의 서로 다른 눈색과 차이가 있다. 검은 단발·흰 줄무늬·머리 장식은 보인다.
- 1.0: 0.6보다 정면에 가까운 선 자세다. 검은 단발·흰 줄무늬와 파랑·빨강 눈색 차이가 보인다. 온실 빛은 대조군보다 부드러워 보인다.

### M6

- 없음: 원본보다 머리색이 밝고 청록·파랑 채도가 높아 보인다. 머리에 손을 댄 자세, 회색 후드·검은 반바지·목 장식이 보이며 배경과 피부의 대비가 강하다.
- 0.6: 대조군보다 머리색이 회청색 쪽으로 보이고 머리카락·얼굴·옷의 선과 명암 경계가 뚜렷해진다. 파랑 눈·귀 장식·후드·반바지가 보인다.
- 1.0: 0.6과 큰 형상이 비슷하고 배경 빛과 인물 위치가 조금 다르다. 회청색 머리·파랑 눈·귀 장식·후드·반바지가 보인다.

### M7

- 없음: 흰 머리·파랑 눈·검은 상의·회백색 바지의 큰 형태가 보인다. 붉은 우산과 배경 청록·보라 네온 빛이 강하다.
- 0.6: 우산이 밝은 베이지색으로 바뀌고 바닥 청록빛이 줄어든다. 왼쪽 네온 일부가 남으며 흰 머리·파랑 눈·검은 상의·바지 형태가 보인다.
- 1.0: 하늘이 더 밝은 보라·주황색이고 네온 표현이 눈에 띄게 줄었다. 밝은 우산, 흰 머리·파랑 눈·검은 상의·바지 형태가 보인다.

그림체 변화는 배경의 채도·빛·선과 면에서 눈에 띈다. 캐릭터 변화 크기는 캐릭터별로 다르고 1.0이0.6보다 항상 더 크게 변한다고 단정하지 않는다. N1의 V 손 모양 유지와 K1.0 정면 자세 등 연출도 달라진다. 부위 개수·연결과 동일성 보장은 할 수 없다.

## 자동 관찰과 파생 후보

자동 관찰은 실행 실패나 사용자 거부가 아니다. single_attempt 설정으로 아래 이유가 있어도 추가 시도하지 않았다.

| 칸 | 기록된 추가 시도 사유 | hair_auto 결정 | 실제 표시 |
|---|---|---|---|
| N1-none | 확신 있는 특징 위반: hair_color | not_applicable | raw.png |
| N1-0.6 | 없음 | not_applicable | raw.png |
| N1-1.0 | 없음 | not_applicable | raw.png |
| K-none | 없음 | not_applicable | raw.png |
| K-0.6 | 확신 있는 특징 위반: eye_color | not_applicable | raw.png |
| K-1.0 | 없음 | not_applicable | raw.png |
| M6-none | 확신 있는 특징 위반: skin | auto_selected | raw.png |
| M6-0.6 | 확신 있는 특징 위반: skin | no_issue | raw.png |
| M6-1.0 | 확신 있는 특징 위반: skin | no_issue | raw.png |
| M7-none | 없음 | not_applicable | raw.png |
| M7-0.6 | 없음 | not_applicable | raw.png |
| M7-1.0 | 없음 | not_applicable | raw.png |

M6-none에는 hair_auto decision=auto_selected와 hair-auto.png가 있으나 outcome.result는 raw.png다. 후보를 최종처럼 바꾸지 않고 불일치를 기록했다. 다른 hair-restored 후보와 자동 태그·색 검사도 구조화 근거 및 runs에 보존했다. K0.6 눈색 차이는 직접 열람과 자동 heterochromia_lost 기록이 함께 있지만 사용자 품질 불합격을 대신 기록하지 않았다.

## 장면과 시간 측정 근거

### N1 — scene_seed 404

The character is stepping forward with one hand raised in a small wave, with a playful grin, in a cozy cafe with wooden tables in the late afternoon. Lighting: warm sunlight streaming through a window. Color design: a lavender and peach palette in the background and lighting only. Eye-level view, natural body proportions. Full-body shot: the whole character is visible from head to feet, with a detailed background.

| 강도 | pipeline elapsed_seconds | worker.seconds | 차이(분석·준비·후처리 등) |
|---|---:|---:|---:|
| 0.0 | 664.511 | 316.676 | 347.835 |
| 0.6 | 665.471 | 322.682 | 342.789 |
| 1.0 | 575.252 | 320.840 | 254.412 |

### K — scene_seed 303

The character is walking with a light step, one arm swinging forward, with a dreamy look, in a greenhouse full of plants in the late afternoon. Lighting: bright clean daylight with soft bounce light. Color design: a coral and turquoise palette in the background and lighting only. Three-quarter view at eye level, natural body proportions. Full-body shot: the whole character is visible from head to feet, with a detailed background.

| 강도 | pipeline elapsed_seconds | worker.seconds | 차이(분석·준비·후처리 등) |
|---|---:|---:|---:|
| 0.0 | 561.352 | 306.491 | 254.861 |
| 0.6 | 580.152 | 317.095 | 263.057 |
| 1.0 | 576.572 | 317.230 | 259.342 |

### M6 — scene_seed 505

The character is tucking a strand of hair behind one ear with one hand, with a soft, relaxed expression, in a quiet museum hall at midday. Lighting: bright clean daylight with soft bounce light. Color design: a warm orange and teal palette in the background and lighting only. Slightly low camera angle, natural body proportions. Full-body shot: the whole character is visible from head to feet, with a detailed background.

| 강도 | pipeline elapsed_seconds | worker.seconds | 차이(분석·준비·후처리 등) |
|---|---:|---:|---:|
| 0.0 | 569.652 | 307.145 | 262.507 |
| 0.6 | 573.571 | 322.119 | 251.452 |
| 1.0 | 589.052 | 330.219 | 258.833 |

### M7 — scene_seed 101

The character is holding an open umbrella over one shoulder, with a soft, relaxed expression, at a shrine with red gates at dusk. Lighting: colorful neon accent light on one side. Color design: a coral and turquoise palette in the background and lighting only. Eye-level view, natural body proportions. Full-body shot: the whole character is visible from head to feet, with a detailed background.

| 강도 | pipeline elapsed_seconds | worker.seconds | 차이(분석·준비·후처리 등) |
|---|---:|---:|---:|
| 0.0 | 568.663 | 310.619 | 258.044 |
| 0.6 | 581.622 | 321.824 | 259.798 |
| 1.0 | 583.722 | 320.751 | 262.971 |

두 측정값은 범위가 다르다. 차이는 분석·준비·후처리 등의 묶음 차이이며 순수 GPU 오버헤드라고 단정하지 않는다. 반올림 칸 시간 합과 시작/종료 시각 차이는1초 차이로, 정밀하게 맞춰 새 시간을 만들지 않았다.

## 보존 자료와 검증 범위

- codex-audit에 before/after, preflight, runtime-env-check, launch-process, process-check, launch-snapshot을 보존했다. 사용자 원본·기존 미추적·기억·판정을 수정하지 않았다.
- C 표시 폴더 inputs/results에 입력 및 결과 사본을 보관했다. 보고·증거 JSON·sheet·오류 기록·결과 사본을 프로젝트와 SHA 대조했다. display-manifest-35.json이 복사 검증 근거다.
- 사용자 판정은 미입력이다. quality_pass=null, human_judgements=[]를 유지했고 품질 합격수나 판정을 만들지 않았다.

## 사용자 판정

1. 그림체가 눈에 띄게 바뀌었나(0.6 / 1.0)?
2. 같은 캐릭터로 보이나, 특징이 무너진 칸은 어디인가?
3. 'AI 같은 느낌'이 줄었나, 늘었나?
