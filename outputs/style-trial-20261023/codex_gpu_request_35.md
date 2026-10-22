# GPU 시험 요청 35 — 그림체 LoRA 맛보기(S0.5) 4캐릭터 × 3강도 (승인 2026-10-23)

## 요약

- **목적:** 직접 학습(S1) 전에 공개된 Qwen 그림체 LoRA 하나를 빌려 세 가지를 본다.
  - 효과 크기: 그림체가 눈에 띄게 바뀌는가
  - 동일성 섞임: 그림체가 바뀔 때 특징(머리색·눈색·머리 모양·부위·옷)이 같이 무너지는가
  - 기술 동작: 가속 LoRA와 함께 얹혀 메모리 상한 안에서 도는가
- **시험 전용이다(docs/STYLE_STAGE_DESIGN.md 3-3 오염 방지 선).**
  - 결과는 `outputs/style-trial-20261023/` 에만 두고 trace에 `trial: style_lora_borrowed`가 붙는다.
  - 품질 기준선·운영 집계·D-118 판정 20건에 넣지 않는다. 목표 그림체도 아니다.
  - 사용자 화면·제품 기본값에는 연결되지 않는다(기본 꺼짐).
- **실행 전 조건:**
  1. ~~LoRA 내려받기~~ — 사용자 승인(2026-10-23)으로 Claude가 내려받음. SHA `dc84f766…8df9`(저장소 LFS SHA와 일치)를 `genai_lab/style_loras.py`에 기록. 작업 프로세스가 불러오기 직전에 다시 대조한다.
  2. ~~GPU 실행~~ — 사용자 "GPU 실행 승인할게, 코덱스한테 인계할게"(2026-10-23).
  3. **사용자 GUI 생성(studio-runs/4b0e5e2c…)이 끝난 뒤 시작한다** — GPU를 함께 쓰지 않는다.
- **실행:** Codex. 코드: 이 문서를 커밋한 HEAD(Claude 커밋). 실행 중에는 코드 수정·커밋 없음.

## 빌린 LoRA

| 항목 | 값 |
|---|---|
| 저장소 | `prithivMLmods/Qwen-Image-Edit-2511-Anime` |
| 리비전 | `977f8c2ed22c8581c8750370bfce34058b253598` |
| 파일 | `Qwen-Image-Edit-2511-Anime-2000.safetensors`, 236,117,056 B |
| 놓는 곳 | `D:/genai-cache/models/style-loras/Qwen-Image-Edit-2511-Anime/` (기준 모델 폴더 옆) |
| SHA-256 | `dc84f7662ed724ebfb9b8c18967eff0eabd5eaf17c2d206e2e09062db9ff8df9` |
| 라이선스 | Apache-2.0 |
| 지시 문장 | `Transform into anime.` (지시문 맨 앞에 붙음) |
| 학습 자료 | 공개 안 됨 → 완성 기준 4(허가 자료 학습)를 증명할 수 없어 **제품에 쓰지 않는다** |

## 입력 (고정 — `outputs/style-trial-20261023/cases.json`)

- 예전 시험에서 사용자가 확정한 입력을 그대로 쓴다(성별·부위·특징표).

| 줄 | 기준 그림 | 성별·부위 | 장면 seed | 출처 |
|---|---|---|---|---|
| N1 | G11-r17/N1/inputs/canvas.png | 여성, 꼬리·날개 | 404 (카페, 손 흔들기) | free-r20/N1-1 |
| K | G19-r16/K/inputs/canvas.png | 여성 | 303 (온실, 걷기) | free-r20/K-1 |
| M6 | eval-smoke-26/sources/M6-hoodie.jpg | 남성 | 505 (박물관, 머리 넘기기) | free-r30-rerun1/M6-S3 |
| M7 | eval-new-male-31/sources/M7-white.jpg | 남성 | 101 (신사, 우산) | free-r31/M7-S1 |

- 줄마다 **같은 장면·같은 seed(42)** 로 세 장: 그림체 LoRA 없음 / 강도 0.6 / 강도 1.0 → 모두 12장.
- 앉은 자세 장면(seed 202)은 지원 범위 밖이라 뺐다.

## 실행

1. 환경 점검(필수 실패 0이어야 진행):

   ```
   D:\genai-cache\venv\Scripts\python.exe scripts\check_runtime_env.py
   ```

2. 실행:

   ```
   D:\genai-cache\venv\Scripts\python.exe scripts\style_trial_run.py outputs\style-trial-20261023\cases.json outputs\style-trial-20261023
   ```

   - 상태 파일 `outputs/style-trial-20261023/batch-status.json`, 끝나면 `summary.json`·`sheet.png`([원본 | 없음 | 0.6 | 1.0]).

**설정:** 8단계 가속(lightning8, CFG 1.0), 재시도 없음(`single_attempt`), 메모리 상한 6.5GiB.
- 세 장이 모두 같은 가속 설정이라 차이는 그림체 LoRA 하나뿐이다.
- 가속과 그림체 LoRA를 함께 얹는다(`set_adapters(["lightning8", "style"], [1.0, 강도])`). 작업 기록 `style_lora.adapters`로 확인.

**예상 시간:** 8단계 기준 한 장 수 분 → 합계 약 1~2시간(요청 30·31 가속 실측 참고, 확정 아님).

## 중단·기록 규칙

- `memory_stopped` 또는 PyTorch reserved > 6.5GiB → 그 칸은 실패로 기록하고 다음으로 간다. **상한을 완화하지 않는다.**
- 그림체 LoRA SHA 불일치 → 실행기가 그 칸을 실패로 기록한다. 다른 파일로 바꾸지 않는다.
- 안전검사 거부는 우회하지 않는다.
- 그림체 어댑터 불러오기·적용 오류(CPU 확인: diffusers 0.38 변환 뒤 가속 LoRA와 같은 키 형식, 다만 조절층 img_mod·txt_mod 120곳은 가속 LoRA가 안 쓰는 곳) → 그 칸 실패로 기록하고 오류 전문을 보고한다. 코드를 고치지 않는다.

## 보고 (판정은 사용자 — Codex는 관찰만)

- 칸마다: 상태, 걸린 시간, 최고 reserved, `style_lora` 기록(어댑터 이름·강도·SHA).
- `sheet.png` 경로.
- 관찰(판정 아님): 그림체 변화 크기, 머리색·눈색·부위가 바뀐 칸.
- 사용자 판정 질문(4줄 이내):
  1. 그림체가 눈에 띄게 바뀌었나(0.6 / 1.0)?
  2. 같은 캐릭터로 보이나(특징이 무너진 칸)?
  3. 'AI 같은 느낌'이 줄었나, 늘었나?
