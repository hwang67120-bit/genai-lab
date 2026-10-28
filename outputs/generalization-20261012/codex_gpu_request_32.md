# GPU 시험 요청 32 — 어두운 배경 원본을 흰 배경으로 바꾸면 나아지는가 (승인 2026-10-21)

M3·M6(어두운 배경 원본)은 머리색·자세 실패가 잦았다. 원본 조명(검은 배경의 테두리 빛, 푸른 조명)이 색에 섞이고, 길쭉한 검은 그림이 흰 캔버스 위 '검은 기둥'으로 들어가는 것이 원인 후보다. **기준 그림의 배경 하나만** 바꿔 8단계로 훑는다(운영 규칙 11절 7항). 상태: **승인** — 사용자 "요청 32 승인, 진행해줘"(2026-10-21). 실행: **Codex**. 실행 코드: 이 문서 커밋 이후 HEAD.

## 입력

- 원래 배경: `eval-smoke-26/sources/M3-braid.jpg`, `eval-smoke-26/sources/M6-hoodie.jpg`.
- 흰 배경: `bg-white-trial/M3-white-canvas.png`, `bg-white-trial/M6-white-canvas.png`(768×1024, 요청 29 기준 분할 합집합 ∪ 배경색과 먼 픽셀로 인물만 남김, `make_white_bg.py`). 한계: M6 머리 가장자리에 짙은 남색이 조금 남음.
- 특징표·성별·부위: `eval-set-26/proposed-inputs.json`의 M3·M6 그대로(사용자 확정). 특징 측정(머리색 기준 등)은 각 기준 그림에서 다시 잰다 — 배경 변화의 일부로 본다.

## 줄(8장)

| 줄 | 기준 | 장면 |
|---|---|---|
| M3-S2-orig / M3-S2-white | 원래 / 흰 배경 | sitting on a simple stool in a cozy cafe |
| M3-S3-orig / M3-S3-white | 원래 / 흰 배경 | walking along a snowy street at night |
| M6-S2-orig / M6-S2-white | 원래 / 흰 배경 | sitting on a simple stool in a cozy cafe |
| M6-S3-orig / M6-S3-white | 원래 / 흰 배경 | walking along a snowy street at night |

- 호출: `run_free_generation(..., base_kind="original", gender="male", body_parts=[], selection_seed=42, experiment=dict(lightning8=True, single_attempt=True))`, seed 42. 제품 기본값 그대로(없는 부위 부정 지시문, 머리색 자동 보정 문턱 7, 재시도 R3 — single_attempt라 재시도 없음).
- 시간: 줄당 약 10분 → 약 1시간 반. 상태 파일 `batch-status.json`.

## 판정과 다음

- 판정 묶음: 원본 | 원래 배경 결과 | 흰 배경 결과(같은 장면끼리). 쌍마다 '흰 배경이 나음/같음/나쁨'과 실패 분류표.
- 채택 후보: 4쌍 중 3쌍 이상 흰 배경이 나음 + 다른 특징 악화 없음 → '어두운 배경 원본은 인물을 오려 흰 배경으로' 공통 전처리 후보 → 같은 줄 40단계 확인 뒤 사용자 결정. 미달이면 기록만.
