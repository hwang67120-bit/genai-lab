# GPU 시험 요청 33 — 같은 캐릭터의 승인 그림을 머리 참조로 추가하면 동일성이 나아지는가 (초안, 승인 전)

설계·근거: `docs/CHARACTER_MEMORY_REUSE_REVIEW.md`(H1).

- 상태: **초안, GPU 미승인**(사용자 2026-10-22 "GPU 실행은 아직 승인하지 않습니다"). 아래 '선행 구현'이 끝나고 사용자가 승인하기 전에는 실행하지 않는다.
- 10/22 보완: 참조 4개를 고정했다(`memory-ref-33/manifest.json`, 검토 문서 ⑨). 눈으로 보니 승인 크롭은 원본보다 나은 색·형태 정보를 주지 않는다(⑨ 9-1). 그래서 이 시험의 주목적은 **퇴행·따라옴·비용 확인**이다.
- 실행: Codex. 실행 코드: 선행 구현 커밋 이후의 HEAD.
- 한 요인 비교다. A(현재) vs C-add(승인 그림의 머리 크롭을 image 3으로 추가 + 그 그림을 설명하는 문장 1개). 나머지는 모두 같다.
- 8단계로 거르고, 통과 기준을 넘을 때만 40단계로 확인한다(CONTRACT 11절 7·8항).
- 기록 재사용(B)은 시험군이 아니다. 같은 특징표가 이미 저장값으로 들어가므로 생성 조건이 A와 같다.

## 선행 구현 (Claude, 사용자 승인 뒤, GPU 전)

1. `genai_lab/character_memory.py`: 캐릭터 ID·이미지 SHA 색인·승인 등록 기록(검토 문서 ③ 구조).
2. 시험 선택지 `memory_reference_add`(bool, 기본 꺼짐), `FreeGenerationSpec` 기억 참조 필드와 검증(승인 기록 SHA 대조), 작업 프로세스 image 3, 지시문 조건 `memory_reference` 1개.
3. 참조는 **이미 고정했다**(2026-10-22, `memory-ref-33/make_refs.py` → `manifest.json`, 다시 만들어도 같은 SHA 확인). 구현은 이 파일을 SHA로 대조해 쓰고 다시 만들지 않는다.
4. CPU 시험:
   - 선택지를 끄면 지시문·그림이 요청 32와 바이트 단위로 같다.
   - 켜면 image 3과 문장 1개만 다르다.
   - 다른 캐릭터의 승인 그림·8단계 승인 그림·같은 장면 그림은 거부한다.

## 입력

- 기준: 요청 29와 같은 원본(`eval-smoke-26/sources/M3-braid.jpg`, `M6-hoodie.jpg`, N1·K 원본).
- 특징표·성별·부위: `eval-set-26/proposed-inputs.json` 그대로(사용자 확정).
- 참조 후보: 요청 29에서 사용자가 통과시킨 40단계 결과(`judgement-29/ledger.json`, SHA 확인함 2026-10-22).
  - M3: S1·S3 raw
  - M6: S3 hair_auto, S2 hair_restored
  - N1: S1·S2·S3 raw
  - K: S1·S3·S4 raw
- 참조 모양: 승인 그림을 그 그림 자신의 분할로 머리 크롭하고, 머리 밖 픽셀은 흰색으로 칠한다. 좌표는 그 그림의 768×1024 캔버스 기준이다.

## 줄(12장)

| 줄 | 장면 | 성격 | 고른 참조(선행 구현 3번에서 채움) |
|---|---|---|---|
| M3-S2 | sitting on a simple stool in a cozy cafe | 알려진 머리색 실패 | M3-S3 raw · `memory-ref-33/M3/memory-reference.png` (cddb6351…) |
| M3-S4 | kneeling on one knee on a beach at sunset | 기억에 없는 장면 | 같음 |
| M6-S4 | kneeling on one knee on a beach at sunset | 알려진 머리색 실패 | M6-S3 hair_auto · `memory-ref-33/M6/memory-reference.png` (95c8f4f8…) |
| M6-S1 | standing in a sunny park with trees in the background | 기억에 없는 장면 | 같음 |
| N1-S4 | kneeling on one knee on a beach at sunset | 정상·퇴행 확인 | N1-S1 raw · `memory-ref-33/N1/memory-reference.png` (a8d65742…) |
| K-S2 | sitting on a simple stool in a cozy cafe | 정상·헤어스타일 퇴행 확인 | K-S1 raw · `memory-ref-33/K/memory-reference.png` (009bf573…) |

- 줄마다 A와 C-add 1장씩 만든다.
- 호출: `run_free_generation(..., selection_seed=42, experiment=dict(lightning8=True, single_attempt=True[, memory_reference_add=True]))`, seed 42. 그 밖은 제품 기본값이다.
- 재현 확인: A의 M3-S2가 요청 32 `free-r32/M3-S2-orig/qwen/raw.png`와 픽셀이 같아야 한다. 다르면 멈추고 보고한다.
- 시간: 약 2시간 20분. 상태 파일은 `batch-status.json`.
- 기록할 것: 줄마다 작업 시간, PyTorch 최고 reserved, `memory_selection` trace.

## 8단계 결과의 해석 범위(시험 전 고정, 검토 문서 ⑨ 9-3)

- 8단계는 제품 설정과 다르다: true CFG 1.0(부정 지시문·추가 문장이 CFG로 증폭되지 않음), 가속 LoRA, 재시도 없음.
- 따라옴·중대 퇴행·메모리 초과 → C-add 중단 근거.
- 개선 → 채택·개선 주장 없음. 40단계 확인 요청으로만 이어진다.
- 차이 없음 → 색 항목은 결론 없음, C 보류. 40단계 확인은 사용자가 따로 결정한다(자동 진행 안 함).
- 8단계 결과를 40단계 운영 집계에 합치지 않는다.

## 중단 조건

- `memory_stopped` 또는 reserved > 6.5GiB → C-add 중단. A만 마치고 보고한다. **상한은 시험 중에도 자동으로 완화하지 않는다.** 램 48GB는 GPU(8GB) 여유가 아니다.
  - 참고: 3장 입력은 40단계에서 6.461GiB였다(요청 25). 조건 그림은 크기와 관계없이 약 1024² 면적으로 커지므로 작은 크롭도 같은 규모다.
- 기억 참조 SHA 불일치 → 즉시 중단.
- 사용자 판정 전이라도 Codex 관찰로 참조의 배경·자세가 결과에 그대로 보이는 줄이 처음 3줄 중 2줄 이상 → 중단하고 보고한다.

## 판정과 다음

- 판정 묶음 6개. 묶음마다 원본 | 결과 X | 결과 Y를 놓고, 시험군은 가린다. 열쇠는 따로 보관한다.
- 항목: 종합 통과, 실패 분류, 머리색, 얼굴·헤어스타일, 피부, 의상·장식, 부위, 요청 반영, **이전 참조에서 따라온 것**, 동일성 비교(X/같음/Y).
- 통과·퇴행 기준: 검토 문서 ⑥(시험 전 고정).
  - 알려진 실패 2줄 중 1줄 이상 개선, 나쁨 0
  - 나머지 4줄 퇴행 0
  - 따라옴 0
  - 요청 반영 손실 0
  - 메모리 초과 0
- 충족하면 40단계 확인 요청(약 10장, 6~8시간)을 따로 제안한다. 미달이면 B만 채택하고 C는 보류한다(검토 문서 ⑦).
