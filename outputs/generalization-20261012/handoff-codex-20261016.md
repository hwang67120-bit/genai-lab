# Codex 인계 — 2026-10-16 (요청 26: 고정 평가 세트 A)

사용자 결정: "특징표 그대로 승인, 요청 26 승인하고 푸쉬 해줘", "코덱스한테 요청 26 인계 메모 작성해 줘". 교대 그대로 — Claude 구현·커밋, **Codex GPU 시험**. 요청서: `codex_gpu_request_26.md`(승인). 실행 코드: **`66d50f14` 이후 HEAD**(브랜치 `codex/onepass-implementation`, 푸시됨).

## 1. 먼저 확인할 것 — 환경이 바뀌었다(D: 고장)

- 10/16 mydesk의 D:·G:가 물리적으로 사라졌다. 모델·venv는 **`C:\genai-temp-d\genai-cache`에 임시로 다시 설치**하고 `subst D: C:\genai-temp-d`로 기존 `D:\genai-cache` 경로를 살렸다. 새 드라이브가 오면 폴더를 옮길 예정.
- **재부팅하면 subst가 풀린다.** 실행 전 `D:\genai-cache\venv\Scripts\python.exe`가 보이는지 확인, 없으면 `subst D: C:\genai-temp-d`.
- 인증서: Avast가 HTTPS를 가로채 pip·HF 내려받기에 PEM이 필요(`C:\genai-temp-d\genai-cache\windows-roots.pem`, PIP_CERT·REQUESTS_CA_BUNDLE·SSL_CERT_FILE). 모델은 모두 받아 둠 — 실행 중 내려받기 없음.
- 모델 대조: Qwen 2511(rev 6f3ccc0b, 매니페스트 SHA)·GGUF Q4_K_M(rev 0d33d969, 설정 SHA)·l2d_sam·WD 태거·DWPose. `drive-recovery-20261016/download_report.json` = 29 일치, 4 예전 기록 없음(WD·DWPose), 불일치 0. 설치·내려받기 스크립트도 같은 폴더.
- 복구 뒤 CPU 전체 시험 2,400 passed(`66d50f14`). 복구 환경으로 Claude가 GPU 생성 6장(통과 시험 M3·M5·F5 2장·M6·F5 피부 재실행)을 정상으로 돌렸다 — 메모리 상한 초과·계약 불일치 없음.
- 실행은 UNC 경로(`\\192.168.0.109\win_g\genai-lab\...`)로.

## 2. 요청 26 요약

- 목적: 자유 생성 1차 완료 판정(CONTRACT 10절 A, D-133). 사람 판정으로 **1장 통과율**과 **선택 통과율**(재시도·머리색 보정 후보 포함)을 따로 잰다. 완료 = 선택 통과율 70%(21줄 중 15) 이상 + 같은 실패가 2명 이상 반복 없음 + 처음 보는 캐릭터 통과 줄 있음.
- 캐릭터 7명: N1·N2·K(요청 21 확정 특징표) + 처음 보는 M2·M3·M6·F5(제안 특징표를 사용자가 그대로 승인). 입력 전부 `eval-set-26/proposed-inputs.json`(status `user_confirmed`, 줄 목록 포함). 새 캐릭터 기준 = `eval-smoke-26/sources/*.jpg`(원본).
- 장면 4종(S1 공원 서기·S2 카페 의자·S3 눈 밤거리 걷기·S4 해 질 녘 해변 무릎 꿇기), 캐릭터마다 3종 → 21줄.
- 제품 함수 `run_free_generation` 그대로: seed 42, 확신 위반만 43 재시도, 골격 유도 꺼짐, 원본 기준.
- 실행기 `codex-preflight-26/run_26.py`(요청 25 구조): 인자 없이 CPU 사전 검사, `--run` 21줄, **끝난 줄 건너뛰고 이어서 실행 가능**하게. 한 줄 약 35분, 전체 약 12시간+.

## 3. 사전 검사에 넣어 줄 것

- 입력 SHA, 특징표 `user_confirmed`, 줄 수 21, 연출별 줄 수(S1 5·S2 5·S3 6·S4 5).
- 줄마다 요청 검사 통과, `classify_pose`가 장면의 지원 자세와 같음, `mentions_background` True(흰 배경 문장이 지시문에 없어야 함), 장면 고유 색 문장(D-129) 있음.
- 피부 특징이 있는 M2·F5는 지시문에 `dark skin` 낱말이 없고 피부 톤 유지 문장(`SKIN_TONE`, D-134)이 있어야 함.
- 코드·설정 SHA 잠금, 배치 상태 파일(`batch-status.json`) — Claude는 이 파일이 running이면 genai_lab 코드를 건드리지 않는다.

## 4. 이번 기간 코드 변경(Claude, 모두 CPU 시험 통과)

| 커밋 | 결정 | 내용 |
|---|---|---|
| 66c300d9 | — | 성인 남성 상의 탈의: 1boy ≥0.7·topless_male ≥0.5·1girl <0.5이면 nipples/topless를 차단 대신 검토(사용자 결정). loli/shota 등은 계속 차단 |
| 57e997e1 | D-134 | 피부는 태그 낱말 대신 "그림 1과 같은 피부 톤 유지" 문장(F5에서 `dark_skin` 낱말이 피부를 확연히 어둡게 함 → 수정 후 ΔE 36.6→11.5, 사용자 성공). 완료 기준 C의 개선 1회차 |
| 66d50f14 | D-135 | 배경 요청 낱말 보강(꽃밭·정원·초원 등이 흰 배경 문장과 충돌하던 문제). 요청 26 장면 문장의 인식·지시문은 그대로 |

## 5. 통과 시험(eval-smoke-26, Claude 실행) 결과 — 참고

- M2 드레드·M5 강아지 소년·M6 후드: 정상. M3: 땋은 머리 길어짐·리본 사라짐(사용자 성공 처리). F5: 피부 어두워짐 → D-134 뒤 성공. M1: 앞만 가리는 천 의상으로 노출 차단 → 사용자 제외, M6로 대체. M5는 N2와 겹쳐 예비.
- 기록: `eval-smoke-26/report.md`, 비교 그림 `compare-*.png`.

## 6. 기록 요청(요청서 '기록' 절)

- 캐릭터별 비교 그림 7장(원본 | 장면 3줄, 재시도·머리색 보정 후보는 옆에 작게). 사용자 판정은 작은 묶음으로(한 장에 4~6짝 이내).
- 자동 기록: 조건별 관찰, 재시도 이유, 머리색 보정 후보, text encode·생성 메모리, 실제 지시문.
- 보고서 `codex_gpu_report_26.md`. 집계는 사용자 판정 뒤 `scripts/studio_judgement_summary.py`.

## 7. GPU 시험 규칙(그대로)

- 사전 CPU 대조·SHA 잠금. 계약 불일치·메모리 상한(6.5GiB) 초과·모델 SHA 불일치는 즉시 보고, 자동 재실행 금지.
- 거부·실패 줄은 기록하고 다음 줄로. 끊기면 이어서 실행(이미 끝난 줄은 다시 만들지 않음).
- 안전: 성인 캐릭터만. 입력은 승인된 7명뿐 — 다른 그림을 넣지 않는다.

## 8. 열린 일(사용자 결정 대기, 요청 26과 별개)

1. 요청 20·21 줄별 사용자 판정(일부만 있음: K-1 눈, N2-2 머리).
2. 꼬리 가림 미반영 — 운영 판정 20건 뒤로 미룸(Claude 권고).
3. 편집본 기준 평가는 나중에.
