# 요청 37 Python 3.10 배포 후보 검증 결과

P4·P5·P6 검증 완료: 요청서 범위의 결과가 통과했고, 준비 단계·안쪽 실행기·바깥 실행기 종료 코드는 모두 0이다.
이번 결과는 고정된 Python 3.10.6 후보의 검증이며 최종 배포 승인이 아니다. 그림 품질 판정이나 새 사용자 합격 판정은 만들지 않았다.

## 실행 범위와 파일 보존

- 시작 HEAD = 종료 HEAD: 6473095ed059c2e7281c398157e1d62fc705db1e. 추적 파일·요청서·cases·고정 목록 1136개 재대조 변경 0, 기존 git status 그대로.
- 후보: D:/genai-test/배포 후보 GenAILab-20261023-embed. 파일 44,991개, 6,692,910,585바이트.
- 고정 지문: 4fe8cf0b60cc5ac2d9983f3544c25d4d81ab131c74d46679dde03f661726da03. P4·P5 각각 verify-before/after 일치.
- 모델 폴더: D:/genai-test/model-home-20261023. 이동·삭제·교체 없음. Python·패키지·후보·운영 코드 수정/설치/다운로드/커밋/푸시 없음.
- 모든 원본·시도·실패·미리보기·JSON·로그는 D:/genai-test/gpu-p4-20261023 및 gpu-p5-20261023에 보존. 저장소에는 보고서와 작은 증거만 복사.
- P6는 후보 런타임+Windows로 좁힌 PATH에서 P4/P5를 실행한 것으로 함께 확인했다. 별도 반복, 추가 재시도, 개발 환경 대조 생성은 하지 않았다.
- launching_shell_gpu_vars: P4 [], P5 []. 바깥 진행·파일 대조만 개발 Python이며 검증 대상 분할/Qwen과 작업실은 후보 runtime-main이다.

## 결과와 프로세스 종료

| 단계 | 결과 | 프로세스 종료 | 후보 대조 |
|---|---|---|---|
| P4 | 통과 5 / 검토 필요 0 / 예상 밖 실패 0 | 안쪽 0, 바깥 0 | 전·후 일치 |
| P5 | A3·B2 생성·저장 완료 | 준비 0, 안쪽 P5 0, 바깥 0 | 전·후 일치 |
| P6 | 좁힌 PATH에서 위 결과 재현 | 같은 실행의 종료 코드로 확인 | 별도 실행 없음 |

- P4: 2026-10-24 00:38:55 ~ 2026-10-24 00:41:22 KST. 분할 안쪽 80.9초(파일 대조 제외), 벽시계 2분 27초.
- P5: 2026-10-24 00:42:00 ~ 2026-10-24 03:18:36 KST. 벽시계 2시간 36분 36초. 사용자 흐름 2026-10-24 00:46:45 ~ 2026-10-24 03:16:25 (2시간 29분 40초).
- 준비 화면 → 모델 폴더 선택 → 깊은 검사 214.6초, 문제 0 → 작업실. 준비 화면의 files_verified는 GPU 품질 승인과 구별했다.
- 실제 프로세스 목록 확인 시각: 2026-10-24T03:20:50.7165470+09:00. 진행기/작업실/Qwen은 모두 사라졌고 CPU 증거 수집기만 있었다.
- 보고 예외: run-report.json의 result.cases.A3/B2는 null이다. 바깥 요약이 안쪽 cases.<id>.flow.state 대신 .state를 읽는 기록 구조 문제다. summary·inner flow·실제 저장 파일과 SHA·종료 코드를 직접 대조했다. 후보 파일은 고치지 않았다.

## P4 분할 비교

| 건 | 결론 | 시간(초) | 보호/얼굴/장식 픽셀 개발=후보 | 배열·파일 | 최고 reserved(GiB) |
|---|---|---:|---|---|---:|
| B2-outfit-parts | 통과 | 24.3 | 18727/5809/0 | 19개 배열·npz SHA 일치 | 5.078125 |
| B2-base-parts | 통과 | 13.7 | 18625/5691/0 | 19개 배열·npz SHA 일치 | 5.078125 |
| B2-result-parts | 통과 | 13.6 | 12945/4239/0 | 19개 배열·npz SHA 일치 | 5.078125 |
| A3-base-parts | 통과 | 13.8 | 39471/8929/1098 | 19개 배열·npz SHA 일치 | 5.078125 |
| A3-result-parts | 통과 | 13.1 | 기준·후보 모두 예정된 거절 | 같은 검사에서 거절 | 기록 없음 |

- 모든 입력 캔버스 SHA가 요청 34 기록과 같다. 완료 4건은 배열·npz SHA 완전 일치이므로 미해결 배열 차이, 별도 검토 기록 또는 겹친 그림이 필요하지 않았다.
- A3-result-parts: 문구 '얼굴·머리를 찾지 못했습니다. 이 그림은 원본 편집형 의상 교체를 쓸 수 없습니다.'와 parts 단계, outfit_edit_worker.py 37행 segment, ValueError가 모두 같다.
- 요청 34·현재 소스·고정 후보의 이 거절 코드 SHA 모두 f80994c72e2f0bd85baba7fb596578c1391d49b5ba77e4ce22432957c250eefd. 같은 파일·단계·검사에서 같은 이유로 거절했다. 보호 영역/얼굴 중 어느 쪽이 비었는지는 개발 기록에 없어 구분하지 않았다.
- 분할 Python: D:\genai-test\배포 후보 GenAILab-20261023-embed\runtime-main\python.exe. 체크포인트 SHA: dda82b916f6c340bc5a6c3a077ff1214f9906906abe3707b1076d55b508f253f.
- 예상된 거절 작업은 정상 분할 완료로 바꿔 세지 않았다. P4의 '통과'는 예정된 거절 재현을 포함하는 시험 판정이다.

## P5 두 줄

| 줄 | 기술 결과 | 행 전체 시간 | Qwen 작업 시간 합 | 최고 reserved(GiB) | 자동 시도·선택 |
|---|---|---:|---:|---:|---|
| A3 | 생성·저장 완료 | 00:40:44 (2444초) | 2087.831초 | 6.367188 | ② seed42 1회, 1번 raw 선택 |
| B2 | 생성·저장 완료 | 01:48:56 (6536초) | 5735.625초 | 6.361328 | ① seed42 1회; ② seed42/43 2회, 1번 raw 선택 |

- B2 ②의 두 시도는 요청 34와 같이 '확신 있는 특징 위반: skin'으로 제품 자체 재시도가 실행됐다. 두 시도 점수가 같아 1번 raw가 선택됐다. 실행기나 사람이 추가한 재시도는 없다.
- A3 자동 머리색 보정은 '결과 분할 없음', B2는 '무채색 계열(평균 채도 6.9)'으로 적용되지 않았다. 파생 후보를 저장 결과로 바꾸지 않았다.
- trace의 user_approved 및 저장 버튼에 따른 approve_saved_by_user는 이 자동 시험의 저장 행동이다. 실제 사용자 품질 합격으로 세지 않았다.
- 메모리는 모든 관련 memory/run/run.final/progress/result 기록을 대조한 작업 프로세스 PyTorch reserved 실측 최고값이다. limit/cap 설정값을 최고값으로 세지 않았고 dedicated/shared VRAM 사용량으로 해석하지 않았다.
- 모든 Qwen 요청·memory.json 상한 6.5GiB, 실측 초과 0, memory_stopped 0. 최고 측정은 텍스트 인코딩의 reserved_bytes였으며 마지막 generation_memory 값만 쓰지 않았다.

## 시도별 실제 호출·CPU 준비물·결과 비교

| 줄/단계/seed | Qwen 초 | 최고 reserved | CPU model_inputs | raw SHA | 다른 픽셀 |
|---|---:|---:|---|---|---:|
| A3/free-generation/42 | 2087.831 | 6.367188 | 일치 | 일치 | 0 |
| B2/outfit-edit/42 | 1948.565 | 6.328125 | 일치 | 일치 | 0 |
| B2/free-generation/42 | 1909.920 | 6.361328 | 일치 | 일치 | 0 |
| B2/free-generation/43 | 1877.140 | 6.361328 | 일치 | 일치 | 0 |

- 네 시도 모두 40단계, CFG 4.0, guidance 1.0, group_block1, 896×1184, seed 42(재시도 43), 가속/스타일 LoRA 없음. 실제 call_arguments는 대응 개발 시도와 완전히 같다. 경로를 제외한 설정 값도 같다.
- qwen-inputs의 그림 6개(A3 2, B2① 2, B2② 2) SHA 일치. 각 worker의 model_inputs 파일 SHA도 요청 관련 파일을 제외하고 이름별 전부 일치한다.
- B2① conditions.json의 파일 SHA는 다르지만 JSON 대조 결과 garment_reference.path/source 두 경로만 다르다. 그림·프롬프트·실제 호출·model_inputs·raw에는 차이가 없어 예상된 경로 기록 차이로 분류했다(미해결 차이 없음).
- raw 네 장과 실제 저장 두 장은 대응 요청 34 파일과 SHA가 같다. RGBA 픽셀 배열 비교도 모두 다른 픽셀 0 / 1,060,864, 최대 채널 차이 0이다. 재시도 수·이유·선택도 같다.
- 분류: 생성 계산·CPU 준비 그림·결과는 '차이 없음'. 기록 JSON의 두 경로는 배포 위치 변경에 따른 차이. 런타임만 바꾼 개발 대조는 필요하지 않아 제안/실행하지 않았다.
- 네 시도 모두 callback index 0~39 40회, preview 오류 기록 없음, 최종 preview-039.png 존재. 남아 있는 preview는 각 037/038/039 세 장이며 중간 40장 모두를 보존했다고 주장하지 않는다. preview는 잠재값 선형 색 근사이며 최종 raw가 아니다.
- 각 Qwen 작업 프로세스 기록에서 명세 27개·GGUF SHA 검사 완료를 확인했다. 이는 제품의 기존 검사 기록이며 별도 전체 모델 재검사나 검사 정책 변경을 추가하지 않았다.
- Qwen/분할 Python = 후보 runtime-main/python.exe. Qwen torch/diffusers/transformers/bitsandbytes/gguf 패키지 위치 전부 후보 안. 관절 환경 설정 = 후보 runtime-pose/python.exe.
- 실제 패키지 기록: torch 2.9.1+cu128, diffusers 0.38.0, transformers 4.57.6, bitsandbytes 0.50.2, gguf 0.19.0. 실행 중 버전 변경 없음.

## 저장 경로와 증거

- A3: D:\genai-test\gpu-p5-20261023\flow\A3-saved.png (SHA 1daa0ce2d0c4437fd59ebee4150ad759e44c79ddd28686235b3b723c99c17ad5).
- A3 scene seed 1216248804: The character is stepping forward with one hand raised in a small wave, with a dreamy look, in a garden with stone paths at golden hour. Lighting: dappled sunlight falling through leaves. Color design: a deep blue with gold accents palette in the background and lighting only. Eye-level view, natural body proportions. Full-body shot: the whole character is visible from head to feet, with a detailed background.
- B2: D:\genai-test\gpu-p5-20261023\flow\B2-saved.png (SHA a355a69461e19849c34a5f47281b1d71bb04ded1480a0166cdc191128b622f1d).
- B2 scene seed 2026898827: The character is reaching up to touch a hanging branch, with a dreamy look, in a snowy mountain village at sunset. Lighting: dappled sunlight falling through leaves. Color design: a pastel pink and mint green palette in the background and lighting only. Eye-level view, natural body proportions. Full-body shot: the whole character is visible from head to feet, with a detailed background.

- 원자료: D:/genai-test/gpu-p4-20261023 및 D:/genai-test/gpu-p5-20261023. 대응 개발 기준은 cases-p5.json의 dev_run·dev_saved.
- 작은 원본 기록 사본: codex-audit-37/p4/ 및 p5/. before.json·after.json·refusal-code-sha.json으로 소스와 거절 코드 근거 확인.
- 상세 SHA·최고 메모리 근거 파일·호출값·비교값·종료 코드는 codex-evidence-37.json. C 표시 폴더 사본은 report-copy-sha.json으로 대조한다.

사용자 조치: 없음. 최종 배포용 Python 선정·교체와 배포 승인은 이번 검증과 별개이며 자동 진행하지 않는다.
