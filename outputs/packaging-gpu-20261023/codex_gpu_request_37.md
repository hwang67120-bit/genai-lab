# GPU 시험 요청 37 — 배포 후보 P4(머리 분할)·P5(Qwen 사용자 흐름) (승인 대기)

## 요약

- **상태: 승인 대기 — GPU 실행 전 사용자 승인이 따로 필요하다.** 사용자 "P4·P5는 요청서 작성까지만 진행하고 GPU 실행은 별도로 승인받으세요"(2026-10-23).
- **질문 하나:** 공식 내장형 Python 3.10.6으로 만든 배포 후보에서, GPU 작업 프로세스 두 종류(머리 분할·Qwen)가 개발 환경과 같은 결과를 내는가.
- **새 품질 시험·설정 탐색이 아니다.** 비교 기준은 이미 있는 개발 결과·실행 기록(요청 34, 2026-10-22)이다. 입력·장면·seed·설정은 그 기록 그대로다.
- **실행:** Codex. 코드: 이 문서를 커밋한 HEAD(Claude 커밋). 실행 중 코드·배포 후보 수정 없음.
- 예상 시간: P4 약 5분, P5 약 2.5시간(준비 검사 4분 + A3 약 37분 + B2 약 108분 — 요청 34와 같은 재시도가 나오면).

## 배포 후보 (고정)

| 항목 | 값 |
|---|---|
| 폴더 | `D:\genai-test\배포 후보 GenAILab-20261023-embed` (경로에 공백·한글) |
| 고정 목록 | `outputs/packaging-gpu-20261023/candidate-manifest.json.gz` — 파일 44,991개, 6.23 GiB, 지문 `4fe8cf0b60cc5ac2d9983f3544c25d4d81ab131c74d46679dde03f661726da03` |
| Python | 공식 내장형 3.10.6(`python-3.10.6-embed-amd64.zip`) — MD5 공개값 일치, GPG 서명 정상(키 지문 `7ED10B6531D7C8E1BC296021FC624643487034E5`, 아래 '출처'), 실행 파일 Authenticode 서명 Python Software Foundation |
| 패키지 | 기존 복사 빌드 후보(D-174)와 **파일·해시까지 같음**(주 29,538개·관절 15,091개 파일). 버전 변경 없음 |
| 앱 코드 | 기존 후보와 같음(292개 파일) |
| P1~P3 | 이 후보로 다시 통과(D-175, CPU) |

- 실행기가 시작 전·끝난 뒤 고정 목록과 대조한다. **시작 전 불일치면 아무것도 돌리지 않는다.**

## 비교 조건 — 요청 34(개발)와 무엇이 같고 다른가

| 조건 | 요청 34(개발) | 이번(배포 후보) | 구분 |
|---|---|---|---|
| 입력 그림 | A3 캐릭터 / B2 캐릭터·의상 | 같음(SHA 대조, 다르면 그 줄 건너뜀) | 고정 |
| 장면 | 자동 조합 seed A3 1216248804 / B2 2026898827 | **같은 seed로 고정**(실행기가 이 실행 안에서만 `choose_scene`을 바꿔 끼움, 문장 일치 확인) | 고정 |
| 생성 seed·설정 | 42/43, 40단계, CFG 4, guidance 1, `group_block1`, 6.5GiB | 같음(설정 파일은 경로만 다르고 값 같음) | 고정 |
| 모델 파일 | `D:\genai-cache\…` | 시험용 모델 폴더 `D:\genai-test\model-home-20261023`(같은 파일, 준비 검사에서 SHA 확인) | 위치만 다름 |
| 작업실 Python | 개발 venv(시스템 3.10.6 기반) | 후보 runtime-main(내장형 3.10.6) | **검증 대상** |
| Qwen 작업 프로세스 Python | `qwen-test-venv` | 후보 runtime-main(`@app`) | **검증 대상** |
| 머리 분할 Python | 개발 venv | 후보 runtime-main | **검증 대상** |
| 하위 프로세스 환경 변수 | 실행 셸 전체 상속 | 필요한 이름만(child_environment, D-170) + PATH = 후보 + Windows | **검증 대상** |
| 코드 | 1eb011f9 | 이 HEAD — 그 사이 변경은 기록·검사·환경 변수·경로·미리보기 파일 이름뿐, 생성 계산 변경 없음(아래) | 영향 없을 것으로 봄 |

- 1eb011f9 이후 제품 코드 변경: 노출 검사 기록 방식(D-158, A3·B2는 막히지 않은 입력), 그림체 LoRA 맛보기(기본 꺼짐), 모델 폴더·배포판 경로(D-166·D-174), 하위 프로세스 환경 변수(D-170), 필수 모델 검사(D-172), 미리보기 파일 이름(D-157).
- PATH를 후보 + Windows로 좁힌 상태라 **P6(PATH 좁혀 P4·P5 반복)도 이 실행으로 함께 확인**한다.
- 띄운 셸의 GPU 관련 변수(`CUDA_*`·`PYTORCH_*` 등)는 넘기지 않는다. 이름만 `run-report.json`의 `launching_shell_gpu_vars`에 남는다. 요청 34는 셸 전체를 상속했으므로, 이름이 있으면 조건 차이로 보고한다.

## P4 — 머리 분할 작업 프로세스 (먼저)

- 요청 34의 분할 5건을 **같은 캔버스(SHA 대조)** 로 후보 런타임에서 다시 돌린다: B2 의상 편집·기준 그림·결과 분석, A3 기준 그림·결과 분석.
  - A3 결과 분석은 요청 34에서도 분할이 "얼굴·머리를 찾지 못했습니다"로 거절한 건이다(기준 결과 없음). 이 건은 **같은 거절 문구가 나오면 같음**으로 본다.
- 통과: 5건 모두 개발과 같은 결론(분할 4건 완료 + 거절 1건), 분할된 4건의 `parts.npz` 배열이 개발 결과와 같음(`all_equal`), 보호·얼굴·장신구 픽셀 수 같음, 체크포인트 SHA 같음, 분할 프로세스 Python이 후보 안. 최고 reserved 기록(요청 34: 5.08GiB).
- **P4가 실패하면(오류·후보 밖 실행) P5를 시작하지 않고 보고한다.** 배열이 다르면 P5는 진행하되 다른 픽셀 수를 그대로 보고한다(원인 구분은 아래 '차이 구분').

## P5 — Qwen 사용자 흐름 2줄 (의상 없음 A3 · 의상 있음 B2)

- 사용자 첫 실행 그대로: 가짜 사용자 폴더에서 **준비 화면 → 모델 폴더 선택 → 깊은 검사 → 기록** → 그 기록으로 작업실 → 줄마다 캐릭터(·의상) 고르고 '이미지 생성' → 끝나면 '저장'.
  - 저장 버튼 누름은 자동 시험 행동이다. 사용자 품질 합격으로 집계하지 않는다.
- 통과: 두 줄 생성·저장 완료, 모든 Qwen 기록 `memory.limit_gib` 6.5 이하이고 최고 reserved ≤ 6.5GiB, `memory_stopped` 없음, 작업 프로세스 Python이 모두 후보 안, 끝난 뒤 후보 파일 목록 그대로.
- 비교(통과 기준과 별도로 기록):
  - CPU 준비물: `qwen-inputs`의 그림 SHA, 요청 SHA를 뺀 `model_inputs` SHA가 요청 34와 같은가.
  - 결과: 시도마다 `raw.png` SHA가 요청 34와 같은가. 다르면 픽셀 차이(다른 픽셀 비율·최대 차이)를 적는다.
  - 재시도 수·재시도 이유·선택된 시도가 요청 34와 같은가.

## 차이 구분 — 패키징 문제인가, 생성 변동인가

| 관찰 | 분류 |
|---|---|
| 후보에서만 나는 오류(불러오기 실패·DLL·후보 밖 경로·권한) | **패키징 문제** |
| CPU 준비물(분할 입력 캔버스·qwen-inputs) SHA가 다름 | **패키징 문제 후보**(CPU 단계는 같은 입력이면 같아야 함) |
| CPU 준비물은 같고 분할 배열·raw만 다름 | **원인 미확정** — 런타임 차이인지 GPU 실행 변동인지 이 실행만으로는 가를 수 없다 |
| 모두 같음 | 차이 없음 |

- '원인 미확정'이면 **대조 실행을 제안만 하고 돌리지 않는다**(사용자 승인 필요). 제안할 대조: 이 HEAD 코드·같은 입력·같은 장면 seed·같은 모델 파일·같은 설정 값으로 **런타임만 개발 환경**(개발 venv + `qwen-test-venv`, `configs/qwen_edit_settings.json`)으로 바꾼 한 번. 결과가 후보와 같으면 런타임 문제가 아니다.
- 사용자 지시: 개발 환경을 다시 만드는 일은 기본 절차가 아니다. 대조도 지금 있는 개발 환경을 그대로 쓴다.

## 실행기 미리 확인 (Claude, GPU 없이 `--dry`, 2026-10-23)

- P4: 후보 대조 시작 전·후 일치, 5건 모두 캔버스 SHA 일치, 분할 Python = 후보 runtime-main, 분할 모델·도구 = 시험용 모델 폴더 안.
- P5: 후보 대조 전·후 일치, 첫 실행 준비 검사 225초 → `files_verified`(GPU 생성 검증 '아직 안 됨') → 작업실, 빠른 검사 두 흐름 통과, A3·B2 입력 SHA·장면 문장 요청 34와 일치, Qwen 작업 프로세스 Python = 후보 runtime-main(`@app`), 관절 = runtime-pose. 띄운 셸 GPU 변수 없음.
- 기록 `outputs/packaging-gpu-20261023/dry/`. 첫 P4 미리 확인에서 A3 결과 분석(개발에서 거절된 건)을 기준 결과로 읽으려다 실패해 실행기를 고쳤다(거절 비교) — 생성·분할은 돌리지 않았다.

## 실행

1. 다른 GPU 작업(사용자 GUI 포함)이 없는지 확인한다. 시험용 모델 폴더 `D:\genai-test\model-home-20261023`가 원래 이름으로 있는지 확인한다.
2. P4:

   ```
   D:\genai-cache\venv\Scripts\python.exe -X utf8 packaging\run_candidate_gpu.py p4 "D:\genai-test\배포 후보 GenAILab-20261023-embed" outputs\packaging-gpu-20261023\candidate-manifest.json.gz D:\genai-test\model-home-20261023 D:\genai-test\gpu-p4-20261023 Z:\genai-lab
   ```

   - 결과 `D:\genai-test\gpu-p4-20261023\segment\inner-p4.json`, `run-report.json`.
3. P4가 오류 없이 끝났을 때만 P5:

   ```
   D:\genai-cache\venv\Scripts\python.exe -X utf8 packaging\run_candidate_gpu.py p5 "D:\genai-test\배포 후보 GenAILab-20261023-embed" outputs\packaging-gpu-20261023\candidate-manifest.json.gz D:\genai-test\model-home-20261023 D:\genai-test\gpu-p5-20261023 Z:\genai-lab outputs\packaging-gpu-20261023\cases-p5.json
   ```

   - 결과 `D:\genai-test\gpu-p5-20261023\flow\`(줄별 `runs\<run id>\`, `A3-saved.png`·`B2-saved.png`, `summary.json`), `setup\report-first.json`, `run-report.json`.
- 출력 폴더는 새 폴더여야 한다(있으면 실행기가 멈춤). 결과는 `D:\genai-test` 아래에 두고, 보고서와 작은 요약만 `outputs/packaging-gpu-20261023/`에 복사한다.

## 중단·기록 규칙

- `memory_stopped` 또는 PyTorch reserved > 6.5GiB → 그 줄 실패로 기록하고 다음 줄. **상한을 완화하지 않는다.**
- seed·장면·설정을 바꿔 다시 돌리지 않는다. 실패한 줄을 반복하지 않는다.
- **배포 후보 파일을 고쳐야 할 것 같으면 고치지 말고 멈춘다.** 보고에 바꿔야 할 파일·내용과, 그 변경으로 다시 해야 할 검증(P1~P5 중 어느 것)을 적는다.
- 모델 SHA 검사 정책을 바꾸지 않는다(작업 프로세스마다 약 200초 해시 — 그대로).
- 시험용 모델 폴더·기존 설치·공유 캐시를 지우거나 옮기지 않는다. 사용자 허깅페이스 토큰 파일을 읽지 않는다(실행기는 가짜 사용자 폴더를 쓴다).

## 보고

- 시작 HEAD·종료 HEAD, 후보 대조 결과(`verify-before`·`verify-after`), `launching_shell_gpu_vars`.
- P4: 5건별 상태·시간·`all_equal`·다른 픽셀 수·픽셀 수 3종(개발/후보)·최고 reserved·분할 Python 경로.
- P5: 준비 검사 결과·시간, 줄별 상태·전체 시간·Qwen 작업 프로세스 시간·시도 seed·재시도 이유·선택 시도·최고 reserved, CPU 준비물 SHA 비교, raw SHA 비교(다르면 픽셀 차이), 작업 프로세스 Python 경로.
- 차이마다 위 '차이 구분' 표의 분류 하나. '원인 미확정'이면 대조 실행 제안(돌리지 않음).
- 그림 품질 판정은 하지 않는다(새 품질 시험이 아님). 저장 그림 경로만 적는다.

## 출처 (내장형 Python 검증 근거, 2026-10-23)

- 파일: https://www.python.org/ftp/python/3.10.6/python-3.10.6-embed-amd64.zip (8,585,728 B), 서명 `.asc`, 서명 키 https://keybase.io/stevedower/pgp_keys.asc
- 공개 MD5 `37303f03e19563fa87722d9df11d0fa0`(python.org 3.10.6 릴리스 페이지) — 일치.
- GPG: 임시 키 저장소에서 `GOODSIG`·`VALIDSIG 7ED10B6531D7C8E1BC296021FC624643487034E5`, 서명일 2022-08-01, 서명자 "Steve Dower (Python Release Signing)". 키 신뢰도 표시는 `[unknown]`(새로 들여온 키에 신뢰를 부여하지 않았기 때문 — 지문을 python.org 공개값과 대조해 확인). 키에 함께 있는 두 번째 지문 `8E4CA3708D521D8056B1FD9DE314D10907F87583`은 암호화 전용 하위 키(용도 e)이고, 서명은 주 키로 만들어졌다.
- **3.10.6은 지금 개발 환경과 맞추기 위한 패키징 검증용 선택이다.** 최종 배포용 Python 버전의 보안 수정·지원 기간 적합성은 별도 확인 항목으로 남긴다(INSTALLER_PACKAGING_PREP 4-4-2).
- Authenticode: python.exe·pythonw.exe·python310.dll·python3.dll = Python Software Foundation(유효), vcruntime140.dll = Microsoft(유효). 제품 버전 3.10.6. python310.dll은 이 PC의 시스템 설치본과 해시 같음.
