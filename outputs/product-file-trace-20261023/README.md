# 제품 의존 파일 관찰 원자료 (2026-10-23)

설치형 패키징 준비(docs/INSTALLER_PACKAGING_PREP.md 2절, D-166·D-168)의 근거 자료. "왜 이 파일을 설치 목록에 넣었는가"를 다시 확인할 때 쓴다.

## 폴더

| 폴더 | 무엇 | GPU | 대신한 단계 | 도구 |
|---|---|---|---|---|
| `base4-first/` | 사용자 화면 네 단계(A 입력 분석·B 기준 분석·C 옷 바꾸기·D 자세 장면) 첫 관찰 | **켜져 있었음**(PowerShell 빈 값 = 변수 삭제). 머리 분할 4회 GPU, 각 최고 예약 5.08GiB | Qwen 생성 | `run_trace.py`(첫 판, 핸들 표본 없음) |
| `startup-first/` | 작업실 사용자 화면 시작(생성 없음) | 켜져 있었음(생성·분할 없음) | — | `run_startup.py` |
| `aborted-cuda-env-20261023/` | 두 번째 관찰 첫 시도 — GPU 숨김 검사에서 시작 전에 멈춤(실행 없음) | — | — | `scripts/product_file_trace/scenarios.py` |
| `outfit_retry/` · `hair_restore/` · `hair_auto/` · `memory_link/` · `product_buttons/` · `base4/` | 조건부 경로 관찰 | **숨김**(`CUDA_VISIBLE_DEVICES=-1`, `torch.cuda.is_available()=False` 확인) | Qwen 생성, 머리 분할(CUDA 필요) | `scripts/product_file_trace/scenarios.py` + 핸들 표본 |
| `env-probe/` | 실행 환경 점검(D-169): 환경마다 기반 Python·경로·.pth·모듈 불러오기와 출처(격리 실행 / 실제 실행 조건) | 숨김 | — | `env_probe.py` |
| `env-base4/` · `env-startup/` | 사용자 화면 단계·시작 화면을 다시 돌리며 프로세스별 Python·모듈 위치(`modules-<pid>.json`)·하위 프로세스 명령(`opens.jsonl`의 popen) 기록, 요약 `env-report.json` | 숨김 | Qwen 생성, 머리 분할 | `scenarios.py` + `GENAI_OPEN_TRACE_MODULES=1`, `env_report.py` |
| `env-base4-explicit/` | 하위 프로세스 명시적 환경 변수(D-170) 검증: 부모에 시험용 변수(VIRTUAL_ENV·CONDA_PREFIX·PYTHONSTARTUP·임의 토큰)를 넣고 네 단계 재실행, 하위 프로세스 `env_keys`(이름만) 기록 | 숨김 | Qwen 생성, 머리 분할 | `scenarios.py` + `GENAI_CHILD_ENV_EXTRA` |
| `worker-imports/` | Qwen·머리 분할 작업 프로세스의 함수 안 import까지 불러온 모듈(모델 안 올림) | 숨김 | — | `worker_imports.py` |
| `packages/` | 관찰 모듈 → 배포 패키지·버전·라이선스 표시(`main.json`·`pose.json`) → `packaging/requirements-*.txt` | — | — | `package_list.py` |
| `model-home/` | 모델 폴더 배치 확인(D-171): 빈 폴더·분석 모델만·전체(시험용 복사본 `D:\genai-test\model-home-20261023`) 점검(`check.json`, Qwen SHA 포함)과 제품 단계 장면 4개 재관찰, 요약 `summary-*.json` | 숨김 | Qwen 생성, 머리 분할 | `model_home_check.py`·`scenarios.py` |
| `packages/evidence-main.json` | opencv·PySide6·setuptools 사용 근거(불러온 파일이 어느 배포 기록과 해시가 같은지) | — | — | `package_evidence.py` |
| `d172/` | HF 토큰 차단·필수 모델 공통 검사(D-172) 확인: 빈 폴더·WD 태거 없는 폴더(문구 수정 전·후)·전체 폴더(준비 화면 수준 SHA) 점검, 전체 폴더 제품 단계. 사용자 토큰 파일 열기 0회 | 숨김 | Qwen 생성, 머리 분할 | `model_home_check.py`·`scenarios.py` |
| `setup-screen/` | 최초 모델 준비 화면(D-173) CPU 검증: 실제 검사 프로세스로 전체 폴더 깊은 검사·40초 뒤 취소·WD 없는 폴더, 화면 멈춤 여부(타이머), 시작 판단. 기록은 시험용 파일 | 숨김 | — | `setup_screen_check.py` |
| `p2p3/` | 배포 후보 격리 검증(D-174 P1~P3): `run2/`(release·trace 4단계, 판정 `results/trace-verdict.json`, 실행기 `results/launcher/`), `attempt1-missing-run-py/`(첫 시도 — run.py 누락), `candidate-build-report.json`. 가짜 사용자 폴더의 기록 JSON 포함 | 숨김 | Qwen 생성, 머리 분할 | `packaging/run_isolated_check.py`·`analyze_isolated_trace.py`·`launcher_check.py` |

- 각 폴더: `opens.jsonl`(원자료: 열린 파일 기록), `status.json`(단계 결과·대신한 단계), `console.log`.
- `summary-first-gpu-on.json`, `summary-gpu-hidden.json`: `scripts/product_file_trace/summarize.py`로 만든 분류 요약.
- `product_buttons/product-controls.json`: 사용자 화면에 보이는 조작 목록.
- 저장소에 넣지 않은 것: 단계별 실행 결과 그림·중간 파일, **캐릭터 기억 사본**(`memory_link/character-memory-copy`, 사용자 데이터).

## 읽는 법

- 열린 파일 = 그 실행 경로에서 쓴 근거. 열리지 않은 파일 = 그 실행에서 쓰지 않았다는 뜻일 뿐, 제품 전체에서 불필요하다는 뜻이 아니다.
- 파이썬 open 감사와 몇몇 라이브러리 감싸기로 기록한다. 핸들 표본은 열려 있는 동안만 잡는다. 열었다 바로 닫은 C 확장 파일은 빠질 수 있다.
