# 모델 로딩 지연 원인 확인

실행: Claude, 2026-09-29. 생성 0회. 제품 소스 수정 없음.

## 결론: D: 드라이브(모델 저장소) 하드디스크 고장 징후
- D: = TOSHIBA HDWD110 1TB HDD (Disk 0). Windows 상태 `HealthStatus: Warning`, `OperationalStatus: Predictive Failure`.
- 시스템 로그 disk 이벤트 ID 7 ("장치에 잘못된 블록이 있습니다", \Device\Harddisk0\DR0): 2026-05-27 부터 매일 기록, 9월 들어 증가 (9/17 61회, 9/29 오늘 23회).
- D:\genai-cache 에 모델(Animagine, IP-Adapter, T2I-Adapter, 태거 등)과 실행 환경(venv, catvton-venv)이 있다.

## 측정
| 단계 | 처음(캐시 없음) | 두 번째(OS 캐시) |
|---|---|---|
| import torch / diffusers / 클래스 | 6.0 / 23.4 / 20.8초 | 3.3 / 2.9 / 6.5초 |
| T2I-Adapter 301 MiB | **119초 (2.5 MiB/s)** | 0.1초 |
| CLIP ViT-H 2411 MiB | **175초 (13.8 MiB/s)** | 0.8초 |
| Animagine 파이프라인 | 5.7초 (이미 캐시에 있었음) | 0.8초 |
| 합계 | **350.8초** | 15.3초 (로컬 스크립트 12.0초) |

- 같은 디스크에서 다른 파일(SD1.5 unet)은 순차 읽기 101 MiB/s, 또 다른 파일(2386 MiB)은 10.9 MiB/s — 파일(디스크 위치)에 따라 10배 차이. 불량 블록 재시도로 느려지는 전형적 모양.
- Z:(\\192.168.0.109\win_g 네트워크 공유)에서 스크립트를 실행하는 영향은 작다 (따뜻한 상태 15.3초 대 12.0초).
- 오전에 로딩이 빨랐던 것은 모델 파일이 OS 메모리 캐시에 있었기 때문. 캐시가 밀려나면 고장 난 디스크에서 다시 읽으며 10~20분 걸렸다.

## 권장 (사용자 결정 필요, 시험 범위 밖)
1. 급함: D:\genai-cache 를 다른 드라이브로 복사. G: (SAMSUNG NVMe 1TB, 여유 약 1019 GB, 상태 정상) 이 적합. 디스크가 더 나빠지기 전에.
   복사 후 outputs/environment-lock-20260928/lock.json 의 모델 파일 SHA-256 으로 무결성 확인.
2. 경로: configs/animagine.yaml 등 D:/genai-cache 경로 변경은 제품 설정 수정이라 10/4 이후 Codex 작업 또는 사용자가 직접.
3. 배포 참고: 모델은 SSD 에 두도록 설치 안내. 캐시가 따뜻하면 로딩 12~15초.

## 백업 복사 결과 (사용자 승인, 2026-09-29)
- robocopy D:\genai-cache → G:\genai-cache (원본은 읽기만). 1시간 38분, 67.494 GB.
- 파일 86,745 중 86,740 복사, **5개 실패** (오류 1392 "파일 또는 디렉터리가 손상되어 읽을 수 없음" — D: 파일 시스템 손상):
  venv 의 pip 캐시 파일 4개(`pip\_internal\commands\__pycache__\freeze/hash/help/index.cpython-310.pyc`)와 `pip-26.2.1.dist-info\licenses\...\idna\LICENSE.md`. 디렉터리 2개(pip 라이선스 하위) 생성 실패.
  모두 pip 내부 파일이라 모델·시험 결과와 무관(.pyc 는 자동 재생성, 라이선스 문서는 동작 무관).
- 검증: 환경 고정 기록(lock.json)의 모델 파일 37개 SHA-256 을 G: 사본에서 대조 → **37/37 일치**. 파일 수·용량 D 86,740 / 67.49 GiB = G 86,740 / 67.49 GiB (읽을 수 있는 파일 기준).
- 상태: G: 는 백업. 실행 경로(configs, venv)는 아직 D: 를 가리킨다. venv 는 D: 경로가 박혀 있어 G: 에서 바로 실행 안 될 수 있다 — G: 로 전환하려면 venv 재생성 + 설정 경로 변경 필요.

## G: 전환 (사용자 승인 "전환 먼저 하자", 2026-09-29)
- 방식: 코드·설정 수정 없이 정션. `D:\genai-cache` → `D:\genai-cache.old-failing` 로 이름 변경(삭제 안 함), `D:\genai-cache` 를 `G:\genai-cache` 로 가리키는 정션 생성.
  venv 기본 Python 은 C:\Program Files\Python310 이라 venv 내부 경로(D:\genai-cache\...)가 그대로 G: 로 연결된다. 저장소의 D:/genai-cache 경로 20개 파일 수정 불필요.
- 확인: venv python 실제 위치 G:\genai-cache\venv, pip 동작, catvton-venv(torch 2.5.1+cu121, easy_dwpose) import 정상.
- 로딩(캐시 없음 상태 포함): 합계 350.8초 → **27.3초** (어댑터 119→0.2초, CLIP 175→1.0초).
- 재현: strength-5pose 검증 이미지 1장(ordinary GUN seed 1, 세기 1.5)을 다시 생성 → **SHA 일치** (outputs/load-latency-repro-20260929).
- 남은 위험: 정션 자체는 D: 에 있다. D: 가 완전히 인식되지 않으면 경로가 끊기므로, 장기적으로 설정 경로를 G: 로 바꾸는 작업(10/4 이후 Codex)과 디스크 교체가 필요. D:\genai-cache.old-failing 은 이전 상태 보존용으로 남겨 둠.
