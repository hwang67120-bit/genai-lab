# 저장장치 손실 대응과 복구 기록 — 2026-10-16

로컬 보존 완료 — C 사본의 SHA 검증을 마쳤으며, 사용자는 복구 문서 한 파일만 원격에 푸시하도록 승인했습니다. D드라이브에서 잃은 실행 환경은 별도로 재설치해야 합니다.

## 사고와 확인된 사실

- 사용자 보고: D드라이브는 물리 고장으로 제거했으며 수리 불가입니다. 파일시스템 복구나 포맷은 수행하지 않았습니다.
- 사용자 계획: Claude가 C드라이브에 임시 실행 환경을 설치하고, 새 드라이브가 도착하면 다시 설치합니다. C 설치 완료와 새 드라이브 문자는 아직 미확인입니다.
- Windows에서 D드라이브와 D:/genai-cache의 Python·모델 경로가 접근되지 않았습니다.
- Z드라이브의 실제 공유 대상은 \\192.168.0.109\win_g 입니다. Z:\genai-lab과 \\192.168.0.109\win_g\genai-lab은 같은 프로젝트입니다.
- Linux 경로 /home/mycom/win_g/genai-lab에서 Windows가 만든 임시 파일을 읽고 수정했으며, Windows의 Z/UNC 두 경로에서 수정 내용과 SHA 일치를 확인했습니다. 검사 파일은 삭제했습니다. 확인 시각: 2026-10-16 16:56 KST.
- 해당 Linux 경로의 findmnt 결과는 /dev/sda2, ext4, / 입니다. Z/UNC/Linux 경로는 독립된 세 백업이 아닙니다.
- 요청 25의 여섯 첫 raw는 D 제거 후에도 읽혔고 codex-evidence-25.json에 기록된 SHA와 같았습니다. 이 확인은 새 환경의 생성 성공이나 이미지 품질 승인이 아닙니다.

## 무엇이 어디에 보관되는가

| 대상 | 현재 위치 | 복구·검증 기준 |
|---|---|---|
| 코드·설정·기록 문서 | 공유 프로젝트와 GitHub 원격 | 커밋 SHA·원격 브랜치 일치 |
| 진행 중 미커밋 코드 | 현재 작업 폴더·로컬 WIP 복구 브랜치·C 임시 사본 | 원격 푸시 대상에서 제외. 미검증 작업 스냅샷 |
| 입력·승인 이미지·실행 원본·trace/판정 | inputs, approved-bases, outputs 등 | Git 제외 파일이 많으므로 별도 파일 사본과 SHA 목록 필요 |
| 모델·가상환경 | 기존 D:/genai-cache 손실, C 재설치 진행 | requirements 및 모델 revision/SHA 기준으로 재구성 |
| 표시용 비교 그림 | C:/Users/user/.codex/visualizations/2026/09/16/01a0a967-02b7-7f31-9719-aa82a8e8f790 | 일부 비교 자료 사본이며 전체 결과 백업과 구분 |

모델 파일과 가상환경은 Git에 들어 있지 않습니다. 원격 Git이 최신이어도 실행 환경과 모든 생성 결과가 백업됐다는 뜻은 아닙니다.

## 이번 보존 작업

- 기준 완료 커밋: 25753f8431f46b17942dc7c290bf131568d14dc9 (D-133).
- 시작 시 origin/codex/onepass-implementation과 현재 HEAD의 ahead/behind는 0/0이었습니다.
- 작업 브랜치와 실제 index를 바꾸지 않고 미커밋 작업을 로컬 복구 커밋으로 보존했습니다. 코드가 담긴 브랜치는 codex/recovery-20261016-170214-local-wip으로 구분하고 원격에 보내지 않습니다. 원격 codex/recovery-20261016-170214에는 이 복구 문서 한 파일만 기존 완료 커밋 위에 추가합니다. 다른 작업자의 진행 중 변경을 병합하거나 완료로 표시하지 않습니다.
- 보존할 미커밋 변경: docs/MEMORY_OPTIMIZATION_PLAN.md, genai_lab/free_generation.py, tests/test_free_generation.py.
- 아직 추적되지 않은 분석 도구: scripts/audit_garment_delivery.py, tests/test_garment_delivery_audit.py.
- 위 Python 파일 네 개의 AST 구문 검사 통과, 기존 diff --check 통과. 재설치 중인 앱 환경의 기능·CPU 회귀·GPU 시험은 실행하지 않았습니다.
- 사용자 입력 이미지·승인 이미지·개인 자료는 새로 Git에 추가하지 않습니다. reference-exclusions.json의 Git 공개 금지와 파생물 공개 금지를 유지합니다. 해당 자료를 포함할 수 있는 C 사본은 비공개 복구용입니다.
- C 임시 백업 루트: C:/Users/user/.codex/visualizations/2026/09/16/01a0a967-02b7-7f31-9719-aa82a8e8f790/recovery-20261016-170214
- files 아래 현재 파일을 삭제·미러링 없이 복사합니다. 제외: .git, Venv/venv/.venv, __pycache__, .pytest_cache, .msys* 임시 파일과 연결 경로. 모델·가상환경은 별도 재설치 대상입니다.
- .git 전체는 복사하지 않았습니다. 기존 완료 코드는 원격에 있고, 미커밋 코드 복구 커밋은 로컬 브랜치와 C의 pending-work.bundle에 보존합니다. 이번 원격 추가 대상은 복구 문서뿐입니다. source-manifest.json·verification.json·robocopy.log·git-backup.json으로 파일 사본·문서 푸시·로컬 WIP 보존을 구분합니다.
- 백업 상태: 72,811개 파일·14,333,600,798 bytes(약14.33GB) 복사 및 SHA 전체 대조 완료. 불일치0개, 재복사0개, 원본 목록 작성 중 변경0개. 검증 완료: 2026-10-16 17:15:22 KST. 복사 로그 실패0개. SHA 검증173.95초. 복구 문서와 백업 관리 파일은 별도 보존합니다.
- 사용자 승인 범위: 2026-10-16 응답 “복구 문서만 푸시 승인”. docs/STORAGE_RECOVERY_20261016.md 한 파일만 https://github.com/hwang67120-bit/genai-lab 의 codex/recovery-20261016-170214에 푸시합니다. 미커밋 코드·MEMORY_OPTIMIZATION_PLAN.md·분석 도구·테스트·입력/생성 이미지는 이번 원격 커밋에 넣지 않습니다. 초기 6파일 푸시는 자동 승인 검토에 차단됐으며, 이번에는 사용자 선택에 맞춰 문서 전용 커밋을 별도로 만듭니다. 실제 원격 SHA 확인은 git-backup.json에 남깁니다.

현재 파일 사본은 파일별로 검증하는 시점의 사본입니다. 다른 작업자가 동시에 수정할 수 있으므로 원자적 파일시스템 스냅샷으로 주장하지 않습니다. 검증 이후의 변경은 다음 커밋/백업에서 다시 보존해야 합니다.

## 실행 환경 복구 순서

1. 기존 프로젝트는 Z/UNC/Linux 경로 그대로 사용합니다. C 설치 때문에 프로젝트를 옮기지 않습니다.
2. Claude가 설치한 실제 C Python·Qwen Python·모델·GGUF 경로와 Python/라이브러리 버전을 먼저 확인합니다. C:/genai-cache라고 추측해 경로를 기록하지 않습니다.
3. configs/qwen_edit_settings.json은 사고 시점에 D 경로를 가리킵니다. 새 경로가 확정되면 현재 실행 설정만 변경하고, 과거 잠금 파일·request/run/trace·승인 snapshot은 수정하지 않습니다.
4. configs/qwen_model_files.json의 파일 SHA와 qwen_edit_settings.json의 model_revision, gguf_revision, gguf_sha256을 대조합니다. 파일이 존재한다는 것만으로 모델 복구 완료로 보지 않습니다.
5. 기본 환경은 requirements.txt, Qwen 환경은 README 및 Qwen 설치 문서와 기존 실행 증거를 참고합니다. requirements 범위와 과거 실제 설치 버전은 다를 수 있으므로 새 환경의 실제 버전을 별도 기록합니다.
6. 앱 import/CPU 검증을 먼저 수행합니다. 새 GPU 검증은 별도 사용자 승인 범위에서 진행합니다. 이전 GPU 결과를 새 환경에서 재현했다고 쓰지 않습니다.
7. 새 드라이브가 오면 실제 문자·용량을 확인하고 같은 방식으로 구성·SHA·CPU 검증을 다시 수행합니다. 작업 코드 변경과 경로 변경을 섞지 않습니다.

새 환경의 모델 manifest 확인 기준:
- model_revision: 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9
- gguf_revision: 0d33d9692b4b26212297240d87b0d4719aa4fd06
- GGUF: qwen-image-edit-2511-Q4_K_M.gguf
- gguf_sha256: 8677bac90627adbbc11efab87b1870e701c4eb3689ee865a3de8ab81b705a723
- artifact manifest SHA: 73561d21617e1dfb136ccd0d81868217ca7d593691be35deada96e9d3477f66e

## 이후 기록과 보존 기준

작업별로 요청·승인 범위, 기준 커밋, 바꾼 조건/고정 조건, 실제 테스트·실행 결과, 입력/출력 SHA, 자동 관찰·Codex 판정·사용자 판정, 실패 이유, 로그 경로를 분리해 남깁니다. 실패·재시도·보류 기록을 덮어쓰지 않습니다.

작업 종료 때 git status를 확인하고 의미 있는 단위로 커밋·일반 push 후 원격 SHA를 확인합니다. 진행 중 작업은 별도 복구 브랜치의 미검증 스냅샷으로 보존합니다. 강제 push와 기존 작업의 reset/checkout은 사용하지 않습니다.

Git에 없는 inputs/approved-bases/outputs/logs는 별도 저장장치에 추가 복사하고 파일 목록·SHA·검증 결과를 함께 보관합니다. 이 C 사본은 지금 공유 원본과 별도 사본이지만 C 고장·PC 전체 손실까지 대비한 최종 백업은 아닙니다. 새 장치 도착 후 독립된 보관 위치에 한 부 더 복사하는 것이 남은 사용자 결정입니다.

다음 조치: 승인된 문서 전용 푸시의 원격 SHA를 확인하고 기록합니다. 새 환경 설치가 끝나면 실제 경로·버전과 CPU 검증 결과를 이어 기록합니다.


## 로컬 복구 파일

C 백업의 pending-work.bundle은 기준 완료 커밋25753f8431f46b17942dc7c290bf131568d14dc9 이후 복구 커밋을 담습니다. 새 clone에서 bundle의 codex/recovery-20261016-170214-local-wip 브랜치를 fetch하면 미커밋 작업 스냅샷을 복원할 수 있습니다. 이 WIP 브랜치는 로컬 전용이며 원격 문서 브랜치와 구분합니다. GitHub를 사용할 수 없어도 files에 현재 코드 사본이 남지만 Git 이력 전체를 오프라인 보관한 것은 아닙니다. 자세한 복원 순서는 C 백업의 README.md에 있습니다.
