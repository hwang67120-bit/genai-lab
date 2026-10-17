# Codex 인계 — 2026-10-17 새 D: 드라이브로 옮기기 (분담)

사용자: "이제 D드라이브 생겼어 작업하는 프로젝트는 여기다(D:\Code) 옮기고 모델은 모델폴더(D:\Model)에 옮겨 … 잠시 분담해서 하자".

## 바뀐 환경

- 새 D: = WD Purple 1TB HDD(WDC WD10PURX, NTFS, 비어 있음), 폴더 `D:\Code`, `D:\Model`. 가상 D:(subst)는 더 이상 쓰지 않는다 — **subst 걸지 말 것**(진짜 D:와 충돌).
- 임시 설치본 `C:\genai-temp-d\genai-cache`(37GB)는 그대로 둔다(옮기기 확인 뒤 지울지 사용자 결정).
- 메모리 의심(10/17): 요청 26이 12/21에서 'GGUF SHA 불일치'로 멈춤. 캐시 거친 읽기 SHA는 틀리고(97e1ee05, 반복 같음) 디스크 직접 읽기(FILE_FLAG_NO_BUFFERING)는 정상(8677bac) → RAM 페이지 캐시 손상 의심. 사용자에게 재부팅·`mdsched.exe` 권고함. 작업 프로세스의 모델 SHA 검사가 막아 주므로 그대로 유지.

## Claude 몫 — 모델·실행 환경

1. `C:\genai-temp-d\genai-cache` → `D:\Model\genai-cache` 복사(robocopy, 진행 중).
2. 정션 `D:\genai-cache` → `D:\Model\genai-cache`. 코드 기본값·configs·venv(.pth, pip 실행 파일)가 모두 `D:/genai-cache`를 가리키므로 **코드·설정 수정 없이** 그대로 동작. `D:/Model` 경로로 설정을 직접 바꾸는 일은 나중에 별도.
3. 모델 SHA 대조(디스크 직접 읽기 + 일반 읽기 둘 다 — 메모리 이상도 함께 확인), CPU 전체 시험.
4. 끝나면 이 문서 아래 '진행 기록'에 결과를 적는다.

## Codex 몫 — 프로젝트 옮기기와 요청 26

1. `\\192.168.0.109\win_g\genai-lab` → `D:\Code\genai-lab` 복사. **추적 안 되는 파일·무시된 파일 전부 포함**(outputs/, approved-bases/, inputs/training_candidates/, character.review.json, docs의 추적 안 된 문서 등). 원본(Z:)은 지우지 않는다 — 사용자 결정. 크기(Claude 측정 10/17): 16.8GB, 97,329개(outputs 14.0GB·87,151개, .git 2.6GB, Venv 0.08GB). 네트워크 목록 읽기만 8분 걸림 — robocopy `/MT` 권장.
2. 확인: 파일 수·크기 비교, `git status`가 같음, `git fsck`. 브랜치 `codex/onepass-implementation`, HEAD `f592b4f0` 이후.
3. **요청 26 이어서 실행 시점**: 권고는 **Claude의 모델 SHA 확인이 끝난 뒤, 지금 위치(UNC) 그대로 남은 9줄(M3-S1부터)을 이어서 돌리고, 프로젝트 전환은 요청 26이 끝난 뒤**. 이유: 입력 경로(`eval-set-26/proposed-inputs.json`의 UNC)와 잠금 스냅샷이 지금 위치 기준이라, 중간에 위치를 바꾸면 계약을 다시 잠가야 한다. 새 위치에서 이어서 돌리려면 경로 대응과 재잠금을 먼저 사용자에게 보고.
4. 모델 경로는 정션 덕분에 바뀌지 않으므로 요청 26 잠금(모델 SHA·설정)은 그대로 맞아야 한다. 다르면 즉시 보고, 자동 재실행 금지.
5. 남은 하드코딩 경로 참고: `studio_generation.py`의 `proportion_sketch_root = G:/genai-cache/...`(G: 없음, 자유 생성에는 안 쓰임). 경로 정리는 별도 작업으로 제안만.

## 순서

1. 사용자: 재부팅(메모리 캐시 비우기)·가능하면 메모리 진단.
2. Claude: 모델 복사 → 정션 → SHA·CPU 확인 → 기록.
3. Codex: 프로젝트 복사·확인(2와 동시에 해도 됨, 다른 디스크 경로) → 요청 26 남은 9줄 → 끝나면 작업 위치를 `D:\Code\genai-lab`로 전환.

## 진행 기록

- (Claude) 모델 복사 시작 2026-10-17, robocopy → `D:\Model\genai-cache`.
- (Claude) 복사 완료 13:18: 파일 73,824개·37.009GB, 실패 0(7분). 정션 `D:\genai-cache` → `D:\Model\genai-cache` 생성, venv·.pth 경로 그대로 동작.
- (Claude) 모델 SHA 33개(`drive-recovery-20261016/verify-new-d-20261017.json`): 일반 읽기 33/33 일치. 디스크 직접 읽기 첫 회는 큰 파일 3개(text_encoder 3·4, GGUF)가 틀렸으나 **다시 직접 읽으면 셋 다 정상**(GGUF는 D:에서 2회·C:에서 1회 모두 8677bac). 같은 파일·같은 방법이 때에 따라 다른 값 → 디스크 파일은 정상, **읽는 순간 메모리에서 데이터가 깨지는 하드웨어 이상**(RAM 의심이 더 강해짐). 작업 프로세스는 시작마다 모델 SHA를 검사하지만, 불러온 뒤 메모리 안에서 깨지는 것은 못 잡는다.
- (Claude) CPU 전체 시험 2,400 passed(새 D: 정션 환경, `f592b4f0`).

### Codex 프로젝트 복사 기록 — 2026-10-17

- UNC 원본을 D:/Code/genai-lab로 전체 복사했다(.git, 무시된 outputs, 입력, 미추적 파일 포함). 원본 Z:는 삭제하지 않았으며 subst도 사용하지 않았다. 작업 기준은 아직 UNC다.
- 검증 자료와 최종 결과: C:/Users/user/.codex/visualizations/2026/09/16/01a0a967-02b7-7f31-9719-aa82a8e8f790/project-move-20261017/ 아래 launch.json, source-final-inventory.json, destination-audit*.json, final-verification.json, report.md를 확인한다. 최종 완료 판정은 final-verification.json의 실제 결과를 따른다.
- 첫 전체 SHA 대조에서 과거 결과 outputs/generalization-text-20260926/cases/muscular-female/partial/on/raw-inpaint.png 사본 1개의 SHA가 원본과 달랐다. 틀린 사본은 위 증거 폴더의 mismatched-copy/에 보존했다. 원본 SHA를 확인하고 D:의 해당 사본만 재복사했으며, 이후 3회 모두 원본 SHA와 일치했다(single-file-recopy.json). 원본은 수정하지 않았다. 이 조치 뒤 전체 SHA를 한 번 더 대조한다.
- 원본 Linux Git fsck와 D: Git fsck 모두 exit 0. UNC에서 Windows Git이 경고를 내는 현상은 별도이며, D: Git 접근은 실행 도구의 권한 문제를 해소한 읽기 전용 검사로 확인한다. Git 상태와 전체 파일 검증 최종값은 위 JSON에 보존한다.
- 사용자는 재부팅만 완료했고 메모리 진단은 아직 하지 않았다고 확인했다. 새 모델의 일반 읽기/직접 읽기 결과가 처음에는 달랐고 재검에서는 맞았다는 기록이 있으므로 원인 해소를 확정하지 않는다. 이번 프로젝트 사본에서도 SHA 불일치가 관측됐으므로 GPU는 재개하지 않았다. 운영 코드·설정·커밋·푸시는 변경하지 않았다.
- 요청26은 12행 완료, M3-S1 중단(첫 seed42 raw 있음), 미시작 8행이다. 기존 실행기는 중단 폴더를 보존하고 미시작 행만 생성하므로 인계의 '남은 9행'을 새 GPU 생성 9회로 해석하지 않는다. 계약 중단 기록을 자동으로 지우거나 재생성하지 않았다.
- 요청26을 기존 UNC 위치에서 재개한 뒤에는 새 출력이 생기므로, 작업 위치 전환 직전에 D:로 최종 증분 복사·대조가 다시 필요하다.
- (사용자 결정 16:3x) 램 진단: 두 개 함께 14:37 오류, 채널 A `B538380E` 단독 16:15 오류. 그 뒤 두 개 다시 꽂음(16:27 재부팅, 32GB 인식), Claude 간이 검사(램 무늬 12GB·모델 직접 읽기 3회) 깨끗. 사용자: **"이대로 작업 이어 나아가자"** — 추가 진단 없이 진행. 요청 26 남은 9줄 재개 가능. 모델 SHA 불일치·작업 프로세스 비정상 종료·이상한 결과가 다시 나오면 램을 먼저 의심하고 즉시 보고(자동 재실행 금지). 의심 램은 `B538380E`(채널 A).
