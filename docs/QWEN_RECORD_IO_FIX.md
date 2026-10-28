# Qwen 진행 기록의 Windows 교체 충돌 수정

2026-10-03. 기준 커밋 abeb9cdb. 생성 설정·모델·버전·지시문 변경 없음. GPU 실행 없음.

## 원인과 선택

부모가 run.json을 읽는 순간 worker가 같은 경로를 replace하면 Windows의 삭제 공유가 없는 읽기 핸들이 교체를 막을 수 있다. 접근 거부는 이 원인 외에도 생길 수 있으므로 모든 WinError 5를 이 문제로 단정하지 않는다. 이번에는 실제 핸들 잠금으로 같은 오류를 재현했다.

근거: [Microsoft CreateFileW / FILE_SHARE_DELETE](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)는 이름 변경에도 삭제 접근이 필요하며 공유 플래그는 핸들을 닫을 때까지 적용된다고 설명한다. [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)의 교체 성공·오류 계약에 맞춰, 같은 디렉터리에 완전한 임시 파일을 닫아 둔 뒤 발행한다.

선택은 **진행 파일 분리 + 제한된 PermissionError 재시도**다. 폴링 간격은 0.2초를 유지한다. 한 파일을 더 느리게 읽는 방식으로 우연한 성공을 기대하지 않는다.

## 구현

- qwen_record_io.write_json: 고유 임시 이름에 직렬화·flush·fsync·close 후 replace. PermissionError만 20/40/80/160/200ms 이후250ms 간격으로 재시도. 총 대기 최대2초, 최대12회 교체 시도, 무한 재시도 없음. SMB 호출 자체의 OS 대기 시간을 강제로 끊는 타임아웃은 아니다.
- 영구 접근 거부는 JSONReplaceError로 노출하고 완전한 `.tmp` 경로를 남긴다. 기존 목적 파일은 덮어쓰기 실패 시 그대로다. 다른 I/O 오류는 재시도하지 않는다.
- worker 진행 상태는 progress/의 순번 파일로 발행한다. 단일 작성자이며 발행한 파일은 다시 쓰지 않는다. 부모는 이 파일만 읽고 부분·읽기 오류를 견딘다.
- 종료 상태 completed/failed/cancelled는 run.final.json에 먼저 보존한 뒤 run.json에 반영한다. 부모는 프로세스 종료 후에만 최종 run.json을 읽는다.
- worker 강제 종료는 finally가 실행되지 않으므로 부모가 종료 대기 후 마지막 진행 자료를 바탕으로 run.json/run.final.json을 남긴다. 미수집 측정은 metrics_incomplete로 구분한다. 저장된 raw.png는 삭제·재생성하지 않고 해시를 남긴다.
- 최종 run.json 교체가 잠금 상한을 넘으면 실행은 오류다. 부모가 잠금 해제 후 run.final.json을 canonical 위치로 복구해도 해당 실행 오류를 조용히 성공으로 바꾸지 않는다.
- 원시 결과가 메모리에만 있는 순간 강제 종료·디스크 고장이 발생했을 때까지 저장을 보장하는 기능은 아니다. 이미 저장된 raw와 완전한 종료 사본을 보존하며, 저장 불가능은 오류로 남긴다.

## 다른 경로 점검

| 파일 | 쓰는 쪽 / 읽는 시점 | 처리 |
|---|---|---|
| progress/*.json | worker / 실행 중 부모 | 이름 재사용 없음, 부모는 완전 발행본만 읽음 |
| run.json | worker 시작·종료, 부모는 종료 후 복구 가능 | 실행 중 부모 폴링 제거, 제한 재시도 |
| run.final.json | worker 종료, 없을 때만 종료 후 부모 | 최종 사본, 같은 실행에서 덮어쓰기 금지 |
| launcher.json | 부모만 작성 | 공통 제한 재시도, 최종 저장은 finally |
| request.json | 부모가 저장 완료한 뒤 worker 시작 | 실행 중 변경 없음 |
| cancel.request | 부모가 생성, worker는 존재만 확인 | 교체 없음; 프로세스 종료 후 부모가 최종 상태 기록 |
| analysis/draft/user_review/runtime/model-files/prepared-request | 기존 write_json 사용자 | 함수 서명·JSON 내용 유지, 저장 내구성·제한 재시도만 추가 |
| raw.png·model_inputs | worker 작성, 종료 후 읽기 | 생성·저장 연산 변경 없음 |

write_json은 qwen_pose_edit에서 재노출해 기존 import 호출부를 유지했다. 신규 모듈은 모델·GPU를 import하지 않는다. 실패 폴더에 재실행하는 guard에 run.final.json과 progress도 포함했다.

## CPU 확인 및 인수인계

- fault injection 단위 테스트는 OS에 무관하게 처음3회 PermissionError 후 성공, 상한 초과, 다른 I/O 오류 즉시 전파, tmp·최종 사본·raw 보존을 확인한다.
- 가짜 worker로 완료·실패·취소의 완전한 최종 기록과 부모의 progress 전용 폴링을 확인한다.
- Windows 실제 검사: scripts/qwen_record_io_check.py. outputs/qwen-record-fix-20261003/windows_check.py도 같은 코드로 연결된다. GPU·모델 없이 실행한다.
- Windows SMB 실측: 기존 replace WinError5 재현, 제한 재시도 약0.922초 후 성공, 긴 잠금의 오류·최종 사본 보존·복구 성공, 스냅숏40개 발행/40개 관찰 및 최종 기록 일치. 원시 보존 검사는 고정 바이트 표본으로 수행했다.
- 전체 테스트 최종 수치와 변경 SHA는 outputs/qwen-record-fix-20261003/results.md에 기록한다.
- GPU 재현은 QWEN_POSE_GPU_HANDOFF.md의 새 20261003b 폴더에서 CPU 확인→raccoon B→통과 시 ordinary-female B 순서. Codex는 GPU를 실행하지 않았다.
