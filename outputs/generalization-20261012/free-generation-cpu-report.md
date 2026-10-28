# 자유 생성 구현·CPU 검증 기록 (2026-10-14)

완료: 공통 함수와 작업실 연결 및 관련 CPU 회귀 검사 233건이 통과했다. GPU 생성과 결과 품질 측정은 하지 않았다.

## 실행 환경과 기존 변경 확인

- 시작 OS: Windows NT 10.0.26300.0, 프로젝트 `\\192.168.0.109/win_g/genai-lab`.
- 코드 작성·수정: SSH `mycom-linux`의 Linux `/home/mycom/win_g/genai-lab`에서만 수행. 테스트는 Windows 기존 `D:/genai-cache/venv/Scripts/python.exe` 사용. 설치·다운로드 없음.
- 시작 HEAD와 해당 원격 브랜치: 둘 다 `5999b8e5450120450ac79811f29e378e17d1d0ea`. 시작 시 이후 커밋·푸시 없음.
- 시작 미커밋 중 사용자 지정 작성분 `free_generation.py`, `qwen_free_generation.py`, `qwen_pose_worker.py`만 내용을 검토해 이번 기능으로 이어서 작성했다.
- 다른 작업의 `docs/MEMORY_OPTIMIZATION_PLAN.md`, `approved-bases/`, `character.review.json`, `inputs/training_candidates/`, `scripts/audit_garment_delivery.py`, `tests/test_garment_delivery_audit.py`, `tests/.msys...`는 수정·스테이징·커밋 대상에서 제외했다.

## 연결된 흐름과 품질 경계

`run_free_generation`은 기준 원본/사용자 승인 의상 편집본 + 확정 특징표 + 연출 요청을 받아 입력을 검증하고 같은 바이트를 보존한다. 입력 사실과 출처, 조건 선택·제외 이유, 실제 모델 이미지·문장·SHA, 생성 결과 자체 분할과 특징별 관찰, 대상 시도·이미지 SHA·판정 주체를 trace로 연결한다. Qwen 실패·취소·메모리 중단은 기록 후 예외이며 원본을 성공 결과로 반환하지 않는다.

작업실의 자유 생성 버튼은 별도 옷 참조 없이 현재 캐릭터를 기준으로 시작한다. 저장된 승인 편집본을 캐릭터 입력으로 선택하면 `.review.json`/SHA/사용자 확인을 검증한다. 모르는 특징·개수·가림은 확인 불가다. 자동 특징 통과로 동일성이나 품질을 통과 처리하지 않는다. 의상 세부·동일성·요청 반영·심한 이상/노출을 사람이 네 항목으로 승인해야 저장할 수 있다. PNG/검토 기록/사람 판정은 같은 이미지 SHA에 연결한다.

D-123 다섯 결정을 유지한다. 같은 기준에서 파생한 머리 크롭을 추가 시각 참조로 전달하며 좌표·픽셀·SHA를 확인한다. 다른 참조나 머리 픽셀 합성은 없고 동일성을 보장하지 않는다. 출력 마스크를 새로 분석하며 원본 마스크를 적용하지 않는다. 연결 요소 수는 관찰값이고 실제 귀·꼬리·날개 개수/연결은 확인 불가다.

무작위 연출 seed는 실행마다 한 번 고르고 저장하며 호출자가 지정하면 재현된다. 생성 seed 42와 재시도 43에서도 연출·참조·프롬프트가 같게 유지된다. 빈 요청의 기본 자세만 전신이고 명시적 구도 요청을 전신 강제로 덮어쓰지 않는다. 문장 패턴은 보조 검사이며 눈 내리는 거리/눈을 감은 표정을 허용한다. 일반 의미 이해나 안전 보장으로 표현하지 않는다.

## CPU 검증

검사 시 `CUDA_VISIBLE_DEVICES`를 빈 값으로 설정하고 Qt offscreen을 사용했다. 새 GPU 생성·분할·Qwen 모델 로드는 모의 처리했다.

대상 12개 파일:

- `tests/test_free_generation.py`, `tests/test_studio_free_generation.py`
- `tests/test_qwen_outfit_edit.py`, `tests/test_outfit_edit_pipeline.py`
- `tests/test_character_features.py`, `tests/test_character_verification.py`
- `tests/test_studio_outfit_edit.py`, `tests/test_studio_ui.py`
- `tests/test_qwen_pose.py`, `tests/test_qwen_tail.py`, `tests/test_qwen_record_io.py`, `tests/test_studio_gui_fixes.py`

최종 결과: **233 passed in 79.43s**. 실행 옵션은 `-B -m pytest 위_12개_파일 -q -p no:cacheprovider`. Linux `git diff --check`도 통과했다.

초기 자유 생성 계약 검사에서는 테스트 JSON 읽기의 Windows 기본 cp949 때문에 3건이 실패했고 UTF-8 명시로 수정했다. GUI 수집은 샌드박스 UNC의 `WinError 65`로 막혀 해당 CPU 검사 프로세스만 종료하고 GPU 비활성 상태로 일반 실행 환경에서 다시 확인했다. 중간 86건·231건 통과 후 연출 선택 재현과 요청 구도 우선 검사를 추가해 최종 233건을 통과했다.

검증 범위: 입력/승인/SHA 계약, 실제 조건 기록, 다른 이미지 크롭 주입 거부, 결과 자체 마스크 사용, 실패/취소/검토 대기/안전 관찰 차단, 동일 연출 재시도, 원본 구분, GUI 시작/취소/사람 승인/저장, 공유 Qwen 요청/메모리 정책/기존 편집·자세·꼬리 경로 회귀.

## 미검증과 다음 결정

GPU 자유 생성의 실제 동일성·의상 세부·부위 연결·요청 반영·메모리/시간은 미검증이다. 기존 의상 편집의 미해결 품질을 해결했다고 간주하지 않는다. HTTP 서버·RAG·학습은 이번 범위에 없다.

요청 20 초안에 목적·고정 조건·여섯 비교·사람 판정 기준·약 3~4시간(전부 재시도 시 약 6~8시간)·Qwen reserved 6.5GiB 정책·결과별 다음 결정·종료 기준을 적었다. 이는 과거 실행 근거의 예산 추정이며 자유 생성 실측이 아니다.

다음 사용자 결정: 요청 20에 사용할 사용자 승인 편집본 세 장과 확정 특징표를 지정하고 GPU 시험 실행 여부를 승인한다.
