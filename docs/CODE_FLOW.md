# 코드 읽기: 작업실에서 이미지 저장까지

2026-10-04 현재 코드의 **실제 호출 관계**를 정리한다. 제품 계약을 새로 결정하는 문서는 아니다.
[FLOW.md](FLOW.md)는 제품 계약, 이 문서는 구현을 따라 읽는 지도다.
먼저 [studio_controller.py](../genai_lab/studio_controller.py)의 `StudioController.start`부터 읽는다.
주 흐름은 파일 앞쪽, 작업 실행·오류 처리는 그 뒤, 확인창 배치는 맨 뒤에 모았다.

## 1. 한 번에 읽는 주 흐름

```text
사용자가 캐릭터·의상을 고르고 '이미지 만들기'
  gui_main.GenAILabWindow.start_generation
    → StudioController.start
      실행 환경·로컬 모델 확인, 요청 폴더 만들기
      → analyze_inputs [작업 스레드 / CPU 하위 프로세스]
        원본 사본·해시 → 캐릭터 골격 → 얼굴 크롭·캐릭터/의상 태그
      → inputs_ready [GUI 스레드]
        confirm_inputs: 얼굴·의상 설명·성별을 사용자가 확인
        취소 → 생성하지 않고 입력 선택으로 돌아감
        승인 → build_request: 승인 정보로 프롬프트·4개 seed 요청 조립
      → generate_onepass_request [작업 스레드]
        자세 없이 캐릭터+의상 이미지 생성
      → generated [GUI 스레드]
        결과 4장을 검토 대기로 표시
      → select: 비교할 이미지 선택 (선택 변경 시 승인 해제)
      → approve: 캐릭터·의상·노출을 사용자 확인
      → save: 승인한 이미지의 해시 재확인 → PNG와 검토 기록 내보내기
```

이 화살표는 모두가 동기 호출이라는 뜻이 아니다.
`launch(action, complete)`가 `StudioTask`를 실행하고, 종료 신호가 `finished`에 도착하면
성공 시 `complete(result)`, 실패 시 `fail(error)`를 호출한다.
분석 완료 함수는 `inputs_ready`, 생성 완료 함수는 `generated`다.
화면·사용자 설정은 GUI 스레드에서 처리한다.

## 2. 단계별 입력·출력과 멈추는 곳

| 순서 | 읽을 함수 | 들어오는 정보 → 나가는 정보 | 취소·오류 시 |
|---|---|---|---|
| 1 | [StudioController.start](../genai_lab/studio_controller.py) | 선택 경로 → 요청 폴더·실행 설정 | 로컬 모델/환경 확인 실패 시 오류 표시 |
| 2 | [analyze_inputs](../genai_lab/studio_generation.py) | 캐릭터·의상 → 분석 사전(얼굴 SHA, 태그 분류, slim 등) | CPU 분석 오류·시간 초과·취소 시 생성하지 않음 |
| 3 | [inputs_ready / confirm_inputs](../genai_lab/studio_controller.py) | 분석 정보 → 사용자가 확인한 성별·의상 태그 | 미확인·취소 시 생성하지 않음 |
| 4 | [build_request](../genai_lab/studio_generation.py) | 승인 정보 → OnePassRequest | 얼굴 파일 변경·부적합 태그 등은 오류 |
| 5 | [generate_onepass_request](../genai_lab/onepass_generation.py) | 요청 → 생성 후보 묶음 | 실패·취소 시 부분 결과를 완성 후보로 제시하지 않음 |
| 6 | [generated / select](../genai_lab/studio_controller.py) | 묶음 → 현재 비교 후보 | 후보를 바꾸면 다시 승인해야 함 |
| 7 | [StudioResults.approve](../genai_lab/studio_generation.py) | 사용자 확인 → 승인 SHA·검토 기록 | 완료 기록·이미지 SHA 불일치 시 승인 불가 |
| 8 | [StudioResults.export](../genai_lab/studio_generation.py) | 승인 후보 → PNG·review.json | 미승인·이미지 변경·기존 파일 덮어쓰기 거부 |

생성 직후는 `awaiting_user_review`, 사용자 확인 후는 `user_approved`, 내보낸 뒤는 `saved`다.
자동 미적 품질 판정을 통과했다는 의미가 아니다.
`reject`는 승인을 해제하고, `discard`는 결과를 사용하지 않음으로 기록한다. 둘 다 자동 재생성을 하지 않는다.
실패·취소 시 원시 부분 산출물은 디스크에 남을 수 있지만 저장할 완성 결과로 표시하지 않는다.

## 3. 필요할 때만 여는 세부 파일

| 궁금한 것 | 파일·함수 |
|---|---|
| 화면 배치와 버튼 연결 | [studio_ui.py](../genai_lab/studio_ui.py) — build_studio |
| 입력 선택, 작업실 진입, 이전 경로 연결 | [gui_main.py](../gui_main.py) — GenAILabWindow |
| 전체 순서·승인·작업 상태 | [studio_controller.py](../genai_lab/studio_controller.py) — StudioController |
| 실행 경로 설정·분석·요청 조립·저장 | [studio_generation.py](../genai_lab/studio_generation.py) — StudioRuntime, analyze_inputs, build_request, StudioResults |
| CPU 분석 실체 | [studio_analysis.py](../genai_lab/studio_analysis.py) — pose, features |
| 얼굴 크롭·태그 분류에 쓰는 규칙 | [onepass_character.py](../genai_lab/onepass_character.py) |
| 사용자 성별 계약 | [onepass_gender.py](../genai_lab/onepass_gender.py) — prepare_onepass_gender |
| 프롬프트와 의상 어휘 | [onepass_prompt.py](../genai_lab/onepass_prompt.py), [onepass_garment_vocabulary.py](../genai_lab/onepass_garment_vocabulary.py) |
| 모델 호출·일정·실행 기록 | [onepass_generation.py](../genai_lab/onepass_generation.py) — prepare_onepass_inputs, generate_onepass_request |
| 기본 생성·프롬프트 설정 | [onepass_generation_settings.py](../genai_lab/onepass_generation_settings.py), [onepass_prompt_settings.py](../genai_lab/onepass_prompt_settings.py) |

캐릭터 입력에서 골격을 분석하는 것은 얼굴 크롭·체형 분석에 쓰기 위한 것이다.
현재 작업실의 생성 요청은 `proceed_without_pose`, `pose_tags=()`다.
캐릭터 분석에서 골격이 나왔다고 생성에 자세 제어가 적용되는 것은 아니다.

## 4. 별도 경로와 이전 경로를 구분하기

- 현재 '이미지 만들기': `start_generation → self.studio.start()`.
- 저장 이미지 자세 편집: `open_qwen_pose_editor → QwenPoseDialog`.
  현재 코드는 런타임 설정, 완성 이미지, 골격 PNG를 직접 선택한다.
  새 이미지 생성의 자동 후속 단계가 아니다. 이번 정리에서 연결·동작을 바꾸지 않았다.
- 이전 생성: `start_legacy_generation` 및 기존 Base/정밀화 흐름.
  코드가 남아 있지만 현재 만들기 버튼의 기본 경로가 아니다.
- 기존 `GenerationOrchestrator` / 제품 CLI는 작업실 경로와 구분해서 읽어야 한다.
  같은 생성 서비스를 사용한다고 README만 보고 가정하지 않는다.
- `outputs/`의 시험 코드와 결과는 운영 구현 자체가 아니다.
  단, 현행 `StudioRuntime.head_cache` 기본값은 outputs 아래 기존 모델 캐시를 가리킨다.
  이 배포 의존성은 이번 리팩터링에서 바꾸지 않았다.

## 5. 파일을 옮겨도 동작이 바뀌지 않았는지 확인

핵심 회귀는 [test_studio_generation.py](../tests/test_studio_generation.py)의
`test_create_button_to_review_and_export_without_legacy_fallback`이다.
실제 Qt 버튼 신호로 입력 확인 → 가짜 생성 백엔드 → 후보 변경 → 사용자 승인 → 파일 저장을 확인한다.
실패·취소, 이미지 변경, 미승인 내보내기, 기존 파일 덮어쓰기 방지도 같은 파일에서 검사한다.

이번 정리는 함수 이름·본문·인자·기본값을 유지하고 배치와 안내를 바꾸는 범위다.
CPU 테스트 통과와 실제 GPU 출력 동일성은 별개다. GPU 생성은 이번 범위가 아니다.

동작을 이해하기 위한 공식 문서:
- [Python 클래스와 메서드](https://docs.python.org/3/tutorial/classes.html)
- [Qt QThread와 스레드 간 신호](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html)

Qt 실행은 신호로 이어지므로 파일의 줄 순서와 실제 실행 순서는 다를 수 있다.
위 주 흐름은 메서드를 사용자 작업 순서로 배치해서 이 차이를 따라 읽기 쉽게 만든 것이다.


### 2026-10-04 중간 리팩터링 검증

- 변경 전 작업실 테스트: 17 passed (17.47초).
- 변경 후 관련 CPU 회귀: 183 passed (49.12초).
- 검사 파일: test_studio_generation.py, test_studio_ui.py, test_onepass_generation.py,
  test_onepass_prompt.py, test_gui_workflow.py, test_gui_contract_integration.py.
- 명령: `python -m pytest -p no:cacheprovider tests/test_studio_generation.py tests/test_studio_ui.py tests/test_onepass_generation.py tests/test_onepass_prompt.py tests/test_gui_workflow.py tests/test_gui_contract_integration.py -q`
- 환경: Windows 기존 venv, QT_QPA_PLATFORM=offscreen, CUDA_VISIBLE_DEVICES 빈 값,
  PYTHONDONTWRITEBYTECODE=1. GUI 생성 테스트는 가짜 백엔드를 사용했다.
- studio_controller.py와 gui_main.py는 정의 순서·설명문을 제외한 실행 구문 트리 동일 확인.
- 새 문서의 로컬 링크와 git diff 공백 검사 통과.
- 전체 tests/ 실행, 실제 모델 생성, 사용자 수동 GUI 검증은 이번에 수행하지 않았다.
