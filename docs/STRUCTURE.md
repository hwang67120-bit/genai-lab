# 기능별 파일 책임과 연결

> 이 문서는 구현 브랜치의 [a1bfd0d9](https://github.com/hwang67120-bit/genai-lab/tree/a1bfd0d910ec9f321d9f58f65bc95e1429163c64)를 기준으로 작성했습니다. main에는 설명 문서만 반영하며, 작업실·Qwen·꼬리 기능 구현은 [PR #6](https://github.com/hwang67120-bit/genai-lab/pull/6)에서 별도로 검토합니다. 코드·테스트 링크는 검토한 구현 커밋을 가리킵니다.

2026-10-04 현재 작업실 코드 기준입니다. 사용자 흐름은 [CODE_FLOW.md](CODE_FLOW.md), 문제 해결 과정과 수치는 [ENGINEERING_REPORT.md](ENGINEERING_REPORT.md)에 설명합니다. 이 문서는 그 과정이 어느 파일에 있는지 찾는 데 사용합니다.

## 전체 흐름을 연결하는 진입점

[studio_controller.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/studio_controller.py)는 입력부터 저장까지의 실행 순서를 연결합니다. 파일 앞부분에는 시작·입력 확인·생성 완료·후보 선택·승인·저장 메서드를 배치했습니다. 모델 계산은 각 단계에서 호출하는 세부 모듈이 담당합니다.

화면 배치와 계산이 파일 하나에 모두 들어 있지는 않습니다. 사용자의 작업 순서를 연결하는 부분, 이미지에서 정보를 추출하는 부분, 생성기를 호출하는 부분, 승인한 결과를 저장하는 부분을 나눴습니다. 이 구분의 목적은 파일 수를 늘리는 것이 아니라 **어느 단계가 정보를 만들고 어느 단계가 그 정보를 사용하는지 명확히 하는 것**입니다.

## 현재 작업실의 다섯 책임

| 책임 | 입력 → 출력 | 담당 파일 | 다른 부분에 맡기는 일 |
|---|---|---|---|
| 화면과 작업 순서 | 선택·확인·취소 → 다음 작업 실행 | [gui_main.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/gui_main.py), [studio_ui.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/studio_ui.py), [studio_controller.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/studio_controller.py) | 모델 추론·태그 계산은 세부 모듈에 위임 |
| 입력 분석 | 캐릭터·의상 이미지 → 얼굴 참조·설명 후보 | [studio_analysis.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/studio_analysis.py), [onepass_character.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_character.py), [clothing_analysis.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/clothing_analysis.py) | 생성과 최종 이미지 승인은 하지 않음 |
| 요청 준비 | 사용자가 확인한 정보 → 생성 문장·설정·4개 seed | [studio_generation.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/studio_generation.py)의 `build_request`, [onepass_gender.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_gender.py), [onepass_prompt.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_prompt.py) | 입력의 미확인 항목을 임의 승인하지 않음 |
| 생성 실행 | 생성 요청 → 이미지·실행 기록 | [onepass_generation.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_generation.py) | 캐릭터·의상 품질의 최종 사용 여부는 사용자가 판단 |
| 승인과 저장 | 선택 후보·사용자 확인 → PNG·검토 기록 | [studio_generation.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/studio_generation.py)의 `StudioResults` | 새 이미지 생성이나 자동 재시도는 하지 않음 |

`studio_generation.py`에는 현재 입력 분석 프로세스 관리, 요청 준비, 승인·저장 책임이 함께 있습니다. 파일마다 하나의 책임만 완전히 분리됐다고 설명하지 않습니다. 클래스와 함수의 경계로 구분한 현재 상태입니다.

## 데이터가 오가는 방식

분석은 완료될 때까지 기다리는 긴 작업이므로 별도 CPU 프로세스에서 실행합니다. 작업실의 작업 스레드가 그 프로세스의 완료를 기다리고, 완료 신호를 받은 GUI가 입력 확인창을 띄웁니다. 생성도 작업 스레드에서 수행하며, 진행 상태와 결과 표시를 GUI에 전달합니다.

따라서 파일을 위에서 아래로 읽는 순서와 실제 실행 시간이 완전히 같지는 않습니다. `launch`는 작업을 시작하는 지점, `finished`는 종료 후 GUI가 다음 단계로 넘어가는 지점입니다. 모델이 실행되는 동안 화면을 갱신하고 취소 요청을 받을 수 있도록 나눈 구조입니다.

기술 근거: [Qt의 스레드와 신호 설명](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html). 현재 연결 검사는 [작업실 테스트](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/tests/test_studio_generation.py)에 있습니다.

## 어휘·검사·설정의 관리 위치

| 수정하려는 내용 | 위치 | 현재 적용 범위 |
|---|---|---|
| 의상 명사·피복·주머니 판단 | [onepass_garment_vocabulary.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_garment_vocabulary.py) | 생성 문장 조립 |
| 꼬리·귀 문구의 등록 태그 검사 | [reference_tag_policy.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/reference_tag_policy.py) | 입력 문구 검사. 생성 결과 안전 판정이 아님 |
| 자세 검사와 사용자 선택 정책 | [onepass_pose.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_pose.py), [onepass_input_settings.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_input_settings.py) | 자세 입력 처리 모듈. 현재 작업실은 외부 자세 미적용 |
| 검출·분할 도구 연결 | [onepass_input_backends.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_input_backends.py) | 도구의 결과를 공통 입력 형태로 변환 |
| 문장 기본값·토큰 길이 처리 | [onepass_prompt_settings.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_prompt_settings.py), [onepass_prompt_tokenizers.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_prompt_tokenizers.py) | 두 텍스트 인코더용 문장·청크 준비 |
| 생성 크기·단계·메모리 한도 | [onepass_generation_settings.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/onepass_generation_settings.py) | 1회 생성 실행 |
| 분석 Python·모델 캐시 위치 | [studio_generation.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/studio_generation.py)의 `StudioRuntime` | `GENAI_STUDIO_RUNTIME`이 가리키는 JSON으로 경로 변경 가능 |

`configs/animagine.yaml`은 현재 작업실의 모든 설정을 관리하는 파일이 아닙니다. 특히 분석 환경 경로와 1회 생성 설정은 위 위치를 확인해야 합니다.

## 별도 기능: 저장한 이미지의 자세 편집

Qwen 편집은 작업실 생성과 다른 실행 환경을 사용합니다. 화면은 [qwen_pose_gui.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_pose_gui.py), 보존 항목 준비는 [qwen_preservation_analysis.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_preservation_analysis.py)와 [qwen_preservation.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_preservation.py), 지시문 조립은 [qwen_pose_prompt.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_pose_prompt.py)가 담당합니다.

사용자가 확인한 요청을 [qwen_pose_edit.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_pose_edit.py)가 별도 프로세스로 전달하고 [qwen_pose_worker.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_pose_worker.py)가 모델을 실행합니다. 실행 환경은 [qwen_pose_settings.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_pose_settings.py), 진행·종료 기록 파일 처리는 [qwen_record_io.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_record_io.py), 사용자 결과 검토는 [qwen_pose_review.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/qwen_pose_review.py)에 있습니다.

## 남아 있는 이전 경로와 시험 자료

[GenerationOrchestrator](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/generation_orchestrator.py), [generator.py](https://github.com/hwang67120-bit/genai-lab/blob/a1bfd0d910ec9f321d9f58f65bc95e1429163c64/genai_lab/generator.py), 기존 Base·의상 정밀화 관련 모듈은 저장소에 남아 있습니다. 현재 만들기 버튼은 작업실 제어기로 진입하므로, 파일이 존재한다는 이유로 기본 GUI에서 실행된다고 판단하면 안 됩니다. 기존 명령행 도구와 `run.py`도 현재 작업실 검증과 구분합니다.

`outputs/`에는 시험 계획·라벨·스크립트·측정·생성 파일이 있습니다. 제품 코드는 원칙적으로 `genai_lab/`에서 읽지만, 현재 머리 검출기 기본 캐시가 `outputs/` 아래를 참조하는 예외가 남아 있습니다. 다른 장비에 배포할 때 해당 캐시를 별도 관리하거나 경로를 지정해야 합니다.

[DATA_MODELS.md](DATA_MODELS.md)의 객체 목록과 [FLOW.md](FLOW.md)의 기존 공통 실행 계약에는 이전 경로 설명이 포함돼 있습니다. 현재 문서에서는 그 차이를 명시하고 과거 기록을 삭제하지 않았습니다. 이번 문서 변경으로 코드나 제품 정책을 바꾸지는 않았습니다.
