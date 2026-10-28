# 의상 제거 전처리 로그

2026-09-06: 연산·승인·통과 기준은 변경하지 않고 진단 기록만 추가했다.
프로그램 재시작 후 기존 의상 제거 과정을 실행하면 콘솔의 [의상 제거]와
outputs/debug_removal/<시간_UUID>.jsonl에 기록된다. 추가 모델/GPU 생성은 없다.
이미지 파일 자체는 자동 저장하지 않는다.

## 기록 범위

- clothing_removal_preprocessing: 원본 RGB 해시, 실행 설정, 입력 마스크, 최종 후보 및 오류
- analysis_runner: 외부 분석 실행 시작/반환/예외, 반환 코드, 표준 출력·오류의 마지막 2,000자
- automatic_mask_repair: 보정 전·후와 자동 의상 힌트
- mask_refinement: 보호·외곽 제한 전·후
- neutralization: 제거 마스크, 중립화 픽셀 수, 영역 밖 변경 수, 결과 이미지 해시
- selected_clothing_coverage: 검증 대상·보호 제외·변경 누락 픽셀 수
- body_initialization: 피부·기본복·배경 영역, 색상 표본 수와 초기 이미지 해시

각 단계의 시작/완료/예외, 시간, 이미지 모드/크기/픽셀 해시를 기록한다.
L 모드 마스크는 128 이상 픽셀 수, 내부 0 영역 개수/면적, 큰 영역 8개의
위치(x,y,width,height)를 기록한다. 이는 기하학적 빈 영역이며 실제 의상 잔재
판정이 아니다. 이미지 경계로 열린 누락 영역은 내부 0 영역 통계에 포함되지 않는다.
선택된 의상 마스크에서 빠진 픽셀과 AI가 처음부터 의상으로 못 잡은 픽셀은 다르다.
이 로그만으로 실제 의상이 전부 제거됐다고 판정하지 않는다.

중첩된 전처리 단계는 같은 실행 ID를 사용한다. 별도로 호출되는 4단계 초기화는
새 ID를 사용하며 원본 RGB 해시와 입출력 마스크 해시로 대조한다.
따라서 전체 GUI 세션/생성까지 하나의 ID로 연결한 기능은 아직 아니다.
단계 시간에는 진단 계산 비용도 일부 포함된다. 로그 저장 실패는 콘솔에 경고하며
원래 연산 결과나 예외를 바꾸지 않는다. 로그는 자동 삭제하지 않으므로 누적 용량을 관리해야 한다.

먼저 실패 실행의 JSONL에서 마지막 failed 또는 completed 단계와
그 전후 마스크의 selected_pixels / enclosed_zero_pixels / pixel_sha256을 비교한다.
실제 잔재 여부는 해당 마스크·원본 검토 화면과 함께 판단한다.

## 근거

- [Python logging](https://docs.python.org/3/library/logging.html): 전용 콘솔 로거
- [OpenCV connectedComponentsWithStats](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html):
  이진 영역의 면적 및 경계 상자 측정

수치 계산·기록 동작의 근거이며 마스크의 의미적 정확도를 보장하는 근거는 아니다.
