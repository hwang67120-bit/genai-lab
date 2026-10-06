# 꼬리 특징 안내 구현과 CPU 확인

2026-10-06. 꼬리 고치기 창에 화면 안내용 분석을 연결했다. 원본 7개와 생성 꼬리 22개의 기존 측정값이 29/29 재현됐다. 이 결과는 측정 재현이며, 사람 판단의 정확도를 새로 검증했다는 뜻은 아니다.

## 사용 흐름과 변경하지 않은 입력

원본에서 사각형 선택 → 별도 CPU 프로세스에서 SAM2 분리 → 원본 크롭과 녹색 마스크 겹침 그림 비교 → 색 가족과 무늬·곡선 안내 확인 순서다. 분석을 기다리지 않고 기존 편집을 진행할 수 있다. 분석을 끄거나 영역을 다시 고르거나 창을 닫으면 진행 중 프로세스를 중단한다. 제한 시간은 120초이며, 모델 없음·실행 오류·시간 초과는 안내만 표시한다.

분석 결과는 무늬 콤보, 끝 모양, TailEditSpec, Qwen 요청, 프롬프트에 넣지 않는다. 기존 인식 설명 기능은 기본 OFF다. 분석용 마스크는 편집용 꼬리 사각형 크롭을 대체하지 않는다. 색 이름은 임의로 붙이지 않고 여러 색/한 가지 색 계열만 표시한다.

## 코드를 읽는 순서

| 시작점 | 입력과 결과 |
|---|---|
| genai_lab/qwen_tail_gui.py: TailInputDialog.queue_complexity_analysis | 선택을 바꾸면 이전 결과를 폐기하고 350ms 뒤 분석 요청 |
| genai_lab/tail_complexity_gui.py: TailComplexityRunner.start | 별도 CPU 프로세스 실행, 취소·시간 초과·오래된 결과 분리 |
| genai_lab/tail_complexity_worker.py: analyze_image | 원본 SHA 확인 → 상자 분리 → 측정 → 마스크·겹침 그림 저장 |
| genai_lab/tail_complexity.py: measure_tail | 잠긴 v3 수치와 표시 전용 판단 보류 생성 |
| genai_lab/qwen_tail_gui.py: TailInputDialog.complexity_finished | 현재 영역과 원본이 같은 결과만 화면에 표시 |
| genai_lab/studio_controller.py: edit_tail / launch_tail_edit | 안내 기록을 별도 저장하고 기존 Qwen 요청은 그대로 실행 |

## 실행 환경과 기록

기존 캐시의 facebook/sam2.1-hiera-tiny만 CPU에서 읽는다. 확인한 revision은 de431c4043854a71d8101e17995dfe596bf101a5다. 모든 모델 읽기는 local_files_only이며, 자식 프로세스에 HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, CUDA_VISIBLE_DEVICES=빈 문자열을 적용한다. 다운로드·설치를 하지 않았다.

캐시 위치는 기존 ClothingMaskExtractionSettings.cache_dir를 재사용한다. 필요한 경우 GENAI_TAIL_SAM_CACHE로 기존 캐시 위치만 지정한다. GPU 설정이나 생성 설정을 추가하지 않는다.

편집 폴더의 complexity/run.json에 tail_complexity를 기록하고 마스크·겹침 그림을 복사한다. Qwen 종료 뒤 edit/launcher.json에도 같은 메타데이터를 기록한다. worker의 request.json, run.json, run.final.json은 수정하지 않는다. 분석 이미지 파일 복사 실패는 artifact_errors로 기록하며 편집을 막지 않는다.

## 확인 결과

| 검사 | 결과 |
|---|---|
| 기존 측정 재현 | 원본 7/7, 생성 22/22. 색 가족·깊은 골 수 일치, 각도 허용오차 ±1도 이내. 마스크 면적도 일치 |
| 안내 ON/OFF | 같은 승인 입력으로 요청 SHA와 프롬프트 SHA 동일. 컨트롤러 경유 요청도 동일 |
| GUI 입력 | 분석 후 무늬 미선택, 끝 모양 빈칸, 인식 기본 OFF 유지 |
| 오류 처리 | 모델 없음, 실행 실패, 시간 초과, 취소, 오래된 결과, 분석 이미지 파일 없음 검사 |
| 실제 CPU GUI | 라이브러리·모델 준비 8.329초, 이미지 분석 2.782초, worker 전체 11.188초, 화면 완료 12.906초 |
| RAM | 실제 GUI 분석 worker 최고 1,370,877,952바이트(약 1.28GiB). 29건 일괄 검사 최고 약 1.58GiB |
| 화면 응답 | 50ms 타이머 201회, 관측 최대 간격 0.125초. 자동 offscreen 검사이며 사용자의 직접 조작 평가는 아님 |

측정 재현: outputs/tail-complexity-implementation-20261006/replay/results.json
실제 GUI: outputs/tail-complexity-implementation-20261006/gui-final/run.json, completed.png
전체 회귀: 1644 passed, 2 warnings, 183.18초. 경고는 기존 Pillow getdata 폐기 예고다. 상세는 같은 폴더 results.md에 기록했다.

Qt 프로세스가 종료될 때 finished/errorOccurred 연결을 해제하고 삭제하도록 했다. 이 정리 전에는 뒤이어 실행한 기존 인식 확인 창 테스트가 종료됐고, 정리 후 해당 순서가 통과했다.

## CPU 재확인 명령

Windows의 프로젝트 폴더에서, 기존 운영 Python을 사용한다.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:QT_QPA_PLATFORM = 'offscreen'
$env:CUDA_VISIBLE_DEVICES = ''
& 'D:\genai-cache\venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests
& 'D:\genai-cache\venv\Scripts\python.exe' -m scripts.verify_tail_complexity --dataset outputs/tail-complexity-measure-20261006 --output outputs/tail-complexity-implementation-20261006/replay-new
```

재현 스크립트는 기존 입력·결과를 읽기만 하고 새 output 폴더에만 기록한다. 이미 있는 output 폴더는 재사용하지 않는다.

## 남은 한계와 사용자 확인

무늬와 곡선은 추정이며 확인이 필요하다. 중심선 길이/굵기 <3이면 수치는 남기되 두 안내를 숨긴다. 이 보류 규칙은 아직 검증되지 않았다. 원래 시험의 사람 라벨은 Claude 잠정 판정이며 사용자 확인 전이고, 나선 사례도 없다. 원래 분리 실패 3/22를 해결하는 변경은 하지 않았다. 사용자가 녹색 영역이 실제 꼬리를 잡았는지 확인해야 한다.

직접 GUI 확인은 꼬리 사각형을 선택해 두 그림과 안내가 보이는지, 분석을 껐다 켜도 무늬·끝 모양이 바뀌지 않는지만 확인하면 된다. 이번 변경 확인에 Qwen GPU 재생성은 필요하지 않다. 자세 GUI, 비키니, 생성 경로는 수정하지 않았다.
