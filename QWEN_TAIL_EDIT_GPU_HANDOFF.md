# Qwen 꼬리 고치기 GPU 확인 인수인계

코드 연결과 CPU 검증은 완료됐고 전체 1,593개 테스트가 통과했습니다. **GPU 재현은 미검증**입니다.
작성일: 2026-10-06. 구현: Codex, 이후 GPU 확인: Claude/mydesk.

## 확인 범위

사용자가 생성한 4장 중 1장을 선택한 뒤 꼬리 고치기를 실행합니다. 원래 캐릭터에서 사각형으로 지정한 꼬리만 Picture 2로 전달합니다. 무늬 있음/없음은 기본값 없이 필수 선택이고, 끝 모양 한 문장은 선택입니다. 사용자 확인 전에는 실행하지 않습니다.

처음 생성한 후보와 기록은 덮어쓰지 않습니다. 편집본을 비교하고 확인한 경우에만 저장 대상이 바뀝니다. 취소·오류·미채택이면 이전 결과를 유지합니다. 편집본 위에 연쇄 편집하지 않고 선택한 최초 후보를 기반으로 새 실행을 만듭니다.

자세 GUI·비키니·SDXL 생성 조건·자동 꼬리 분석·선택 영역 제거는 추가하지 않았습니다. 시험의 고정 문장 `exactly one tail`은 재현을 위해 유지했으며 새로운 개수 지시나 부정 문구는 추가하지 않았습니다.

## 구현 진입점과 설정

| 역할 | 파일·함수 |
|---|---|
| 버튼 → 확인 → 실행 → 비교 | `genai_lab/studio_controller.py`: `edit_tail`, `tail_edited`, `tail_failed` |
| 원본 좌표 사각형·선택 화면 | `genai_lab/qwen_tail_gui.py`: `TailCropCanvas`, `TailInputDialog` |
| 크롭·지시문·입력 무결성 | `genai_lab/qwen_tail_edit.py`: `prepare_tail_spec`, `make_tail_request`, `validate_tail_request` |
| 별도 프로세스 실행 | `run_tail_edit(request, directory, TailEditWorkflow(spec))` |
| 공통 Qwen 실행 | `qwen_pose_edit._run_image_edit`, `qwen_pose_worker`의 `task_kind=tail_edit` 분기 |
| 편집본 승인·내보내기 | `studio_generation.StudioResults.adopt_tail_edit`, `approve`, `export` |
| R1~R5 재현 준비 | `scripts/verify_qwen_tail_replay.py` |

기존 `QwenPoseSettings` JSON을 그대로 사용합니다. 실행 경로·모델 경로를 새로 고정하지 않습니다. GUI는 `GENAI_QWEN_SETTINGS` 환경 변수에서 JSON 경로를 읽으며, 없으면 사용자가 파일을 선택합니다. 선택한 경로는 현재 작업실 세션에서 재사용합니다. 배포 실행기에서 이 변수를 설정하면 매번 파일을 고를 필요가 없습니다.

추가 기본값: 끝 모양 빈 문자열, 무늬 미선택, 확인 미체크. 꼬리 편집 seed는 시험값 **209212001만 허용**합니다. 끝 모양은 ASCII 한 줄·300자 이내·한 문장입니다. 기존 금지 어휘를 문장 안에서도 검사하고 의상 명사·종 이름을 추가 검사합니다. 유한 어휘 검사이며 모든 문장의 뜻을 판별하지는 않습니다.

모델 설정은 Q4_K_M / nf4 텍스트 인코더 후 해제 / group_block1 / zero_cond_t / 40단계 / true CFG 4 / guidance 1 / 부정 공백 1개입니다. 기존 패키지 버전·모델 해시 검사와 앞 단계 GPU 해제 검사를 재사용합니다.

## 완료한 CPU 검증

| 확인 | 결과 |
|---|---|
| 전체 `tests/` | 1,593 통과, 기존 Pillow getdata 사용 중단 예정 경고 2건 |
| 새 꼬리 테스트 | 49 통과 |
| 고정 지시문 | 무늬 있음·없음·나선 문장 3종 바이트 일치 |
| 실험 입력 R1~R5 | 크롭 SHA·긍정·부정·seed 5/5 일치 |
| 실제 GUI 신호, 가짜 Qwen 프로세스 | 실행·채택·저장, 실패·취소 원본 유지 통과 |
| 좌표 | 원본 936×2048, 투명 합성, 반대 방향 드래그, 여백, 창 크기 변경 확인 |
| 화면 | 합성 도형으로 입력 창 배치 확인. 실제 캐릭터 GUI 사용 확인은 아래에서 수행 |

실행 명령:
```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:QT_QPA_PLATFORM='offscreen'
$env:CUDA_VISIBLE_DEVICES=''
& D:\genai-cache\venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/ -q
```
225.75초. GPU 모델 실행·다운로드·설치 없음. 기록: `outputs/qwen-tail-implementation-20261006/results.md`, `replay-preflight/summary.json`.

## Claude 실행 순서

아래에서 `$runtime`에는 **이미 검증한 기존 Qwen 실행 환경 JSON의 실제 경로**를 지정합니다. 환경 설치·다운로드·모델 설정 변경은 하지 않습니다. 실행마다 새 출력 폴더를 사용하며 실패한 실행을 덮어쓰거나 자동 재시도하지 않습니다.

먼저 CPU에서 설정까지 포함한 R1~R5 대조를 합니다. 모델을 로드하지 않습니다.
```powershell
$runtime = '기존 Qwen runtime.json의 실제 경로'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& D:\genai-cache\venv\Scripts\python.exe scripts/verify_qwen_tail_replay.py --settings $runtime --output "outputs/qwen-tail-gpu-verify-$stamp/preflight"
```
모든 `matches`가 true여야 합니다. 검증 스크립트는 저장소의 시험 기록과 원시 결과 SHA도 확인합니다. 전체 원본은 CPU에서 크롭을 다시 만드는 데만 쓰고, 모델 입력은 기반 이미지와 크롭 두 장뿐입니다.

그다음 GPU 실행은 **한 번에 한 사례만** 수행합니다. 다른 생성·분석 프로세스를 종료하고, CPU 시험 때 설정한 GPU 비활성 환경 변수를 해제합니다.
```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& D:\genai-cache\venv\Scripts\python.exe scripts/verify_qwen_tail_replay.py --settings $runtime --case R1 --execute --output "outputs/qwen-tail-gpu-verify-$stamp/R1"
```
R1 완료 후 기록을 확인하고 R2~R5를 각각 실행합니다. `--execute`가 없으면 생성하지 않습니다. 다섯 건 약 2.5시간은 과거 실행에서 얻은 예상이며 현재 시간은 실제 로그로 기록합니다.

| 사례 | 기대 raw SHA-256 |
| R1 | `81c1c665a33323eeb437e77a502f6afb4dbefa7bd5200eab955a215893efe6c4` |
| R2 | `c5d208903f3b6ecdd4123ad7b3d3bfbc326a9fa70406039f79993c134f70bc60` |
| R3 | `e8791f48a713d9f0d03eb1dd7faa7a2c261ed6bb235540fb026e8f558674409f` |
| R4 | `4324f017759b33c1dcc7255f28ee0f596d02e18be8561659839b7841009586aa` |
| R5 | `4faff67e5a1e0b7be7ea53eb03ad201c7e95a706bce15e8618cf33184a80e594` |

각 실행에서 raw SHA, 실제 모델 입력 2장, 전처리 크기, 지시문, 단계 수, 패키지/모델 기록, 메모리와 시간을 확인합니다. 미리보기는 c8e5331b와 동일하게 기반 실제 크기로 복원합니다. raw SHA가 다르면 원본을 보존하고 차이를 보고하며, 설정을 바꾸어 맞추거나 재시도하지 않습니다. SHA 일치는 다섯 사례 재현 근거이며 다른 캐릭터의 성공을 보장하지 않습니다.

## 실제 GUI 확인

1. 원본 꼬리 태그가 있는 캐릭터의 생성 결과에서, 최종 승인 전에도 꼬리 고치기 버튼이 보이는지 확인합니다. `twintails`만 있는 캐릭터에는 없어야 합니다.
2. 무늬를 고르지 않거나 확인을 체크하지 않으면 실행되지 않아야 합니다. 원본 전체·빈 영역은 사용할 수 없어야 합니다.
3. 그림을 확대해 보거나 창 크기를 바꿔도 같은 원본 영역을 자르는지 확인합니다. 크롭에 옷·다리가 포함됐는지는 사용자가 확인합니다.
4. 실행 중에는 후보·입력 변경과 중복 실행이 막혀야 합니다. 취소해도 4장은 남아야 합니다.
5. 원본 캐릭터 / 고치기 전 / 고친 뒤를 비교하고, 미채택하면 이전 결과가 유지되는지 봅니다.
6. 채택 후 저장한 PNG가 해당 `edit/product.png`와 같은 SHA인지 확인합니다. 원래 후보 raw와 기록은 그대로여야 합니다.
7. 다른 후보를 선택하면 편집본 채택과 승인이 해제되어야 합니다. 자세 사진 선택이나 자세 검출 화면이 새로 열리지 않아야 합니다.

GUI 실행 전 `$env:GENAI_QWEN_SETTINGS=$runtime`을 지정할 수 있습니다. 실제 GUI 검증은 별도 신규 생성 횟수를 임의로 늘리지 말고, 승인된 재현 계획 안에서 수행합니다. 진입점 재현만 확인했다면 실제 GUI 확인을 완료했다고 보고하지 않습니다.

## 남는 한계

큰 고리와 굵기 불일치, 약 30분의 편집 시간은 해결되지 않았습니다. 사각형이 꼬리만 담았는지 자동 분석하지 않습니다. 사용자 입력의 의미를 완전히 검사하지 않습니다. 참고 이미지에서 드러나지 않는 모양을 자동으로 보완하지 않습니다.

기존 Qwen 보존 명세의 R10·R11에 있는 고정된 “꼬리 2개” 기술은 별도 후속 검토 대상입니다. 이번 꼬리 경로는 해당 자세 보존 명세를 호출하지 않으며, 자세 코드의 그 문장은 변경하지 않았습니다.

**다음 작업: Claude가 R1~R5 GPU 재현과 실제 GUI 확인 결과를 보고합니다. 푸시는 사용자 확인 후 진행합니다.**

최종 자원 해제 보완 후 꼬리 관련 49개를 다시 실행해 모두 통과했습니다(29.20초). 이전 파이프라인의 후크 메서드 참조까지 해제한 뒤 편집을 시작하는지 확인했습니다.
