# 작업실 배경 정리와 의상 경고 검증 인수인계

2026-10-08 구현. 요청 원본: `outputs/gui-test-review-20261008/codex_request_gui_fixes.md`.

## 변경 결과와 범위

작업실에서 생성한 네 장을 CPU로 흰 배경 처리한 뒤 비교·승인·저장에 사용한다. 의상 설명의 충돌은 입력 확인창에 표시하고, 사용자가 수정한 설명과 남은 경고를 기록한다. 부정 문구에서 단어를 삭제하는 처리는 없다.

GPU 생성, 모델 다운로드·설치, 자동 노출 검사, 자세·2단계 비율 GUI 연결은 이번 범위에 없다. 기존 생성 원본과 `run.json`은 그대로 둔다. 커밋·푸시는 사용자 확인 후 진행한다.

## 사용자가 거치는 과정

1. 캐릭터와 의상을 선택하고 입력 확인창을 연다.
2. 바지·치마 동시 검출, 바지·스타킹 동시 검출, 의상 설명에 섞인 자세, 부정 문구와 겹친 단어가 있으면 경고가 보인다.
3. 필요한 경우 `옷 설명 자세히 보기·수정`을 열어 잘못된 단어를 직접 지운다. 변경하면 기존 입력 확인 체크가 해제된다. 경고만으로 차단하거나 자동 삭제하지 않는다.
4. 기존 조건으로 네 장을 생성하고 생성 모델을 해제한다. 이어 CPU 배경 정리 모델을 한 번 불러와 네 장을 순서대로 처리한 뒤 해제한다.
5. 정리된 결과를 비교하고 승인한다. `원본(raw) 보기`는 배경 정리 전 생성 이미지를 보여 준다. 기존 `원본 크기로 보기`는 현재 저장 대상의 확대 보기다.
6. 저장할 때는 승인한 이미지의 SHA를 다시 검사한다. 꼬리 고치기도 배경 정리본을 기준으로 시작하며, 편집 결과를 채택하면 그 편집본을 저장한다.

## 배경 합성과 실패 처리

`proportion_foreground.AnimeForeground`와 `proportion_inputs.white_background`를 재사용한다. 모델은 기존 캐시의 `skytnt/anime-seg`, revision `493cb60893f47441b26ec4fb9a306bce9e342982`, `isnetis.onnx`다. `FOREGROUND_SHA`와 맞을 때만 CPU 세션을 연다. 캐시 루트는 기존 `GENAI_STUDIO_RUNTIME`의 `model_cache` 설정을 따른다.

흰 배경 합성은 **부드러운 알파 전체를 사용한다.** 0.5 이진화·최대 덩어리 선택은 스케치 지도 작성 규칙이며, 흰 배경 합성에는 적용하지 않는다. 따라서 떨어져 보이는 꼬리 조각을 최대 덩어리 밖이라는 이유로 없애지 않는다.

모델 없음·SHA 불일치·CPU 추론 실패에는 해당 후보의 원본을 표시하고, 결과 영역에 **배경 정리 안 됨**을 계속 표시한다. 실패 이유는 `background.json`에 남는다. 한 후보 추론 실패는 다른 후보의 정리를 막지 않는다. 원본 SHA 불일치나 기록 저장 실패는 전체 실행 오류로 올려 정상 완료처럼 표시하지 않는다. 취소하면 원본은 실행 폴더에 보존하고 결과 승인 화면으로 넘어가지 않는다.

몸에 붙은 줄·덩어리·장식은 남을 수 있다. 기존 보고의 9/30 잡티 정리 7/24는 이번 GUI 수정의 성공률이 아니다. 새 성인 자료의 육안 비교가 필요하다. 배경 정리는 노출 검사와 별개이며, 사용자가 결과를 확인하는 기존 절차를 유지한다.

## 설정과 기록

| 위치 | 담는 정보 |
| --- | --- |
| `configs/studio_garment_warnings.json` | 충돌 단어 묶음, 자세·동작 단어, 안내 문구, 부정 문구 겹침 검사 사용 여부 |
| `inputs/approval.json`의 `garment_review` | 최종 승인한 의상 설명, 실제 부정 문구, 경고, 사용한 경고 설정 |
| 각 seed 폴더의 `garment-review.json` | 생성 결과와 연결된 동일한 경고 기록. 자동 제거 없음 |
| 각 seed 폴더의 `background.json` | 성공·실패, raw/product SHA, 모델 경로와 기대 SHA, CPU 처리 시간, 실패 이유 |
| 각 seed 폴더의 `product.png` | 배경 정리 성공 시 미리보기·승인·저장 대상 |
| `user-review.json`, 내보낸 `.review.json` | 사용자 승인 대상 SHA, 배경 정리 기록, 원본과 편집 이력 |

다른 경고 규칙 파일은 `GENAI_STUDIO_RUNTIME` JSON의 `garment_warning_rules`에 경로를 지정한다. 규칙을 바꿔도 생성 설명 자체가 자동으로 바뀌지는 않는다. `panties`, `underwear`, `buruma`, `nsfw` 등 기존 부정 문구는 유지한다.

겹치는 부정 단어를 해당 요청에서 제거하는 안은 후속 결정 대상이다. 이번 코드는 이 안을 실행하는 옵션을 제공하지 않는다.

## Claude CPU 검증

**기존 실행 폴더에 덮어쓰지 않는다. GPU를 실행하지 않는다.** 성인으로 확인된 기존 GUI 생성 결과만 별도 검증 폴더에 복사한다. 요청서 `analysis.md`의 분석 제외 자료는 그대로 제외한다.

1. 결과별 원본 PNG와 해당 `run.json`을 새 seed 폴더에 복사한다. `run.json`의 raw SHA와 복사본 SHA를 먼저 비교한다. 경로·입력·인물을 매핑한 목록을 따로 보존한다.
2. 아래 API로 CPU 정리를 실행한다. `OnePassCandidate.path`는 복사본, `record_path`는 복사한 기록이어야 한다. 배치 디렉터리는 새 폴더다. 원본 기록을 고쳐 성공 상태를 만들면 안 된다.

```python
from genai_lab.studio_background import prepare_backgrounds
# batch: OnePassBatch, candidates: 복사한 raw/run.json의 OnePassCandidate들
prepare_backgrounds(batch, runtime.model_cache, progress=print)
```

3. `background.json`의 성공 여부와 raw SHA 불변을 먼저 확인한다. 원본·정리본을 나란히 보고 회색 배경, 붉은 장식, 얼굴·머리·의상·꼬리 경계 손실을 각각 기록한다. 실행 성공과 육안 성공을 나누어 집계한다.
4. 기존 입력 승인 기록 10건의 `garment_tags`, `negative`를 아래 함수에 넣는다. 원본 approval은 수정하지 않고 새 JSON에 경고를 저장한다. 실제로 겹친 단어와 경고가 맞는지 확인한다.

```python
from genai_lab.studio_garment_warnings import garment_warnings
warnings = garment_warnings(approval['garment_tags'], approval['negative'])
```

5. GUI에서 경고 → 수동 수정 → 경고 갱신 → 재확인을 확인한다. 새 GPU 생성 없이 저장된 배치를 `StudioResults` 또는 `StudioController.generated`에 전달해 정리본·raw 보기와 저장 파일이 맞는지 확인할 수 있다.
6. 모델 경로가 없는 테스트용 런타임으로 CPU 처리하면 정리 실패가 표시돼야 한다. 실제 캐시 파일을 지우거나 수정하지 않는다.

## CPU 테스트 범위

`tests/test_studio_gui_fixes.py`는 정리본 생성·원본 기록 불변·모델 없음/잘못된 SHA·일부 추론 실패·취소·변조 거부·재실행 덮어쓰기 거부·정리본 기준 꼬리 편집·경고 규칙·부정 문구 불변·실제 GUI 신호와 저장을 검사한다. 이미지와 생성 백엔드는 합성 자료·가짜 구현을 쓰므로 모델 품질을 평가하는 시험은 아니다.

전체 CPU 테스트 1,831개 통과(181.49초, 기존 Pillow 사용 중단 예정 경고 2건). 이후 작은 창의 보기 버튼을 같은 줄로 정리하고 관련 테스트 98개를 재실행해 통과했다(27.76초). GPU 실행은 없었다. 로그: `outputs/gui-fixes-cpu-targeted.log`, `outputs/gui-fixes-cpu-full.log`.

합성 자료로 입력 경고·정리본·실패 안내 화면을 확인했다. 1024×700 창에서 결과 보기 버튼과 미리보기가 겹치지 않으며, 승인·저장 버튼이 창 안에 있다. 캡처는 `outputs/gui-fixes-verification/input-warnings.png`, `result-product.png`, `result-fallback.png`에 있다. 화면 없는 Qt 캡처에만 설치된 맑은 고딕을 명시했으며, 운영 글꼴 설정은 변경하지 않았다.

## 변경 파일

- `genai_lab/studio_background.py`: CPU 정리와 원본/결과 분리 기록.
- `genai_lab/studio_garment_warnings.py`, `configs/studio_garment_warnings.json`: 설정 기반 경고.
- `genai_lab/studio_generation.py`: 승인 기록, 정리본 검증·저장.
- `genai_lab/studio_controller.py`, `genai_lab/studio_ui.py`: 입력 경고, CPU 후처리, 원본 보기 연결.
- `tests/test_studio_gui_fixes.py`, `tests/test_studio_generation.py`: 새 계약과 기존 흐름 회귀 검사.

다음 작업은 Claude의 기존 성인 raw CPU 비교와 사용자의 GUI 확인이다. 이번 변경의 GPU 생성 품질은 검증하지 않았다.