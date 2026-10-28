# 원본 비율 전달 시험 경로

확인한 원본 비율을 문장으로 변환해 기존 1회 생성기에 전달하고, 생성 결과를 같은 기준으로 비교하는 선택 실행 경로다. CPU 연결 검증용 구현이며 생성 품질 개선은 GPU 미검증이다. GUI와 기본 생성 요청은 변경하지 않았다.

## 읽기 시작할 함수

`genai_lab/body_proportion_trial.py`의 `generate_body_proportion_trial`이 주 흐름이다.

원본 비율 확인 → 원본·얼굴 SHA 확인 → 비율 문장 및 토큰 계획 생성 → 기존 generate_onepass_image 1회 호출 → 생성 결과 측정 → 별도 비교 기록.

- `body_proportions.py`: 좌표에서 비율 측정, 자동 검출 후보, 확인한 참조, 수치 대조.
- `body_proportion_trial.py`: 기존 요청을 비율 시험 요청으로 변환하고 실제 생성 함수에 전달. 생성 후 검토 기록과 실패 처리를 담당.
- `tests/test_body_proportion_trial.py`: GPU 없이 실제 생성 기록 함수까지 연결해 검증.

## 이번에 연결한 것과 남은 것

| 항목 | 상태 |
|---|---|
| 머리·발끝·어깨 좌표에서 비율 계산 | 구현 |
| 기존 DWPose·머리 검출 결과를 검토 후보로 변환 | 함수 구현. 기존 모델 결과를 호출자가 넘김 |
| 확인한 비율을 실제 긍정 문장과 청크에 반영 | 구현 |
| 기존 골격·얼굴·성별·의상·부정 조건 보존 | 구현·CPU 검사 |
| 생성 결과를 같은 기준으로 비교하고 별도 기록 | 구현·CPU 검사 |
| 모델이 숫자 문장을 따라 동일성을 개선하는지 | GPU 미검증 |
| 귀·뿔·머리카락을 제외한 머리 기준 자동 확정 | 미구현. 기존 오검출을 자동 승인하지 않음 |
| 운영 GUI에서 비율 확인·선택 | 미연결 |
| 팔·다리·몸통별 수치 전달, 공간 마스크 제어 | 이번 범위 밖 |

기존 BODY_PROPORTION_PRESETS.md의 체형 시트·Canny ControlNet은 이전 Animagine Base 경로의 별도 기능이다. 이번 1회 생성 숫자 문장 시험과 혼용하거나 이미 연결된 것으로 취급하지 않는다.

## 측정과 확인 계약

`measure_visible_proportions`는 이미지 파일과 머리 사각형, 보이는 발바닥의 y 좌표, 선택적인 어깨 두 점을 받는다. 원본 파일 SHA와 좌표도 기록한다.

머리 높이 = 머리 사각형 높이. 전체 높이 = 머리 위부터 발바닥까지. 등신 수 = 전체 높이 / 머리 높이. 어깨 비율 = 어깨 두 점 사이 거리 / 머리 사각형 너비.

동일한 머리 정의를 원본과 결과에 적용해야 한다. 귀·뿔·머리카락을 잘못 포함하지 않았는지, 발끝이 보이는지, 옷이 어깨를 가렸는지, 자세·원근 때문에 비교가 왜곡되는지 검토한다. 머리 영역을 검출했다는 사실만으로 `geometry_reviewed=True`를 지정하지 않는다. 가려진 어깨는 None으로 남길 수 있다. 발끝이 없으면 등신 수를 만들지 않고 생성용 확정을 거부한다.

`BodyProportionReference`에는 원본 측정과 그 원본에서 만든 얼굴 참조 SHA를 묶고 `confirmed=True`를 명시한다. 잘못된 원본 파일·다른 얼굴 참조는 생성 전에 거부한다. 좌표와 확인을 대신하는 자동 해부학적 복원은 없다.

기존 DWPose와 머리 검출 결과는 `suggest_proportions_from_pose`에 전달할 수 있다. 이 결과는 발목까지의 지표와 머리 상자 폭을 이용한 참고값이다. `heads_tall`은 항상 None이고 상태는 needs_review다. 발목을 발끝으로 바꾸거나 머리카락 포함 상자를 검증된 머리로 사용하지 않는다.

## 생성 조건

예를 들어 확인한 값이 4등신, 어깨 폭이 머리 폭의 1배이면 다음 두 문장을 기존 긍정 문장 뒤에 붙인다.

```text
body proportions of 4 head lengths tall, shoulders 1 head widths wide
```

문장과 목표값을 `run.json`의 `prompt.rules.body_proportion_condition`에 기록한다. 원래 긍정 문장, 원본/얼굴 SHA, 측정 좌표, 확인 여부, 전후 토큰 수도 남긴다. 원본 비율값은 그대로 기록하고 문장 숫자는 유효숫자 3자리로 표현한다.

이것은 숫자를 통한 텍스트 조건이며 출력 좌표를 고정하는 기능이 아니다. 효과를 확인하지 않은 상태에서 운영 기본값으로 켜거나 SD/chibi 태그로 자동 치환하지 않는다. 생성기의 기존 청크 함수를 사용하므로 75토큰을 넘으면 필요한 청크가 늘어난다. 부정 문장은 같고, 청크 수를 맞추기 위한 빈 청크가 추가될 수 있다.

`prepare_body_proportion_inputs(inputs, None)`은 입력 객체 자체를 그대로 반환한다. 비율을 사용하지 않는 기존 프롬프트·토큰·설정에는 변화가 없다.

## 실행 예

아래의 좌표는 사용자가 확인한 실제 이미지 좌표로 전달해야 한다. 기존 `inputs`, `backend`, `tokenizers`, `settings`를 재사용하며 모델 경로나 사용자 성별 저장소를 새로 설정하지 않는다.

```python
from genai_lab.body_proportions import measure_visible_proportions, BodyProportionReference
from genai_lab.body_proportion_trial import generate_body_proportion_trial

original = measure_visible_proportions(
    source_image,
    head_box=confirmed_source_head_box,
    sole_y=confirmed_source_sole_y,
    shoulders=confirmed_source_shoulder_points,
    geometry_reviewed=True,
)
reference = BodyProportionReference(original, inputs.face_sha256, confirmed=True)

result = generate_body_proportion_trial(
    backend, inputs, reference,
    source_image=source_image, tokenizers=tokenizers,
    seed=seed, directory=new_run_directory, settings=settings,
    measure_output=measure_generated_image,
)
```

`measure_generated_image(path)`는 그 생성 이미지 자체를 측정해 BodyProportionMeasurement를 반환하는 함수다. 원본 좌표를 결과에 그대로 적용해서는 안 된다. 결과 좌표를 아직 확인하지 않았다면 geometry_reviewed=False로 반환한다. 이때 수치가 맞아도 통과하지 않는다.

GPU 해제 후 CPU 측정이나 사용자 확인을 진행하려면 `prepare_body_proportion_inputs` → 기존 `generate_onepass_image` → 기존 backend.close() → `review_body_proportion_candidate` 순으로 별도 호출할 수 있다. 편의를 위한 통합 함수는 백엔드 소유권을 넘겨받지 않으며 자동 해제·재로딩하지 않는다.

## 결과와 실패

생성 기록 run.json을 덮어쓰지 않고 `body-proportion-review.json`에 비교 결과를 남긴다.

| 상태 | 의미 |
|---|---|
| within_tolerance | 확인된 측정값이 수치 범위 안에 있음. 캐릭터 동일성 승인 아님 |
| outside_tolerance | 확인된 측정값이 목표 범위를 벗어남 |
| unmeasurable_or_unreviewed | 필요한 좌표가 없거나 영역 검토 전 |
| measurement_failed | 측정 오류·대상 SHA 불일치 등. 생성 raw는 보존 |
| measurement_cancelled | 생성 후 측정 취소. raw 보존, 취소 예외 유지 |

기본 상대 오차 0.05는 검증되지 않은 비교용 설정이며 제품 합격 기준이 아니다. 결과에 tolerance_validated=False와 product_approved=False를 기록한다. 오류 없이 생성됐다는 기록과 비율이 맞는다는 기록을 분리한다. 기존 비교 기록은 덮어쓰지 않는다.

## GPU 인수인계

1. 원본의 머리 기준·발끝·어깨 위치를 확인하고 비율 참조와 이미지 SHA를 잠근다. 이전 머리카락 포함 상자/발목 지표 4.223을 실제 4.223등신으로 그대로 입력하지 않는다.
2. 골격·얼굴·성별·의상·seed·설정을 같게 하고 비율 조건 없음/있음을 비교한다. 기존 얼굴 참조가 정상인 단일 참조 경로를 사용한다.
3. 처음에는 일반 의상, 원본 1장, seed 2개로 총 4장 이내. 대조군은 기존 SHA를 재현하고, 오류·일정 불일치·reserved 6.5 GiB 초과 시 중단한다.
4. 얼굴·의상·비율을 따로 판정한다. 결과별 확인 좌표로 비교하며 모르는 부위는 모른다고 남긴다. 지표가 좋아져도 얼굴이 훼손되면 동일성 성공이 아니다.
5. 문구가 전달됐지만 비율이 달라지지 않으면 텍스트 방식의 효과가 확인되지 않은 것으로 보고한다. 자동 인식기를 확대하거나 GUI 기본값을 바꾸지 않는다.

이번 구현 작업에서는 GPU 생성·모델 다운로드·설치를 하지 않았다. 커밋·푸시는 별도 요청 전까지 하지 않는다.

## 이번 CPU 검증 결과

최종 274 passed in 24.58s. 전체 tests/ 실행 결과가 아니라 다음 관련 5개 파일의 결과다.

```text
python -B -m pytest -p no:cacheprovider tests/test_body_proportion_trial.py tests/test_onepass_generation.py tests/test_onepass_prompt.py tests/test_studio_generation.py tests/test_studio_ui.py -q
```

CUDA_VISIBLE_DEVICES 빈 값, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, QT_QPA_PLATFORM=offscreen, PYTHONDONTWRITEBYTECODE=1로 Windows 운영 venv에서 실행했다. GPU 생성은 하지 않았다. 원본 4등신 목표에 합성 측정 6등신이 들어오면 outside_tolerance가 기록되는 것, 문구가 실제 생성 함수와 run.json으로 전달되는 것, 원본·얼굴·결과 SHA 불일치 거부, 긴 긍정 청크 확장, 측정 실패·취소 시 raw 보존을 검사했다.

로그: outputs/body-proportion-implementation-20261007/final-tests.log.

## 사용자 지정 비율 목표

GPU 대조용으로 실측값과 사용자 지정 목표를 구분한다. BodyProportionReference의 basis 기본값은 reviewed_measurement다. 이 경로는 기존처럼 geometry_reviewed=True와 confirmed=True를 요구한다.

사용자가 목표를 직접 지정한 시험은 basis=user_declared_target로 명시한다. 이 경우 geometry_reviewed는 반드시 False여야 하며 confirmed는 목표를 사용하기로 했음을 뜻한다. 목표 숫자가 원본을 자동 분석한 실측 정답이라는 뜻이 아니다. 생성 조건과 비교 기록에 basis를 함께 남긴다. 결과의 geometry_reviewed=False는 여전히 자동 통과할 수 없다.

2026-10-07 첫 GPU 비교의 목표는 사용자가 앞서 제시한 약 4등신·어깨 폭 1머리 폭이다. 원본을 자동 측정한 값으로 바꾸어 기록하지 않는다. 시험 폴더는 outputs/body-proportion-target-ab-20261007이다.
