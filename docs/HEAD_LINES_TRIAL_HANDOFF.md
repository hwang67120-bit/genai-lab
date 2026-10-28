> 2026-10-09 후속 수정: 피부색 방식으로 R6 얼굴 분리 CPU 검사가 통과했다(얼굴/머리 14.7%). 새 미리보기 확인 후 GPU 시험이 가능하며 아직 생성은 하지 않았다. [최신 인수인계](HEAD_SKIN_MASK_HANDOFF.md)를 먼저 읽는다. 아래는 초기 SAM2 구현과 당시 실패 기록이다.
# 머리 안쪽 선 옵션 구현과 CPU 확인

머리 안쪽 선 옵션과 시험 실행기를 구현했다. 다만 R6 실제 CPU 검사에서 SAM2가 얼굴 피부 대신 머리 영역의 99.9%를 선택해, 약속한 70% 기준으로 중단했다. GPU 12장 시험은 아직 시작할 수 없다. 얼굴 분리 방식부터 보완해야 한다.

## 구현 범위

`ProportionOptions.head_lines`에 `none`, `hair`, `all`을 추가했다. 기본은 `none`이다. GUI, 1회 생성 경로, 마무리 경로, 다운로드·설치, GPU 실행, 커밋·푸시는 이번 작업에 포함하지 않았다.

원본 정규화 이미지와 사용자가 확인한 머리 마스크를 그대로 사용한다. 새 좌표 변환은 없다.

| 단계 | 고정한 동작 |
| --- | --- |
| 선 추출 | RGB 그레이 변환 → GaussianBlur σ=1 → Canny 100/200 |
| 잡음 제거 | 머리 마스크 안에서 8-연결 조각을 검사하고 20픽셀 미만 제거 |
| 선 두께 | 요청서에서 허용한 3×3 팽창 1회. 명칭은 2px 옵션이지만 실제 모든 선이 균일한 2픽셀 폭이라는 뜻은 아님 |
| 얼굴 분리 | SAM2.1-hiera-tiny CPU, 기존 로컬 캐시. 코 양성 점 + 머리 마스크 외접 상자 |
| 얼굴 제외 | 얼굴 마스크를 머리 영역에 한정하고 2픽셀(5×5 팽창) 넓혀 머리카락 선에서 제외 |
| 합성 | 기존 1단계 몸 바깥선 + 머리 외곽 4px + 안쪽 선. 기존 골격·얼굴 참조·프롬프트·seed·어댑터 강도와 단계 유지 |

코 관절 신뢰도 0.3 미만 또는 누락, 얼굴 영역 비어 있음, 얼굴 영역이 머리의 70% 초과이면 중단한다. 다른 방식으로 자동 전환하거나 기준을 완화하지 않는다.

## 데이터 흐름과 읽기 시작할 함수

- `genai_lab/head_lines.py:prepare_head_lines`: 원본·머리·코 확인 → CPU 얼굴 분리 → 선과 마스크 저장 → 검토용 겹침 이미지 저장. 자동 승인하지 않는다.
- `genai_lab/head_lines.py:checked_lines`: 확인 여부와 이미지·미리보기·SAM2 파일의 SHA를 검증한다.
- `genai_lab/head_lines.py:compose_head_lines`: 검토된 안쪽 선을 기존 스케치에 합친다. `none`이면 파일을 읽거나 인식하지 않고 원래 스케치를 그대로 반환한다.
- `scripts/head_lines_trial.py:prepare_case`: C0 1단계 원본으로 기존 스케치를 CPU에서 다시 계산해 SHA를 확인하고 두 조건의 미리보기와 잠금을 만든다.
- `scripts/head_lines_trial.py:run_prepared`: 사용자가 확인한 잠금 SHA를 받아 입력과 코드를 재검사한 뒤 CONTOUR만 실행한다. BASE를 다시 생성하지 않는다.

`head_lines_file`, `head_lines_sha256`, `head_lines_confirmed`는 미리보기 결과를 생성 경로에 연결하는 계약이다. 제품 함수는 확인되지 않은 선을 받으면 GPU 로딩 전에 중단한다.

## R6 실제 CPU 결과

자료: `outputs/hair-lineart-design-20261008/codex-cpu-r6-01/`.

| 확인 항목 | 실측 |
| --- | --- |
| 코 신뢰도 | 약 0.8323 |
| 코 좌표 | (365.9167, 215.6396) |
| 머리 상자 | [252, 39, 487, 316] |
| 얼굴로 분리된 비율 | 0.9989388427, 약 99.9% |
| SAM2 예측 점수 | 약 0.9498 |
| 준비 시간 | 약 29.34초 |
| 결과 | `face_too_large`, GPU 생성 0장 |

`head-lines/mask-preview.png`를 확인했다. 얼굴뿐 아니라 앞머리와 옆머리까지 같은 마스크로 잡혔다. 높은 SAM2 예측 점수는 사용자가 원하는 '피부만 분리'가 맞다는 증거가 아니었다.

확인된 사실은 **현재의 코 점 + 머리 상자 조건으로 R6의 피부만 분리하지 못했다**는 것이다. 다른 캐릭터에서도 반드시 실패한다거나, 안쪽 선 조건 자체가 효과 없다는 결론은 아직 낼 수 없다. 실제 선 조건의 생성 효과는 시험하지 않았다.

실패 시에도 `head-lines/head-lines.json`, `face-raw.png`, `mask-preview.png`, 상위 `status.json`과 로그를 남겼다. 통과한 것처럼 확인 SHA를 발급하거나 `all`로 자동 진행하지 않았다.

## CPU 검증

- 기존 R6 6개 seed: 스케치와 흰 배경 결과 모두 이전 파일 SHA 일치. GPU 생성 없이 기존 이미지만 사용했다.
- 관련 테스트: 최초 80개 통과. 시험 실패 시 상태 파일 기록 보강과 테스트 1개 추가 후 관련 81개를 다시 실행해 모두 통과했다(51.29초).
- 전체 CPU 회귀: 1,961개 통과(241.27초), 기존 Pillow 폐기 예정 경고 2건. 전체 검사 시작 이후 추가한 실패 기록 테스트는 위의 최종 관련 81개 검사에 포함해 검증했다.
- 테스트에는 `none` 불변, Canny 결정성·영역 밖 제거, 얼굴 제외, 누락·빈 마스크·70% 초과 중단, 입력 변경 거부, 확인 전 실행 거부, CONTOUR만 생성, 실패 후 다음 조건 중단을 포함한다.

로그:
- `outputs/hair-lineart-design-20261008/codex-tests-final.log`
- `outputs/hair-lineart-design-20261008/codex-full-pytest.log`
- `outputs/hair-lineart-design-20261008/codex-none-replay-01/result.json`
- `outputs/hair-lineart-design-20261008/codex-cpu-r6-01.log`

## 실행기 사용 순서

한 번의 준비·실행은 한 캐릭터를 대상으로 한다. R6는 seed 209212001·209212002, GUI는 완료된 4장 실행의 첫 seed 2개를 쓴다. 세 캐릭터를 모두 통과시키면 새 생성은 `3명 × hair/all × 2 seed = 12장`이다.

C0는 완료된 기존 CONTOUR 결과를 그대로 비교 자료로 사용한다. 각 C0의 프롬프트·얼굴·모델·seed는 변경하지 않는다. 따라서 기존 R6 의상을 새 후드 의상으로 자동 교체하지도 않는다. 캐릭터 사이 의상 조건이 다른 경우에는 전체 결과를 같은 의상 일반화 시험으로 해석하지 않는다.

CPU 준비 예시(기존 폴더는 덮어쓰지 않는다):

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\head_lines_trial.py' --studio-run '<완료된 GUI 실행 폴더>' --output '<새 CPU 미리보기 폴더>'
```

`--r6 --foreground-model 'D:\genai-cache\huggingface\models--skytnt--anime-seg\snapshots\493cb60893f47441b26ec4fb9a306bce9e342982\isnetis.onnx'`로 잠금 R6 입력도 받는다. **현재 R6는 이미 실패 원인을 확인했으므로 같은 조건을 그대로 반복할 필요가 없다.**

CPU 검사가 통과하면 머리카락 초록·얼굴 주황·선 자홍색으로 겹쳐 보여 준다. `hair-preview.png`, `all-preview.png`는 머리 선 확인용이고, 각 조건의 `preview_<seed>.png`는 최종 스케치까지 겹친 그림이다. 사용자가 특히 얼굴 마스크가 맞는지 먼저 확인한다.

GPU 실행은 확인된 준비 폴더를 별도 호출로 받는다. 미리보기 잠금 SHA를 명시하지 않으면 실행하지 않는다.

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\head_lines_trial.py' --prepared '<확인된 CPU 미리보기 폴더>' --review-sha '<확인한 preflight SHA>' --run --output '<새 GPU 결과 폴더>'
```

이 명령은 준비·사용자 확인이 통과한 뒤에만 사용한다. 실패 마스크는 확인으로 우회할 수 없다. 모델 강도 [1.2,0.5], 공통 0~10단계, VRAM 6.5 GiB 중단 기준과 기존 흰 배경 처리는 유지한다.

## Claude에 전달할 다음 판단

```text
머리 안쪽 선 옵션과 시험 실행기는 구현됐습니다. 다만 R6 CPU 검사에서 코 양성 점+머리 상자 SAM2가 얼굴 피부 대신 머리의 99.9%를 선택해 70% 기준으로 멈췄습니다. GPU 생성은 0장입니다.

먼저 codex-cpu-r6-01/head-lines/mask-preview.png와 head-lines.json을 확인해 주세요. 이 결과로 머리 선 생성 효과까지 실패했다고 판단하지 말고, 얼굴 피부 분리 입력의 실패로 구분해 주세요.

70% 기준을 올리거나 실패한 hair를 건너뛰어 all로 자동 진행하지 마세요. 피부만 선택할 수 있는 입력 방식의 수정안을 CPU 비교 조건으로 먼저 제안해 주세요. 코 점과 머리 전체 상자가 피부 의미를 강제하지 못했다는 것이 이번 관찰입니다. 새 분리 방식은 아직 구현하거나 시험하지 않았습니다.

라쿤·근육질 남성은 사용자가 확인한 GUI 비율 생성 폴더가 필요합니다. 준비가 통과한 캐릭터도 최종 겹침 그림을 사용자에게 보여 준 다음에만 GPU를 실행해 주세요.
```

다음 할 일은 얼굴 분리 방식의 CPU 보완 설계와 라쿤·근육질 남성의 완료된 GUI 실행 폴더 확보다. R6 실패 기록은 그대로 보존한다.
