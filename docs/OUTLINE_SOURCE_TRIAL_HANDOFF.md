# 원본 몸 윤곽 시험 실행기 인수인계

원본 정규화 이미지의 바깥선을 전달하는 시험 옵션을 추가했다. 기본값은 기존 `base`이며 GUI에는 선택지를 추가하지 않았다. **GPU 생성은 실행하지 않았고, 새 방식의 체형·의상 보존 효과는 아직 모른다.**

## 이번 비교에서 달라지는 것

| 항목 | 기존 base | 시험 original |
|---|---|---|
| 몸 바깥선 출처 | 각 seed의 1단계 생성 결과 | 원본 정규화 이미지 |
| 스케치 준비 | seed마다 별도 그림 | CPU로 한 번 만든 그림을 모든 seed에 재사용 |
| 1단계 생성 | 기존대로 실행 | 결과를 사용하지 않으므로 생략 |
| 머리 윤곽·좌표 | 사용자가 확인한 기존 윤곽 | 동일 |
| 골격·얼굴·문장·seed | 기존 입력 | 비교할 실행에서 그대로 복원 |
| 어댑터·일정 | 골격 1.2 + 스케치 0.5, 공통 0~10단계 | 동일 |
| 최종 배경 처리 | 기존 CPU isnet-anime 흰 배경 합성 | 동일 |

1단계 생략은 요청서에서 Codex 판단에 맡긴 사항이다. 원본 모드에서는 1단계 결과가 제어 이미지·문장·얼굴 참조에 사용되지 않으므로 불필요한 생성을 생략했다. 각 최종 생성은 기존처럼 해당 seed의 난수 생성기를 사용한다. `base_stage=skipped_unused_outline`과 빈 BASE 완료 목록으로 구분한다. 재현 결과는 Claude의 GPU 시험으로 확인해야 한다.

`make_sketch`의 선 굵기·알파 문턱·최대 덩어리 선택·머리 주변 6px 삭제 규칙은 수정하지 않았다. 원본 옷의 외곽선도 함께 들어간다. 이는 옷이 따라오는지를 재려는 이번 시험의 조건이며 체형만 분리한 지도가 아니다.

## 읽기 시작점과 변경 파일

- `genai_lab/proportion_inputs.py`: ProportionOptions.outline_source 추가. base가 기본이며 base/original 외 값은 거부한다.
- `genai_lab/proportion_generation.py`: generate_proportion_batch의 원본 모드 분기, build_original_maps의 CPU 단일 추론·공유 지도, 출처와 생략 기록.
- `genai_lab/proportion_backend.py`: 생성 설정 변경 없이 실제 윤곽 출처·원본 SHA를 생성 기록에 추가.
- `scripts/body_outline_source_trial.py`: R6 잠금 입력 또는 완료된 GUI 실행 폴더를 읽는 시험 전용 실행기. 기본 동작은 CPU 사전 검사다.
- `tests/test_outline_source_trial.py`: 두 모드 픽셀 대조, 원본 단일 추론·1단계 미호출, 파일 변경·미완료 거부, GUI 입력 복원, 명시적 GPU 옵션 검증.

기존 시험 실행기의 보호 장치는 수정하지 않았다. 잠금 자료·판정 기준·자료 폴더도 바꾸지 않았다. 커밋·푸시는 별도 확인 전까지 하지 않는다.

## 실행기 입력 계약

**R6:** 기존 잠금된 BASE 계획에서 문장·토큰 청크·얼굴·골격을 그대로 읽는다. 머리 확인 및 모델 다운로드 기록의 SHA를 검사한다. seed는 209212001, 209212002, 209212003이다. 원본 윤곽은 CPU로 다시 계산하며, 잠금 SHA `a253333d1e3e7f89d60987d927958c48cd1aabe5ea3286a4e9f6196dd9f32f56`와 일치해야 한다.

**GUI 실행 폴더:** `--studio-run`에는 `generation/`이 아니라 그 부모인 실행 식별자 폴더를 지정한다. 사용자가 원본 비율 참고를 켜고 완료한 4장 묶음만 받는다. 기존 preflight 잠금, run 완료 상태, 입력 승인, 사용자 머리 확인, 원본·사본·얼굴·골격·머리·스케치·생성 결과 파일 SHA를 검사한다. 실제 문장과 토큰 청크·얼굴 일정·모델 경로·seed를 복원하고 윤곽 출처만 original로 바꾼다. 새로운 태깅·분석·문장 조립·seed 추첨은 하지 않는다. 원본 파일이 이동하거나 바뀌어도 비교 조건 확인이 안 되므로 중단한다.

기존 잠금 자료의 옛 코드 해시를 현재 코드로 덮어쓰지 않는다. 새 실행 기록에 현재 제품 코드 해시와 읽은 자료 해시를 별도로 남긴다.

## CPU 확인 결과

- 관련 테스트 65건 통과. 첫 검사에서 얼굴 파일 변경은 정상 거부됐으나 테스트가 기존 예외 종류를 예상하지 못한 1건이 있었다. 기존 오류 처리는 유지하고 테스트의 예상 예외를 수정했다.
- R6 원본 스케치: 실제 CPU isnet-anime으로 계산한 SHA가 잠금 SHA와 일치. 한 번 만든 그림을 3개 seed가 공유한다. [사전 검사](../outputs/body-outline-test-20261008/codex-preflight-r6-01/trial-preflight.json).
- 기존 방식 회귀: 저장된 BASE에서 재계산한 스케치 6/6, 저장된 CONTOUR에서 재계산한 흰 배경 6/6이 이전 결과와 SHA 일치. **새 GPU 생성 결과의 SHA를 확인한 것은 아니다.** [CPU 대조](../outputs/body-outline-test-20261008/codex-base-regression-01/result.json).
- GUI 실행 복원은 가짜 생성기를 사용하는 CPU 테스트로 확인했다. 아직 늑대의 실제 원본 비율 GUI 실행 폴더가 없어 해당 폴더의 실제 사전 검사는 하지 않았다.
- 전체 CPU 회귀 테스트 **1,873개 통과**, 171.59초. 기존 Pillow 사용 폐기 예정 경고 2건 외 실패 없음. [전체 테스트 기록](../outputs/body-outline-test-20261008/codex-full-tests.log).

## Claude 실행 순서

다른 GPU 작업이 종료된 상태에서 R6 B 3장, 늑대 B 4장을 실행한다. 아래 명령은 GPU 실행용이다. **사전 검사만 하려면 `--run`을 빼고 다른 새 출력 폴더를 지정한다.** 사전 검사에서도 CPU isnet 추론은 수행한다. 기존 폴더를 덮어쓰거나 자동 재시도하지 않는다.

R6 B 3장:

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\body_outline_source_trial.py' --r6 --foreground-model 'D:\genai-cache\huggingface\models--skytnt--anime-seg\snapshots\493cb60893f47441b26ec4fb9a306bce9e342982\isnetis.onnx' --output '\\192.168.0.109\win_g\genai-lab\outputs\body-outline-test-20261008\r6-original-gpu-01' --run
```

늑대 B 4장: 먼저 사용자가 GUI에서 늑대 + 빨간 수영 반바지로 원본 비율 참고를 켜고 A 4장을 완료한다. 아래의 `<실행ID>`만 해당 폴더명으로 바꾼다. 전경 모델 경로는 A 기록을 그대로 사용하므로 별도로 넘기지 않는다.

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\body_outline_source_trial.py' --studio-run '\\192.168.0.109\win_g\genai-lab\outputs\studio-runs\<실행ID>' --output '\\192.168.0.109\win_g\genai-lab\outputs\body-outline-test-20261008\wolf-original-gpu-01' --run
```

출력의 `trial-preflight.json`과 SHA에는 복원 입력·출처·CPU 지도·모델·코드·자료 해시를 남긴다. GPU 실행은 그 아래 `generation/`의 기존 제품 경로를 사용한다. 원본 그림은 `CONTOUR_<seed>/raw.png`, 흰 배경 최종 결과는 `seed-<seed>/product.png`다. `run.json`과 `maps.json`에서 원본 정규화 SHA·스케치 SHA·출처를 확인할 수 있다. `trial-status.json`의 GPU 요청 여부와 실제 생성 완료 목록은 구분한다.

**중단:** 입력·모델·지도 SHA 불일치, 머리 좌표 오류, 빈 마스크, 적용 일정 불일치, reserved 6.5 GiB 초과, 생성 실패·취소는 기존 규칙대로 중단한다. 완료된 중간 파일은 남기며 BASE를 성공 결과로 대신 반환하지 않는다.

**판정:** 잠금된 criteria.md를 그대로 사용한다. 늑대 체형 개선 3/4 이상만으로 채택하지 않고, 의상 실루엣 전이·요청 의상 적용·비율 오차 조건까지 함께 확인한다. 이번 구현은 원본 윤곽 전달 수단이며 체형 개선 성공을 의미하지 않는다.
