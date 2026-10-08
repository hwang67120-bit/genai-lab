# 2단계 비율 생성 내부 연결과 GPU 재현 인수인계

내부 생성 경로를 연결했다. 기본 OFF이며 GUI는 연결하지 않았다. CPU 픽셀 대조에서 R6 6건의 스케치와 흰 배경 결과가 기존 시험과 SHA까지 일치했다. **새 내부 경로의 GPU 생성 재현은 아직 확인하지 않았다.**

## 입력부터 결과까지

기존 제품 진입점 `generate_onepass_request()`에 `proportion` 선택 인자를 추가했다. 생략하거나 `enabled=False`면 기존 단일 어댑터 생성 함수를 그대로 실행한다. 머리 자료나 새 모델은 필요하지 않다.

ON일 때는 다음 순서로 실행한다.

1. 확인된 머리 마스크·4px 윤곽, 정규화 이미지와 골격의 기준, 얼굴·골격 SHA, 로컬 모델 SHA를 검사한다. 이 경로는 골격이 있는 입력만 받는다. 자세 없는 입력을 임의로 골격 경로로 바꾸지 않는다.
2. 기존 SDXL 경로로 각 seed의 1단계 이미지를 만든 뒤 파이프라인을 해제한다.
3. CPU isnet-anime으로 각 이미지의 알파를 구한다. 알파 > 0.5 최대 덩어리의 바깥선을 4px로 그리고, 원본 머리 마스크를 6px 넓힌 영역 안의 기존 선을 지운 뒤 확인된 머리 윤곽을 합친다. CPU 세션을 해제한다.
4. 골격 1.2 + 스케치 0.5 MultiAdapter로 같은 seed·프롬프트·얼굴 참조를 사용해 다시 생성한다. 두 어댑터는 공통 0~10단계, IP는 기존 입력의 초기값에서 11단계부터 0.9로 전환한다. 파이프라인을 해제한다.
5. CPU isnet-anime 알파로 흰 배경에 합성한다. raw는 그대로 보존하고 별도 product.png를 만든다.

자동 재시도나 실패 시 BASE를 성공 결과로 반환하는 처리는 없다. reserved 6.5 GiB 초과, 일정 불일치, SHA 불일치, 빈 마스크, 취소 시 중단한다. 완료된 중간 이미지는 보존한다. 실패·취소와 실패한 단계를 status.json과 run.json에 남긴다.

## 머리 확인 계약

`prepare_head_outline(normalized_file, control_file, mask_file, directory, nose=..., neck=...)`는 **기존 골격과 같은 좌표의 머리 마스크**를 받아 4px 윤곽과 겹친 미리보기를 만든다. 자동 검출·자동 승인은 하지 않으며 반환값의 `confirmed`는 False다. 사람이 범위를 확인한 뒤에만 True로 바꿔 전달한다. 사각형 선택 UI와 분할 도구 연결은 다음 GUI 작업 범위다.

머리 윤곽은 머리카락을 포함한 확인 범위다. 등신 수 측정에서 사용하는 해부학적 머리 상자와 구분하며 귀·뿔 포함 여부를 시스템이 추측하지 않는다. 운영 함수에서는 좌표를 새로 변환하거나 윤곽을 임의로 확대하지 않는다.

## 설정과 호출

| 설정 | 기본값·제약 |
|---|---|
| `ProportionOptions.enabled` | False |
| `head` | None. ON은 확인된 HeadOutline 필수 |
| `sketch_root` | None. ON은 검증된 로컬 스케치 모델 경로 필수 |
| `foreground_model` | None. ON은 로컬 isnetis.onnx 경로 필수 |
| `foreground_sha256` | None. ON은 고정한 isnet-anime SHA와 일치해야 함 |
| `sketch_revision` | cc3c4e3362296c6825c370b83838306723ece983 |

어댑터 강도 [1.2, 0.5], 28단계, CFG 5.5, 736×1232, 공통 0~10단계, IP 전환과 6.5 GiB 한도는 검증된 구성으로 고정했다. 다른 설정이면 생성 전에 거부한다. 새 태그·체형 문구를 추가하지 않는다.

스케치 파일 SHA는 config `2f5ed7ae…`, fp16 가중치 `9f88c533…`이며 전체 값은 `proportion_inputs.py`에 있다. isnet-anime SHA는 `f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99`이다. 모델을 내려받거나 설치하는 코드는 없다.

호출 예(확인된 head와 로컬 경로를 준비한 뒤):

```python
from genai_lab.onepass_generation import generate_onepass_request
from genai_lab.proportion_inputs import ProportionOptions, FOREGROUND_SHA

options = ProportionOptions(
    enabled=True, head=confirmed_head, sketch_root=sketch_root,
    foreground_model=isnet_file, foreground_sha256=FOREGROUND_SHA,
)
batch = generate_onepass_request(request, new_directory, proportion=options)
```

OFF는 기존 `OnePassBatch`, ON은 `ProportionBatch`를 반환한다. ON의 각 후보는 `candidate.raw`에 2단계 raw 후보를, `candidate.path`에 흰 배경 결과를 둔다. 원본 SHA와 product SHA를 서로 대신 사용하지 않는다. 결과는 `proportion_unreviewed`, 자동 승인 불가 상태다. 기존 GUI의 raw 승인·저장 함수에 그대로 끼워 넣지 말고, GUI 승인 시 이 구분을 연결해야 한다.

## 기록과 읽기 시작점

- `genai_lab/proportion_generation.py::generate_proportion_batch`: 전체 처리 순서, 단계별 모델 해제, 실패 기록. 4개 seed 제품 요청과 6개 seed 재현이 같은 함수를 쓴다.
- `genai_lab/proportion_inputs.py`: 사람 확인 머리 입력, 모델 검증, 스케치·합성 규칙.
- `genai_lab/proportion_backend.py`: 실제 MultiAdapter 호출과 스텝별 일정·VRAM 검사.
- `genai_lab/proportion_foreground.py`: 오프라인 CPU 전경 추론. 시험과 동일한 입력 크기·BGR·알파 복원 규칙.
- `scripts/verify_proportion_product.py`: 기존 R6 시험 자료를 제품 함수로 재현하는 도구. 기본은 사전 검사만, `--run`을 명시해야 GPU 생성한다.

새 실행 폴더에는 preflight.json과 SHA, status.json, run.json, BASE_<seed>/raw.png, sketch_<seed>.png, maps.json, CONTOUR_<seed>/raw.png, seed-<seed>/product.png를 남긴다. 각 2단계 run.json에는 실제 두 모델·강도·적용 단계·스케치 SHA·어댑터 호출 횟수·VRAM이 기록된다. 사전 기록에는 입력·모델·코드 SHA를 포함한다. 기존 폴더는 덮어쓰지 않는다.

## CPU 확인 결과

- 관련 회귀 검사: 169개 통과 후 구식 callback·머리 미리보기 미승인·2단계 SHA 실패 테스트를 추가했다. 최종 전체 테스트는 **1,810개 통과**, 188.11초였다. 기존 Pillow 사용 폐기 예정 경고 2건이 있었고 실패는 없었다.
- 실제 CPU isnet-anime 처리: 기존 BASE로 다시 만든 스케치 6/6 SHA 일치, 기존 gpu-05 raw를 흰 배경으로 처리한 결과 6/6 SHA 일치. 18.48초, GPU 생성 0장.
- [CPU 픽셀 대조 기록](../outputs/proportion-product-verify-20261008/cpu-01/result.json), [관련 회귀 검사](../outputs/proportion-product-cpu-20261008-final.log), [전체 회귀 검사](../outputs/proportion-product-full-20261008.log).

커밋·푸시는 구현 검증 후 사용자의 별도 요청(2026-10-08)에 따라 진행한다. 함께 검증한 인식·꼬리 분석 수정도 포함한다. 자료 폴더와 잠금 시험 자료는 수정하지 않았다.

## Claude GPU 재현 순서

기존 gpu-05와 모든 잠금 자료는 그대로 둔다. 새 폴더에서 **6 seed × BASE·CONTOUR = 12장**을 생성한다. 이전 BASE를 그대로 가져다 쓰는 우회가 아니라 제품의 처음부터 끝까지 연결을 확인하기 위해서다. CPU 검사 도구는 기존 결과를 읽었지만 GPU 재현 도구는 BASE도 새로 생성한다.

- seed 순서: 209212003, 209212001, 209212002, 209212004, 209212005, 209212006.
- BASE는 gpu-02/gpu-04 raw SHA, 스케치는 잠금 maps.json SHA, CONTOUR는 gpu-05 raw SHA와 각각 대조한다.
- 하나라도 다르거나 VRAM·일정 오류가 나면 즉시 중단한다. 자동 재시도·강도 조정·기준 갱신을 하지 않는다.
- raw 재현과 별도로 흰 배경 결과·얼굴·의상·바지 아랫단 색을 기존 gen4와 비교한다. raw SHA가 같아도 기존의 색 문제를 해결했다는 뜻은 아니다.
- 먼저 다른 GPU 작업과 모델이 해제된 상태인지 확인한다. 아래 명령은 **Claude 실행용이며 Codex는 실행하지 않았다.** `--run`을 빼면 파일·모델 사전 검사만 한다. 사전 검사와 생성은 서로 다른 새 출력 폴더를 사용한다.

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\verify_proportion_product.py' --run --foreground-model 'D:\genai-cache\huggingface\models--skytnt--anime-seg\snapshots\493cb60893f47441b26ec4fb9a306bce9e342982\isnetis.onnx' --output '\\192.168.0.109\win_g\genai-lab\outputs\proportion-product-verify-20261008\gpu-01'
```

남은 한계: GPU 운영 재현 미검증, R6 한 캐릭터만 품질 근거가 있음, 바지 아랫단 색·머리 테두리 문제 유지, 생성 시간 약 2배. GUI 연결·자동 머리 검출·자세 UI·금지 조합 생성은 이번 작업에 포함하지 않았다.
