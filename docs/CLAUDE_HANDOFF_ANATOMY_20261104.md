# Claude 인계: 그림체·인체 생성과 자동 보정 현황

작성일: 2026-11-04. 상태: CPU 보강과 회귀 검증 완료, 인체 생성 오류 해결은 미완료.

## 먼저 읽을 결론

사용자는 동일성·그림체가 유지되는 결과를 확인했지만, 일부 입력과 시드에서 팔 중복·팔 잘림·잘못된 손가락이 계속 발생한다고 판정했다. 최근 E32/E35의 최초 생성과 보정 결과도 불합격이다. 이 판정을 자동 검사 결과로 덮어쓰면 안 된다.

최근 보강은 **보정 후 손이 두 개 검출됐다는 이유로 정상 복구라고 자동 채택하던 경로를 막는 작업**이다. 정상 인체를 생성하는 방법이나 손가락 복구 모델을 새로 구현한 것이 아니다. 최신 코드로 새 GPU 생성은 아직 하지 않았다.

2026-11-04 인계 전 현재 소스를 CPU로 검사하여 **196개 시험과 하위 시험 7개 통과, 종료 코드 0**을 확인했다. 이는 코드 계약·회귀 검사이며 이미지 품질 합격률이 아니다.

## 사용자 목표와 유지할 결정

- 목표는 원본 캐릭터성, 사용자가 승인한 그림체, 지정 의상, 정상 인체를 함께 유지하는 제품이다. 속도·메모리 최적화보다 이 조건의 실제 앱 검증이 먼저다.
- 원본 상반신에 보이지 않는 하체를 억지로 확장하지 않는다. 원본에서 보이는 범위와 요청 구도를 보존한다.
- 실패·중간·탈락 이미지도 보여 준다. 성공한 시드만 골라 일반적인 성공처럼 보고하지 않는다.
- 동일성, 그림체, 의상, 인체, 오염, 구도는 구분해서 판단한다. 자동 저장·검출 통과·정상 종료는 사용자 품질 승인이 아니다.
- 사용자 거절은 후보 SHA·시드·사유로 기록한다. 같은 시드로 조건을 바꾸어 다시 생성하되, 거절한 결과를 정답 또는 새 참조 이미지로 사용하지 않는다.
- 사용자는 원격 UI 조작의 토큰 비용 때문에 클라우드 실행을 직접 하는 방식을 선호한다. 새 유료 실행의 비용·시간·중단 조건은 별도로 확인한다. 과거 잔액을 현재 잔액으로 쓰지 않는다.
- 모든 코드 작성·수정은 Linux에서 한다. 프로젝트 저장·문서·테스트는 Windows D:다. 한국어로 보고한다.

## 실행 환경과 저장 위치

| 용도 | 확인한 위치 |
|---|---|
| Windows 저장소 | `D:/Code/genai-lab` |
| Linux 작업 경로 | `/home/mycom/win_g/genai-lab` |
| Linux 실제 공유 경로 | `/home/mycom/win_g/windows-code/genai-lab` |
| Linux 호스트 | `mycom-14U530-MFB3L`, SSH 별칭 `mycom-linux` |
| Windows 기본 시험 Python | `D:/genai-cache/venv/Scripts/python.exe` |
| CPU 자세 검출 환경 | `D:/genai-cache/catvton-venv/Scripts/python.exe` |
| 결과 기록 | `D:/Code/genai-lab/outputs/style-training-prep-20261025/` |

Linux와 D: 경로는 같은 저장소다. 서로 독립된 백업이 아니다. 이전 UNC 작업 경로만 보고 Linux 로컬 저장소로 오인하지 않는다. 이번 작업 시작 시 Windows OS와 Linux 연결·경로를 확인했다.

## 가장 중요한 모델 구분

사용자가 최종 그림체 **V1**으로 지정한 파일과 현재 앱 설정/최근 E32·E35 실행 파일이 서로 다르다.

| 구분 | 파일 및 SHA256 |
|---|---|
| 사용자 지정 V1 최종본 | `outputs/style-training-prep-20261025/flux-style-lora-s26-20261101/final-pytorch_lora_weights.safetensors` / `5c13c86be80bad7c063cc6a23f50b7ef1f51a49bcfb5de554591199157e9c67b` |
| 현재 앱 설정 및 E32/E35 시험용 step500 | `D:/genai-cache/models/flux-style-lora-gnlbq4-s500/pytorch_lora_weights.safetensors` / `2403f169d5629b580070012a5c9b8bafba8bc8aaaef8c912dff749eed336a523` |
| V1에 자세 200단계를 추가한 검증 후보 | SHA `5725111d972182ea0d73d75d877c9f34e6aed7976875e08d34eaa7d67e6cec56` |

V1과 step500은 각각 66,866,416바이트·120개 텐서지만 값이 다르다. 현재 `configs/flux_model_files.json`을 V1로 임의 변경하지 않았다. 최근 코드 수정 효과를 분리하려고 E32/E35 재생성에서는 기존 step500 조건을 유지했다. 따라서 이를 V1 생성 검증이라고 부르면 안 된다.

기본 모델 경로는 `D:/genai-cache/models/FLUX.2-klein-4B-e7b7dc27`이다. 다음 생성 전에는 사용자 지정 V1과 실제 로드 경로·SHA를 먼저 일치시켜야 한다. 과거 step500 비교와 V1 비교는 서로 다른 실험으로 보존한다.

## 지금까지의 실제 시험과 해석

| 시험 | 확인한 결과 | 해석과 제한 |
|---|---|---|
| 긴 지시 PC 앱 검증 | 6조건·18후보, 약 29분 36초, 종료 0. 실제 토큰 261~312개 전달 확인 | 긴 지시 잘림 문제의 해당 범위 검증. 이후 사용자가 인체 결함을 지적했으므로 품질 합격으로 계산하지 않는다 |
| 자세 기준 시험 | 개발 32조건, 시드 42·43·44 | 사용자의 단계 진행 승인과 모든 개별 이미지 합격을 동일시하지 않는다 |
| 단계 3 CPU 재검사 | 96쌍의 이전/이후 기록 대조 | 손 검출이 정상처럼 보여도 실제 세 번째 팔을 놓친 사례가 있다 |
| V1 자세 추가 학습 | 200단계, 120텐서 모두 finite, V1과 같은 키·120텐서 변화, runner exit 0 | 학습 완료는 확인. 인체 개선 성공은 아님 |
| V1 대 자세 추가본 | 6입력 × 2시드 × 2가중치 = 24장, 12쌍 | 그림체는 대체로 유지됐으나 기존 시드43 팔 중복 지속. 사용자는 시드42가 상대적으로 양호하다고 관찰 |
| V1 강도 비교 | 6입력 × 2시드 × 강도 0.6/0.8/1.0 = 36장 | 시드43 팔 중복은 세 강도 모두 지속. 강도만 낮추면 해결된다는 가설은 지지되지 않음 |
| E32/E35 수정 후 재생성 | E32 seed44, E35 seed43, 실행 종료 0 | 최초·보정 결과 모두 사용자가 손가락/팔/손 결함으로 불합격 판정 |
| 최근 보정 전후 CPU 보강 | 저장 기록 E32 자동 채택 가능→보류, E35 보류 유지 | 실패 결과의 자동 승격을 막음. 새 생성 또는 자동 복구 효과는 미검증 |

V1 비교는 참조·문장·시드가 같은 쌍에서 가중치만 변경했고 자동 보정·얼굴 합성·색 보정은 제외했다. 전체 앱 최종 품질 시험과 구분해야 한다. 강도 1.0의 12장은 이전 V1 비교 결과와 PNG SHA까지 일치했다.

V1 비교 24장은 약 12분 4초, 강도 비교 36장은 약 14분 38초였다. 이는 해당 PC·조건의 실측이며 미래 실행 시간 보장이 아니다. PyTorch reserved 최고는 각각 6.171875GiB, 6.115234375GiB였다. 장치 전체 메모리나 메모리 누수 부재를 뜻하지 않는다.

긴 지시 시험 reserved 6.123047GiB는 로딩 후 초기화된 생성 구간 값이다. 로딩을 포함한 전체 최고값으로 쓰지 않는다. 설정 상한 6.5GiB와 실측을 혼동하지 않는다.

## 현재 제품 흐름과 읽을 코드

주 흐름은 `genai_lab/style_redraw_worker.py`의 `generate_flux`에서 읽는다.

1. 원본 캐릭터·선택 의상·특징·장면·보이는 범위를 받아 생성 문장을 구성한다.
2. `flux_prompt_delivery.py`가 길이와 실제 임베딩 전달을 확인한다.
3. FLUX에 원본 참조 이미지와 선택 의상 참조, 텍스트 조건을 전달해 최초 후보를 생성한다.
4. 특징·의상 자동 보정과 선택적인 수인 복원·얼굴·색 보정에서 제안 이미지를 만든다.
5. 최초 및 보정 이미지의 인체 근거와 보정 전후 차이를 검사한다. 미확인 결과를 자동 합격으로 올리지 않는다.
6. 원본 후보·중간·보류 제안·최종 후보와 사유를 보존하고, 사용자에게 선택·거절을 받는다.

| 파일 | 책임 |
|---|---|
| `genai_lab/flux_prompt_delivery.py` | 전체 문장 조건 생성·길이/형태/유한값 검사 |
| `genai_lab/style_redraw_prompt.py` | 특징·의상·장면·프레이밍 문장 구성 |
| `genai_lab/style_redraw_quality.py` | 특징·의상과 보이는 영역의 검사 계약 |
| `genai_lab/style_redraw_anatomy.py` | 인체 근거 집계·검토/자동 채택 가능 상태 구분 |
| `genai_lab/style_redraw_pose_evidence.py` | 로컬 CPU 자세 검출 근거 |
| `genai_lab/style_redraw_arm_evidence.py` | 팔 관절·손 대응·화면 경계 근거 |
| `genai_lab/style_redraw_correction_evidence.py` | 보정 전후 손/관절 위치·수 변화 비교 |
| `genai_lab/style_redraw_pipeline.py` | 사용자 거절 후 원본 참조·같은 시드 재시도 흐름 |
| `genai_lab/studio_product.py` | 거절 사유 등 제품 입력 |
| `genai_lab/style_redraw_results.py` | 검토 필요·자동 검사 한계 표시 |
| `scripts/product_flow_run.py` | 실제 제품 경로 시험 진입점 |

현재 FLUX 호출에는 **골격 지도나 관절 좌표를 실제 생성 조건으로 주입하는 연결이 없다**. 별도 자세 모듈이 존재한다는 이유만으로 현재 생성에 골격 제어가 적용됐다고 설명하면 안 된다.

### 긴 지시 전달

256토큰 이하는 기존 경로를 유지하고, 257~512는 전체 문장을 인코딩하여 `prompt_embeds`로 전달한다. 끊어진 문자열을 뒤에 붙이는 방식이나 문장 조각별 생성이 아니다. `[1,N,7680]` 임베딩, `[1,N,4]` 텍스트 위치와 유한값을 검사한다. 512 초과는 조용히 잘라 생성하지 않고 실패로 기록한다. 실제 GPU 증거는 261~312 범위이며 모든 512토큰 조건의 품질을 보장하지 않는다. `flux_img2img.py` 경로도 같은 전달 함수를 쓴다.

### 사용자 거절과 자동 학습의 구분

`retry_rejected_candidate`는 거절 사유와 후보 SHA·시드를 보존하고 원본 캐릭터·의상·마스크를 사용한다. 같은 시드에서 보정 조건을 바꾸는 재생성이지, 사용자 거절마다 모델 가중치를 온라인 학습하는 구현이 아니다. 동일 출력 SHA를 새 성공으로 취급하지 않는다.

## 최근 보강의 원인·내용·한계

E32 최초 결과는 손 1개에 손목 2개가 대응되어 미확인이었다. 의상 보정 후 손 2개가 다른 위치에 검출되자 이전 로직은 제안 이미지의 검사만 보고 자동 채택했다. 검출 수 증가가 손가락·팔 복구의 증거가 아니었던 것이 확인된 결함이다. E35는 기존에도 경계 위험 등으로 보류됐지만 이미지 자체는 여전히 잘못됐다.

`compare_correction_anatomy`와 `guard_correction_anatomy`를 추가해 의상/특징, 수인, 얼굴, 색 보정 네 경로에서 이전 후보와 제안을 함께 검사한다.

- 기존 인체 미확인, 손 수 변화, 손 위치의 일대일 대응 변화, 어깨·팔꿈치·손목 이동을 각각 기록한다.
- 좌표 누락·비정상 수치·이미지 크기 불일치는 미검증으로 남긴다. 손 검출 순서만 바뀐 경우는 움직임으로 오판하지 않는다.
- 이동 기준은 캔버스 대각선 4%다. 학습으로 확정한 정확도 기준이 아니며 정상 변화도 보류할 수 있다.
- 경계 검사는 손목·손 상자에 팔꿈치를 추가했다. 경계 접근은 잘림 의심이지 실제 잘림 확정이 아니다.
- 보정이 보류되어 이전 파일을 유지했더라도 성공으로 승격하지 않는다. 최초 이미지 역시 정답이 아니라 비교 대상이다.
- 손가락 형태 및 여분 팔의 부재를 검증하지 못한다는 한계를 사용자 화면에 남긴다.

저장 이미지 36개의 자기 비교에서 위치 변화 오판은 0건이었다. 기존 자동 채택 가능 15건은 15건으로 유지됐다. 이는 같은 입력에 대한 대조 시험이지 36장의 인체 합격이나 새 이미지 일반화 성능이 아니다.

전체 화면과 겹친 4개 영역으로 확대 검출을 시험했지만, E32 형태 결함을 구별하지 못하고 E01 배경 손 오검출이 생겼다. 따라서 제품 판정에는 넣지 않았다. 단순 확대 검출이나 금지 문장 증가만으로 해결했다고 주장하지 않는다.

이번 보강에서는 원본·의상·장면·시드·해상도·생성 프롬프트·모델 설정을 변경하지 않았다. 상반신을 전신으로 확장하는 변경도 없다. 이번 인계 작업에서도 GPU/클라우드를 실행하지 않았다.

## CPU 검증과 복구 근거

인계 전 GPU 비활성 및 허브 오프라인 환경에서 아래 시험을 실행했다. 196개 및 하위 시험 7개 통과, 31.44초, 종료 0.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
& D:/genai-cache/venv/Scripts/python.exe -B -X utf8 -m pytest tests/test_identity_autofix.py tests/test_studio_product.py tests/test_style_part_combine.py tests/test_style_redraw_flux.py tests/test_style_redraw_pipeline.py tests/test_flux_prompt_delivery.py tests/test_garment_delivery_audit.py tests/test_garment_framing_resume.py tests/test_style_redraw_anatomy.py tests/test_style_redraw_arm_evidence.py tests/test_style_redraw_correction_evidence.py tests/test_style_redraw_quality.py tests/test_style_redraw_inputs.py tests/test_face_detail.py -q
```

명령은 저장소 루트에서 실행한다. 기존 환경 변수가 있다면 시험 후 복원한다. 현재 검증 로그는 `outputs/style-training-prep-20261025/handoff-commit-20261104/cpu-tests.log`, 종료 기록은 같은 폴더의 `cpu-test-exit.json`이다. 원격에 포함한 작은 검증 기록은 `docs/evidence/ANATOMY_CPU_20261104.json`이다.

최근 수정 전 코드 사본과 전후 SHA·패치는 `anatomy-detail-cpu-20261104/source-before/`, `source-before.json`, `source-after.json`, `changes.patch`에 있다. 다른 작업까지 덮어쓰는 전체 되돌리기 대신 관련 변경만 비교해 복구한다.

## 다음 담당자가 진행할 순서

1. V1 최종본과 앱 실제 로드 파일의 차이를 먼저 정리한다. 사용자 승인 V1과 시험 조건을 명시하고, 이전 step500 기록은 그대로 보존한다.
2. 현재 FLUX에 실제로 연결 가능한 공간 조건 방식과 국소 손·팔 복구 방식을 조사한다. 텍스트만 바꾸는 방법과 실제 관절 제어를 구별한다. 기존 모델이 그 입력을 받는지 CPU 계약·배선부터 확인한다.
3. 자동 복구를 구현한다면 고칠 영역, 유지할 얼굴/의상/체형/크롭, 실패 반환 상태를 먼저 명시한다. 검출 근거만으로 성공 처리하지 말고 보정 전후 비교·사용자 거절 기록을 연결한다.
4. GPU 비교는 한 조건만 바꾸고 원본·의상·장면·해상도·시드·모델 SHA를 고정한다. 시드42뿐 아니라 실패 시드43 및 E32 seed44를 포함한다. 시간/메모리/실패 중단 기준을 정한 뒤 실행한다.
5. 남성·여성, 짧은/긴 체형, 전신·상반신, 겹침/가림 입력을 나누어 평가한다. 이미 조정에 쓴 입력은 미사용 검증 자료라고 부르지 않는다. 실패 이미지도 비교판에 남긴다.
6. 사용자에게 동일성·그림체·의상·인체·오염·구도를 별도로 판정받는다. 개선이 없으면 수치 반복이나 유료 재학습을 자동으로 이어가지 않는다.

생성 오류의 단일 원인은 아직 확정되지 않았다. 시드가 달라 오류가 발생한다는 관찰은 조건과 확률적 생성의 취약성을 보여 주지만, 시드43 자체가 원인이라는 설명은 아니다. 자세 학습 200단계와 강도 조절에서 개선되지 않은 반대 증거를 유지한다.

## 메모리·속도 최적화의 위치

`docs/MEMORY_OPTIMIZATION_PLAN.md`에 단계별 시간·메모리를 먼저 측정하고 병목별로 Python/네이티브/GPU/Rust 적용 여부를 결정하는 계획이 있다. 현재 주요 행렬 연산을 순수 Python이 계산한다고 가정하지 않는다. Rust 전환만으로 VRAM 전송·어텐션·GPU 연산 병목을 해결했다고 볼 수 없다. 메모리 예약 캐시와 실제 누수도 구별한다. 이 인계에서 Rust 전환이나 성능 최적화를 실행하지 않았다.

## 로컬 원자료 안내

아래 경로는 모두 `D:/Code/genai-lab/outputs/style-training-prep-20261025/` 아래이며 Git 원격에 이미지·모델을 포함하지 않는다.

| 폴더 | 읽을 기록 |
|---|---|
| `flux-full-prompt-gpu-20261103/` | `REPORT.md`, GPU 조건 증거, 이후 사용자 판정 |
| `pose-baseline-pc-20261103/` | 기준 실행·최초/중간/최종 이미지 |
| `pose-stage3-cpu-20261103/` | `result-ko.md`, CPU 재검사 |
| `pose-v1-20261104/cloud-run-20261104/` | `verification.json`, 학습 결과·회수 로그 |
| `pose-v1-20261104/pc-ab/` | `assessment.ko.txt`, 24장 비교·명세 |
| `pose-v1-20261104/pc-strength/` | `assessment.ko.txt`, 36장 강도 비교 |
| `anatomy-independent-cpu-20261104/` | 36개 검출 기록, 일반화 합격 아님 |
| `arm-framing-fix-20261104/` | `result-report.ko.txt`, `lora-comparison.json`, `user-review-20261104-220331.json`, `review/E32-comparison.jpg`, `review/E35-comparison.jpg` |
| `anatomy-detail-cpu-20261104/` | `result-report.ko.txt`, `cpu-replay.json`, `multiview-probe.json`, 전후 소스·패치 |
| `handoff-commit-20261104/` | 이번 시험 로그·커밋/푸시 증거 |

최근 보강 보고서 SHA256: `b2d253b7326589db5157532672bc9c18a7ff40ef7b96e93c4e463b60212bcb3d`.
E32/E35 사용자 불합격 기록 SHA256: `a676d0dc51aeef8bdaebac26fb99863faca6d74c8086dd3d564c8562307c989c`.

## 이번 저장소 인계 범위

브랜치는 `codex/onepass-implementation`, 원격은 `https://github.com/hwang67120-bit/genai-lab.git`이다. 작업 시작 HEAD는 `54c8b9ef748c3e55ad4f4514a15cc68ea2e276ae`, 확인한 원격 HEAD는 `fa30de73a61385d53756f426fa58308825dc52f0`이며 로컬에 기존 커밋 46개가 앞서 있었다. 이력 분기 없이 같은 브랜치에 일반 푸시한다. main 병합이나 강제 푸시는 하지 않는다.

이번 커밋은 미커밋 상태였던 긴 지시 전달·의상/거절 처리·인체 검사·회귀 시험·관련 문서를 함께 보존한다. `scripts/style_checks/two_pass_garment_s58.py`는 이전 실험용 도구이며 현재 기본 자동 보정으로 실행하지 않는다. `docs/STORAGE_RECOVERY_20261016.md`와 `docs/SYNTHETIC_STYLE_CLOUD_TRAINING_PLAN.md`는 당시 기록이다. 과거의 준비 상태나 저장 위치를 현재 사실로 적용하지 않는다.

참조 이미지, 학습 후보, 승인 원본, 모델 가중치, `character.review.json`, 생성 원자료와 임시 파일은 이번 푸시에서 제외하고 로컬에 보존한다. 따라서 코드 푸시가 모델·학습 자료까지 백업했다는 뜻은 아니다. 최종 커밋과 원격 SHA 확인 결과는 로컬 `handoff-commit-20261104/push-verification.json`에서 확인한다.

사용자는 이 문서를 Claude에게 전달하면 된다. 다음 GPU 실행은 이 문서만으로 자동 시작하지 않는다.
