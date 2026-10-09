# 얼굴 참조 마무리 블록 오프로드와 GPU 재시험

얼굴 참조 ON 마무리 경로에 블록 단위 오프로드를 구현했다. CPU 검증과 GPU 재시험 준비까지 수행했으며, 실제 GPU 메모리·시간·품질은 아직 확인하지 않았다. GPU는 Claude가 새 `product-ip-03` 폴더에서 실행한다.

## 확인한 현상과 변경 이유

직전 `product-ip-02`는 첫 고해상도 재처리에서 reserved 7,470,055,424 bytes(약 6.96 GiB), allocated 6,946,447,872 bytes(약 6.47 GiB)를 기록하고 멈췄다. 얼굴 참조 사전 계산과 효율적 어텐션만으로는 reserved 6.5 GiB 한도를 통과하지 못했다.

IP 가중치의 GPU 상주가 기여했을 가능성은 있지만, 이 수치만으로 초과분 전부를 IP 가중치 탓이라고 확정하지 않는다. 이번에는 UNet 전체를 한꺼번에 올리는 방식에서 필요한 블록을 올렸다가 내리는 방식으로 바꾼다. 효과는 동일한 원본 16장의 재시험에서 판단한다.

## 적용 범위와 처리 흐름

얼굴 참조를 켰을 때만 적용한다. 기본값, GUI, 생성 단계, 모델 파일, 프롬프트는 바꾸지 않았다. 끈 경로는 기존 `enable_model_cpu_offload()` 호출을 그대로 사용한다.

**텍스트·얼굴 정보 준비 → 계산 모델 CPU 이동 → VAE 입력 처리 → VAE CPU 이동 → UNet 블록별 계산·CPU 이동 → VAE 출력 처리 → 남은 모델 CPU 이동** 순서다.

| 대상 | 적용 방식 |
| --- | --- |
| UNet | Diffusers `block_level`, `num_blocks_per_group=1` |
| IP의 키·값 층 | 소속 down/up 블록과 동반 이동. mid 블록의 IP는 기본 UNet 루트 그룹에 포함 |
| 이미지 투영층 | 기본 UNet 루트 그룹과 이동. UNet 호출 동안 GPU에 있고 호출 뒤 CPU로 이동 |
| 텍스트·얼굴 인코더·VAE | Accelerate의 모델 단위 CPU 오프로드 |
| 스트림·더 잘게 나누는 방식 | 사용하지 않음. `use_stream=False`, `record_stream=False`, `non_blocking=False` |
| 실패 시 대응 | 기록 후 중단. 같은 프로세스에서 다른 방식으로 전환하거나 자동 재시도하지 않음 |

설치된 Diffusers에서 블록으로 나뉘지 않는 mid 블록·입출력 층·투영층은 루트 그룹으로 묶인다. 따라서 모든 가중치가 한 층씩만 GPU에 존재한다는 뜻은 아니다. 이 구성을 바꾸기 위한 추가 분할은 하지 않았다.

라이브러리가 마무리 종료 시 모델 단위 훅을 다시 설치하면 UNet의 그룹 훅과 충돌한다. ON에서는 텍스트·VAE 훅을 별도 객체가 관리하고 성공·실패 시 직접 내려보낸다. UNet에는 Accelerate 훅을 달지 않는다. VAE는 UNet 첫 계산 전에 내려보낸다.

IP 가중치가 그룹에 포함됐는지 파라미터별로 확인하며, 누락·중복이면 생성 전에 중단한다. 별도 GPU 상주 방식으로 자동 전환하지 않는다. 실행 중 IP 층의 장치·dtype·호출 수를 기록하고 각 단계 종료 시 UNet이 CPU로 내려갔는지도 확인한다.

## 유지한 품질·중단 조건

- 1.5배, 출력 1104×1848, 강도 0.35, 28단계, CFG 5.5, 원본 seed, IP 0.9.
- 얼굴 정보 사전 계산과 재사용, 기존 효율적 어텐션, VAE 타일링.
- 머리 상자·확대·합성 규칙, 흰 배경 처리, 원본 보존.
- reserved 6.5 GiB 초과 시 중단. 배율 축소·재시도 없음.
- OFF 결과의 기존 F1·F2 SHA 재현 계약 유지. 이번에는 GPU로 SHA를 다시 확인하지 않았다.

## 기록을 읽는 위치

`finishing-status.json`과 각 이미지의 `finishing.json`에 `offload`를 추가했다. 준비 실패도 실패 상태와 오프로드 설정을 보존한다. 이미지 생성 설정이 바뀌었다는 뜻은 아니다.

| 기록 | 의미 |
| --- | --- |
| `offload.groups` | 실제 그룹 위치와 가중치 바이트 수 |
| `offload.ip_membership` | 각 IP 가중치의 그룹·dtype·바이트 수 |
| `offload.ip_parameter_bytes` / `ip_resident_bytes` | IP 전체 가중치 크기 / 그룹 밖 별도 상주 가중치 크기(현재 0) |
| `stages.hires` / `stages.face` | 각 단계 최대 reserved·allocated, 계산 시간, UNet·IP 호출 수 |
| `ip_device_observations` | 실제 IP 층 실행 장치·dtype·호출 수 |
| `offload_release_seconds` | 단계 종료 후 모델 정리 시간 |
| `seconds_including_release` | 모델 정리까지 포함한 단계 시간 |
| `failed_stage_metrics` | 실패 직전까지 얻은 실행·메모리 기록 |

`ip_resident_bytes=0`은 별도 상주가 없다는 뜻이다. 루트 그룹에 속한 투영층과 mid 블록은 UNet 호출 동안 GPU에 존재한다. 그룹별 가중치 크기는 실제 최대 VRAM이나 예상 절감량과 같지 않다. 중간 계산값과 메모리 할당자의 예약 공간도 필요하다.

## CPU 확인

- 관련 테스트 73개 통과. 설치된 Diffusers의 작은 실제 UNet과 IP 층으로 CPU 전후 출력 일치, 반복 호출, 그룹 소속, VAE 해제 순서, 실패 기록·정리를 확인했다.
- ON 재시험 원본 16장 사전 검사 통과.
- 전체 CPU 회귀 1,946개 통과(251.69초). 기존 Pillow 폐기 예정 경고 2건만 남았다. OFF 16건 사전 검사도 통과했다.
- GPU 실행, 다운로드·설치, 커밋·푸시는 하지 않았다.

CPU 테스트는 CUDA 전송·실제 모델의 최대 메모리·이미지 품질을 증명하지 않는다.

관련 로그는 `outputs/finishing-test-20261008/product_path/codex-ip-offload-*.log`에 저장했다.

## Claude 실행과 판정

기존 재시험 실행기를 그대로 사용한다. `product-ip-01`, `product-ip-02`를 덮어쓰거나 지우지 않는다. `product-ip-03`이 이미 존재하면 새 실행 전에 그 이유부터 확인한다.

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\verify_finishing_face_reference.py' --run --output '\\192.168.0.109\win_g\genai-lab\outputs\finishing-test-20261008\product-ip-03'
```

1. 원본 16장을 복사해 마무리만 실행한다. 새 기준 이미지 생성은 하지 않는다.
2. reserved·allocated 최대값, 고해상도·얼굴 보정 시간, 정리 포함 시간과 이미지 전체 시간을 기록한다. 기존 OFF 약 26초·11초(합계 약 37초)와 비교하되 입력·측정 범위 차이를 함께 표시한다.
3. 블라인드 4짝을 사용자에게 보여 주고, 눈색·머리 장식 보존을 별도로 평가한다. 실행 성공과 품질 통과를 나누어 보고한다.
4. 한도 초과·장치 불일치·오프로드 실패면 첫 실패에서 중단하고 로그·중간 결과를 보존한다. 자동 전환·재시도는 하지 않는다.

ON 이전 조건은 메모리 한도 때문에 완료 결과가 없으므로, 오프로드 ON/OFF 간 실제 GPU 픽셀 일치를 주장하지 않는다. 사용자 선정 기준(4짝 중 3짝 이상 마무리본 선호, 캐릭터 변화 0)을 충족하는지 판단한 뒤 기본값 변경 여부를 별도로 결정한다.

다음 작업은 Claude의 `product-ip-03` GPU 재시험이다. 헤어 작업과 기본값 변경은 이번 범위에 포함하지 않았다.
