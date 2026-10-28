# 마무리 얼굴 참조 메모리 수정과 재시험

> 후속 GPU 시험에서도 한도를 초과했다. 현재 구현과 새 재시험 명령은 [블록 오프로드 인수인계](FINISHING_IP_OFFLOAD_HANDOFF.md)를 따른다. 아래는 product-ip-02를 준비했던 기록이다.

얼굴 참조 ON 경로의 메모리 사용 방식을 수정했다. 실제 GPU 절감량·6.5 GiB 통과 여부·품질은 아직 확인하지 않았다. 같은 조건으로 `product-ip-02`에서 Claude가 검증한다.

## 확인한 중단 원인

`product-ip-01`은 첫 이미지의 고해상도 재처리 첫 UNet 호출에서 멈췄다. 얼굴 참조와 강도 0.9는 전달됐지만, 최대 reserved가 한도를 넘었다.

| 기록 | 값 |
| --- | --- |
| 최대 reserved | 7,493,124,096 bytes, 약 6.98 GiB |
| 최대 allocated | 6,955,565,568 bytes, 약 6.48 GiB |
| 둘의 차이 | 537,558,528 bytes, 약 512.7 MiB |
| 중단 기준 | reserved 6.5 GiB |

allocated는 당시 실제 할당량이고 reserved에는 할당자가 확보한 여유 공간도 포함된다. 이 기록만으로 캐시가 유일한 원인이라고 단정하지 않는다. 첫 호출에서 멈췄으므로 이후 단계의 최대 사용량도 모른다.

## 바꾼 처리 순서

**프롬프트 준비 → 얼굴·CFG 음성 임베딩 한 번 계산 → 인코더를 CPU로 이동 → CUDA 캐시 반환 → 임베딩만 재사용해 재처리·얼굴 보정** 순서다.

동일한 얼굴을 사용하는 한 배치(4 seed)에서 얼굴 준비 함수는 한 번 실행한다. Plus-Face는 실제 얼굴과 빈 이미지 각각의 hidden state가 필요하므로, 인코더 forward는 이때 두 번 수행한다. 빈 이미지의 임베딩을 임의의 0 텐서로 대체하지 않는다.

계산된 임베딩은 CPU에 보관한다. 두 img2img 단계에는 `ip_adapter_image_embeds`만 전달하며 `ip_adapter_image`를 다시 넘기지 않는다. 설치된 Diffusers 함수가 이를 GPU로 옮긴다. 사전 계산이 끝난 뒤 인코더 forward를 다시 부르면 중단한다. 인코더는 CPU에 남고 마무리 파이프라인 종료 시 함께 정리된다.

임베딩 SHA·dtype·shape·크기를 기록한다. 각 재처리 전 얼굴 파일과 캐시를 검사하고, 각 단계의 첫 UNet 호출에 실제 전달된 임베딩 바이트도 캐시와 대조한다. 이후 호출도 기존 IP 강도·임베딩 전달 검사를 유지한다.

## 선택한 attention 방식

기존 `AttnProcessor2_0`와 `IPAdapterAttnProcessor2_0`, 가중치, dtype을 유지하고 PyTorch SDPA의 **EFFICIENT_ATTENTION만 허용**한다. 적용 범위는 UNet 호출 내부다. VAE와 얼굴 인코더는 기존 계산 방식을 유지한다.

일반 attention slicing은 processor를 교체할 수 있어 IP 처리까지 바뀔 위험이 있다. 이번에는 이를 사용하지 않았다. 설치된 PyTorch 2.9.1+cu128은 Flash Attention이 빌드되지 않았다고 보고하므로 Flash를 선택하지 않았다.

이 선택은 모델이나 attention 수식을 바꾸지 않고 메모리 절약 커널을 사용하려는 것이다. **GPU 결과 차이가 가장 작다는 것을 실측한 것은 아니다.** 커널 차이로 부동소수점 결과가 달라질 수 있으며, GPU 지원 여부와 품질은 재시험으로 확인한다.

절약 모드가 지원되지 않으면 실패 기록을 남기고 멈춘다. 큰 attention 행렬을 만드는 math 경로로 자동 전환하지 않는다. UNet이 끝나거나 예외가 나면 원래 SDPA 설정을 복원한다. 얼굴 참조 OFF에서는 이 설정에 들어가지 않는다.

## GPU 없이 확인한 크기 근거

로컬 모델의 safetensors 헤더와 설정만 읽었다. 모델 추론이나 GPU 할당은 하지 않았다.

| 대상 | 계산한 크기 |
| --- | --- |
| 얼굴 인코더 파라미터·텐서 원소 | 632,077,057개 |
| 전부 FP16일 때의 텐서 크기 | 약 1.18 GiB, 활성값·추가 버퍼 제외 |
| CFG 포함 얼굴 임베딩 예상 shape | `[2, 1, 257, 1280]` |
| FP16 임베딩 예상 크기 | 1,315,840 bytes, 약 1.25 MiB |

257은 224×224 입력을 14×14 패치로 나눈 256개 토큰과 별도 토큰 하나를 합한 값이다. 크기는 `CFG 2 × 이미지 1 × 토큰 257 × 채널 1280 × 2 bytes`로 계산했다. 실제 실행에서는 나온 텐서의 크기와 SHA를 따로 기록한다.

기존 CPU offload도 인코더를 옮기므로, **1.18 GiB가 이전 최대값에서 그대로 빠진다는 뜻은 아니다.** 이번 변경은 인코더를 다시 올리는 일을 없애고 UNet 전에 캐시를 반환한다. 절감량은 메모리 재사용·단편화·attention 커널에 따라 달라진다.

근거: `outputs/finishing-test-20261008/product_path/codex-memory-size-evidence.json`.

## 유지한 조건과 기록

1.5배(1104×1848), strength 0.35, 28단계, CFG 5.5, IP 0.9, seed·프롬프트·모델, 얼굴 상자·합성 규칙은 유지했다. 기본값은 OFF다. GUI·2단계 비율 경로·헤어 후속 작업은 변경하지 않았다.

6.5 GiB를 넘으면 사전 계산 단계에서도 중단한다. 배율 축소·강도 변경·자동 재시도는 없다. 이미 만들어진 원본과 중간 결과는 보존한다.

| 기록 위치 | 추가 내용 |
| --- | --- |
| `finishing-status.json` | `face_embedding_preparation`: 준비 횟수, 인코더 호출 수, 메모리, CPU 이동 후 할당량, SHA·dtype·shape, attention 정책 |
| 후보별 `finishing.json` | 같은 준비 기록과 각 재처리의 `ip_embeddings`, `attention_policy`, IP 호출 관찰·메모리 |
| 실패 기록 | 준비 실패 또는 재처리 실패와 당시 메모리·호출 관찰 |

`attention_policy`에는 `sdpa_efficient_only`, `unet_only`, math fallback false를 남긴다. 얼굴 참조 OFF의 모델 로드·파이프라인 인자·계산 방식은 유지했다. OFF의 실제 GPU SHA 재현은 이번에 실행하지 않았다.

## CPU 검증

- 관련 테스트 66개 통과. 설치된 Diffusers의 이미지 입력 경로와 캐시 입력 경로의 임베딩 바이트 일치를 확인했다. 시험 인코더를 쓴 CPU 검증이며 실제 얼굴 모델의 GPU 추론은 아니다.
- 사전 계산 한 번, 인코더 두 번, CPU 이동 후 캐시 반환, 이후 인코더 재실행 차단, 얼굴·임베딩 변경 차단을 검사했다.
- 정상·오류 상황 모두 UNet 밖에서 SDPA 설정이 복원되는지 검사했다. OFF 경로가 사전 계산에 진입하지 않는지도 확인했다.
- 기존 ON용 원본 16장과 OFF용 잠금 시험 16건 모두 기존 실행기의 CPU 사전 검사를 통과했다.
- 최종 전체 회귀 테스트: CPU 1,939개 통과, 289.97초. 기존 Pillow 사용 중단 예고 2건. GPU 생성 없음.
- 로그: `codex-ip-memory-tests.log`, `codex-ip-memory-full-pytest.log`, `codex-ip-memory-preflight.log`, `codex-ip-memory-off-preflight.log`(모두 `outputs/finishing-test-20261008/product_path/`).

## Claude 재시험 명령과 판정

실행기는 바꾸지 않았다. 기존 16장을 복사하고 마무리와 흰 배경만 실행한다. 실패 기록이 남은 `product-ip-01`은 그대로 둔다. `product-ip-02`도 이미 있으면 다른 새 이름을 사용한다.

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\verify_finishing_face_reference.py' --run --output '\\192.168.0.109\win_g\genai-lab\outputs\finishing-test-20261008\product-ip-02'
```

먼저 캐시 준비 기록의 인코더 호출 수 2, CPU 이동, 두 마무리 단계의 임베딩 SHA 일치, IP 0.9·호출 수, 최대 메모리를 확인한다. 얼굴이 검출되지 않으면 기존대로 얼굴 보정은 생략된다.

완료하면 원본 raw와 새 마무리본의 블라인드 4짝을 만든다. 이전과 같은 첫 seed를 사용하고, 눈색(C)·머리 장식(A·B)과 얼굴 영역 색 차이를 별도로 기록한다. 마무리 선택 3/4 이상·캐릭터 바뀜 0 조건을 유지한다. 한도를 통과했다는 사실과 품질이 좋아졌다는 판정은 분리한다.

이번에는 GPU 실행·다운로드·설치·커밋·푸시를 하지 않았다. 실제 메모리 감소와 절약 커널 지원 여부는 Claude 재실행에서 확인해야 한다.
