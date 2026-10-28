# 마무리 얼굴 참조 선택 기능과 GPU 검증 인수인계

후속 GPU 시험은 메모리 한도 초과로 중단됐다. 최신 메모리 수정과 `product-ip-02` 실행 안내는 [마무리 메모리 인수인계](FINISHING_IP_MEMORY_HANDOFF.md)를 따른다. 아래는 최초 얼굴 참조 연결 기록이다.

얼굴 참조 선택 기능을 구현했다. 기본값은 OFF를 유지하며, GPU 실행·품질·실제 VRAM은 아직 검증하지 않았다. 6.5 GiB 초과 시 중단하며 강도나 해상도를 낮춰 재시도하지 않는다.

## 문제와 이번 시험의 구분

지난 제품 경로 블라인드에서는 마무리 2짝, 원본 2짝이 선택됐고, 마무리에서 C의 눈색과 A·B의 머리 장식이 달라졌다. 얼굴 참조 없이 마무리한 것이 원인이라는 설명은 아직 가설이다.

이번 시험은 **이미 생성한 동일한 16장에 얼굴 참조만 추가**한다. 새로운 기준 이미지를 생성하지 않는다. 같은 seed·프롬프트·토큰 ID·원본으로 비교하므로, 원본이 달라져 생기는 차이를 줄인다. 근거 요청서는 `outputs/finishing-test-20261008/product_path/codex_request_finishing_ip.md`다.

## 사용자 흐름과 설정

기존 실행 설정 `GENAI_STUDIO_RUNTIME`에서 `finishing_face_reference`를 켜면 된다. GUI 체크박스는 추가하지 않았다.

```json
{"finishing": true, "finishing_face_reference": true}
```

기존 설정 파일을 복사해 시험용 설정에서 두 항목만 추가한다. 다른 모델·경로·품질 태그 설정은 유지한다. 마무리를 끄면 얼굴 참조 옵션만 켜도 마무리를 실행하지 않는다. 2단계 비율 생성에는 두 옵션 모두 적용하지 않는다. 실제 운영 설정 파일과 기본값은 바꾸지 않았다.

마무리 흐름은 **기존 원본 → 1.5배 재처리 → 얼굴 보정 → 흰 배경 → 사용자 검토** 그대로다. 생성 때 사용한 확인된 얼굴 이미지를 두 img2img 단계에 동일하게 전달한다.

| 항목 | 고정 값 |
| --- | --- |
| IP 모델 | h94/IP-Adapter, 생성 기록과 같은 revision·파일·image encoder |
| 가중치 | `sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors` |
| 참조 이미지 | 해당 요청의 `inputs/face.png`, 원본 생성 기록의 SHA와 대조 |
| IP 강도 | 두 마무리 단계 모두 0.9, 실제 UNet 전체 호출 |
| 재처리 | 1.5배, 1104×1848, strength 0.35, 28단계, CFG 5.5, 기존 seed·Euler a |
| 실제 호출 수 | 각 img2img 9회; 얼굴 검출이 없으면 얼굴 보정은 기존대로 생략 |
| 얼굴 영역 | 기존 머리 상자·1.6배 확장·1024×1024·흐린 마스크 합성 유지 |

0.9는 기준 생성 후반에 쓰던 강도를 가져온 시험 조건이다. 이것이 색과 장식을 보존한다는 결론은 GPU 결과로 확인해야 한다.

## 검증과 기록

`checked_reference()`는 생성 기록에서 얼굴 SHA, IP 모델 경로·revision·가중치와 image encoder를 대조한다. 얼굴 이미지가 다르거나 한 배치의 얼굴 참조가 섞이면 GPU 파이프라인을 올리기 전에 중단한다. 각 재처리 직전에도 같은 파일 바이트를 검증한다.

IP는 CPU offload를 설치하기 전에 로드한다. image encoder도 offload 순서에 포함한다. 각 UNet 호출 직전에 attention processor의 실제 scale과 `image_embeds` 전달 여부를 확인한다. 잘못된 scale이나 임베딩 누락은 중단 사유다.

`finishing.json`에 다음 내용을 남긴다.

- `face_reference: on/off`, `ip_scale`, `face_file`, `face_sha256`, 모델·revision.
- 재처리·얼굴 보정 각각 `ip_applied_calls`, 호출별 실제 scale·이미지 임베딩 전달 여부.
- 단계별 시간·GPU 메모리. 실행 실패 시 확인한 호출 수와 메모리도 별도 보존.

OFF에서는 image encoder와 IP를 추가로 로드하거나 파이프라인에 얼굴 인자를 넘기지 않는다. 기존 F1·F2 재현 실행기는 얼굴 참조를 명시적으로 OFF로 고정했다. 코드·가짜 파이프라인 검증과 잠금 자료의 SHA 확인까지 했으며, 이번 수정 뒤 실제 GPU의 OFF SHA 재현을 다시 수행한 것은 아니다.

## 기존 16장으로 마무리만 재실행

새 실행기는 `scripts/verify_finishing_face_reference.py`다. 기본 입력은 `outputs/finishing-test-20261008/product_path/run-*` 네 건이다.

1. 네 실행·16 seed·완료 상태·원본 SHA·동일 얼굴·모델·프롬프트를 사전 검사한다.
2. 새 폴더에 원본, 생성 요청 기록, 입력 승인 기록, 얼굴 이미지를 바이트 그대로 복사한다. 생성 기록을 새로 만든 것으로 바꾸지 않는다. 복사 출처는 별도 `replay-source.json`에 기록한다.
3. 기록된 토큰 ID까지 그대로 복원한다. 생성 태그를 다시 분석하거나 토큰을 다시 계산하지 않는다.
4. 얼굴 참조 ON으로 마무리와 흰 배경만 실행한다. 원본 생성 함수는 호출하지 않는다.
5. 실패한 경우 새 폴더의 중간 파일과 실패 기록을 보존하고 멈춘다. 원본 폴더를 덮어쓰거나 자동 재시도하지 않는다.

CPU 사전 검사만 실행하는 명령:

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\verify_finishing_face_reference.py'
```

Claude가 GPU로 16장 마무리를 실행할 명령이다. 아래 폴더가 이미 있으면 새 이름을 사용한다.

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\verify_finishing_face_reference.py' --run --output '\\192.168.0.109\win_g\genai-lab\outputs\finishing-test-20261008\product-ip-01'
```

각 `run-*`의 `generation/finishing-status.json`, 후보별 `finishing.json`, 최상위 `replay.json`을 확인한다. 완료 여부와 실제 품질은 별도로 판정한다.

## Claude 판정

기존 순서와 seed를 유지한다. 각 실행의 첫 seed를 사용해 원본 raw와 IP 마무리본 네 쌍을 블라인드로 제시한다. 판정 전에 성공한 seed로 바꾸지 않는다.

사용자 선택과 별개로 이전 IP 없는 마무리본 대비 C의 눈색, A·B의 머리 장식 보존과 얼굴 영역 색 차이의 95백분위를 기록한다. 이전 시험과 같은 좌표·색 공간·측정 규칙을 쓴다. 규칙 변경이 필요하면 비교 전에 보고한다.

**기본값 ON 추천 조건은 마무리 선택 3/4 이상이며 캐릭터 바뀜 0이다.** 조건을 충족해도 자동으로 기본값을 바꾸지 않는다. 한도 초과, 실패, 다른 부위의 변형을 별도로 보고한다.

## CPU 확인

- 얼굴 참조 ON: 기존 제품 원본 4건·16장 사전 검사 통과. 실제 16장 복사·프롬프트와 토큰 ID 복원·얼굴 SHA 대조까지 CPU로 확인했고 원본도 그대로다. GPU 생성 없음.
- 복사 확인 기록: `outputs/finishing-test-20261008/product_path/codex-ip-copycheck-01/cpu-copy-result.json`. 이 폴더는 CPU 복사 확인용이며 GPU 시험은 별도 새 폴더에서 실행한다.
- 얼굴 참조 OFF: 기존 16건 F0·F1·F2 잠금 SHA 사전 검사 통과.
- 최종 전체 CPU 테스트 1,930개 통과, 303.32초. 기존 Pillow 사용 중단 예고 2건. 이번 얼굴 참조 테스트 17개에는 OFF 로드 불변, ON 모델·scale·전달, SHA 변경 차단, 복사 무결성, 메모리 중단, 생성 없는 재실행을 포함한다.
- 로그: `outputs/finishing-test-20261008/product_path/codex-ip-preflight.log`, `codex-ip-off-preflight.log`, `codex-ip-tests.log`, `codex-ip-full-pytest.log`.

## 코드 읽기 시작점

`genai_lab/studio_finishing.py`의 `finish_batch()`에서 확인된 얼굴 참조를 준비하고, `genai_lab/finishing_backend.py`에서 실제 파이프라인과 호출을 맡는다. 얼굴 검증과 호출 관찰은 `genai_lab/finishing_reference.py`에 모았다. 설정은 기존 `StudioRuntime`에 한 항목만 추가했다.

이번에는 코드와 CPU 테스트까지만 진행했다. GPU 실행·다운로드·패키지 설치·기본값 변경·커밋·푸시는 하지 않았으며, 체형 후속 작업은 진행하지 않았다.
