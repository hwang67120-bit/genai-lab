# 실제 파이프라인 시스템 RAM 사용 내역

ordinary-female × 페라리, 실제 GenerationOrchestrator 경로를 1회 실행했습니다. 상태: **completed**. 연속 RSS 피크 **15.971 GiB** (17149.05 MB), 2026-09-24T16:34:57+09:00, 단계 `base / post-sampling analysis and gates`.

RSS는 부모와 모든 재귀 하위 프로세스의 합입니다. OS 전체 RAM 사용량이나 고유 물리 페이지 합은 아닙니다. 표의 GB는 요청 코드와 같은 1024³ 바이트(GiB)입니다. 이전 13,481.98 MB는 십진 MB이므로 단위를 구분합니다.

코드 `73dd0c8f270cf33b3e6249b62cd3edd83f3d27b0`. 운영 소스·설정·계약 문서를 수정하지 않았습니다. 추적 파일 변경: `[]`. 계측은 outputs 내 Python 프로파일 훅·프로세스 한정 sitecustomize로 삽입했고, 종료 시 훅과 환경을 원복했습니다. dtype·오프로드·모델 로드 방식은 변경하지 않았습니다. 생성 호출: `{'base': 1, 'inpaint': 1}`. 재시도하지 않았습니다.

## 표 1 — 체크포인트

실제 관측 시각 순서입니다. 동일 코드가 반복되는 것은 별도 프로세스 또는 재로드입니다. 델타는 직전 행의 전체 RSS와의 차이이며, 동시 실행·해제·버퍼·단편화도 포함하므로 해당 모델 가중치만의 값으로 확정하지 않습니다. gc.collect()는 표의 해당 PID에서 실행했습니다. 다른 프로세스의 GC를 원격 강제하지 않았습니다.

| # / PID | 지점 | RSS GB | 델타 GB |
|---|---|---:|---:|
| 00 / 4872 | process start (instrumentation initialized) | 0.0213 | — |
| 01 / 4872 | torch import complete | 0.4623 | +0.4410 |
| 02 / 4872 | diffusers import complete | 0.5399 | +0.0777 |
| 03 / 4872 | SDXL from_pretrained complete | 0.8416 | +0.3017 |
| 05 / 4872 | load_ip_adapter complete | 0.8467 | +0.0051 |
| 04 / 4872 | enable_model_cpu_offload complete | 0.8467 | +0.0000 |
| — / 4872 | reference analysis before | 0.8469 | +0.0001 |
| 00 / 26300 | process start (instrumentation initialized) | 0.8752 | +0.0283 |
| 01 / 26300 | torch import complete | 1.2245 | +0.3493 |
| 02 / 26300 | diffusers import complete | 1.2736 | +0.0491 |
| — / 26300 | isnet ONNX before load | 1.3869 | +0.1133 |
| — / 26300 | onnx_other ONNX before load | 1.3869 | +0.0000 |
| 09 / 26300 | isnet ONNX session created | 1.5766 | +0.1896 |
| 10 / 26300 | CatVTON AutoMasker initialized | 1.8230 | +0.2464 |
| 16 / 26300 | process exit / instrumentation removed | 2.2908 | +0.4678 |
| — / 4872 | processor/config preparation (not model weights): grounding_dino from_pretrained complete | 0.9580 | -1.3328 |
| — / 4872 | processor/config preparation (not model weights): grounding_dino from_pretrained complete | 0.9630 | +0.0050 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 0.9630 | +0.0000 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 0.9630 | +0.0000 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 0.9632 | +0.0002 |
| 07 / 4872 | sam2 from_pretrained complete | 0.9610 | -0.0022 |
| 06 / 4872 | grounding_dino from_pretrained complete | 0.9669 | +0.0059 |
| — / 4872 | reference analysis after | 2.6595 | +1.6926 |
| — / 4872 | wd14 ONNX before load | 2.6627 | +0.0032 |
| — / 4872 | onnx_other ONNX before load | 2.6627 | +0.0000 |
| 08 / 4872 | wd14 ONNX session created | 3.0187 | +0.3560 |
| 11 / 4872 | Base stage before | 2.6669 | -0.3517 |
| 04 / 4872 | enable_model_cpu_offload complete | 2.6669 | +0.0000 |
| — / 4872 | wd14 ONNX before load | 3.3265 | +0.6596 |
| — / 4872 | onnx_other ONNX before load | 3.3265 | +0.0000 |
| 08 / 4872 | wd14 ONNX session created | 3.6789 | +0.3524 |
| — / 4872 | base actual sampling before | 4.5766 | +0.8977 |
| 04 / 4872 | enable_model_cpu_offload complete | 13.1213 | +8.5446 |
| — / 4872 | base actual sampling after | 13.1112 | -0.0100 |
| 00 / 33176 | process start (instrumentation initialized) | 13.2198 | +0.1085 |
| 01 / 33176 | torch import complete | 13.5659 | +0.3461 |
| 02 / 33176 | diffusers import complete | 13.6146 | +0.0487 |
| — / 33176 | isnet ONNX before load | 13.7250 | +0.1104 |
| — / 33176 | onnx_other ONNX before load | 13.7250 | +0.0000 |
| 09 / 33176 | isnet ONNX session created | 13.9124 | +0.1874 |
| 10 / 33176 | CatVTON AutoMasker initialized | 14.1594 | +0.2470 |
| 16 / 33176 | process exit / instrumentation removed | 14.6243 | +0.4648 |
| — / 4872 | processor/config preparation (not model weights): grounding_dino from_pretrained complete | 13.2369 | -1.3873 |
| — / 4872 | processor/config preparation (not model weights): grounding_dino from_pretrained complete | 13.2408 | +0.0039 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 13.2408 | +0.0000 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 13.2408 | +0.0000 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 13.2408 | +0.0000 |
| 07 / 4872 | sam2 from_pretrained complete | 13.2408 | -0.0000 |
| 06 / 4872 | grounding_dino from_pretrained complete | 13.2446 | +0.0038 |
| 00 / 24900 | process start (instrumentation initialized) | 14.2325 | +0.9879 |
| 01 / 24900 | torch import complete | 14.5788 | +0.3463 |
| 02 / 24900 | diffusers import complete | 14.6269 | +0.0481 |
| — / 24900 | isnet ONNX before load | 14.7364 | +0.1095 |
| — / 24900 | onnx_other ONNX before load | 14.7364 | +0.0000 |
| 09 / 24900 | isnet ONNX session created | 14.9192 | +0.1828 |
| 10 / 24900 | CatVTON AutoMasker initialized | 15.1931 | +0.2739 |
| 16 / 24900 | process exit / instrumentation removed | 15.9684 | +0.7753 |
| 12 / 4872 | Base stage after | 13.5171 | -2.4513 |
| 13 / 4872 | refinement stage before | 13.5171 | +0.0000 |
| 00 / 14484 | process start (instrumentation initialized) | 13.5542 | +0.0371 |
| 01 / 14484 | torch import complete | 13.9010 | +0.3468 |
| 02 / 14484 | diffusers import complete | 13.9492 | +0.0482 |
| — / 14484 | isnet ONNX before load | 14.0585 | +0.1093 |
| — / 14484 | onnx_other ONNX before load | 14.0585 | +0.0000 |
| 09 / 14484 | isnet ONNX session created | 14.2424 | +0.1839 |
| 10 / 14484 | CatVTON AutoMasker initialized | 14.5158 | +0.2734 |
| 16 / 14484 | process exit / instrumentation removed | 15.2923 | +0.7765 |
| — / 4872 | processor/config preparation (not model weights): grounding_dino from_pretrained complete | 13.5301 | -1.7622 |
| — / 4872 | processor/config preparation (not model weights): grounding_dino from_pretrained complete | 13.5340 | +0.0039 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 13.5340 | +0.0000 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 13.5340 | +0.0000 |
| — / 4872 | processor/config preparation (not model weights): sam2 from_pretrained complete | 13.5340 | +0.0000 |
| 07 / 4872 | sam2 from_pretrained complete | 13.5348 | +0.0007 |
| 06 / 4872 | grounding_dino from_pretrained complete | 13.5373 | +0.0025 |
| 04 / 4872 | enable_model_cpu_offload complete | 14.3833 | +0.8461 |
| 04 / 4872 | enable_model_cpu_offload complete | 14.3833 | +0.0000 |
| — / 4872 | inpaint actual sampling before | 14.3868 | +0.0034 |
| 04 / 4872 | enable_model_cpu_offload complete | 14.6088 | +0.2221 |
| — / 4872 | inpaint actual sampling after | 14.6088 | +0.0000 |
| — / 4872 | processor/config preparation (not model weights): grounding_dino from_pretrained complete | 13.9329 | -0.6759 |
| — / 4872 | processor/config preparation (not model weights): grounding_dino from_pretrained complete | 13.9369 | +0.0039 |
| 06 / 4872 | grounding_dino from_pretrained complete | 13.9395 | +0.0026 |
| 04 / 4872 | enable_model_cpu_offload complete | 14.6591 | +0.7196 |
| 04 / 4872 | enable_model_cpu_offload complete | 14.6591 | +0.0000 |
| 14 / 4872 | refinement and its internal gates after | 14.6524 | -0.0068 |
| 15 / 4872 | measurement and gates complete (inside finalize boundary) | 14.6524 | +0.0000 |
| 16 / 4872 | process exit / instrumentation removed | 12.8434 | -1.8090 |

미관측 체크포인트: 없음. 미관측은 계측 누락 여부를 실행 경로와 함께 확인해야 하며 곧바로 미로드로 단정하지 않습니다.
14·15는 finalize_selected_candidate가 정밀화와 내부 측정·게이트를 끝낸 반환 경계입니다. 14는 순수 확산 시간만의 경계가 아닙니다. 실제 확산 호출 직전·직후를 별도 행으로 추가했습니다.

## 표 2 — 모듈

실제 parameters + buffers 원소 수 × element_size 합입니다. 같은 역할이 여러 번 로드되면 최대 관측 크기 1개를 대표로 표시하고, 모든 개체·체크포인트 스냅샷은 modules.json에 보존했습니다. device는 이 대표 스냅샷 시점입니다. 오프로드로 이후 바뀔 수 있습니다. 같은 객체의 일반 사전학습 이름과 파이프라인 속성 이름은 중복 합산하지 않았습니다.

| 모듈 | dtype | device | 크기 GB |
|---|---|---|---:|
| automasker/AutoMasker.densepose_processor.predictor.model | torch.float32 | cuda:0 | 0.2381 |
| automasker/AutoMasker.schp_processor_atr.model | torch.float32 | cuda:0 | 0.2490 |
| automasker/AutoMasker.schp_processor_lip.model | torch.float32 | cuda:0 | 0.2490 |
| grounding_dino/GroundingDinoForObjectDetection | torch.float32 | cpu | 0.6419 |
| ip_adapter/image_encoder | torch.float16 | cpu | 3.4364 |
| sam2/Sam2Model | torch.float32 | cpu | 0.1171 |
| sdxl/text_encoder | torch.float16 | cpu | 0.2292 |
| sdxl/text_encoder_2 | torch.float16 | cpu | 1.2939 |
| sdxl/unet | torch.float16 | cpu | 5.4366 |
| sdxl/vae | torch.float16 | cpu | 0.1558 |

ONNX는 parameters 접근 대신 로드 전후 RSS 델타를 기록했습니다. 이 값은 가중치 크기가 아닙니다. 반복 로드 전체 값은 modules.json에 있습니다. 아래는 종류별 최대 관측 로드 델타입니다.

| ONNX | 공급자 | RSS 델타 GB |
|---|---|---:|
| isnet | CPUExecutionProvider | +0.1897 |
| wd14 | CPUExecutionProvider | +0.3560 |

## 표 3 — 요청한 합계와 잔여

**이 표는 물리 RAM의 정확한 분할이 아닙니다.** SDXL/IP/보조 모델은 논리 텐서 크기(GPU 텐서 포함), 런타임·ONNX는 RSS입니다. 서로 다른 시점의 최대 역할 크기를 더하므로 상주 시점도 같지 않습니다. 요청한 뺄셈을 그대로 제공하되 잔여는 **원인 미확정**입니다.

| 항목 | GB | 비고 |
|---|---:|---|
| 런타임 (torch+diffusers import) | 0.5399 | 부모 체크포인트 02, 지연 import 전체를 포함하지 않음 |
| SDXL 가중치 | 7.1155 | 표 2 sdxl 역할 합, UNet에 로드된 adapter 가중치 포함 가능 |
| IP-Adapter 이미지 인코더 | 3.4364 | image_encoder 논리 크기 |
| 보조 모델 합 | 2.0408 | 텐서 1.4951 + ONNX 로드 RSS 델타 0.5457; 동시 상주 합 아님 |
| 귀속 안 된 잔여 | 2.8386 | 전체 피크 − 위 합계, 원인 미확정 |

## 연속 피크 및 Q1~Q3

- 연속 샘플 663개, 목표 간격 1초, 실제 최대 간격 5.983초. 측정 구간 2026-09-24T16:26:13+09:00 ~ 2026-09-24T16:37:15+09:00.
- 체크포인트 최대 15.9684 GiB (PID 24900, `process exit / instrumentation removed`), 연속 최대 15.9713 GiB. 차이 0.0029 GiB.
- Q1 CPU + float32 관측: **예**. sam2/Sam2Model, grounding_dino/GroundingDinoForObjectDetection.
- 피크 시점 프로세스별 RSS: [(4872, 14.208), (12060, 0.0048), (24900, 1.7585)] GiB.
- Q2 아래는 부모의 살아 있는 약한 참조로 확인한 보조 모델/ONNX와 프로세스 목록입니다. RSS 유지 자체를 모델 상주의 증거로 쓰지 않았습니다. 하위 프로세스가 살아 있는 동안의 세부 모델은 별도 PID 체크포인트에 남겼습니다.

| 단계 | RSS GB | 보조 모듈 | ONNX | 살아 있는 PID |
|---|---:|---|---|---|
| 11 | 2.6669 | 관측 없음 | 관측 없음 | [4872] |
| base actual sampling before | 4.5766 | 관측 없음 | wd14 | [4872] |
| base actual sampling after | 13.1112 | 관측 없음 | wd14 | [4872] |
| 12 | 13.5171 | 관측 없음 | 관측 없음 | [4872] |
| 13 | 13.5171 | 관측 없음 | 관측 없음 | [4872] |
| inpaint actual sampling before | 14.3868 | 관측 없음 | 관측 없음 | [4872] |
| inpaint actual sampling after | 14.6088 | 관측 없음 | 관측 없음 | [4872] |
| 14 | 14.6524 | 관측 없음 | 관측 없음 | [4872] |
| 15 | 14.6524 | 관측 없음 | 관측 없음 | [4872] |

- Q3 연속 최대와 체크포인트 최대 차이 > 1 GiB: **아니오**.
차이가 있더라도 일시적 이중 사본·pinned memory·단편화·이미지 버퍼 중 무엇인지는 이 측정으로 확정하지 않았습니다.

## 실행 및 제한

- 해상도: [736, 1232], seed 209210001, profile {'reference_scale': 0.8, 'inpaint_strength': 0.9, 'inference_steps': 28}.
- Base 270.6894476999996초, 정밀화(내부 검사 포함) 120.77715989999979초. 프로파일 훅과 GC가 추가된 측정 실행의 시간이며 성능 비교값으로 사용하지 않습니다.
- 계측 오류 6건. checkpoints.json에 보존. ONNX 상위 생성자의 중첩 반환에서 아직 없는 _providers 조회가 실패한 경우, 원래 InferenceSession 초기화는 계속됐고 바깥 생성자 완료의 세션 공급자·델타 관측 여부를 별도로 확인했습니다. 오류를 숨기거나 재시도하지 않았습니다.
- raccoon은 측정하지 않았으며 더 큰 중간 버퍼 때문에 RAM을 더 쓸 수 있습니다.
- 프로세스 시작 값은 Python/psutil/계측기 import 이후 첫 체크포인트입니다. OS 프로세스 생성 순간 RSS와는 다릅니다.
- 샘플러 목표는 1초지만 실제 최대 간격은 5.983초였습니다. 따라서 엄밀한 매초 측정은 충족하지 못했고, 이 값은 관측 피크입니다. 샘플 사이의 더 짧은 피크를 놓칠 수 있습니다. 하위 프로세스 RSS 합에는 공유 페이지가 중복 포함될 수 있습니다.
- 모델 resident 여부는 Python 객체 생존 관측이며 모든 네이티브 allocator 버퍼의 해제를 증명하지 않습니다.
- 얼굴 보조 검출 YOLO는 실행 로그에서 사용이 확인되지만 이번 계측의 모듈 등록 대상에 포함되지 않아 개별 바이트를 계측하지 못했습니다. genai_lab/reference_face_observation.py:62~73의 캐시 로더는 확인했으나 해당 객체 생존을 직접 측정하지 않았습니다. 표 3의 보조 모델 합은 관측한 항목에 한정됩니다.
- 최적화는 수행하거나 권고하지 않았습니다.

## 계측 근거

[psutil Process RSS / children](https://psutil.io/) · [PyTorch Tensor.element_size](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.element_size.html). 사용한 코드 및 원시 JSONL은 이 출력 폴더에 보존했습니다.

## 동일 조건 및 원복 확인

Base·Final·input_source·input_garment SHA-256이 기존 페라리 기준선과 모두 일치했습니다. 실제 Inpaint는 0.80/0.90/28, 콜백 25회였고 Base 1회·정밀화 1회만 호출했습니다. 00~16 체크포인트가 모두 관측됐습니다. ONNX 내부 생성자 조회 오류 6건에 대해 바깥 생성자 정상 완료 기록도 각각 6건 확인했습니다. 상세는 verification.json에 있습니다.

## VRAM 추가 요청 — 기존 기록 복구

동일 실행은 이미 종료돼 신규 계측을 소급할 수 없습니다. [표 4·5 및 Q4~Q6](vram-addendum.md)에 저장된 자료와 미계측 항목을 구분했습니다. 원시 복구 결과: vram-existing-records.json. 새 생성은 수행하지 않았습니다.
