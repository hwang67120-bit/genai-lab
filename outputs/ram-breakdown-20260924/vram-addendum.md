# 표 4·5 추가 — 종료된 동일 실행의 VRAM 기록

직전 RAM 계측 실행은 이미 종료됐습니다. 저장된 자료만 복구했으며 새 생성은 하지 않았습니다. 체크포인트의 CUDA allocator 수치와 디노이징·VAE 내부 상주 상태는 소급 복원할 수 없습니다. 미계측은 0이 아닙니다.

## 표 4 — VRAM 체크포인트

요청 코드가 있는 체크포인트와 실제 생성 호출 경계를 표시합니다. 전체 스냅샷은 vram-existing-records.json에 있습니다. GPU 상주 표기는 저장된 모듈 device 스냅샷 기준이며 첫 파라미터와 dtype/device별 텐서 합을 기록한 것입니다.

| # / PID | 지점 | allocated GB | reserved GB | GPU 상주 모듈 |
|---|---|---:|---:|---|
| 00 / 4872 | process start (instrumentation initialized) | 미계측 | 미계측 | 관측된 모듈 없음 |
| 01 / 4872 | torch import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 02 / 4872 | diffusers import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 03 / 4872 | SDXL from_pretrained complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 05 / 4872 | load_ip_adapter complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 04 / 4872 | enable_model_cpu_offload complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 00 / 26300 | process start (instrumentation initialized) | 미계측 | 미계측 | 관측된 모듈 없음 |
| 01 / 26300 | torch import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 02 / 26300 | diffusers import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 09 / 26300 | isnet ONNX session created | 미계측 | 미계측 | 관측된 모듈 없음 |
| 10 / 26300 | CatVTON AutoMasker initialized | 미계측 | 미계측 | AutoMasker.densepose_processor.predictor.model, AutoMasker.schp_processor_atr.model, AutoMasker.schp_processor_lip.model |
| 16 / 26300 | process exit / instrumentation removed | 미계측 | 미계측 | 관측된 모듈 없음 |
| 07 / 4872 | sam2 from_pretrained complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 06 / 4872 | grounding_dino from_pretrained complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 08 / 4872 | wd14 ONNX session created | 미계측 | 미계측 | 관측된 모듈 없음 |
| 11 / 4872 | Base stage before | 미계측 | 미계측 | 관측된 모듈 없음 |
| 04 / 4872 | enable_model_cpu_offload complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 08 / 4872 | wd14 ONNX session created | 미계측 | 미계측 | text_encoder_2, image_encoder |
| — / 4872 | base actual sampling before | 미계측 | 미계측 | image_encoder |
| 04 / 4872 | enable_model_cpu_offload complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| — / 4872 | base actual sampling after | 미계측 | 미계측 | 관측된 모듈 없음 |
| 00 / 33176 | process start (instrumentation initialized) | 미계측 | 미계측 | 관측된 모듈 없음 |
| 01 / 33176 | torch import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 02 / 33176 | diffusers import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 09 / 33176 | isnet ONNX session created | 미계측 | 미계측 | 관측된 모듈 없음 |
| 10 / 33176 | CatVTON AutoMasker initialized | 미계측 | 미계측 | AutoMasker.densepose_processor.predictor.model, AutoMasker.schp_processor_atr.model, AutoMasker.schp_processor_lip.model |
| 16 / 33176 | process exit / instrumentation removed | 미계측 | 미계측 | 관측된 모듈 없음 |
| 07 / 4872 | sam2 from_pretrained complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 06 / 4872 | grounding_dino from_pretrained complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 00 / 24900 | process start (instrumentation initialized) | 미계측 | 미계측 | 관측된 모듈 없음 |
| 01 / 24900 | torch import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 02 / 24900 | diffusers import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 09 / 24900 | isnet ONNX session created | 미계측 | 미계측 | 관측된 모듈 없음 |
| 10 / 24900 | CatVTON AutoMasker initialized | 미계측 | 미계측 | AutoMasker.densepose_processor.predictor.model, AutoMasker.schp_processor_atr.model, AutoMasker.schp_processor_lip.model |
| 16 / 24900 | process exit / instrumentation removed | 미계측 | 미계측 | 관측된 모듈 없음 |
| 12 / 4872 | Base stage after | 미계측 | 미계측 | image_encoder |
| 13 / 4872 | refinement stage before | 미계측 | 미계측 | image_encoder |
| 00 / 14484 | process start (instrumentation initialized) | 미계측 | 미계측 | 관측된 모듈 없음 |
| 01 / 14484 | torch import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 02 / 14484 | diffusers import complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 09 / 14484 | isnet ONNX session created | 미계측 | 미계측 | 관측된 모듈 없음 |
| 10 / 14484 | CatVTON AutoMasker initialized | 미계측 | 미계측 | AutoMasker.densepose_processor.predictor.model, AutoMasker.schp_processor_atr.model, AutoMasker.schp_processor_lip.model |
| 16 / 14484 | process exit / instrumentation removed | 미계측 | 미계측 | 관측된 모듈 없음 |
| 07 / 4872 | sam2 from_pretrained complete | 미계측 | 미계측 | image_encoder |
| 06 / 4872 | grounding_dino from_pretrained complete | 미계측 | 미계측 | image_encoder |
| 04 / 4872 | enable_model_cpu_offload complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 04 / 4872 | enable_model_cpu_offload complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| — / 4872 | inpaint actual sampling before | 미계측 | 미계측 | 관측된 모듈 없음 |
| 04 / 4872 | enable_model_cpu_offload complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| — / 4872 | inpaint actual sampling after | 미계측 | 미계측 | 관측된 모듈 없음 |
| 06 / 4872 | grounding_dino from_pretrained complete | 미계측 | 미계측 | image_encoder |
| 04 / 4872 | enable_model_cpu_offload complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 04 / 4872 | enable_model_cpu_offload complete | 미계측 | 미계측 | 관측된 모듈 없음 |
| 14 / 4872 | refinement and its internal gates after | 미계측 | 미계측 | 관측된 모듈 없음 |
| 15 / 4872 | measurement and gates complete (inside finalize boundary) | 미계측 | 미계측 | 관측된 모듈 없음 |
| 16 / 4872 | process exit / instrumentation removed | 미계측 | 미계측 | 관측된 모듈 없음 |

## 표 5 — 저장된 단계별 피크

| 단계 | max allocated GB | max reserved GB | 피크 시점 |
|---|---:|---:|---|
| Base candidate generation | 6.0517 | 미계측 | 미계측 |
| Base orchestrator (generation + validation) | 미계측 | 6.4180 | 미계측 |
| refinement orchestrator (generation + internal gates) | 미계측 | 6.4160 | 미계측 |
| measurement only | 미계측 | 미계측 | 미계측 |

**서로 다른 행의 최대값을 빼서 allocated–reserved 차이로 해석하지 않았습니다.** Base 후보 구간과 오케스트레이터 전체 구간의 측정 범위가 다르고, generator.py 내부 피크 리셋도 있습니다. 부모 프로세스 CUDA allocator 값은 분석용 하위 프로세스 VRAM을 합산한 값이 아닙니다.

## Q4~Q6 관찰

- Q4: 판정불가. Base 실제 호출 직전 및 Base 완료/정밀화 시작 경계에서 image_encoder의 cuda:0 상주가 관측됐습니다. 정밀화 실제 Inpaint 호출 직전에는 cpu였습니다. 디노이징 중 계속 GPU에 남았는지는 기록이 없어 판단하지 않습니다.
- Q5: 판정불가. IP-Adapter 인코딩·UNet 디노이징·VAE 디코드별 피크 시점이 없습니다.
- Q6: 판정불가. 동일 시각 allocated/reserved 쌍이 없습니다. 차이가 1 GiB를 넘는지 확인할 수 없습니다.

정확한 표 4·5와 Q4~Q6에는 동일 조건의 RAM+VRAM 계측 실행 1회가 새로 필요합니다. 앞선 재시도 금지 및 동일 실행 조건 때문에 사용자 확인을 요청했으며, 승인 전에는 생성하지 않습니다. 최적화는 권고하거나 적용하지 않았습니다.
