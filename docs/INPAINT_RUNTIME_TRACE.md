# 신체 복원·의상 합성 실행 계측

Spring AOP의 around advice와 비슷하게 호출 전후를 감싸는 진단 계층이다. 생성 모델·시드·연산 설정·입출력은 변경하지 않는다. 텐서나 이미지 참조를 기록하지 않는다.

## 확인 위치

새 실행의 `outputs/debug_benchmark/<run>/runtime_trace.jsonl`에 시작·종료·실패와 호출 횟수, PID, 메모리 스냅샷을 저장한다. 기존 부모 진행 감시 스레드가 약 1초마다 파일을 읽어 PowerShell에 `[복원 계측]`으로 출력한다. 기존 `stderr.log`는 종료 후 저장되는 방식 그대로이며 실시간 계측은 별도 파일이다.

10초마다 현재 중첩 연산과 지속 시간을 기록한다. `pipeline.generate > pipeline.prepare_latents > vae.encode > vae.offload_pre_forward`처럼 어디까지 진입했는지 볼 수 있다. 종료 기록 없는 마지막 시작은 중단 후보이지 원인 확정이 아니다. GPU/드라이버 정지나 GIL 점유 시 heartbeat 스레드 자체가 지연될 수 있다.

## 계측 범위

- ControlNet/파이프라인 로딩, CPU offload 설치
- encode_prompt, prepare_latents, prepare_mask_latents, prepare_control_image, _encode_vae_image
- text_encoder/2.forward, vae.encode/decode, controlnet.forward, unet.forward
- 위 모듈에 기존 Accelerate `_hf_hook`가 있으면 pre_forward/post_forward를 추가 계측한다. 버전에 따라 없는 메서드는 unavailable로 표시한다. 기존 hook을 대체하거나 제거하지 않는다.
- cleanup 직전/직후 자원과 부모 프로세스의 runner 전후 자원(metadata.json)

RAM RSS는 현재 상주 메모리, Windows private는 프로세스 private 메모리이며 Linux에서는 vms로 명시한다. psutil이 없으면 unavailable이며 자동 설치하지 않는다. 시스템 available RAM 및 프로세스 누적 I/O도 기록한다. 페이지 파일 사용/페이지 폴트나 GPU 전송량을 직접 측정하지 않으므로 이 수치만으로 paging이나 누수를 확정할 수 없다.

VRAM allocated/reserved/peak는 해당 프로세스 PyTorch 캐시 할당자 값이며 다른 앱을 포함한 전체 GPU 사용량이 아니다. CUDA 컨텍스트가 없으면 초기화하지 않고 unavailable 상태로 남긴다.

## 계측 영향과 한계

시간은 **호스트 호출 경과 시간**이다. CUDA synchronize를 삽입하지 않으므로 정확한 GPU 커널 시간으로 해석하면 안 된다. offload pre/post 시간도 순수 전송 시간만은 아니다. 시작/종료마다 소량의 자원 조회 비용은 있다. 최대 2048개 큐와 별도 파일 쓰기 스레드로 느린 디스크 I/O를 연산에서 분리하며, 초과 이벤트 수는 dropped_events로 기록한다. 강제 종료 직전 큐의 일부 기록은 유실될 수 있다. 쓰기 실패는 stderr 경고만 남기고 생성 예외는 그대로 보존한다.

현재 계측은 원인 진단용이며 메모리 누수 수정이나 속도 향상을 보장하지 않는다. GPU 실제 모델 검증은 별도다.

근거: [Accelerate ModelHook](https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.hooks.ModelHook), [PyTorch CUDA 메모리와 비동기 실행](https://docs.pytorch.org/docs/stable/notes/cuda.html), [Python wraps](https://docs.python.org/3/library/functools.html#functools.wraps).
