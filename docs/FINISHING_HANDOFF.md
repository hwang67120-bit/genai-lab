# 품질 태그와 고해상도 마무리 인수인계

구현 완료, GPU 미검증. 품질 태그와 마무리는 모두 임시 기본값 OFF이며, 품질 태그의 최종 기본값은 사용자 결정 대기다. 운영 설정 파일은 변경하지 않았다.

## 적용 범위와 실행 흐름

기준은 `outputs/finishing-test-20261008/codex_request_finishing.md`와 같은 폴더의 `run_finish.py`다.

1회 생성에서 사용자 확인을 마치면 기존 얼굴 참조를 사용해 4장을 생성한다. 생성 파이프라인을 해제한 다음, 마무리를 켠 경우에만 별도 img2img 파이프라인을 올린다. 이 파이프라인에는 IP-Adapter를 로드하지 않는다.

**생성 원본 → 1.5배 재처리 → 얼굴 보정 → 파이프라인 해제 → CPU 흰 배경 정리 → 결과 확인·승인·저장** 순서다. 마무리 결과는 1104×1848이고 원본 736×1232도 그대로 남는다.

| 설정 | 동작 | 현재 기본값 |
| --- | --- | --- |
| `quality_tags` | 진단 ②b의 품질 문구를 사용하고 기존 분할 인코딩으로 전달 | false, 사용자 결정 대기 |
| `finishing` | 고해상도 재처리와 얼굴 보정을 추가 | false |

두 설정은 독립적이다. 두 설정을 켜더라도 **원본 비율 참고(2단계 비율 생성)**에는 적용하지 않는다. 기존 비율 경로의 입력·프롬프트를 유지한다. 별도의 GUI 체크박스는 추가하지 않았으며, 기존 `GENAI_STUDIO_RUNTIME` 실행 설정으로 켜고 끈다.

## 품질 문구 변경

긍정 문구에서 `best quality, coherent anatomy`를 제거하고 끝에 `masterpiece, best quality, very aesthetic, absurdres`를 붙인다. 부정 문구에서 아래 다섯 항목만 제거한다.

`different character, different hairstyle, different hair color, different eye color, missing character features`

시험에서 정한 품질 부정 문구를 중복 없이 추가한다. `nsfw, panties, underwear, buruma`, 성별·손·배경 관련 문구는 유지한다. 9/30 확정 부정 문구를 바꾸는 선택 사항이라는 점을 실행 기록에도 남긴다.

## 마무리의 고정 조건과 실패 처리

| 항목 | 값 |
| --- | --- |
| 재처리 | Lanczos 1.5배, 1104×1848 |
| img2img | strength 0.35, 28단계, CFG 5.5, 같은 seed |
| 스케줄러 | 기존 `scheduler_config()`의 Euler a |
| 메모리 | CPU offload, VAE tiling, 최대 reserved 6.5 GiB |
| 머리 검출 | 기존 캐시 `head_detect_v2.0_s`, CPU 별도 프로세스, 오프라인 |
| 얼굴 선택 | 점수 0.5 이상 중 최고 한 개 |
| 얼굴 보정 | 긴 변 ×1.6 상자, 이미지 경계로 제한, 1024×1024 재처리 후 원래 자리로 합성 |
| 합성 마스크 | 안쪽 여백 `max(2, 짧은 변//8)`, GaussianBlur(여백/2) |

설치된 Diffusers의 28단계·강도 0.35는 실제 UNet 9회 호출이다. 실행 중 호출 수·취소·메모리를 검사한다. 해상도를 1.25배로 낮추는 시험용 OOM 재시도는 넣지 않았다.

머리가 검출되지 않거나 점수가 기준보다 낮으면 `face_detail: skipped`와 사유를 기록하고 고해상도 결과를 유지한다. 모델 파일 누락·검출 프로세스 오류·취소·메모리 초과는 정상적인 검출 없음과 구분해 실패/취소로 기록한다. 원본과 이미 완료된 중간 파일은 보존한다.

흰 배경 처리만 실패하면 마무리 이미지를 표시하며, 화면에 실패 사실을 안내한다. 얼굴 보정이 실패한 결과를 완료된 결과로 승인시키지는 않는다.

## 기록과 코드 읽는 순서

읽기 시작점은 `genai_lab/studio_controller.py`의 `begin_generation()`이다. 생성이 끝난 뒤 `finish_batch()`를 호출하고, 그 결과에 흰 배경을 적용한다.

| 파일 | 책임 |
| --- | --- |
| `genai_lab/finishing_prompt.py` | 시험과 같은 품질 문구와 분할 인코딩 계획 |
| `genai_lab/studio_finishing.py` | 후보별 재처리·얼굴 보정·기록·검증 |
| `genai_lab/finishing_backend.py` | 별도 img2img 파이프라인과 GPU 한도 검사 |
| `genai_lab/finishing_head_worker.py` | CPU 머리 검출 |
| `genai_lab/studio_background.py` | 마무리 결과를 받아 흰 배경 처리 |
| `genai_lab/studio_generation.py` | 설정 읽기와 최종 결과 승인·저장 |
| `scripts/verify_studio_finishing.py` | Claude용 잠금 시험 재현 실행기 |

각 후보 폴더에 `raw.png`·기존 `run.json`은 변경 없이 남긴다. `finished_hires.png`, `finished.png`, `product.png`와 각각의 SHA가 추가된다. `finishing.json`에는 seed, 실제 문구와 토큰 계획, 머리 상자·점수·합성 상자, 단계별 시간·GPU 메모리·스케줄러를 남긴다. `finishing-status.json`은 배치 완료·실패·취소 상태를 기록한다. 승인 기록과 저장 이미지의 `.review.json`에도 마무리 기록을 연결한다.

## 해상도 변경의 영향

결과 표시·승인·저장은 실제 이미지 크기를 사용한다. Qwen 꼬리 고치기의 참조 사각형은 원본 캐릭터 좌표이므로 생성 결과의 확대와 별개다. Qwen 기반 이미지 크기는 1104×1848로 전달하고, 미리보기는 기존 직접 리사이즈 방식으로 이 크기에 복원한다. 꼬리 복잡도 분석은 원본 꼬리 크롭을 받으므로 이번 확대의 영향을 받지 않는다.

**남은 확인:** Qwen의 기존 출력 크기 규칙은 같은 종횡비에서 800×1312를 선택한다. 따라서 좌표는 맞아도 편집 뒤에 고해상도 세부 표현이 줄어들 수 있다. 이번에는 Qwen의 크기 규칙이나 설정을 바꾸지 않았다. 실제 보존 품질은 GPU 확인이 필요하다.

## CPU 확인 기록

- 관련 테스트 108개 통과. 문구 16건의 글자 일치, 설정 OFF, 비율 경로 제외, 실제 호출 인자, 순서, 오류·취소, 중간 파일 보존, 승인·저장, 좌표를 포함한다.
- 기존 시험 16건 모두 잠금 plan·로그·F0·F1·F2 SHA와 프롬프트 사전 검사 통과. GPU 실행 없음.
- 기존 첫 F1에 실제 CPU 머리 검출 실행: 상자 `[338,19,754,428]`가 시험과 일치하고 점수 0.917938(시험 기록 0.918)이다. 기록은 `outputs/finishing-test-20261008/codex-head-cpu01/heads.json`.
- 전체 테스트: 1,913개 통과, 기존 Pillow 사용 중단 예고 2건. 248.38초, CPU·오프라인·Qt offscreen 조건. GPU 생성 없음.
- 로그: `outputs/finishing-test-20261008/codex-cpu-tests.log`, `codex-preflight.log`, `codex-full-pytest.log`.

## Claude GPU 재현

기존 시험 F0를 새 폴더에 복사하고 제품 마무리 함수로 처리한다. 원본 시험 폴더는 덮어쓰지 않는다. `run.json`에 기존 F0 복사임을 명시하며 새로 생성했다고 기록하지 않는다. F1 또는 F2 SHA가 다르면 완료한 파일과 비교 기록을 남기고 다음 사례 전에 멈춘다. 불일치를 숨기거나 재시도하지 말고, 문구·임베딩·머리 검출 상자·환경 차이와 육안 결과를 보고한다.

먼저 첫 사례 한 건을 재현한다. 아래 `product-gpu-01`이 이미 있으면 새 이름을 쓴다.

```powershell
& 'D:\genai-cache\venv\Scripts\python.exe' -B '\\192.168.0.109\win_g\genai-lab\scripts\verify_studio_finishing.py' --cases 1 --run --output '\\192.168.0.109\win_g\genai-lab\outputs\finishing-test-20261008\product-gpu-01'
```

통과하면 새 폴더에서 `--cases 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16`으로 나머지를 확인한다. 같은 명령에서 `--run`과 `--output`을 빼면 GPU 없이 사전 검사만 수행한다.

## 실제 제품 경로 네 쌍 비교

시험용 런타임 JSON에 아래 설정을 넣고, `GENAI_STUDIO_RUNTIME`을 해당 파일로 지정한 뒤 GUI를 실행한다. 기존 런타임 JSON이 있다면 복사한 시험용 파일에서 두 항목만 추가하고 나머지 경로는 유지한다.

```json
{"quality_tags": true, "finishing": true}
```

원본 비율 참고는 끈다. 기존 시험에서 사용자가 승인한 캐릭터·의상 네 쌍을 그대로 사용한다. 캐릭터당 한 seed를 사전에 정하고 결과를 보고 바꾸지 않는다. 기존 얼굴 참조를 켠 제품 생성 원본과, 바로 그 원본을 마무리한 결과를 한 쌍으로 비교한다. 별도 생성으로 seed를 바꾼 결과를 대조군으로 쓰지 않는다.

마무리 효과 비교에서는 품질 태그를 양쪽에 동일하게 유지한다. 품질 태그 효과를 따로 비교할 때만 같은 seed·입력으로 `quality_tags`를 바꾼다. 결과 이름을 숨긴 비교에서 얼굴·색·체형·의상·추가 장식·세부 품질을 판정하고, 원본과 마무리의 해상도 차이도 함께 기록한다. 6.5 GiB 초과 또는 실행 오류는 중단한다.

사용자 확인 후 품질 태그와 마무리의 기본값을 정한다. 이번 변경은 GPU 생성·다운로드·패키지 설치·커밋·푸시를 하지 않았다. 별도 `outline_source` 작업은 유지했다.
