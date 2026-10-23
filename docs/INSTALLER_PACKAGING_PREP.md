# 설치형 앱 패키징 준비 — 구성 요소·용량·배포 라이선스 1차 목록 (2026-10-23)

> 지도 4절 병행 갈래(D-165). 범위: 지금 PC의 실제 구성을 읽어 목록만 만들었다. 코드·설정 변경, 설치·내려받기, GPU 실행은 하지 않았다.
> 목표 형태: **무료 설치형 Windows 앱, 서버 없이 사용자 PC에서 오프라인 실행**(DEPLOYMENT_API_COMPATIBILITY.md 2차 목표와 같은 방향).

## 결론

- **제품 실행에 필요한 것은 약 37GiB다**(모델 약 30GiB + Python 실행 환경 약 7GiB). 설치 파일 하나에 다 넣기는 크다.
  - 후보: 앱(작은 설치 파일) + **첫 실행 때 모델 받기**(고정 리비전·SHA 대조, 받은 뒤에는 오프라인). 결정은 사용자 몫이다.
- **제품 경로 모델의 라이선스는 대부분 Apache-2.0·MIT다.** 재배포 가능 — 고지·라이선스 문서 동봉 의무가 있다.
  - 예외: `deepghs/ccip_onnx`는 **OpenRAIL** → 재배포 시 사용 제한 조항을 최종 사용자에게 전달해야 한다.
  - see-through 원본 저장소의 라이선스 파일은 **아직 확인 못 했다**(포크는 Apache-2.0이라고 적음).
- **패키징 전에 고칠 것 5가지**(2절): 네트워크 공유 폴더를 가리키는 모델 경로, 설정 속 절대 경로, 실행 환경 3개, 개발용 패키지가 섞인 실행 환경, 제품에 안 쓰는 모델이 같은 캐시에 섞임.

## 1. 구성 요소

### 1-1. 모델 (제품 경로 — 실행 환경 점검 `scripts/check_runtime_env.py` 기준)

| 구성 요소 | 지금 위치 | 용량 | 출처·리비전 | 라이선스(HF 표시) | 설치 묶음 |
|---|---|---|---|---|---|
| Qwen-Image-Edit-2511(문자 인코더·VAE·설정 등) | `D:/genai-cache/models/Qwen-Image-Edit-2511` | 15.70GiB | Qwen/Qwen-Image-Edit-2511 @6f3ccc0b | Apache-2.0 | 필요 |
| Qwen 변환기 GGUF Q4_K_M | `D:/genai-cache/models/Qwen-Image-Edit-2511-GGUF/…Q4_K_M.gguf` | 12.34GiB | unsloth/Qwen-Image-Edit-2511-GGUF @0d33d969 | Apache-2.0 | 필요 |
| 머리 분할 체크포인트 | `D:\genai-cache\huggingface\models--24yearsold--l2d_sam_iter2` | 1.21GiB | 24yearsold/l2d_sam_iter2 | Apache-2.0 | 필요 |
| 머리 분할 코드(see-through) | `D:\genai-cache\tools\see-through` | 0.02GiB | shitagaki-lab/see-through @df019de5 | **확인 필요**(원본 LICENSE 파일) | 필요 |
| WD 태거 v3 | `D:\genai-cache\huggingface\models--SmilingWolf--wd-vit-tagger-v3` | 0.35GiB | SmilingWolf/wd-vit-tagger-v3 | Apache-2.0 | 필요 |
| 관절 분석 DWPose | `D:\genai-cache\huggingface\easy-dwpose`(0.33) · `models--yzd-v--DWPose`(0.33) | 0.33~0.66GiB | yzd-v/DWPose | Apache-2.0 | 필요(같은 가중치 두 벌인지 확인 필요) |
| 캐릭터 분할 anime-seg | `D:\genai-cache\huggingface\models--skytnt--anime-seg` | 0.16GiB | skytnt/anime-seg | Apache-2.0 | 제품 경로 사용 여부 확인 필요 |
| 머리 검출 | **`\\192.168.0.109\win_g\genai-lab\outputs\…\model-cache\hub`** | 0.18GiB(아래 둘 합) | deepghs/anime_head_detection | MIT | 필요 |
| 캐릭터 특징 비교 CCIP | 같은 곳 | (위에 포함) | deepghs/ccip_onnx | **OpenRAIL** — 사용 제한 조항 전달 | 제품 경로 사용 여부 확인 필요 |

**제품에 넣지 않는 것(같은 폴더에 있음):**

| 구성 요소 | 용량 | 이유 |
|---|---|---|
| 8단계 가속 LoRA(lightx2v, Apache-2.0) | 0.79GiB | 시험용(D-141). 제품은 40단계 |
| 빌린 그림체 LoRA(anime_borrowed) | 0.22GiB | 시험 전용·학습 자료 미공개(D-159·D-162) |
| 손 검출(deepghs/anime_hand_detection, OpenRAIL) | 0.04GiB | 오프라인 채점 전용(D-157) |

### 1-2. 실행 환경 (Python)

| 환경 | 용량 | 쓰임 | 메모 |
|---|---|---|---|
| `D:\genai-cache\venv` | 5.93GiB | GUI·분석(PySide6, torch 등) | 시험 도구(pytest 등)도 함께 들어 있음 |
| `D:\genai-cache\qwen-test-venv` | 0.02GiB | Qwen 작업 프로세스 | `.pth`로 위 venv를 참조. 이름이 '시험'용 |
| `D:\genai-cache\catvton-venv` | 0.83GiB | 관절 분석(easy_dwpose·onnxruntime) | 예전 CatVTON용 이름이 남음 |

- 주요 패키지 라이선스(일반적으로 알려진 값, 설치본의 LICENSE로 다시 확인): torch(BSD), diffusers·transformers(Apache-2.0), bitsandbytes·gguf·onnxruntime(MIT), **PySide6(LGPL-3.0 — 동적 연결 유지, 교체 가능하게, 라이선스 문서 동봉)**.

## 2. 패키징 전에 고칠 것 (발견만 — 고치는 것은 사용자 승인 뒤)

1. **머리 검출 모델 캐시가 네트워크 공유 폴더(`\\192.168.0.109\…\outputs\…`)를 가리킨다.** 다른 PC에서는 없다 → 모델 폴더 아래로 옮기고 설정 기본값을 바꿔야 한다.
2. **설정에 절대 경로가 박혀 있다**(`configs/qwen_edit_settings.json`의 `D:/genai-cache/…`, `StudioRuntime` 기본값). 설치 위치 기준 상대 경로나 설치 때 정하는 경로로 바꿔야 한다.
3. **실행 환경이 3개다.** 하나로 합칠 수 있는지(관절 분석 패키지 충돌 여부) 확인해야 한다. 합치지 못하면 이름을 제품용으로 정리한다.
4. **실행 환경에 개발용 패키지가 섞여 있다.** 제품용 최소 목록(requirements)을 따로 만들어야 한다.
5. **제품에 안 쓰는 모델이 같은 캐시에 있다**(1-1 아래 표). 설치 묶음 목록을 '제품 경로에서 실제로 불러오는 파일'로 정하고, 그 목록을 시험으로 고정해야 한다.

## 3. 배포 전 이미 정해진 일 (다른 문서)

- 노출 검사를 `"block"`으로 되돌리고 판정 방식을 다시 만든다(지도 보류 목록, D-158).
- AI 생성 표시(PNG 파일 정보)는 이미 저장마다 들어간다(D-152).
- 그림체 LoRA를 넣는다면 학습 자료의 앱 배포·오프라인 실행 권리를 확인한 뒤다(STYLE_TRAINING_DESIGN 6-3).

## 4. 다음 단계 후보 (사용자 선택)

| 순서 | 내용 | GPU | 비고 |
|---|---|---|---|
| ① | 제품 경로가 실제로 여는 모델 파일 목록을 CPU 시험으로 고정(가짜 실행 + 파일 열기 기록) | 없음 | anime-seg·CCIP·DWPose 두 벌 여부가 여기서 정해진다 |
| ② | 2절 1·2번(경로) 수정: 모델 폴더 한 곳 + 설정의 상대 경로 | 없음 | 코드 변경 — 시험 후 커밋 |
| ③ | 설치 방식 결정: 한 묶음 / 앱 + 첫 실행 모델 받기 | 없음 | 사용자 결정 |
| ④ | 제품용 최소 실행 환경 만들기(새 venv에서 점검 통과) | 없음 | 설치 작업 — 승인 필요 |
| ⑤ | 다른 PC 설치 → 첫 생성(배포 호환성) | 있음 | 지도 4절 2번 |

## 출처 (라이선스 표시, 2026-10-23 확인)

- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) · [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) · [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
- [24yearsold/l2d_sam_iter2](https://huggingface.co/24yearsold/l2d_sam_iter2) · [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through)
- [SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) · [yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose) · [skytnt/anime-seg](https://huggingface.co/skytnt/anime-seg)
- [deepghs/anime_head_detection](https://huggingface.co/deepghs/anime_head_detection) · [deepghs/ccip_onnx](https://huggingface.co/deepghs/ccip_onnx)
