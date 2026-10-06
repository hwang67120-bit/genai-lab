# 꼬리 편집 전 이미지 인식 연결

이미지 인식은 꼬리 고치기 입력을 확인한 뒤 한 번 실행한다. 선택한 결과 이미지에서 보존할 외형을 읽고, 원본의 꼬리 크롭에서 바꿀 특징을 읽는다. 모델 응답을 사용자가 확인한 뒤에만 Qwen 편집 지시문에 추가한다. 자동 승인이나 자동 재생성은 없다.

## 화면 흐름과 읽기 시작점

1. 작업실에서 결과 한 장 선택 → 꼬리 고치기 → 원본의 꼬리 영역 지정.
2. `그림을 읽어 색·형태 설명 보강`을 켜고 실행한다. 준비된 개인 설정이 있으면 켜진 상태로 표시한다. 끄면 기존 지시문을 그대로 사용한다.
3. 별도 인식 프로세스가 꼬리 크롭과 선택 결과를 차례대로 읽고 종료한다. 원본 전체를 Qwen 편집의 두 번째 이미지로 전달하지 않는다.
4. 확인창에서 잘못 읽은 항목을 제외한다. 설명은 실제 지시문에 들어갈 영어이며, 항목 이름과 안내는 한국어다. 무늬 없음과 직접 입력한 끝 문장이 모델 설명보다 우선한다.
5. 사용자가 확인하면 기존 Qwen 편집 → 원본/편집 전/편집 후 비교 → 채택 또는 취소.

주 흐름은 `genai_lab/studio_controller.py`의 `edit_tail → tail_recognized → launch_tail_edit`이다. 인식 실패, 취소, 시간 초과이면 편집을 시작하지 않고 기존 후보를 유지한다.

| 책임 | 파일 및 함수 |
|---|---|
| 인식 입력, 캐시, 프로세스 수명 | `genai_lab/tail_recognition.py`: `recognize_tail`, `run_worker` |
| 실제 로컬 모델 추론 | `genai_lab/tail_recognition_worker.py`: `infer`, `inspect_image` |
| 인식 설명 확인 | `genai_lab/qwen_tail_gui.py`: `review_tail_recognition` |
| 확인한 설명 전달 | `genai_lab/qwen_tail_edit.py`: `assemble_tail_prompt`, `TailEditSpec.verify_image` |
| 명시적 모델 설치 | `scripts/provision_tail_vision.py` |
| 이미지 생성 없는 인식 검증 | `scripts/verify_tail_recognition.py` |

## 모델과 실행 환경

- 모델: `Qwen/Qwen3-VL-2B-Instruct`, Apache-2.0.
- revision: `89644892e4d85e24eaac8bacfd4f463576704203`.
- 가중치 SHA-256: `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
- 설치 파일 11개, 총 4,266,647,442바이트. 패키지 설치·변경 없음.
- 공식 모델 카드: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- 실행 시에는 로컬 모델만 사용하며 다운로드하지 않는다. 파일 명세와 SHA를 확인한 뒤 모델을 적재한다.
- 인식은 기존 운영 Python을 별도 프로세스로 호출한다. GPU 기본값은 bfloat16/SDPA, CPU 검증은 float32. Qwen 이미지 편집 환경은 변경하지 않는다.
- 요구 버전: torch 2.9.1+cu128, transformers 4.57.6, Pillow 12.3.0. 다르면 인식 실행 전 오류 처리.
- 사용자별 설정: `~/.genai-lab/vision-settings.json`; `GENAI_VISION_SETTINGS` 또는 입력창의 설정 파일 선택으로 변경할 수 있다. 운영 코드에 드라이브 문자를 고정하지 않았다.
- 설정 항목: python_executable, model_root, manifest_sha256, cache_dir, model_revision, device. 기본 한도는 max_pixels=262144, max_new_tokens=384, timeout_seconds=240이다. 이미지 해상도 축소는 인식 모델 입력에만 적용한다.

## 메모리와 재사용

이전 생성 모델 해제 → 인식 자원 확인 → 인식 프로세스 → 종료 대기 → 사용자 확인 → Qwen 자원 확인 → 편집 순서다. 인식 모델은 확인 화면과 Qwen 실행 중에 남아 있지 않는다.

꼬리와 선택 결과의 인식 캐시는 따로 둔다. 이미지 SHA, 역할, 모델 revision·명세 SHA, 프롬프트, 전처리·출력 한도, 장치, 버전으로 키를 만든다. 후보만 바뀌면 같은 꼬리 인식은 재사용한다. 크롭이나 인식 조건이 바뀌면 기존 캐시를 쓰지 않는다. 두 항목이 모두 있으면 모델 프로세스를 시작하지 않는다. 캐시가 손상되면 성공으로 넘어가지 않는다.

각 편집 폴더의 `recognition/run.json`은 캐시 적중 수, 전체 시간, 모델 적재 시간, 장치, GPU allocated/reserved 피크(실제 GPU 실행 때)를 남긴다. `tail-response.json`, `basis-response.json`에 원시 모델 문장과 입력·출력 토큰 수를 남긴다. 인식 결과는 `observations.json`, 사용자 확인은 `recognition-review.json`, 최종 전달 내용은 기존 편집 `request.json`에 남는다.

## 검증 범위

CPU 단위 테스트는 입력 SHA 연결, 기존 프롬프트 동일성, 항목 제외, 미확인 처리, 캐시 재사용과 무효화, 취소·시간 초과·실패, 프로세스 종료 후 편집, 기존 결과 보존을 확인한다. 합성 입력으로 확인창 배치도 점검했다.

실제 R1의 원본 꼬리 크롭과 선택 결과를 CPU로 인식한다. 이는 이미지 인식이며 새 이미지를 생성하지 않는다. 인식 정확도와 편집 품질은 별개다. 초기 확인에서 일부 항목이 질문 문구를 복사한 것이 발견되어, 해당 응답은 미확인으로 제외하고 원인을 기록하도록 보완했다. 색이나 모양을 잘못 읽을 수 있으므로 사용자 확인이 필수다.

GPU 인식 속도·메모리 및 인식 문구를 넣은 Qwen 편집 품질은 아직 검증하지 않았다. 기존 R1~R5의 SHA 재현 결론을 새 문구를 넣은 경로에 적용하지 않는다. 인식을 끈 기존 경로의 지시문과 설정은 유지된다.

## Claude 검증 순서

패키지와 모델을 추가로 설치하지 않는다. 운영 venv를 쓰며 경로는 현장 설정을 따른다.

```powershell
# CPU 파일·버전 점검만
& <운영Python> -m scripts.verify_tail_recognition

# 인식만 실행: 기존 selection.json 또는 spec이 있는 preflight.json 사용
# GPU로 확인할 때 CPU 테스트용 CUDA_VISIBLE_DEVICES='' 환경변수를 해제한다.
& <운영Python> -m scripts.verify_tail_recognition --recognize --selection <기존입력기록> --output <새확인폴더>

# 같은 입력 재실행: cache_hits=2, 모델 프로세스 없음 확인
& <운영Python> -m scripts.verify_tail_recognition --recognize --selection <같은입력기록> --output <다른확인폴더>
```

그다음 GUI에서 꼬리 고치기 → 인식 → 틀린 항목 제외 → 확인 → 편집 → 비교·저장을 확인한다. 인식만 확인한 결과로 꼬리 색·외형 보존이 해결됐다고 판정하지 않는다. 인식 과정에서 새로 생긴 설명은 이전 프롬프트와 구분해 비교한다.

## 실제 CPU 관찰의 한계

기존 R1을 읽었을 때 꼬리는 blue / striped / thick / curved / rounded로 나왔다. 원본의 밝은 색과 어두운 띠, 보라색 끝을 세분하지 못했고 가려진 끝 모양을 확정하는 오류 가능성도 남았다. 질문을 정리한 뒤에는 보라색 눈·흰 상의·검은 반바지를 기술했지만, 귀를 cat ears로 추측했다. 질문 반복과 동물 종 이름 추정은 최종 전달 항목에서 제외하고 미확인 사유를 남긴다. 이러한 문법 필터가 색이나 형태의 시각적 정확성을 보장하지는 않는다.

인식 설명을 사용자가 확인하는 이유가 이것이며, 이 결과를 꼬리 색 또는 전체 외형 보존 해결로 보고하지 않는다. CPU 원시 응답과 수정 전후 기록은 outputs/tail-recognition-20261006/ 아래에 남긴다.


최종 CPU 회귀: 1618 passed, 기존 경고 2건, 236.98초. 관련 테스트 74개 통과. 상세 결과와 코드 SHA는 outputs/tail-recognition-20261006/results.md 및 code-sha256.json에 기록했다. 커밋·푸시는 하지 않았다.
