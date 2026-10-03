# Qwen 자세 편집 — B 조건 구현

작성: 2026-10-03. 코드·CPU 검증 단계. GPU 결과 재현은 아직 미검증이다.

## 사용 흐름

기준 이미지 생성 → 생성 자원 해제 → 완성 이미지 분석 → 분석 자원 해제 → 보존 문장 사용자 확인 → 별도 Qwen 프로세스 1회 → 사용자 검토·채택.

기존 생성 화면의 **승인한 완성 이미지 자세 편집 (Qwen, 선택 기능)** 버튼에서 런타임 JSON, 완성 이미지, 정규화한 반전 전 RGB 골격을 고른다. 실행 중인 생성·분석 작업이 있으면 시작하지 않는다. 자동으로 편집하거나 기존 선택 이미지를 덮어쓰지 않는다.

1. 완성 이미지를 보존 기준으로 승인한다.
2. 필요하면 기존 분석 근거·생성 태그 JSON을 불러온다. 완성 이미지 SHA가 다른 근거는 충돌 항목으로 남는다.
3. 분석을 실행한다. 캐시가 없으면 다운로드하지 않고 수동 입력으로 넘긴다.
4. 한국어 사실과 실제 전달할 영어 문장을 확인한다. 후보는 전부 미확인으로 시작한다. 출처·분석 충돌에는 해결 근거가 필요하다.
5. 시점과 손 동작은 선택 입력이며 기본값은 비어 있다. 입력했다면 별도로 확인한다.
6. 전달 문장 전체를 확인한 다음 1회 실행한다. 실패·취소 후 자동 재시도하지 않는다.
7. 결과에서 보존 항목별 유지/변형/소실·추가/가림으로 확인 불가와 자세·손·시점·그림체·구조·노출을 직접 평가하고 채택 여부를 고른다.

## 설정·환경

`QwenPoseSettings`의 JSON 한 곳에 python_executable, model_root, gguf_file, artifact_manifest, artifact_manifest_sha256, analysis_cache_dir를 둔다. 제품 코드에는 실행 드라이브나 환경 이름이 고정되어 있지 않다. analysis_cache_dir는 기존 WD 태거 Hugging Face 캐시 루트이며 비어 있으면 수동 입력이다.

필수 버전: torch **2.9.1+cu128**, diffusers **0.38.0**, transformers **4.57.6**, bitsandbytes **0.50.2**, gguf **0.19.0**. worker는 모델 import 전에 배포 메타데이터를 검사한다. 이 별도 환경이 운영 환경 패키지를 공유하더라도 불일치 시 중단한다. 실행 기록에 실제 패키지 위치를 남긴다. 설치·업데이트·환경 생성은 하지 않는다.

`python -m scripts.prepare_qwen_pose_config --python <기존 Qwen Python> --model-root <기존 모델 폴더> --gguf <기존 GGUF> --analysis-cache <기존 WD 캐시> --output <새 설정 폴더>`로 로컬 파일 해시 명세와 runtime.json을 만들 수 있다. CPU 작업이며 파일을 다운로드하지 않는다. 기존 폴더는 덮어쓰지 않는다. 모델 용량 때문에 해시 계산에는 시간이 걸릴 수 있다.

이 명세는 **선택한 로컬 파일의 잠금**이다. 모델 출처의 진위를 새로 검증하는 도구가 아니다. 검증자가 승인된 시험 캐시·revision과 같은 폴더인지 먼저 확인한다. 매 실행 시 누락·추가·변경된 모델 파일과 GGUF SHA를 검사한다. 경로 밖을 가리키는 심볼릭 링크는 지원하지 않고 중단한다.

고정값: group_block1, 40단계, true_cfg_scale=4.0, guidance_scale=1.0, 기본 seed=209212001. 알 수 없는 오프로드는 오류이며 자동 대체하지 않는다. API의 seed는 명시적으로 지정할 수 있고 GUI는 설정의 기본 seed를 쓴다.

## 분석 재사용 범위

| 자료 | 연결 방식 | 미확정 처리 |
|---|---|---|
| 완성 이미지 WD·의상 상세 | 기존 WdTagSession·analyze_garment_details, CPU·로컬 캐시만 | 위치·존재는 후보이며 사용자 확인 |
| 눈 색 | 기존 analyze_with_eye_review, 현재 이미지 머리 마스크를 API로 제공할 때 | 마스크 없으면 직접 입력 |
| 머리 상세 | 기존 analyze_hair_details, 현재 이미지 머리·얼굴 마스크를 API로 제공할 때 | 마스크 없으면 전체 태그 후보와 미확정 표시 |
| 부위 색 | PartColorDescription.record() 목록 불러오기 | 색이 무늬·위치를 증명하지 않음 |
| 귀·꼬리·장신구 | 기존 검출 보고 불러오기, 원문을 analysis.json에 보존 | 개수·무늬·형태 문장은 직접 입력 |
| 생성 태그 | generation_tags 목록 불러오기 | 요청한 태그이며 실제 이미지의 증거로 확정하지 않음 |

GUI는 새로운 분할을 실행하지 않는다. API에 마스크를 주입하거나 저장 분석을 가져오는 방식이며, 없는 근거는 수동 입력이다. 한국어 자유문장의 자동 영어 번역이나 외부 LLM 호출은 없다.

선택 JSON 형식:

```json
{
  "generation_tags": ["white camisole"],
  "reports": [{
    "source": "eye_color_analysis",
    "image_sha256": "분석에 사용한 이미지의 전체 SHA-256",
    "report": {"status": "suggested", "suggested_tag": "blue eyes"}
  }]
}
```

`source`는 part_color_descriptions, eye_color_analysis, garment_detail_analysis, hair_detail_analysis, extra_parts_analysis, accessory_analysis 중 하나다. report에는 각 기존 모듈의 원래 보고를 넣는다. part_color_descriptions의 report는 record() 목록이다. 이미지가 다르면 근거 충돌을 해결한 뒤에만 확인할 수 있다.

보존 항목 변경은 확인을 무효화하고, 기준 이미지가 분석 중·확인 후 바뀌면 실행을 차단한다. 어휘 검사는 알려진 위반만 잡는다. 자유문장의 모든 의미 충돌을 자동 판정하지 않는다.

## 골격·출력·기록

- `prepare_qwen_control`은 기존 크기·여백 규칙으로 만든 반전 전 RGB 사본을 반환한다. 호출자가 닫는다.
- `PreparedPose.copy_qwen_control()`도 소유권이 독립된 RGB 사본을 제공한다. 기존 control_image는 T2I 반전 후 이미지로 유지한다.
- GUI는 준비된 골격 PNG를 명시적으로 선택한다. RGB 파일이라는 사실만으로 실제 골격이나 색 순서가 올바름을 자동 판별하지 않으므로 표시된 골격을 확인한다.
- 그림 1의 비율로 출력 크기를 항상 명시한다. 736×1232 입력은 raw 800×1312, product 736×1207 + 위12/아래13px 흰 여백이다. raw는 수정하지 않는다.
- outputs/qwen-pose-edits/preparation-*/: analysis.json, draft.json.
- outputs/qwen-pose-edits/edit-*/: request.json, launcher.json, worker.log, run.json, model_inputs/, raw.png, product.png, user_review.json.
- 실제 VL 입력·VAE 입력의 시각화, prompt 전문·SHA, embedding 길이, 호출값, 패키지·모델 해시, raw SHA, 단계 시간, allocated/reserved를 남긴다. Windows 전용·공유 메모리 표본은 현재 미구현이며 null/미측정으로 기록한다.
- PyTorch의 부모 프로세스 할당·예약 메모리 잔존을 검사한다. GPU를 쓰는 별도 외부 프로그램까지 해제하는 기능은 아니다. 검증 전에 다른 GPU 작업을 종료한다.

## 한계와 검증 범위

B는 보존 보장이 아니다. 원 시험은 캐릭터 2명·seed 1개, 모든 조건 동시 충족 0/6이다. 목선·끈·장식·손·원근이 달라질 수 있고 1장 약27분이 걸렸다. 세부 참조 C, 사진 자세 입력, 추가 SDXL 생성, 자동 품질 판정은 범위 밖이다.

CPU 테스트와 실제 저장 B 문장·제어 이미지 대조는 구현 보고에 남긴다. GPU 출력 SHA·실제 취소·해제·메모리는 Claude 인수인계 문서대로 검증해야 한다.

## 근거

- [사용자 승인 설계](../QWEN_POSE_IMPLEMENTATION_DESIGN.md)
- [GPU 인수인계](../QWEN_POSE_GPU_HANDOFF.md)
- 로컬 시험: outputs/qwen-pose-identity-20261003/run_identity.py, preservation.json, prompts.json.
- [Diffusers 0.38 Qwen 편집 파이프라인 소스](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py): 명시 크기·임베딩 전달 계약의 근거. 실제 실행 환경의 설치 소스와 대조 후 사용한다.
- [Python subprocess 문서](https://docs.python.org/3.10/library/subprocess.html): shell=False 별도 프로세스·종료·대기.
