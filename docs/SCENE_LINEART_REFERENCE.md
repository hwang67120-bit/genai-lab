# 장면 선화 + 부위별 RGB 참조: 실험 경로

## 현재 상태

2026-09-08 구현. 기존 참조 생성의 기본 동작은 변경하지 않는다.
scene_lineart.enabled 기본값은 false다.

이 경로는 **검증된 SceneAnalysis를 받아 생성까지 연결하는 기반**이다.
AutomaticSceneAnalyzer는 Protocol이며, 의상 단독 기준점과 특수 부위의
앞뒤 가려짐을 자동 판별하는 실제 모델 구현은 아직 제공하지 않는다.
설정만 켜면 작동하는 완성된 자동 분석 기능으로 안내하면 안 된다.
분석기 없이 활성화하면 모델 로드 전에 명확한 오류로 중단한다.

## 구현한 것

- 장면 분석 결과의 공통 좌표계, 부위별 RGB/영역/강도, 의미 있는 대응점 계약.
- 초기 RGB, 인페인팅, 원본 픽셀 덧씌우기 없이 선화만 ControlNet에 전달.
- 최대 3회의 제한된 분석 재시도. 미검출을 부위 부재로 확정하지 않음.
- TPS 역매핑, 좌표 검증, 청크 평가, premultiplied alpha 보간,
  빈 결과/국소 뒤집힘/허용 영역 이탈 검사.
- 기존 의상 선화를 제외하고 새 의상 및 확정된 전면 부위를 합성.
- 얼굴·헤어를 새 의상이 덮으면 중단. 특수 부위별 RGB 참조 유지.
- 실제 IP-Adapter 입력 직전 PNG/통계 저장과 미리보기/실제 참조 일치 검사.
- SDXL Text2Image ControlNet 로드, CPU offload, 기존 후보 비교·승인 UI 재사용.
- 선화/부위 참조 미리보기와 단계 완료 시 시간·VRAM 로그.

## 연결 계약

1. 기존 GUI의 정규화된 source 캔버스와 승인된 의상 입력을 받는다.
2. scene_lineart.analyzer에 AutomaticSceneAnalyzer 구현 객체를 런타임에 주입한다.
   YAML에서 임의 모듈 경로를 동적 실행하는 기능은 제공하지 않는다.
3. analyze()는 반환 이미지들을 소유한 SceneAnalysis를 반환한다.
   사용자 입력 객체를 그대로 반환하지 말고 복사본을 사용한다.
4. src_points는 garment_rgba의 좌표, dst_points와 모든 구조 마스크는
   source 출력 캔버스의 좌표다. point_labels의 순서는 두 배열에서 동일하다.
5. parts에는 identity(얼굴·헤어)를 반드시 포함한다. 귀/꼬리/날개 등은
   각각 분리한다. 중복 영역은 분석기가 해소해야 한다.
6. 앞뒤가 불확실하거나 기준점이 부족하면 unresolved를 기록한다.
   깊이 중앙값만으로 확정하거나, 의상 단독 입력에서 사람 관절을 강제하지 않는다.
7. close()는 분석기가 소유한 모델과 GPU 텐서 참조를 해제해야 한다.
   생성 모델을 먼저 GPU에 올려 둔 상태에서 전처리를 실행하지 않는다.

keep_mask는 기존 의상과 겹치지 않는다. front_mask는 keep_mask의 부분집합이다.
RGB 추출용 부드러운 알파와 구조용 0/255 마스크를 분리한다.
배치 허용 영역은 원래 의상 마스크의 단순 복제가 아니라 새 의상 디자인과
몸의 위치를 고려해 분석해야 한다. 의상 착용자 이미지의 크롭/리사이즈 변환도
분석기가 기록해야 한다.

## 모델과 환경

- 베이스 모델과 IP-Adapter는 기존 프로젝트 설정 재사용.
- 선택적인 SDXL ControlNet: TheMistoAI/MistoLine.
- 선화 추출기: controlnet_aux.LineartDetector, CPU 실행.
- 선택적 의존성: requirements-scene-lineart.txt.
- ControlNet/SDXL 실험 로더는 local_files_only=True. 캐시를 사전에 준비한다.
- LineartDetector 자체는 캐시에 없으면 모델 다운로드를 시도할 수 있다.
- 새 모델의 실제 다운로드/로드/생성 테스트는 이번 구현 검증에 포함하지 않음.
- 테스트 환경에는 scipy 1.15.3만 --no-deps로 추가했다.

SD1.5 Lineart ControlNet을 SDXL에 혼용하지 않는다. 선화 극성은 실제
추출 결과로 확인한 후 extractor_output_white_lines를 설정한다.
MistoLine 모델 카드의 라이선스/상업적 사용 조건을 확인한다.

## 시간 제한과 품질 한계

추론 callback에서 취소/제한 시간을 검사한다. 이 검사는 **각 단계가 끝나야**
호출된다. 모델 로딩 또는 단일 CUDA 연산의 정지를 강제로 끊는 watchdog은 아니다.
강제 중단이 필요하면 별도 생성 프로세스와 프로세스 수명 관리가 추가로 필요하다.

입력 통계나 단위 테스트 통과는 캐릭터/종족/의상 디자인 일치의 증거가 아니다.
TPS 뒤집힘 검사는 과도한 늘어짐이나 소매 정합성 전체를 보장하지 않는다.
현재 이 경로의 전신 분위기 비교 마스크는 자동 분석 계약에 없으므로 계산 불가로
표시한다. 전체 원본 배경을 대신 비교하거나 생성 참조로 넣지 않는다.

실제 검증에는 동일 시드의 선화 OFF/ON 비교, 얼굴·헤어·꼬리 색상,
의상 단독/착용 사진/가려짐 사례, 두 번째 후보의 시간·VRAM 비교가 필요하다.
실패를 임의의 좌표나 기존 RGB 초기 이미지로 우회하지 않는다.

## 근거

- [SDXL ControlNet Text2Image API](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl)
- [IP-Adapter multiple references and masks](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)
- [SciPy TPS/RBF solvability conditions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)
- [Diffusers memory/offloading](https://huggingface.co/docs/diffusers/optimization/memory)
- [MistoLine model card](https://huggingface.co/TheMistoAI/MistoLine)
- [Lineart extractor implementation](https://github.com/huggingface/controlnet_aux/blob/master/src/controlnet_aux/lineart/__init__.py)
