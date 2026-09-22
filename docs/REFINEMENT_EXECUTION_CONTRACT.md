# 최종 정밀화 실행 계약

## 목적

승인된 Animagine Base 이후의 정밀화는 한 요청에서 하나의 엔진만 실행한다. GUI, CLI, GPU 검증은 모두 `GenerationOrchestrator`와 동일한 요청 컨텍스트를 사용한다.

## 1~7단계 유도 계약

1. `refinement_execution.mode`를 `sdxl_local` 또는 `flux_whole_image`로 명시한다. 두 모드는 동시에 실행되지 않는다.
2. `GenerationOrchestrator.finalize_selected_candidate()`가 봉인된 모드 하나로만 분기한다.
3. 요청마다 `GenerationRunContext`를 생성하고 `inputs`, `base`, `masks`, `refinement`, `diagnostics` 경로를 분리한다. 학습 자료 루트와 생성 결과 루트가 겹치면 시작 전에 거부한다.
4. 요청 시작·종료·실패 경계에서 interrupt, IP-Adapter scale, model hook, 캐시된 inpaint pipeline을 초기화한다.
5. 생성된 Base 좌표에서 최대 유도 범위와 `soft_guidance`를 다시 계산한다. `hard_edit_domain`은 하위 호환 진단 이름이며 binary 강제 편집, latent 복원, RGB 덮어쓰기 또는 최종 합성에 사용하지 않는다. 참조 이미지의 분석 좌표를 Base 좌표로 직접 투영하지 않는다.
6. 시간적 유도는 초기 정체성·인체, 중간 의상 실루엣, 후반 색상·소재·장식 순서로 조건 강도를 전환한다. 단계적 유도는 기술적으로 정상인 직전 출력을 다음 단계 앵커로 사용한다. 엔진이 해당 제어를 지원하지 않으면 `unsupported`로 기록하고 적용한 것으로 간주하지 않는다.
7. GUI는 실행 모드, 요청·적용·미지원 유도, Soft 지도, 보호 감쇠, 단계별 변화와 인물 수 진단을 공개한다. Grounding DINO 인물 수 검출은 현재 `blocking: false`이며 오검출 하나만으로 결과를 폐기하지 않는다.

## 생성 개입과 반환 경계

- `soft_guidance`만 영역별 생성 강도를 제어한다. 경계는 Gaussian feather를 사용하고 공통 계약에 고정 가중치를 하드코딩하지 않는다.
- 얼굴·헤어·사람 귀·검출된 동물귀·헤어 장식 보호 지도는 편집 영역을 잘라내는 차집합 마스크가 아니라 조건 감쇠 지도다.
- 픽셀 손상, 비정상 latent, 다중 인물, 명백한 구조 붕괴와 승인 입력 불일치만 반환을 차단한다.
- 캐릭터·헤어·의상·색상 유사도 미달은 결과와 함께 진단으로 반환한다.
- 중간 단계가 품질 목표에 미달하면 전체 실행을 폐기하지 않고 최신 기술적 안전 결과와 미세조정 대상을 반환한다.

## 모드별 변경 범위

| 모드 | 입력 | 변경 가능 범위 | 실패 시 반환 |
|---|---|---|---|
| `sdxl_local` | 승인 Base, 출력 좌표 Soft 유도 지도, 넓은 최대 유도 범위, 격리 의상 참조 | Soft 지도 강도에 따른 국소 인페인트 | 국소 결과가 기술적 안전 검사를 통과하지 못하면 승인 Base 고정 |
| `flux_whole_image` | 승인 Base, 격리 의상 보드, 승인 프롬프트 계약, 엔진이 지원하는 유도 스케줄 | 전체 이미지의 단계적 재정밀화 | FLUX 결과가 기술적 안전 검사를 통과하지 못하면 승인 Base·Seed 고정 |

두 모드 모두 이미지 병합, hard paste, 마스크 합성 및 마스크 외부 픽셀 강제 복원을 사용하지 않는다. 최대 유도 범위는 진단과 누설 측정에 사용하고, Soft 지도만 실제 조건 강도를 표현한다.

## 진단 산출물

각 요청은 `outputs/generation-runs/<run-id>/`에 다음을 남긴다.

- `run-context.json`: 요청 ID, 모드, 이벤트, 입력 파일 SHA-256
- `inputs/`: 승인 실행 기록과 의상 참조
- `base/`: 선택된 Base와 후보 기록
- `masks/`: 최대 유도 범위, Soft 유도 지도, 보호 감쇠, 충돌 및 오버레이
- `refinement/`: 선택 모드 결과, 시간 유도 스케줄, 단계별 입력·출력 해시, 요청·적용·미지원 제어와 최종 제품 증거
- `diagnostics/`: 인물 수 검출 오버레이

## 근거

- [Diffusers Inpainting](https://huggingface.co/docs/diffusers/main/using-diffusers/inpaint): 흰색 마스크 영역을 편집하고 검은색 영역을 보존하는 공식 마스크 의미.
- [Diffusers IP-Adapter](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter): 이미지 조건의 scale 제어와 영역별 IP-Adapter mask 전달 방식.
- [Grounding DINO 논문](https://arxiv.org/abs/2303.05499): 텍스트 범주로 열린 집합 객체를 검출하는 인물 수 진단 근거.

## 롤백 경계

기본 모드는 `flux_whole_image`다. 유도 변경은 시간 스케줄, Soft 공간 지도와 단계 구성의 독립 프로필로 저장한다. 문제가 생기면 해당 프로필만 이전 값으로 되돌릴 수 있으며, 요청별 산출물이 분리되어 있어 기존 승인 Base와 다른 요청의 입력 파일은 변경되지 않는다.

이 문서는 다음 구현의 기준 계약이다. 실행 코드와 보고서 스키마가 이 항목을 기록하기 전에는 기존 GPU 결과를 시간·공간·단계 유도가 적용된 결과라고 표시하지 않는다.
