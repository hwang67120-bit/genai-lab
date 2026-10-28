# 출력 좌표 기반 의상 국소 정밀화

## 실행 흐름

1. 캐릭터·의상 참조를 분석하고 승인된 태그와 시각 참조를 봉인한다.
2. Animagine Base 후보를 생성한다.
3. 선택된 Base에서 의상·전경·얼굴·헤어·동물귀·헤어 장식·꼬리를 다시 검출한다.
4. 기존 의상 제거 범위와 승인 태그에서 계산한 목표 착용 범위를 결합한다.
5. 얼굴·헤어·동물귀·헤어 장식은 Hard 보호하고, 꼬리는 조건부 보호한다.
6. Hard 편집 허용 범위 내부에서만 침식과 Gaussian feather를 적용해 Soft 조건 마스크를 만든다.
7. Soft 마스크는 SDXL Inpaint의 편집 마스크와 IP-Adapter Attention 마스크에 전달한다.
8. 결과는 합성하거나 Hard paste하지 않고 외부 RGB 변화율을 진단한다.
9. 의상 유사도, 목표 착용 범위, 마스크 충돌, 외부 변화율을 후보 JSON과 로그에 기록한다.

## 마스크 계약

```text
requested_edit =
    source_garment_removal
  | target_garment_coverage
  | user_add

effective_protection =
    hard_protection
  | (conditional_protection & ~conditional_release)

hard_edit_domain =
    requested_edit
  & (foreground | garment_growth_envelope)
  & ~user_exclude
  & ~effective_protection

soft_guidance =
    inward_feather(hard_edit_domain)
  * (1 - soft_boundary_protection)
```

참조 이미지의 픽셀 마스크는 Base 좌표에 복사하지 않는다. 참조 태그는 의상 의미만 제공하며 모든 실행 마스크는 생성된 Base 좌표에서 만들어진다.

## 보호 대상

- Hard 보호: 얼굴 중심, 헤어, 검증된 동물귀, 헤어 장식
- 조건부 보호: 꼬리
- 진단 전용: 독립 사람귀 검출 결과
- 사람귀 보호: 얼굴·머리 보호에 포함
- 교체 대상: 기존 의상과 목표 의상이 덮어야 하는 영역

동물귀 모양 후드가 실제 동물귀로 검출되는 문제는 출력 부위 진단 이미지에서 확인해야 한다. 자동 검출 결과에는 각각의 부위 마스크가 별도 PNG로 저장된다.

## 진단 산출물

선택 후보의 `output_coordinate_regions_directory/local_refinement_masks`에 다음 파일을 저장한다.

- `source_garment_removal.png`
- `target_garment_coverage.png`
- `garment_growth_envelope.png`
- `requested_edit.png`
- `hard_edit_domain.png`
- `hard_protection.png`
- `conditional_protection.png`
- `effective_protection.png`
- `mask_conflict.png`
- `soft_guidance.png`
- `overlay.png`
- `garment_edit_plan.json`
- 검출된 얼굴·사람귀·동물귀·헤어·장식·꼬리 마스크

## 현재 판정 범위

목표 착용 범위는 생성된 Base의 전경, 현재 의상 위치, 승인 의상 태그를 사용하는 제한된 투영이다. 결과는 `REVIEW`로 기록한다. 서 있는 단일 캐릭터를 우선 대상으로 하며, 자세가 복잡하거나 의상 범위가 잘린 참조는 DWPose/SCHP 또는 사용자 승인 마스크가 추가로 필요하다.

의상 유사도가 낮거나 외부 RGB 변화가 관찰되어도 결과를 즉시 폐기하지 않는다. `corrected_review_required`와 수치를 반환해 다음 미세 조정 판단에 사용한다. 크기 불일치, 빈 마스크, 좌표 계약 위반은 실행 전에 중단한다.

## 비합성 계약

의상 정밀화 결과에는 다음 연산을 사용하지 않는다.

- 이미지 병합
- 디코딩 후 Hard paste
- 원본 픽셀 강제 복원
- 참조 이미지 마스크 좌표 복사

따라서 마스크 외부 RGB 0% 변경을 보장하지 않는다. 대신 `outside_changed_ratio`와 평균 절대 차이를 기록한다. 기존 Hard paste 기반 헤어·동물귀·꼬리 보정은 현재 설정에서 비활성화한다.

## 근거

- Diffusers IP-Adapter masking: https://huggingface.co/docs/diffusers/main/using-diffusers/ip_adapter
- Diffusers Inpainting API: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/inpaint
- Blended Diffusion: https://arxiv.org/abs/2111.14818

