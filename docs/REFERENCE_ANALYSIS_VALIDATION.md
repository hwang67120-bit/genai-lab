# 참조 분석 검증 계약

## 목적

참조 이미지에서 동물귀, 헤어 장신구, 꼬리를 찾는 분석기가 실제로 어느 정도 정확한지 생성과 분리해 측정한다. 검증 중에는 이미지 생성, 국소 정밀화, 보호 마스크 적용, 자동 프롬프트 주입을 실행하지 않는다.

## 판정 흐름

1. 사람이 세 부위를 각각 `present`, `absent`, `ambiguous`로 표기한다.
2. 현재 GroundingDINO·SAM 분석 설정으로 동물귀, 헤어 장신구, 꼬리를 독립 분석한다.
3. 검출 후보가 없으면 `no_detection`, 후보가 있으나 마스크 품질이나 문맥 검토를 통과하지 못하면 `uncertain`, 진단 마스크가 있으면 `detected`로 기록한다.
4. 동물귀와 장신구 진단 마스크를 각각 보존하고 중첩 마스크를 별도로 만든다.
5. 두 마스크가 겹치면 `overlap_ambiguous`로 기록한다. 하나를 승자로 선택하거나 서로 빼지 않는다.
6. `no_detection`을 부위 없음으로 자동 확정하지 않는다.
7. 정답과 예측을 비교해 검출 및 좌표 품질을 기록한다.

## 비강제 교차 진단

동물귀와 헤어 장신구는 머리 상단의 비슷한 좌표에 있을 수 있다. 검증기는 이 중첩을 오류로 즉시 차단하지 않는다.

- `animal_ears_mask`와 `hair_accessory_mask`를 독립 보존한다.
- 두 마스크의 공통 픽셀과 작은 마스크 기준 중첩률을 기록한다.
- `winner_selected=false`, `masks_modified=false`를 기록한다.
- `automatic_conditioning_applied=false`를 기록한다.
- 생성·보정·게이트 정책은 변경하지 않는다.

## 공간·연결·교차 클래스 진단

이 진단은 검출 결과를 수정하지 않는 observe_only 단계다.

- **머리 ROI**: hair 문맥 마스크가 있으면 그 경계를 확장해 사용한다. 문맥 마스크가 없으면 이미지 상단 55%를 대체 ROI로 사용하고 normalized_image_fallback이라고 명시한다. 동물귀와 헤어 장신구 후보가 ROI와 겹친 비율을 기록한다.
- **신체 연결 위치**: garment 문맥 마스크가 있으면 의상 경계의 중하단을 꼬리 연결 후보 영역으로 사용한다. 없으면 중앙 골반 대역을 대체 영역으로 사용한다. 꼬리 후보가 이 영역과 접촉하지 않으면 outside_body_anchor로 기록한다.
- **동일 마스크 교차 클래스 충돌**: 관측된 모든 부위 쌍의 작은 마스크 포함률과 IoU를 계산한다. 포함률 0.90 이상이고 IoU 0.60 이상이면 same_mask_cross_class_conflict로 기록한다.

대체 ROI는 정확한 해부학 좌표가 아니므로 결과를 차단 근거로 사용하지 않는다. 모든 진단은 predictions_modified=false, masks_modified=false, automatic_conditioning_applied=false, generation_policy_changed=false를 기록한다.

GroundingDINO는 텍스트 질의로 박스를 찾는 개방형 검출기이고 SAM2는 박스 같은 시각 프롬프트로 마스크를 만든다. 따라서 SAM2의 높은 마스크 품질 점수는 동물귀·장신구 같은 의미 클래스가 맞다는 보장이 아니다. 후단 진단은 박스와 마스크의 공간 관계를 별도 증거로 기록한다.

- GroundingDINO 공식 구현: https://github.com/IDEA-Research/GroundingDINO
- GroundingDINO 논문: https://arxiv.org/abs/2303.05499
- SAM2 논문: https://arxiv.org/abs/2408.00714
- Hugging Face 마스크 생성 문서: https://huggingface.co/docs/transformers/tasks/mask_generation

## 지표

- **detected precision**: 검출했다고 한 비모호 사례 중 실제로 존재한 비율
- **strict present recall**: 실제 존재 사례 중 진단 마스크까지 얻은 비율
- **false positive rate**: 실제 부재 사례에서 검출한 비율
- **uncertain rate**: 비모호 사례 중 자동 결론을 내리지 않은 비율
- **diagnostic coverage**: 비모호 사례 중 `detected` 또는 `no_detection`까지 관측한 비율
- **IoU / Dice**: `present + detected` 사례의 정답 마스크 좌표 일치도
- **protection overlap**: 예측 마스크가 얼굴·사람 형태 귀 보호 정답을 침범한 비율
- **ear-accessory overlap**: 동물귀·장신구 예측 마스크가 겹친 픽셀과 작은 마스크 기준 중첩률

모호한 정답은 결과에서 삭제하지 않고 별도 집계한다. 진단 범위와 정확도 분모에는 포함하지 않는다.

## 데이터 분리

검증 manifest와 파일은 `inputs/validation/reference-parts/` 아래에 둔다. 생성 출력, 승인 결과, 미세조정 대기 결과, 학습 후보 폴더를 검증 입력으로 사용할 수 없다. manifest 밖의 상대 경로나 절대 경로도 거부한다.

## 적용 경계

검증 결과는 현재 분석기의 성능과 혼동 사례를 보여주는 진단 자료다. 검출기가 내부적으로 주입 가능한 후보라고 표시하더라도 검증 결과의 `automatic_conditioning`은 항상 `false`다. 임계값 변경이나 생성 연결은 별도 승인과 A/B 검증 없이는 수행하지 않는다.

## 소형 머리 장신구 확대 관찰

작은 머리핀·머리끈처럼 전체 이미지에서 축소되는 후보는 다음 세 좌표계에서 반복 관찰한다.

1. 전체 이미지
2. hair 문맥 또는 상단 대체 영역에서 얻은 머리 ROI
3. 머리 ROI를 겹치게 나눈 확대 타일

각 후보는 원본 좌표로 복원하고 원본 픽셀 크기, GroundingDINO 입력에서의 예상 픽셀 크기, 검출 점수와 출처 뷰를 함께 기록한다. 기본 전처리 예상치는 짧은 변 800, 긴 변 최대 1333이며 실제 모델 전처리 설정과 다르면 설정값을 갱신해야 한다.

- 같은 클래스와 위치가 둘 이상의 독립 뷰에서 반복되면 localized
- 한 뷰에서만 관찰되면 present_unlocalized
- 같은 위치가 여러 클래스로 해석되면 class_conflict
- 후보가 없으면 not_detected

not_detected는 실제 부재가 아니다. 확대 관찰은 박스만 수집하며 SAM2를 호출하거나 마스크를 생성하지 않는다. localized도 곧바로 생성 조건이 되지 않고 이후 검증 단계의 마스크 후보 자격만 뜻한다.

설치된 Transformers GroundingDINO 프로세서는 한 번의 추론에서 여러 텍스트 라벨을 반환한다. 따라서 클래스마다 모델을 반복 실행하지 않고 전체 질의를 한 번에 전달해 뷰당 한 번만 추론한다. 현재 확대 관찰은 scripts/validate_reference_parts.py의 오프라인 검증에만 연결되어 있으며 GUI·일반 생성 경로에서는 실행하지 않는다.

검증 결과는 predictions.jsonl의 accessory_observations, 집계는 metrics.json, 사례별 요약은 accessory-observations.csv에 저장한다. 모든 기록에는 masks_created=false, sam2_invoked=false, automatic_conditioning_applied=false가 포함된다.

작은 물체를 확대 타일로 추론하는 설계 근거:
https://arxiv.org/abs/2202.06934