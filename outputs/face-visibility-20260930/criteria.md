# 얼굴 크기·방향과 캐릭터 동일성 관계 — 사전 등록

작성: 2026-09-30, Claude. 측정 전에 잠근다. 생성 0회, 다운로드 0회.

## 질문
캐릭터 동일성(VI "예")이 떨어지는 원인이 결과 이미지에서 얼굴이 작거나 정면이 아니기 때문인가?

## 대상
integration-20260929 120장 + pose-newimg-20260929 48장 = 168장. 동일성 라벨은 각 시험에서 이미 잠근 VI 를 그대로 쓴다
(integration labels.json fbfa23ae…, pose-newimg labels.json 098a9e05…). VI "판단 어려움"은 분모에서 제외.

## 측정 (설치된 도구만)
- 머리 크기: imgutils anime head detector (head_detect_v2.0_s, face-crop 시험에서 받은 것, conf 0.4). 가장 큰 머리 상자의 높이(px, 이미지 높이 1232 기준). 못 찾으면 "머리 없음".
- 얼굴 방향 신호: WD 태거(wd-vit-tagger-v3) 점수 — looking_at_viewer(정면 시선), from_above, from_side, profile, from_behind, looking_down, looking_away, hair_over_eyes, hair_over_one_eye.
  "정면 얼굴" = looking_at_viewer >= 0.35 그리고 from_behind·profile < 0.35.
- DWPose 얼굴 관절은 쓰지 않는다(occlusion-diag 에서 뒷모습·옆모습을 정면으로 오판).

## 판정
- FV1 (크기 영향): 머리 높이 중앙값 이상 집단과 미만 집단의 VI "예" 비율 차 >= 25%p.
- FV2 (방향 영향): "정면 얼굴" 집단과 아닌 집단의 VI "예" 비율 차 >= 25%p.
- 둘 다 성립 → 둘 다 원인. 하나만 → 그쪽이 주 원인. 둘 다 불성립 → 크기·방향 외 원인(얼굴 참조 약함 등).
- 보조: 크기 4분위별 VI 비율, 정면 얼굴 × 크기 2×2 표, 자세별 머리 높이·정면 비율, 캐릭터별.

## 범위
outputs/face-visibility-20260930 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음. 전후 제품 파일 해시·git 상태 대조.
