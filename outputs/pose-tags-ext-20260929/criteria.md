# 자세 태그 허용 목록 확장 시험 — 사전 등록

작성: 2026-09-29, Claude. 태거를 새 목록으로 돌리기 전에 잠근다(목록을 결과에 맞추지 않기 위해).

## 질문
pose-newimg(GF)에서 자세 태그가 0개였던 하이앵글(H2)·설명이 부족했던 벽 기대기(P3)·소총(G2)이 실패했다.
자세 설명 태그 허용 목록을 넓히면 처음 보는 자세 이미지의 좋은 결과가 늘어나는가? 이미 잘 되던 이미지(P2)를 망치지는 않는가?

## 확장 허용 목록 (기존 23개 + 추가, Danbooru 일반 자세·방향 태그, 배경 물체 태그 제외)
추가: standing, walking, leaning_back, leaning_forward, leaning_to_the_side, crossed_legs, legs_together, standing_on_one_leg, contrapposto,
hand_on_hip, hands_on_hips, hand_on_own_chin, hand_on_own_face, hand_to_own_mouth, hand_on_own_chest, arms_behind_back, arm_at_side, arms_at_sides, own_hands_together,
arm_across_chest, bent_over, head_tilt, looking_down, looking_up, looking_away, looking_to_the_side, from_side, from_behind, profile, holding_rifle, shouldering, pointing_weapon.
(against_wall·wall 등 배경 물체 태그는 제품 범위(배경 없음) 때문에 넣지 않는다. 어휘에 없는 태그는 기록하고 무시.)
임계 0.35, 원본 자세 이미지에서 태깅. 채택 순서: 점수 높은 순. 긍정 프롬프트가 75토큰을 넘으면 넘치지 않을 때까지 점수 낮은 추가 태그부터 뺀다(한 청크 유지). 뺀 태그는 기록.

## 생성
pose-newimg-20260929 와 모든 것이 같고 자세 태그만 다르다: P2·P3·G2·H2 × 캐릭터 2 × seed 209210001~209210006 = 48장. 비교 대상은 pose-newimg 48장(같은 seed).
자세 태그가 기존과 같게 나온 이미지도 생성한다(같은 입력이면 바이트 동일 기대 — 재현 확인으로 기록).

## 측정
96장(기존 48 + 새 48)을 섞어 블라인드 재라벨(자세·캐릭터별 시트 12장씩, 조건 비공개). pose-newimg 과 같은 항목·정의. DWPose 목표 근접.

## 판정
- TX1 (확장 채택 근거): 새 조건 좋은 결과 >= 기존 조건 + 6 (재라벨 기준), P2 좋은 결과 >= 기존 P2 − 2, 붕괴 증가 <= 2.
- TX2 (효과 없음): TX1 불성립이고 새 − 기존 좋은 결과 <= 2.
- TX3 (부분): 그 사이.
- 보조: 이미지별 좋은 결과·4장 요청 성공, 태그가 바꾼 방향(뒷모습·옆모습 증가 등) 기록.

## 범위
outputs/pose-tags-ext-20260929 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음. 전후 제품 파일 해시·git 상태 대조.
