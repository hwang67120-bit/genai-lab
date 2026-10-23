# 자세·의상·소품 모순 시험 — 사전 등록

작성: 2026-09-29, Claude. 준비·생성 전에 잠근다. 판정자는 이전 결과를 알고 있어 완전 블라인드가 아니다.

## 질문
1. 주머니에 손을 넣은 자세인데 의상에 주머니가 없으면? (hands_in_pockets 태그를 유지할지 뺄지)
2. 총 쏘는 자세인데 소품(총)을 원하지 않으면? (소품 태그를 빼도 자세가 유지되는지)

## 공통 구성 (garment-drift2 C4 와 같음)
정규화 제어 이미지(pose-norm control_*.png), 얼굴 참조(raccoon C6 크롭, ordinary 수동 크롭), 기존 부정 프롬프트, 모델, 스케줄러,
T2I scale 1.5 / factor 0.4, 카드 색 반전, IP 0~10 0.0 → 11~27 0.9, 28단계, CFG 5.5.
프롬프트 형식: [1girl(ordinary만)] + 의상 태그 + 덮지 않는 부위 태그 + 외형·체형·고정 특징 + 자세 태그 + 공통 꼬리.
75 토큰을 넘으면 prompt-chunk-20260929 의 fixed 청크 인코딩을 쓴다 (동등성 검증됨).
seed: 209210001 ~ 209210006 (6개). 캐릭터 2 × seed 6 = 조건당 12회.

## 시험 1: 주머니 (POCKET 자세)
의상: 태그만으로 정의한 가상 의상 "white dress, sleeveless dress" (이미지 없음, 승인 이력 없음 — 이 시험 전용).
덮지 않는 부위 규칙(garment-other criteria.md) 계산 결과: bare legs (다리 덮는 태그 없음. 어깨는 상의가 camisole·tank top 이 아니므로 추가 안 함).
- PA: 자세 태그 hands_in_pockets 유지 (현재 방식)
- PB: hands_in_pockets 제거 (자세 태그 없음)
측정:
- 엄격 의상 GD: dress >= 0.35, 그리고 pants, jeans, shorts, jacket, open_jacket, coat, hoodie, leggings 모두 < 0.35
- 육안: V1 자세 일치(예/부분/아니오), VH 손 상태(주머니에 넣음 / 옷 속에 파묻힘 / 다른 곳(허리·뒤·옆) / 판단 어려움),
  VP 주머니 있는 옷 추가(예/아니오), V3 형체 붕괴, VG 의상(흰 민소매 원피스로 보임: 예/부분/아니오)
판정:
- PK1: PA 의 GD <= PB 의 GD − 3 → 주머니 태그가 주머니 있는 옷을 불러온다
- PK2: PB 자세 "예" >= PA 자세 "예" − 2, 그리고 PB 붕괴 <= PA 붕괴 + 1 → 빼도 자세가 크게 무너지지 않는다
- PK1 과 PK2 가 모두 성립하면 "주머니 없는 옷이면 hands_in_pockets 를 뺀다" 규칙 채택 근거. 아니면 PK3 (근거 불충분).

## 시험 2: 총 (GUN 자세)
의상: 원래 의상 white camisole, black shorts + bare legs, bare shoulders, bare arms (garment-drift2 C4 와 같음).
- GA: 자세 태그 전부 (gun, holding weapon, rifle, holding gun, aiming) — seed 1~3 은 garment-drift2 C4 결과를 재사용, seed 4~6 만 새로 생성
- GB: 소품 태그 gun, holding weapon, rifle, holding gun 제거, aiming 만 유지
측정: 육안 V1, V2(총을 쥐고 겨눔 / 총은 있으나 쥐는 모양 이상 / 총 없음), VE 빈손 조준(총 없이 겨누는 손 모양: 예/아니오), V3 형체 붕괴, VG.
판정:
- GN1: GB 붕괴 <= GA 붕괴 + 1, GB 자세 "예" >= GA 자세 "예" − 2, 그리고 GB 에서 총이 있는 결과 <= 2/12 → 소품을 빼도 된다
- GN2: GB 붕괴 >= GA 붕괴 + 3 또는 GB 자세 "예" <= GA 자세 "예" − 3 → 소품이 자세 안정에 필요
- GN3: GB 에서 총이 있는 결과 >= 3/12 → 태그를 빼도 골격이 총을 부른다
(여러 행 가능)

## 공통 기록
노출 v4 (A·B·C·D) 기록. 라벨은 결과 태깅 전 잠금. 총 새 생성 42회 (PA 12, PB 12, GA 6, GB 12).
