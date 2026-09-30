# 다른 의상 시험 — 사전 등록

작성: 2026-09-28, Claude. 준비·생성 전에 잠근다. 판정자는 이전 결과를 알고 있어 완전 블라인드가 아니다.

## 목적
garment-drift2-20260928 에서 "의상이 덮지 않는 부위 태그"(C4)가 캐미솔·반바지의 흔들림을 크게 줄였다 (엄격 G 10 → 27/30).
이 규칙을 의상마다 기계적으로 계산하는 형태로 정하고, 다른 두 의상에서도 통하는지 본다.

## 노출 부위 태그 규칙 (의상 승인 태그만 보고 계산, 사람 판단 없음)
태그는 공백/밑줄 무시하고 정확히 일치로 본다.
- 다리: 승인 태그에 pants, jeans, leggings, pantyhose, thighhighs, long skirt, bodysuit 중 하나라도 있으면 덮임. 없으면 "bare legs".
- 어깨·팔: long sleeves, jacket, hoodie, coat, shirt, t-shirt, short sleeves 중 하나라도 있으면 덮임.
  없고 상의가 camisole 또는 tank top 이면 "bare shoulders, bare arms". 그 외 소매를 알 수 없는 상의는 추가하지 않음.
- 배: camisole, shirt, bodysuit, dress, hoodie, jacket 중 하나라도 있으면 덮임. 없고 crop top 이 있으면 "midriff".
계산 결과 (이 문서에서 고정):
- 원래 의상 (white camisole, black shorts): bare legs, bare shoulders, bare arms — garment-drift2 C4 와 같음
- 페라리 (bodysuit, black bodysuit, jacket, yellow jacket, long sleeves, hoodie): 추가 없음
- 크롭 (thighhighs, shorts, white shorts, gloves, elbow gloves, crop top, white thighhighs, white gloves): midriff

## 조건 (각 30회: 5 자세 × 2 캐릭터 × seed 209210001~209210003)
- F: 페라리. 프롬프트 = [1girl] + 의상 승인 태그(승인 순서) + 외형·체형·고정 특징 + 자세 태그 + 공통 꼬리. 규칙 추가 태그 없음.
- CA: 크롭, 같은 형식, 규칙 태그 없음.
- CB: 크롭, 같은 형식, 의상 태그 바로 뒤에 "midriff".
그 밖의 모든 것은 garment-drift2 C4 와 같다 (정규화 제어 이미지, 얼굴 참조, 부정 프롬프트(기존 73 토큰), 모델, 스케줄러, T2I 1.5 / 0.4, IP 일정, 28단계, CFG 5.5).
긍정 프롬프트는 두 토크나이저 모두 75 토큰 이하인지 확인하고, 넘으면 그 조건을 생성하지 않는다. 총 90회.

## 측정
- 엄격 의상 G (태거 임계값 0.35):
  페라리 통과 = bodysuit >= 0.35, (jacket, hoodie, yellow_jacket 중 하나) >= 0.35, 그리고
    camisole, shorts, skirt, tank_top, crop_top, bikini, leotard, dress 모두 < 0.35
  크롭 통과 = crop_top >= 0.35, shorts >= 0.35, 그리고 pants, jeans, leggings, jacket, coat, hoodie, skirt, long_skirt, camisole, dress 모두 < 0.35
  (thighhighs, elbow_gloves 는 따로 기록만)
- 육안 라벨 (결과 태깅 전 잠금): V1, V2 (POCKET·GUN), V3 (형체 붕괴 명시), V4 노출, VG 의상 (예 / 부분 / 아니오)
- 노출 v4: A·B 는 기존과 같음 (승인 태그에 있으면 제외), C 는 페라리만 적용 (크롭은 배가 덮이지 않는 의상),
  D = 페라리 bodysuit·jacket 둘 다 < 0.35, 크롭 crop_top·shorts 둘 다 < 0.35 이면 거부.

## 판정 (해석표, 보고에는 행 번호만, 여러 행 가능)
- O1: F 의 G >= 20/30 그리고 붕괴 <= 6 → 다 덮는 의상은 추가 태그 없이도 자세 변화에서 유지
- O2: F 의 G < 20 → 다 덮는 의상도 흔들림
- O3: CB 의 G >= CA 의 G + 6, 그리고 CB 자세 "예" >= CA 자세 "예" − 3 → 규칙 태그가 크롭에도 효과
- O4: O3 미충족 → 크롭에는 효과 불충분
- 보조: 예상 밖 노출 라벨 수, 노출 v4 놓침·오탐, thighhighs·gloves 유지 수.
