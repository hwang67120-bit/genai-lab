# 의상 흔들림 시험 — 사전 등록

작성: 2026-09-28, Claude. 준비·생성 전에 잠근다. 판정자는 이전 결과를 알고 있어 완전 블라인드가 아니다.

## 배경
pose-norm-20260928 (정규화 골격, 채택 방향)에서 의상 통과(camisole 또는 shorts >= 0.35)가 13/30.
반바지가 긴 바지·레깅스로, 캐미솔 위에 재킷·코트가 추가된 결과가 많았다. 자세 태그(hands_in_pockets, from_above, holding_phone 등)가
주머니 있는 옷·겉옷을 부른다는 가설과, 의상 태그의 힘이 약하다는 가설을 가른다.

## 기준 조건 C0 (재생성 없음)
pose-norm-20260928 의 30장 그대로 (정규화 골격, 자세 태그 있음, 기존 프롬프트·부정 프롬프트).

## 새 조건 (각 30회: 5 자세 × 2 캐릭터 × seed 209210001~209210003)
- C1: 자세 태그 제거. 긍정 프롬프트 = pose-range 의 "태그 없음" 형식 (외형·체형·고정 특징 + white camisole, black shorts + 공통 꼬리).
- C2: 자세 태그 유지, 의상 태그를 맨 앞으로. 순서 = [1girl (ordinary만)], white camisole, black shorts, 외형, 체형, 고정 특징, 자세 태그, 공통 꼬리.
- C3: C0 프롬프트 그대로 + 부정 프롬프트 끝에 "pants, long pants, jeans, leggings, pantyhose, skirt, jacket, open jacket, coat, hoodie" 추가.
  두 토크나이저 모두 부정 프롬프트가 75 토큰 이하인지 먼저 확인하고, 넘으면 생성하지 않고 멈춘다.
그 밖의 모든 것(제어 이미지 = pose-norm control_*.png, 얼굴 참조, seed, 모델, 스케줄러, T2I scale 1.5 / factor 0.4, IP 일정, 28단계, CFG 5.5)은 pose-norm 과 같다.
실행 스크립트는 pose-norm run_batch.py 와 같은 코드. 총 90회. M0 측정 안 함.

## 측정
1. 엄격 의상 판정 G (태거, 임계값 0.35, 결과 라벨 잠금 후):
   통과 = shorts >= 0.35 이고, 아래 충돌 태그가 모두 < 0.35:
   pants, jeans, leggings, pantyhose, skirt, long_skirt, jacket, open_jacket, coat, hooded_jacket, hoodie, track_jacket
   (어휘에 없는 태그는 "어휘 없음"으로 기록하고 빼지 않는다.)
   C0 30장도 같은 방법으로 다시 태깅해 계산한다.
2. 육안 라벨 (결과 태깅 전 잠금, C1·C2·C3 90장): V1 자세 일치(예/부분/아니오), V2 소품·손(POCKET·GUN), V3 인체 이상(형체 붕괴 명시),
   VG 의상 육안(흰 캐미솔 + 검은 반바지로 보임: 예 / 부분 / 아니오). C0 는 기존 pose-norm 라벨을 쓰고 VG 만 추가로 적는다.
3. 노출 v4 (pose-norm 과 같은 규칙) 기록.

## 판정 (해석표, 보고에는 행 번호만, 여러 행 가능)
C0 의 G 통과 수를 g0, 육안 "예" 21, 붕괴 4 로 둔다.
- G1: C1 의 G 통과 >= g0 + 6 → 자세 태그가 주요 원인
- G2: C3 의 G 통과 >= g0 + 6, 그리고 C3 육안 "예" >= 18, 붕괴 <= 6 → 부정 프롬프트가 품질 손실 없이 효과
- G3: C2 의 G 통과 >= g0 + 6, 그리고 C2 육안 "예" >= 18, 붕괴 <= 6 → 순서 변경이 품질 손실 없이 효과
- G4: 어느 조건도 g0 + 6 에 미치지 못함
- 보조: 태그 G 와 육안 VG 의 일치율, C1 에서 총·휴대폰·주머니 손 손실, 자세별 G.
