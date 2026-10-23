# M0 깨끗한 기준 재계산 — 사전 등록

작성: 2026-09-28, Claude. 계산 전에 잠근다. 생성 0회, CPU.

## 배경
M0(운영 정체성 재순위와 같은 함수)의 ordinary-female 기준 특징은
outputs/identity-rerank-20260923/crops/56be9809…-identity.png 에서 계산됐다.
이 크롭에는 머리 뒤로 올린 두 팔, 목, 가슴 피부, 캐미솔 끈이 들어 있다. raccoon 기준 크롭은 깨끗하다.
가설: 이 오염 때문에 ordinary-female의 M0가 옷(피부 노출)과 팔 자세에 민감했다.

## 바꾸는 것 (하나만)
ordinary-female의 기준(양성) 특징만 교체한다.
- 깨끗한 기준 크롭 = genai_lab.visual_reference.masked_crop(I01 원본, 수동 마스크)
  · I01 원본: outputs/face-crop-20260928/manifest.json 의 I01 path (SHA-256 dd8a2899…)
  · 수동 마스크: onepass2/face_ref_mask.png 를 face-crop-20260928/crop.py refmask 와 같은 방법으로 원본 좌표에 역투영
    (왼쪽 padding 18 제거, box 274,142,455,359)
  · I01은 갤러리 파일(ca87afc2…)과 다른 파일이지만 같은 캐릭터 입력 이미지다. 이 점을 기록한다.
- 음성 23장(갤러리 중 ordinary-female 이외)의 특징, 인코더, 결과 이미지 쪽 크롭은 모두 그대로 쓴다.
  결과 쪽은 저장된 identity-crop.png 를 다시 인코딩한다.

## 동등성 확인 (먼저)
저장된 결과 크롭 2개(generalize ordinary-female_original_arms_s0, measure/off)를
기존 기준 특징으로 다시 계산해 저장된 self_score와 |차이| <= 1e-4 인지 확인한다. 아니면 멈추고 보고한다.

## 계산 대상
- 새 OFF 마진: measure/identity/off/identity-crop.png. 새 기준값 = 0.5 × 새 OFF 마진.
- generalize ordinary-female 12개 (팔짱 6, 높이 차기 6). 저장 크롭이 없는 실행은 미측정.
- 참고(판정 제외): onepass E3·E4 계열 저장 크롭이 있으면 함께 계산.

## 판정
- P1' 구분: 팔짱 6개 중 새 기준으로 자기 1위 >= 5/6
- P2' 옷 불변: 팔짱 seed 3쌍의 |마진(원래 의상) − 마진(페라리)| 중앙값 / 새 OFF 마진 <= 0.25
  (기존 값 0.3473 과 나란히 적는다)
- 해석표 (보고에는 행 번호만):
  K1: P1'·P2' 통과 → 기준 크롭 오염이 옷 민감성의 주원인이라는 증거
  K2: P2' 통과, P1' 실패 → 옷 민감성은 줄었지만 구분력도 잃음
  K3: P2' 실패 → 기준 오염만으로 설명되지 않음
  K4: 판정 불가
