# 의상 재현 보강 — 사전 등록

작성: 2026-10-01, Claude. 계획·생성 전에 잠근다. 판정자는 e2e 결과(의상 "예" 2/20)를 봤다(완전 블라인드 아님).

## 배경
e2e-20261001: 처음 쓰는 5 시나리오에서 의상 "예" 2/20. 원인은 의상 분석 태그가 원본을 충분히 담지 못함(청록→blue, 카고→pants, 재킷 약함, 마네킹 다리→thighhighs, 크롭 니트→sweater).
제품 흐름 5단계는 사용자가 태그를 확인·수정하는 단계이고, FLOW 6단계는 의상 참조를 시각 조건으로도 전달한다고 정한다. 두 경로를 시험한다.

## 공통
e2e 5 시나리오와 같은 캐릭터·자세·얼굴 크롭·제어 이미지·seed(209210201~204)·생성 구성(T2I 1.2, 조건부 얼굴 IP, 28단계, CFG 5.5).
성별은 사용자 지정(2026-10-01: S1·S3 남성, S2·S4·S5 여성)을 제품 방식으로: 긍정 맨 앞 1boy/1girl + 보조 태그(male focus, masculine silhouette / female focus, feminine silhouette), 부정 맨 앞 반대 성별 차단.

## 조건 (각 20장)
- B (기준): e2e 의상 태그 그대로 + 위 성별 방식.
- T (태그 수정 — 사용자 5단계 확인을 판정자가 대신): 의상 이미지를 보고 고친 태그. 색·형태·길이를 바로잡고 오인 태그(thighhighs, open-chest sweater)는 뺀다. 상표·로고는 넣지 않는다.
  - S1: aqua sweater, cropped sweater, off-shoulder sweater, long sleeves, striped sleeves, aqua shorts, short shorts, dolphin shorts, midriff
  - S2: white shirt, collared shirt, short sleeves, plaid necktie, light blue pants, cargo pants, baggy pants, white footwear
  - S3: grey jacket, jacket on shoulders, white shirt, black neck ribbon, suspenders, black pants, high-waist pants, black footwear
  - S4: white tube top, crop top, white shrug (clothing), turtleneck, long sleeves, white skirt, miniskirt, midriff
  - S5: white crop top, long sleeves, shoulder cutout, buttons, red pleated skirt, miniskirt, midriff
  덮지 않는 부위·주머니 규칙은 같은 규칙으로 다시 계산(중복 제거).
- I (의상 이미지 참조): B 와 같은 태그 + 두 번째 IP-Adapter(ip-adapter-plus_sdxl_vit-h, 로컬 파일, 같은 ViT-H 인코더)에 의상 원본(흰 여백으로 정사각형 패딩).
  의상 IP 일정: 0~10단계 0.0 → 11단계부터 0.5 (자세·구도가 잡힌 뒤 의상만 유도, FLOW 시간적 유도). 얼굴 IP 일정은 그대로.
- TI: T 태그 + I 의상 참조.
새 생성 80장. 최대 reserved 기록(두 번째 어댑터로 늘어날 수 있음, 6.5 GiB 넘으면 기록).

## 측정 (80장, 시나리오별 시트에 4조건 × 4장 섞은 블라인드, 캐릭터·의상 원본 함께)
VG 의상(의상 원본과 같은 옷: 예/부분/아니오), VI 캐릭터, MG 성별(지정 성별과 일치: 예/아니오/모호), V1 자세, 붕괴, VX 노출, NP 인체 비율.
태거 게이트: e2e 와 같은 규칙(게이트 규칙은 각 조건의 의상 태그에서 만든다).

## 판정 (조건별 20장, B 대비)
- 합격: VG 예 >= B + 6, 그리고 VI 예 >= B − 2, 그리고 붕괴 <= B + 1, 그리고 자세 예 >= B − 2, 그리고 보이는 예상 밖 노출 <= B, 그리고 MG 일치 >= B − 1.
- 합격 여럿이면 VG 예가 많은 쪽, 같으면 변경이 적은 쪽(T < I < TI).
- 해석 주의: T 는 사람이 태그를 고치는 단계를 대신한 것(자동 아님), I 는 자동.
- 보조: 시나리오별 VG, 의상 IP 가 캐릭터 얼굴·체형이나 마네킹 형태를 끌어오는지(누설) 기록.

## 범위
outputs/garment-fidelity-20261001 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음(ip-adapter-plus_sdxl_vit-h 는 로컬 캐시에 있음). 전후 제품 파일 해시·git 상태 대조.
