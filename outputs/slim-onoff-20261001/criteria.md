# 마른 체형 규칙 on/off 비교 — 사전 등록

작성: 2026-10-01, Claude. 계획·생성 전에 잠근다. 판정자는 final-int 결과(ordinary 의 KNEEL·HIGH 자세 하락)를 봤다(완전 블라인드 아님).

## 질문
final-int 에서 자세 "예"가 113 → 106 으로 떨어졌고 하락 7 중 6이 마른 체형 규칙이 켜진 ordinary-female 의 KNEEL·HIGH 였다.
다른 조건을 모두 고정하고 마른 체형 규칙만 켜고 끄면 자세 추종 차이가 재현되는가?

## 조건 (final-int F 구성과 같고 긍정 프롬프트의 체형 부분만 다름)
- ON: final-int F 와 같음 — "... blue eyes, slender, skinny, breasts, small breasts, ..."
- OFF: 마른 체형 규칙 없음 — "... blue eyes, breasts, medium breasts, ..." (integration 과 같은 체형 태그)
공통: 최종 부정 프롬프트(72토큰), 세기 1.2·0~10단계, 얼굴 참조 조건부(KNEEL 0.0 / HIGH 0.5 → 11단계 0.9), 28단계, CFG 5.5, 736x1232, 필요 시 청크.

## 표본
캐릭터 ordinary-female, 자세 KNEEL·HIGH, 의상 original·crop (final-int 에서 하락이 있었던 노출 의상). seed 209210101~209210112 (12개).
셀 4개 × seed 12 × 조건 2 = 96장. ON 의 seed 101~104 16장은 final-int 재사용, 새 생성 80장(ON 32 + OFF 48).
재현 확인: final-int F_original_ordinary-female_KNEEL_209210101 를 한 번 더 생성해 raw SHA 일치 확인(불일치면 멈추고 원인 보고).

## 측정 (96장, 의상·자세별 시트 4장에 ON 12 + OFF 12 섞은 블라인드, 캐릭터 원본·자세 원본 함께)
V1 자세(예/부분/아니오: final-int 와 같은 기준 — KNEEL 은 무릎 꿇기 + 한 팔 올림, HIGH 는 위에서 본 서 있는 자세 + 휴대폰/아래 보기),
붕괴, VI 캐릭터, VG 의상, VX 노출, VB 체형 유지(원본처럼 마른 체형: 예/부분/아니오), VC 어려 보임.
보조: 태거 게이트(integration 규칙) 숨김, DWPose 어깨폭/몸통.

## 판정 (조건별 48장)
- SO1 (마른 체형 규칙이 자세를 떨어뜨림): OFF 자세 "예" >= ON 자세 "예" + 5.
- SO2 (차이 없음): |OFF − ON| 자세 "예" <= 2 → final-int 의 하락은 이 규칙 때문이라고 보기 어려움.
- 그 사이(3~4): 판단 보류, 방향만 기록.
- 보조: 셀별 자세·붕괴, 체형 유지 ON 대 OFF(slim-body 에서는 7 → 20/24), 캐릭터·의상·노출·어려 보임.
- SO1 이면 다음 후보(이번엔 안 함): "skinny" 제거, 체형 태그 위치 이동, 자세별 약화.

## 범위
outputs/slim-onoff-20261001 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음. 전후 제품 파일 해시·git 상태 대조.
사용 금지 파일(143960292_19, 참조 의상 폴더의 미성년 사진)은 쓰지 않는다.
