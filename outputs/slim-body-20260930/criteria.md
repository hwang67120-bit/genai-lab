# 마른 체형 대책 시험 — 사전 등록

작성: 2026-09-30, Claude. 생성 전에 잠근다. 판정자는 retarget3 결과를 봤다(완전 블라인드 아님).

## 배경
retarget3: 마른 체형 C19 는 결과가 어깨 넓고 가슴·골반 큰 모델 기본 체형으로 끌림(체형 유지 5/12). 태거는 C19 에서 slim·skinny·petite 를 찾지 못한다(slim·slender·thin 은 어휘에 없음, skinny·petite < 0.05) — 태거만으로는 마른 체형을 감지할 수 없다.

## 감지 규칙 후보 (기하)
캐릭터 이미지 DWPose 어깨폭 / 몸통 길이 < 0.48 이면 "마른 체형". 값: C19 0.41, ordinary-female 0.42 → 해당. C11 0.55, C23 0.66, C02 0.86 → 비해당.

## 조건 (마른 체형으로 판정된 캐릭터에만 적용)
- A (현재): 태거 태그 그대로(medium breasts 포함).
- P (체형 태그 추가): 긍정 프롬프트 체형 위치에 "slender, skinny" 추가.
- PS (P + 가슴 태그 한 단계 낮춤): P 에서 medium breasts → small breasts.
"petite"·"flat chest"·"loli" 같은 어린 체형 연상 태그는 쓰지 않는다(안전).

## 표본
캐릭터 C19(마른 체형, 얼굴 face_C19.png), ordinary-female(얼굴 face_ref_clean.png, 태그 black hair, short hair, blue eyes, medium breasts).
의상 ferrari, 자세 P2·KNEEL·LOW, seed 209210101~209210104, 세기 1.2, 확정 부정 프롬프트, 조건부 얼굴 일정(LOW 만 0.5), retarget3 와 같은 코드.
C19 A 는 retarget3 A 12장 재사용. 새 생성: C19 P·PS 24 + ordinary A·P·PS 36 = 60장.

## 측정 (72장, 캐릭터·자세별 시트에 A 4 + P 4 + PS 4 섞음, 캐릭터 원본 함께, 블라인드)
VB 체형 유지(원본과 같은 마른 체형: 예/부분/아니오), VC 어려 보임(성인·청소년 체형보다 어린아이처럼 보임: 예/아니오), V1 자세, 붕괴, VI 캐릭터, VG 의상.
보조: DWPose 어깨폭/몸통.

## 판정 (조건별 24장, A 대비)
- SB1 (채택 후보): VB "예" >= A + 5, VC "예" <= A, 붕괴 <= A + 1, 자세 "예" >= A − 2, 캐릭터 "예" >= A − 3.
- 둘 다 성립하면 VB 가 많은 쪽, 같으면 변경이 적은 P.
- VC 가 A 보다 늘면 그 조건은 채택 불가(안전).
- 성립 없음 → 태그로 해결 안 됨, 마른 체형은 한계로 기록.

## 범위
outputs/slim-body-20260930 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음. 전후 제품 파일 해시·git 상태 대조.
