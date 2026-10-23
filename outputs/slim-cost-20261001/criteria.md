# 마른 체형 규칙 대가 줄이기 — 사전 등록

작성: 2026-10-01, Claude. 계획·생성 전에 잠근다. 판정자는 slim-onoff 결과를 봤다(완전 블라인드 아님).

## 배경
slim-onoff(SO1): 규칙 ON("slender, skinny, breasts, small breasts" 를 외형 태그 뒤) 은 OFF 대비 자세 "예" 28 → 20, 캐릭터 30 → 22, 체형 유지 10 → 31.
체형 효과는 유지하면서 자세·캐릭터 하락을 줄이는 프롬프트 변형을 찾는다.

## 조건 (slim-onoff 와 같은 요청·seed·구성, 긍정 프롬프트의 체형 부분만 다름)
- ON (재사용): "... blue eyes, slender, skinny, breasts, small breasts, <자세 태그>, solo, ..."
- OFF (재사용): "... blue eyes, breasts, medium breasts, <자세 태그>, solo, ..."
- NS (skinny 제거): "... blue eyes, slender, breasts, small breasts, <자세 태그>, solo, ..."
- MV (위치 이동): "... blue eyes, <자세 태그>, slender, skinny, breasts, small breasts, solo, ..."
- NM (둘 다): "... blue eyes, <자세 태그>, slender, breasts, small breasts, solo, ..."
자세 태그 강화(반복·가중치)는 이번에 넣지 않는다 — 현재 인코딩은 가중치 문법이 없고 반복은 다른 태그 비중까지 바꾼다. 세 변형이 모두 실패하면 다음 후보로.
75토큰 초과(crop × HIGH) 는 기존과 같이 고정 청크. MV·NM 은 체형 태그가 뒤로 가므로 2번째 청크에 들어갈 수 있다 — 기록한다.

## 표본
ordinary-female × KNEEL·HIGH × original·crop × seed 209210101~209210112. 새 생성 NS·MV·NM 각 48 = 144장. ON·OFF 96장은 slim-onoff 이미지 재사용.

## 측정 (240장 전부 이번 세션에서 다시 라벨 — 비교는 같은 라벨 세션 안에서만)
셀별 시트 3장(20장씩, 5조건 섞음, 자세 원본·캐릭터 원본 함께), 블라인드. 정의는 slim-onoff 와 같음:
V1 자세, 붕괴(반투명 팔다리 덩어리·겹친 사람·얼굴을 물건이 대신함·머리 잘림 포함), VI 캐릭터, VG 의상, VX 노출, VB 체형 유지, VC 어려 보임, BG.
태거 게이트(integration 규칙) 숨김.

## 판정 (변형 X 별, 48장, 같은 세션의 ON·OFF 라벨 대비)
- 합격: X 자세 "예" >= OFF − 2, 그리고 X 체형 "예" >= ON − 5, 그리고 X 붕괴 <= OFF + 2, 그리고 X 어려 보임 <= ON, 그리고 X 캐릭터 "예" >= ON.
- 합격 변형이 여럿이면 체형 "예"가 많은 쪽, 같으면 변경이 적은 쪽(NS < MV < NM).
- 합격 없음 → 태그 조정으로 해결 안 됨. 보조 결과로 가장 가까운 변형 기록.
- 이번 세션 라벨에서 ON 과 OFF 의 자세 차이가 3 미만이면(slim-onoff 재현 실패) 판정은 참고로만 둔다.
- 보조: 셀별 표, 같은 seed 짝 비교(X 대 OFF, X 대 ON).

## 범위
outputs/slim-cost-20261001 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음. 전후 제품 파일 해시·git 상태 대조.
사용 금지 파일(143960292_19, 참조 의상 폴더의 미성년 사진)은 쓰지 않는다.
