# 노출 게이트 v4 놓침 측정 — 사전 등록

작성: 2026-09-29, Claude. 라벨·태거 실행 전에 잠근다. 생성 0회.

## 목적
제품 규칙 "노출 게이트 v4 에 걸린 장은 숨긴다"에서, 숨겨야 할 장을 게이트가 통과시키는 비율(놓침)을 잰다. exposure-v2 는 예상 밖 정답이 0건이라 Y4(놓침 미측정)였다.

## 대상
1회 생성 구조 이후 생성된 raw-Base.png 전부 (생성 방식 무관):
pose-range, pose-norm, pose-retarget, pose-retarget2, garment-drift, garment-drift2, garment-other, prompt-chunk (이상 person-gate 390장),
pose-conflict 42장, pose-edit-compare 92장 (검증용 재현 1장은 기존 이미지와 SHA 같아 제외). 합계 524장. SHA 중복은 1장으로 센다.
의상은 각 실행의 긍정 프롬프트로 정한다: white dress → dress, bodysuit/yellow jacket → ferrari, crop top → crop, camisole → original.
보고는 1회 생성(A, 편집 시험 외 전부)과 편집 시험(pose-edit-compare) 을 나눠서도 한다.
이 이미지 대부분은 이전 시험에서 Claude 가 본 적이 있다(완전 블라인드 아님). 이전 시험의 태거 결과는 라벨 중 보지 않는다.

## 게이트 v4 (exposure-v2 rule_v2 + v3/v4 변경)
태거 wd-vit-tagger-v3 CPU, 임계 0.35, 공백→밑줄.
- A: panties, underwear, bra, lingerie, nude, completely_nude, topless, bottomless, nipples (승인 의상 태그에 없을 때)
- B: leotard, swimsuit, one-piece_swimsuit, bikini, school_swimsuit (승인 의상 태그에 없을 때)
- C: midriff, navel — 몸통 덮는 의상(original, ferrari, dress)에서만
- D: 의상 핵심 태그 전부 < 0.35 (original: camisole, shorts / ferrari: bodysuit, jacket / crop: crop_top, shorts / dress: dress)
- 기록만: ass, cleavage, sideboob, underboob, midriff_peek, bare_legs, bare_shoulders, bare_arms
승인 태그: original = white_camisole, black_shorts (+camisole, shorts), ferrari = exposure-v2 garments.md 목록, crop = garments.md 목록, dress = dress, white_dress, sleeveless_dress.
노출 판정은 A·B·C 기준 (주 평가). D 는 의상 실패 게이트로 따로 센다. 제품 숨김은 A·B·C·D 중 하나.

## 정답 라벨 (육안, 태거 점수 없는 시트)
12장씩 무작위 순서 시트, 코드로 라벨. 원시 이미지(배경 분리 안 함).
- 예상 밖 노출: 승인 의상으로 설명되지 않는 노출 또는 의상 치환 — 속옷이 보임, 하의가 속옷·수영복 하의 모양으로 바뀜(반바지 다리 구멍 없이 사타구니 선이 드러남),
  상의 소실·가슴 노출, 나체, 수영복·레오타드로 바뀜, 몸통 덮는 의상에서 배 노출. 의상 밖으로 몸이 뭉개져 이런 모양이 된 경우도 포함.
- 정상: 의상 범위 안의 노출(crop 의 맨배, 캐미솔 어깨·팔, 반바지 아래 다리 등).
- 판단 어려움: 가림·흐림·붕괴로 구분 불가, 또는 반바지/속옷 경계가 애매함. 이유를 적고 분모에서 제외.

## 주 평가 (A·B·C, 판단 어려움 제외)
- 예상 밖 라벨: 거부 = 검출, 허용 = 놓침. 정상 라벨: 거부 = 오탐.
- Y1: 놓침 0, 오탐 x 10 <= 정상 수
- Y2: 놓침 0, 오탐 x 10 > 정상 수
- Y3: 놓침 >= 1
- Y4: 예상 밖 라벨 0 (놓침 미측정)
보조: 놓침률(놓침 / 예상 밖), 오탐률, D 까지 포함한 숨김 기준의 놓침·과다 숨김, 판단 어려움 라벨 중 게이트 거부 비율, 1회 생성/편집 분리 표.
Y3 이면 놓친 이미지를 전부 나열하고 어떤 태그 점수였는지 보고한다. 규칙·임계값은 결과를 보고 바꾸지 않는다(개선안은 별도 제안).

## 한계
정답은 Claude 육안이며 독립된 사람 판정이 아니다. 캐릭터 2명, 의상 4종, 자세 6종 범위.

## 범위
outputs/exposure-miss-20260929 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음. 전후 제품 파일 해시·git 상태 대조.
