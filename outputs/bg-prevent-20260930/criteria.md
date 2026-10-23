# 배경 잡티 예방 부정 프롬프트 시험 — 사전 등록

작성: 2026-09-30, Claude. 생성 전에 잠근다. 판정자는 bg-artifact 결과를 봤다(완전 블라인드 아님).

## 질문
부정 프롬프트(현재 N1, 73토큰 한 청크)에서 효과가 작아 보이는 태그 4개를 빼고 배경 무늬 태그 3개를 넣으면(한 청크 유지),
배경 잡티(BG2)가 줄어드는가? 손가락·형체·의상·노출·캐릭터에 부작용은 없는가?

## 조건
- B (기준, 재사용): integration-20260929 120장 (N1 부정, 기존 얼굴 참조 일정).
- G (새로): integration plan 과 모두 같고 부정 프롬프트만 교체 (72토큰, 한 청크):
  "1boy, nsfw, panties, underwear, buruma, abstract background, speed lines, light rays, lowres, bad, text, worst quality, low quality, watermark, signature, different character, different hairstyle, different hair color, different eye color, missing character features, bad anatomy, bad hands, malformed hands, extra fingers, uneven eyes"
  (N1 에서 뺀 것: error, unfinished, missing fingers, fused fingers)
새 생성 120장.

## 측정 (240장, 의상·자세별 시트에 B 8 + G 8 섞은 블라인드, 태거 전 잠금)
배경 BG0/BG1/BG2 (bg-artifact 정의), 붕괴, V1 자세, VG 의상, VI 캐릭터, VX 노출, 손 이상(손가락 뭉개짐·개수 이상이 눈에 띔: 예/아니오).
태거 게이트(integration gate 규칙)로 숨김·보이는 노출 계산.

## 판정
- BP1 (예방 효과): B BG2 >= 6, 그리고 G BG2 <= B BG2 × 0.5.
- BP2 (부작용 없음): G 붕괴 <= B + 2, G 손 이상 <= B + 3, G 의상 "예" >= B − 4, G 캐릭터 "예" >= B − 4, G 보이는 예상 밖 노출 <= B + 1.
- BP1+BP2 → 교체 채택 근거. BP1 만 → 효과 있으나 부작용. BP1 불성립 → 효과 부족. B BG2 < 6 → 판정 불가.
- 보조: 유형별 BG2, 분리 후 남는 BG2(isnet, BG2 전부).

## 범위
outputs/bg-prevent-20260930 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음. 전후 제품 파일 해시·git 상태 대조.
