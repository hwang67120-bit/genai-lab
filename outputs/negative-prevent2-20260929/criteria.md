# 노출 예방 부정 프롬프트 재시험 (골격 세기 1.2, 한 청크) — 사전 등록

작성: 2026-09-29, Claude. 생성 전에 잠근다. 판정자는 negative-prevent·strength-5pose 결과를 봤다(완전 블라인드 아님).

## 배경과 질문
negative-prevent(세기 1.5, 두 번째 부정 청크 12태그)는 노출 8→4 였지만 붕괴 +4·의상 −5 로 부작용 기준 초과(NP1 만). 그때는 구조가 약한 1.5 였고, 빈 긍정 청크를 붙이는 2청크 구조가 혼재 요인이었다.
이번에는 세기 1.2 에서, 청크 구조를 바꾸지 않고 기존 부정 프롬프트(73토큰, 한 청크) 안의 태그 2개를 노출 태그 3개로 바꾼다.
질문: 이 교체가 블루머·속옷 모양 하의 등 예상 밖 노출을 줄이는가? 부작용은 없는가?

## 조건 (공통: strength-5pose 와 같은 코드 run_bs.py, C4 프롬프트, 정규화 골격, 얼굴, 골격 세기 1.2, 0~10단계, 얼굴 IP 11단계부터, 한 청크)
- N0: 기존 부정 프롬프트 그대로.
- N1: 기존 부정에서 "different character design", "extra digits" 를 빼고 "nsfw" 뒤에 "panties, underwear, buruma" 추가 (73토큰, 한 청크 유지):
  "1boy, nsfw, panties, underwear, buruma, lowres, bad, text, error, missing fingers, worst quality, low quality, watermark, signature, unfinished, different character, different hairstyle, different hair color, different eye color, missing character features, bad anatomy, bad hands, malformed hands, fused fingers, extra fingers, uneven eyes"

## 표본
- 노출 주 평가: 노출이 주로 나오는 무릎·주머니 × 캐릭터 2 × seed 209210001~209210012 = 조건당 48장.
  N0 seed 1~6 은 기존 결과 재사용(무릎: body-weak W12, 주머니: strength-5pose S12), seed 7~12 새로 생성(24). N1 48장 새로.
- 부작용 보조: 총·하이앵글·로우앵글 × 캐릭터 2 × seed 1~6 = 조건당 36장. N0 은 strength-5pose S12 재사용, N1 36장 새로.
- 새 생성 108회. 168장 전부 새 블라인드 시트로 라벨(재사용 이미지도 새 라벨).

## 측정
블라인드 육안(자세·캐릭터별 시트, N0/N1 섞음, 태거·대응표 전 잠금): VX 노출(예상 밖/정상/판단 어려움, exposure-miss 정의: 블루머·속옷 모양 하의, 몸통 배 노출, 레오타드·수영복, 속옷 보임, 상의 소실),
붕괴(있음/없음), V1 자세(예/부분/아니오), VG 의상(흰 캐미솔+검은 반바지), VI 캐릭터. 태거: 노출 v4 A·B·C, D, 엄격 의상 G.

## 판정
- NP1 (예방 효과, 무릎·주머니 48 대 48): N0 예상 밖 >= 6, 그리고 N1 예상 밖 <= N0 예상 밖 × 0.5.
- NP2 (부작용 없음, 전체 84 대 84): N1 붕괴 <= N0 붕괴 + 3, N1 자세 "예" >= N0 − 3, N1 의상 "예" >= N0 − 3, N1 캐릭터 "예" >= N0 − 3.
- NP1+NP2 → 제품 부정 프롬프트 교체 근거. NP1 만 → 효과 있으나 부작용. NP1 불성립 → 효과 부족. N0 예상 밖 < 6 → 판정 불가.
- 보조: 게이트(A·B·C·D) 숨김 후 남는 예상 밖 수, 태거 G.

## 범위
outputs/negative-prevent2-20260929 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음. 전후 제품 파일 해시·git 상태 대조.
