# 프롬프트 청크 시험 — 사전 등록

작성: 2026-09-29, Claude. 구현·생성 전에 잠근다. 판정자는 이전 결과를 알고 있어 완전 블라인드가 아니다.

## 배경
CLIP 텍스트 인코더는 한 번에 77 토큰(내용 75)만 읽는다. garment-other-20260928 에서 raccoon × 크롭 15 조합이 76~80 토큰이라 생성하지 못했다.
UNet 교차 어텐션은 텍스트 길이에 제한이 없으므로, 75 토큰씩 나눠 인코딩한 뒤 이어 붙이면(청크) 한도가 사라진다.

## 청크 인코딩 (시험 코드에서만 구현, 운영 코드 수정 없음)
diffusers 0.38.0 StableDiffusionXLAdapterPipeline.encode_prompt 와 같은 계산을 청크마다 한다:
- 두 토크나이저·텍스트 인코더 각각: 청크 토큰을 [BOS] + 청크 + [EOS] + 토크나이저 pad 로 77 에 맞추고, output_hidden_states 의 hidden_states[-2].
  두 인코더 출력을 마지막 축으로 이어 붙이고(2048), 청크들을 길이 축으로 이어 붙인다 (77 × 청크 수).
- 청크 텍스트가 75 토큰 이하이면 tokenizer(text, padding=max_length, max_length=77, truncation=True) 를 그대로 쓴다 (기본 방식과 같음).
- pooled 임베딩: 두 번째 인코더의 첫 청크 출력.
- 부정 프롬프트: 같은 방식으로 인코딩하고, 청크 수가 긍정보다 적으면 빈 문자열("") 청크를 뒤에 붙여 맞춘다.
- 결과를 prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds 로 파이프라인에 넣는다.
모드:
- fixed: 긍정 프롬프트 전체 토큰을 75 개씩 순서대로 나눈다 (75 이하이면 청크 1개 = 기본 방식).
- semantic: 의미 단위 텍스트 청크 [캐릭터(1girl·외형·체형·고정 특징) + 공통 꼬리] / [의상 태그 + 덮지 않는 부위 태그] / [자세 태그].
  각 청크가 75 를 넘으면 그 청크만 fixed 방식으로 더 나눈다. 빈 청크는 만들지 않는다.
- trunc: 청크 없이 기존처럼 prompt 문자열을 넣는다 (diffusers 가 75 토큰에서 자른다). 잘린 부분을 기록한다.

## 단계 1: 동등성 (3회)
garment-drift2-20260928 C4 의 ordinary-female_KNEEL_C4_209210002, raccoon_POCKET_C4_209210001, ordinary-female_LOW_C4_209210003 을
fixed 모드(청크 1개)로 다시 생성한다. 원시 결과 SHA-256 이 3개 모두 원본과 같아야 한다.
하나라도 다르면 단계 2·3 을 실행하지 않고 멈춘다.

## 단계 2: 넘쳤던 15 조합 (30회)
garment-other-20260928 에서 토큰 초과로 제외된 raccoon × 크롭 15 조합 (CA: GUN·HIGH, CB: GUN·HIGH·LOW, 각 seed 3) 을
trunc 와 fixed 로 각각 생성한다. 프롬프트는 garment-other prompts.json 그대로.

## 단계 3: 의미 단위 청크 (30회)
garment-drift2-20260928 C4 의 30 조합을 semantic 모드로 생성한다 (같은 태그, 같은 seed, 청크만 나눔).
비교 대상은 C4 원본 결과 (재생성 없음): 엄격 G 27, 육안 자세 "예" 21, 붕괴 2.

그 밖의 모든 설정은 garment-other / garment-drift2 와 같다 (정규화 제어 이미지, 얼굴 참조, 기존 부정 프롬프트, 모델, 스케줄러, T2I 1.5 / 0.4, IP 일정, 28단계, CFG 5.5).

## 측정
- 엄격 의상 G: 원래 의상은 garment-drift 규칙, 크롭은 garment-other 규칙 그대로.
- 육안 라벨 (결과 태깅 전 잠금): V1, V2, V3 (형체 붕괴 명시), V4, VG. 단계 2 는 trunc·fixed 를 나란히 보고 라벨한다.
- 노출 v4 기록.

## 판정 (해석표, 보고에는 행 번호만)
- EQ: 단계 1 SHA 3/3 일치 (아니면 EQ-FAIL, 이후 판정 없음)
- 단계 2: K1 fixed 의 G − trunc 의 G >= 3 그리고 fixed 붕괴 <= trunc 붕괴 → 청크가 잘림보다 나음
          K2 |G 차이| < 3 → 차이 불분명 / K3 fixed 의 G − trunc 의 G <= −3 → 청크가 더 나쁨
- 단계 3: S1 (자세 "예" >= 24 또는 G >= 29) 그리고 G >= 25 그리고 붕괴 <= 3 → 의미 단위가 더 좋음
          S3 G <= 24 또는 붕괴 >= 4 또는 자세 "예" <= 17 → 더 나쁨 / S2 그 외 → 비슷함
