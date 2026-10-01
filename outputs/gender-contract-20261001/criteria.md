# 성별 계약 위반 확인과 수정 — 사전 등록

작성: 2026-10-01, Claude. 계획·생성 전에 잠근다. 판정자는 원인(아래)을 이미 알고 있다(완전 블라인드 아님).

## 원인 (조사 결과, 생성 전)
- 제품 계약: 성별은 사용자가 캐릭터 이미지마다 고른 값만 쓴다(genai_lab/character_preferences.py, QSettings). 지정되면 긍정 맨 앞에 1boy/1girl + 보조 태그(male: "male focus, masculine silhouette" / female: "female focus, feminine silhouette"), 부정 맨 앞에 반대 성별 차단(male → 1girl, female → 1boy), 같은 성별 부정어 제거. 지정 안 함이면 성별 태그·차단 없음(clothing_reference_generation.py).
- 자세 시험 경로: pose-identity-20260927 preflight 의 부정 프롬프트(ordinary-female 실행에서 온 "1boy, nsfw, …")를 이후 모든 시험이 모든 캐릭터에 그대로 썼다. 긍정에는 ordinary 만 1girl, raccoon 은 성별 태그 없음.
- raccoon 의 사용자 지정 성별 = 남성(레지스트리 HKCU\Software\GenAILab\CharacterPreferences, 키가 HFTK9dCbgAAkxKb.png 경로와 일치). 따라서 자세 시험의 raccoon 120장 이상은 남성 캐릭터에 여성 차단 없이 "남성 차단(1boy)"만 건 상태로 생성됐다 → 동일성 계약 위반.
- e2e 시험은 태거의 1boy/1girl 을 그대로 썼다(출처가 사용자 아님). 이번 사용자 지정(S1·S3 남성, S2·S4·S5 여성)과 결과적으로 같았지만 보조 태그가 빠졌다.

## 조건 (raccoon, 사용자 지정 남성)
- X (재사용): final-int F 의 raccoon × 의상 3(original, ferrari, crop) × 자세 P2·KNEEL·LOW × seed 209210101~104 = 36장. 긍정 성별 태그 없음, 부정 "1boy, …".
- U (새로, 계약 방식): X 와 모두 같고 성별만 제품 방식 — 긍정 맨 앞 "1boy, male focus, masculine silhouette", 부정 맨 앞 "1girl"(기존 "1boy" 를 대체). 36장.

## 측정 (72장, 의상·자세별 시트에 X 4 + U 4 섞은 블라인드, 캐릭터 원본 함께)
MG 성별 표현(남성/여성/모호: 얼굴·체형·가슴으로 판단), VI 캐릭터, V1 자세, 붕괴, VG 의상, VX 노출.

## 판정
- GC1 (계약 방식이 지정 성별을 지킨다): U 남성 >= 27/36 (75%), 그리고 U 남성 >= X 남성 + 12.
- GC2 (부작용 없음): U 캐릭터 "예" >= X − 3, U 자세 "예" >= X − 3, U 붕괴 <= X + 2, U 보이는 예상 밖 노출 <= X + 1.
- 둘 다 성립 → 자세 경로에 제품 성별 계약을 그대로 적용하는 근거. GC1 만 → 성별은 맞지만 부작용 기록.
- 보조: 의상·자세별 남성 수(노출 의상 original·crop 에서 남성 표현이 무너지는지).

## 범위
outputs/gender-contract-20261001 에만 쓴다. 제품 소스·docs·테스트 수정 없음, 커밋 없음, 다운로드 없음. 레지스트리는 읽기만 했다. 전후 제품 파일 해시·git 상태 대조.
