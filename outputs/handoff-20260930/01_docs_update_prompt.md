# Codex 프롬프트 1 — 2026-09-28~30 결정·발견을 DECISIONS·BACKLOG 에 반영

(10/4 이후 Codex 에 그대로 붙여 넣는다. 작성: Claude, 2026-09-30. 이 문서는 초안이며 docs 는 아직 수정되지 않았다.)

---

## 작업
2026-09-28~30 동안 Claude 가 `outputs/` 에서 시험만 실행했다(docs·코드·설정 수정 없음, 커밋 없음). 그 기간의 **사용자 결정**과 **시험 근거**를 `docs/DECISIONS.md` 와 `docs/BACKLOG.md` 에 반영하라.
- 코드·설정·테스트는 이 작업에서 수정하지 않는다. 문서만.
- `docs/DOCUMENTATION_RULES.md` 를 따른다(수치+조건+근거 경로, 미측정은 미측정).
- 각 근거 폴더의 `results.md` 를 직접 열어 수치를 대조한다. 이 프롬프트의 수치와 다르면 results.md 를 따르고 차이를 보고한다.
- 결정 번호는 현재 최댓값 D-074 다음부터. 아래 묶음은 제안이며, 번호·묶음은 기존 문서 관례에 맞게 조정 가능.
- 모든 결정은 "1회 생성 구조 후보(D-074)의 구성 갱신"이며 **운영 반영은 아직 아니다**(FLOW.md 는 바꾸지 않는다). D-074 의 운영 반영 조건 목록과의 관계를 각 결정에 적는다.

## 제안 결정 (사용자 확정 항목만)

### D-075 1회 생성 후보 구성 갱신
| 항목 | D-074 값 | 새 값 | 근거 (outputs/…/results.md) |
|---|---|---|---|
| T2I-Adapter 골격 세기 | 1.5 | **1.2**, 모든 자세 단일값 | strength-5pose-20260929 (P1: 60장씩, 붕괴 15→2, 자세 예 34→43, 캐릭터 예 15→31, 의상 예 31→48), body-weak-20260929, strength-perspective-20260929 (하이앵글 1.3·1.4 도 불리 → 자세별 세기 불필요) |
| 골격 전처리 | 비율 유지 여백 | **몸 관절 bbox 정규화**(위 .20H·아래 .10H·옆 .15H, 736:1232, 회색 128, 재검출) | pose-norm-20260928 (N2: 자세 일치 8→21/30, 붕괴 10→4) |
| 부정 프롬프트 | exposure-v2 preflight 73토큰 | 아래 **최종본 72토큰, 한 청크** | negative-prevent2-20260929 (보이는 노출 3→0), bg-prevent-20260930 (뚜렷한 배경 잡티 11→6, 부작용 없음). 두 시험 모두 사전 기준 일부 불성립을 사용자가 알고 채택 |
| 얼굴 IP 일정 | 0~10단계 0.0 → 11~27 0.9 | 기본 동일. **자세 이미지 얼굴이 정면 아님**이면 0~10단계 **0.5** → 11~27 0.9 | ip-timing-20260930, ip-timing-int-20260930 (옆 보는 자세 캐릭터 예 8→16/24, 정면 자세 변화 없음). 조건부 방식 자체의 통합 시험은 미실시 |
| 긍정 프롬프트 조립 | — | [1girl/1boy] + 의상 태그 + 덮지 않는 부위 + 외형·체형·고정 + 자세 태그 + 꼬리 | garment-drift2-20260928, garment-other-20260928 |
| 덮지 않는 부위 규칙 | — | 의상 **명사**(수식어 제거) 기준. 다리 덮는 명사 없으면 bare legs / 상의가 camisole·tank top 이고 팔 덮는 명사 없으면 bare shoulders, bare arms / 몸통 덮는 명사 없고 crop top 이면 midriff | garment-drift2 (H1: 엄격 의상 10→27/30), garment-other ("white camisole" 정확 일치 실패 → 명사 비교 필요) |
| 주머니 규칙 | — | 의상에 주머니 있는 명사(shorts, pants, jeans, jacket, hoodie, coat)가 없으면 hands_in_pockets·hand_in_pocket 제거. 쥐는 소품 태그(gun 등)는 유지 | pose-conflict-20260929 (PK1·PK2, GN2·GN3) |
| 토큰 초과 | 잘림 | 75토큰 초과 시 **고정 청크 인코딩**(중요 태그 첫 청크, 넘치는 꼬리만 뒤 청크) | prompt-chunk-20260929 (EQ: 1청크 SHA 3/3 동일, 의미 단위 청크 분리는 악화) |
| 마른 체형 | — | 캐릭터 이미지 DWPose 어깨폭/몸통 < 0.48 이면 "slender, skinny" 추가 + 태거 medium breasts → small breasts. petite·flat chest·loli 등 어린 체형 연상 태그 금지 | slim-body-20260930 (체형 유지 7→20/24, 어려 보임 0) |
| 남성 캐릭터 | — | 긍정 1boy, 부정의 "1boy" 를 "1girl" 로 교체 | retarget3-20260930 (구현 메모, 별도 판정 없음) |

최종 부정 프롬프트:
```
1boy, nsfw, panties, underwear, buruma, abstract background, speed lines, light rays, lowres, bad, text, worst quality, low quality, watermark, signature, different character, different hairstyle, different hair color, different eye color, missing character features, bad anatomy, bad hands, malformed hands, extra fingers, uneven eyes
```

### D-076 결과 제공 방식: 여러 seed 생성 후 사용자 선택
- 요청당 seed 4장, 모델 1회 로드 후 한 장씩 순차 생성(장마다 스케줄러·IP scale 초기화), 완성될 때마다 표시.
- 노출 게이트(A·B·C + D 의상 핵심) 실패 장은 **숨기고 "N장 제외됨"만 표시**. 자동 순위 없음(정체성 재순위는 배포 후). 각 장 seed·선택 여부 기록. "다른 seed로 4장 더".
- 생성 후 isnet-anime 전경 분리 + 흰 배경 합성 유지(캐릭터 손상 0/44, 떨어진 방사형 잡티만 제거).
- "블루머·속옷 모양으로 짧아진 하의"는 숨겨야 할 노출(사용자 기준 2026-09-29).
- 근거: integration-20260929 (통합 통과 IP: 요청 30/30, 보이는 노출 0/120, 의상 111/120, 붕괴 1/120, 장당 20.9초·요청 84초·최대 reserved 6.41 GiB, RTX 4060 8GB), exposure-miss-20260929 (게이트 v4 놓침 26/62 — 예방 프롬프트와 함께 써야 하는 이유), bg-artifact-20260930.

### D-077 입력 검사와 자세 지원 범위
- 골격 검사: K1(인원 ≠ 1)·K2(몸 핵심 관절 없음) → 현재 기본은 "진행" 선택지 없이 다른 이미지 요청(2026-09-30 사용자 확정). 단 **서비스 운영 단계에서 바꿀 수 있도록 고정 코드가 아니라 설정값으로 둔다**(검사별 처리: 거부 / 경고+선택 / 통과). 예: 선화 지원이나 여러 사람 중 1명 선택이 생기면 K1·K2 처리를 설정만 바꿔 조정. K3~K6, **K7 다리 겹침**(정규화 골격에서 무릎·발목 좌우 거리 모두 < 0.15T), 팔·다리 **추측 관절(신뢰도 0.30~0.50) 2개 이상**, **얼굴 방향 태거 조건** 중 하나라도 걸리면 **경고 + 원본 위 골격 그림 + 사용자 "진행 / 다른 이미지" 선택**.
- 얼굴 방향 태거 조건(하나로 세 곳에 사용: 방향 경고·"캐릭터 유사도가 낮을 수 있음" 안내·조건부 IP 일정): 자세 이미지에 WD 태거 looking_at_viewer < 0.35 또는 profile·from_side·from_behind·from_above >= 0.35.
- 지원 범위: 서 있는 기본 자세(손 위치 변화 포함), 무릎 꿇기, 로우·하이앵글(하이앵글 원근 비율 정확도는 낮음). 범위 밖 안내: 벽·의자 등 받침 물체 자세, 선화·스케치, 소품을 특정 방향으로 쓰는 팔 자세(총 조준 등).
- 근거: leg-overlap-20260929 (LO1), guess-joint-drop-20260930 (추측 관절 제외 효과 없음 → 지표로만), input-direction-20260930 (DWPose 코·눈 방향 판정 7/7 놓침, 태거 조건 놓침 0·오경보 6), occlusion-diag-20260930 (DWPose 가 안 보이는 관절을 추측으로 채움), pose-newimg-20260929 (GF: 처음 보는 자세 좋은 결과 17/48), pose-tags-ext-20260929 (TX2).
- BACKLOG 68 을 이 결정으로 갱신(기존 "K3·K5 경고 운영안"은 "K3~K7+추측+방향 → 경고·사용자 선택"으로).

### D-078 얼굴 참조 자동 크롭 (2026-09-28 사용자 결정)
- C6: 애니 머리 검출기 ∩ isnet − DWPose 팔 띠, 팔꿈치·손목이 머리 근처면 다른 이미지 요청, 크롭 미리보기 사용자 확인. 근거 outputs/face-crop-c5-20260928/decision.md. BACKLOG 67 갱신.

## 채택하지 않음 (결정 문서의 "채택하지 않은 것"에 기록)
| 방안 | 근거 |
|---|---|
| 편집 2단계(서 있는 1단계 → img2img+골격) | pose-edit-compare-20260929 (EC2: 자세 예 20 대 11, 시간 21초 대 38초) |
| 골격 세기 강화(2.0)·신호 연장 | body-stability-20260929 (붕괴 증가, 세기 2.0 12/12 붕괴) |
| 자세별 세기(하이앵글 1.3·1.4) | strength-perspective-20260929 |
| 추측 관절을 제어 이미지에서 제외 | guess-joint-drop-20260930 (OC2) |
| 자세 태그 허용 목록 확장·방향 태그 자동 채택 | pose-tags-ext-20260929 (TX2, from_behind 오판 → 뒷모습 12/12) |
| 체형 리타게팅 v1·v2 | pose-retarget-20260928, pose-retarget2-20260928, retarget3-20260930 (RX2: 체형 유지 39/48 현재 방식으로 충분, 비율 오차 오히려 증가) |
| 두 번째 부정 청크(빈 긍정 청크) | negative-prevent-20260929 |
| 태거·DWPose 인원 검사 게이트 | person-gate-20260929 (PG3: 오탐 81%) |

## BACKLOG 갱신
- 16(현재 본선)·66: 선행 조건에 D-075~D-078 반영, 통합 시험(integration-20260929) 결과 추가. 운영 구현 시 **새 옵션 OFF 에서 기존 출력 바이트 동일** 조건은 그대로.
- 67·68: 위 결정으로 "다음" 문구 갱신.
- 70: pose-tags-ext 결과(허용 목록 확장 효과 없음) 반영.
- 71 노출 게이트: exposure-miss(Y3) 반영 — v4 는 단독 합격 불가, 예방 부정 프롬프트와 조합으로 운영(보이는 노출 0/120, integration). 놓침 유형 "블루머형 하의" 기록.
- 72 정체성: identity-cleanref-20260928 (M0 기준 크롭 오염, OFF 마진 0.143→0.048), face-visibility-20260930 (정면 얼굴 80% 대 정면 아님 46%, 머리 크기 무관), 사용자 선택 방식으로 확정(D-076).
- 73 배경 분리: bg-artifact-20260930 (뚜렷한 잡티 10%, 분리 후 7/24 만 제거, 손상 0).
- 신규 제안:
  - **하이앵글 원근 비율 정확도**(strength-perspective: 관절 근접 1.2 에서 1/12).
  - **총 자세 정면 조준 편향**(strength-5pose: 옆 조준 1~2/12).
  - **D: 드라이브 고장**(outputs/load-latency-20260929): D: TOSHIBA HDD Predictive Failure·불량 블록 이벤트. 2026-09-29 사용자 승인으로 D:\genai-cache → G:\genai-cache 복사(모델 SHA 37/37 일치), 원본은 D:\genai-cache.old-failing 으로 이름 변경, D:\genai-cache 는 G: 를 가리키는 정션. 남은 일: 설정·스크립트의 D:/genai-cache 경로를 G: 로 직접 변경(20개 파일), 디스크 교체. 배포 안내에 "모델은 SSD" 추가(캐시 없는 로딩 351초→27초).
  - **조건부 얼굴 IP 일정 통합 재확인**(D-075 의 조건부 방식은 부분 근거만 있음).

## 끝나면
- 변경 요약을 표로 보고(결정 번호, 바뀐 BACKLOG 항목, 근거 경로).
- 이 프롬프트의 수치와 results.md 가 다른 곳 목록.
- 커밋은 사용자 확인 후.
