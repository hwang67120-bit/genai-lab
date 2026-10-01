# 구현 인수인계 — 1회 생성 후보 구조 (2026-10-01)

작성: Claude. 대상: Codex(구현·문서 반영), 사용자(결정).
이 문서는 운영 소스·docs 를 바꾸지 않는다. 근거는 각 시험 폴더의 `criteria.md`(사전 등록)와 `results.md`(결과)다.
상태 용어: **시험 통과** ≠ **사용자 채택** ≠ **운영 구현 완료**. 이 문서의 어떤 항목도 운영 구현 완료가 아니다.
기존 인수인계 초안 `outputs/handoff-20260930/`(01·02)보다 이 문서가 우선한다(이후 시험으로 바뀐 내용이 있음 — 4절).

---

## 1. 시험 마감 목록과 완료 상태

신규 탐색은 이 시점에서 멈췄다. 마감선 안의 시험은 아래 하나였고 완료했다.

| 시험 | 시작 시 상태 | 남은 생성 | 결과 |
|---|---|---|---|
| face-ip-high-20261001 (하이앵글 얼굴 참조 일정·강도) | 사용자 승인·사전 등록 잠금 후 실행 중, 48장 중 39장 완료 | 9장 | 48/48 유효 생성, 라벨·판정 완료 (합격 없음) |

그 밖의 오늘 시험은 마감 전에 완료됐다: final-int, slim-onoff, slim-cost, e2e, gender-contract, garment-fidelity(B·T 만, I·TI 는 8GB VRAM 초과로 차단), garment-pose, high-decomp, front-ctrl. 재시도·추가 생성 없음.

---

## 2. 결과·부작용·해석상 주의점

### 2-1. 오늘 시험 요약 (경로는 `outputs/` 기준)

| 시험 | 질문 | 실제로 바꾼 변수 | 사전 판정 | 결과 수치 | 부작용 | 한계 | 근거 |
|---|---|---|---|---|---|---|---|
| final-int | 조건부 IP + 배경 부정 + 마른 체형(PS)을 통합 구성에 함께 켜도 되는가 | 부정 프롬프트 최종본, 초기 IP 태거 조건부, 마른 체형 규칙 (3개 동시) | FC1 성립, FC2 불성립(자세) | 요청 29/30, 보이는 노출 0, 의상 109, 붕괴 3, 동일성 92→103, 자세 113→106 | 자세 −7 | 3변경 동시 → 변경별 효과 분리 불가. raccoon 성별 태그 없음(아래 성별 문제) | final-int-20261001/results.md |
| slim-onoff | 마른 체형 규칙이 자세를 떨어뜨리는가 | 체형 태그만 on/off | SO1 성립 | 자세 28→20, 체형 유지 10→31, 동일성 30→22 | — | 캐릭터 1명 | slim-onoff-20261001/results.md |
| slim-cost | 규칙 대가를 줄이는 변형이 있는가 | NS(skinny 제거)/MV(위치)/NM | 합격 없음 | NS 자세 27(=OFF)·체형 29(ON 39) | — | 재라벨 세션에서 ON 체형 39(이전 31) → 기준이 엄격해짐 | slim-cost-20261001/results.md |
| e2e | 원본 3장부터 전체 흐름이 완주하는가 | 처음 쓰는 캐릭터·의상 5 시나리오 | E1~E5 성립 | 완주 5/5, 보이는 좋은 결과 17/20, 동일성 17/20, 비율 이상 1, 붕괴 1 | 의상 "예" 2/20 | 사용자 확인 단계는 "그대로 진행" 가정. 성별은 태거값 사용(계약 위반) | e2e-20261001/results.md |
| gender-contract | 자세 경로가 사용자 지정 성별을 지키는가 | 성별 조건만 (기존 방식 ↔ 제품 계약 방식) | GC1·GC2 성립 | 남성 지정 raccoon 이 남성으로 보임 2/36 → 28/36, 동일성·자세 유지 | 의상 "예" 34→23 | 캐릭터 1명 | gender-contract-20261001/results.md |
| garment-fidelity | 태그를 바로잡으면 의상이 좋아지는가 | 의상 태그(분석기 그대로 ↔ 사람이 수정) | T 합격 | 의상 "예" 1→14/20 (종류·주색 수준) | — | 의상 이미지 IP(I·TI)는 VRAM 7.21 GiB·단계당 19초로 실행 불가 | garment-fidelity-20261001/results.md |
| garment-pose | 자세가 의상 재현을 얼마나 흔드는가 | 자세만 (없음/기본/시나리오), 태그 T 고정 | PZ1(실루엣)·TL 성립 | 종류 19/18/18, 실루엣 20/16/13, 색 배치 6/4/2, 세부 5/2/2 | — | 4항목 라벨 첫 사용 | garment-pose-20261001/results.md |
| high-decomp | 하이앵글 붕괴는 어느 조건에서 생기나 | 조건 누적(골격만→+자세태그→+캐릭터→+의상→+얼굴 IP) | D1 | 붕괴 16/11/2/1/4, 동일성 0/0/2/0/11 | — | L1 실패는 최소 프롬프트 탓이 섞임(front-ctrl) | high-decomp-20261001/results.md |
| front-ctrl | L1 실패가 하이앵글 골격 탓인가 | 골격만 (정면/없음), 하이앵글은 재사용 | FB1 | 얼굴 없는 형체 정면 16, 없음 11, 하이앵글 16 / 비율 11·10·0 | — | 하이앵글 쪽에만 "from above" → 골격만 바뀐 비교 아님 | front-ctrl-20261001/results.md |
| face-ip-high | 얼굴 IP 일정·강도로 하이앵글 동일성과 몸 안정을 함께 지킬 수 있나 | 얼굴 IP 일정·강도만 | 합격 없음 | 아래 2-2 | — | 캐릭터 2 × seed 8 | face-ip-high-20261001/results.md |

### 2-2. 얼굴 참조 일정 시험 (face-ip-high) — 항목 분리

| 조건 | 실제 IP scale (28단계) | 몸 구성 실패 SB | 얼굴 없는 형체 SF | 머리 잘림 SC (판정 외) | 자세 "예" | 동일성 "예" |
|---|---|---|---|---|---|---|
| A0 | 0.0 전 구간 | 0 | 0 | 0 | 10/16 | 0/16 |
| A1 현재 조건부 | 0~10 0.5, 11~27 0.9 | 2 | 0 | 3 | 8 | 11 |
| B | 0~10 0.0, 11~27 0.9 | 1 | 0 | 2 | 10 | 4 |
| C | 0~10 0.0, 11~27 0.7 | 1 | 0 | 1 | 9 | 3 |
| D | 0~13 0.0, 14~27 0.9 | 0 | 0 | 1 | 11 | 0 |
- 실패 유형 분리(SB/SF/SC)는 이 시험의 사전 등록 정의다(사후 아님). 하이앵글에서 동일성은 초기 단계 얼굴 참조에서 거의 전부 나오고, 그 참조가 몸 안정을 깎는다 — 일정·강도만으로는 둘을 함께 지키지 못했다.

### 2-3. 해석상 주의점 (기존 문서는 고치지 않고 여기 적는다)
1. **front-ctrl 은 골격 단독 원인을 확정하지 않는다.** 하이앵글 조건에만 `from above` 가 있다. "몸 형태 소실은 하이앵글 골격 탓"은 확정 아님(front-ctrl results.md 정정 절).
2. **최소 프롬프트 실패로 제품의 "태그 개수 경고"를 정당화하지 않는다.** 최소 프롬프트(성별+standing)는 실제 제품 입력이 아니다. front-ctrl 결과의 "외형 태그 적음 + 원근 자세 경고 후보"는 근거 부족 — 이후 연구로 둔다.
3. **의상 "예" 14/20(garment-fidelity)은 종류·주색 수준이다.** 세부 디자인 동일성을 뜻하지 않는다. 세부는 자세가 없어도 5/20(garment-pose).
4. **의상은 네 항목으로 구분한다:** 종류·구성 / 실루엣·길이 / 색 배치 / 세부 디자인. 자세는 실루엣에 영향, 색·세부는 태그 전달 한계.
5. **라벨 세션이 다르면 수치를 합치지 않는다.** 예: 같은 16장(high-decomp L5)이 high-decomp 에서 붕괴 4, face-ip-high 에서 SB 2 + SC 3. 같은 B(integration 120장)의 동일성 "예"가 세션마다 78·92.
6. **"붕괴" 하나에 다른 실패가 섞였던 시험이 있다.** high-decomp·front-ctrl 의 붕괴 수치는 얼굴 없는 형체와 몸 구성 실패를 함께 센 것이다. face-ip-high 부터 분리.
7. **기존 인수인계 초안(handoff-20260930)보다 이후 시험이 우선한다.**
   - 조건부 IP: 통합 확인은 final-int 에서 됐다(초안의 "통합 미검증"은 해소). 다만 하이앵글에서는 몸 안정과 교환(face-ip-high).
   - 마른 체형 규칙(PS): 자세 저하가 확인됐다(slim-onoff). NS 대안은 기준 미달(slim-cost). 사용자 결정 대기.
   - 성별: 초안의 "남성이면 부정 1boy→1girl(태거 기준)" 규칙은 계약 위반 방식이다. 초안 01·02 는 제품 계약 방식으로 이미 고쳐 두었다(gender-contract).
8. **e2e 의 "전체 흐름 통과"는 사용자 확인 단계를 "그대로 진행"으로 가정한 결과**이고 성별은 태거값이었다(이번 사용자 지정과 결과적으로 같았음).

---

## 3. 이번 구현 후보 표

"사용자 채택 근거"는 사용자가 대화에서 직접 확정한 말만 적는다(날짜 2026-09-28~10-01). 시험의 "후보"·"추천"은 채택으로 보지 않는다.

| 기능 | 사용자 채택 근거 | 검증 범위 | 운영 반영 | 시험 구현 위치 | 제품 연결 위치 | 남은 위험 | 이번 처리 |
|---|---|---|---|---|---|---|---|
| 자동 얼굴·머리 크롭 C6 + 미리보기 확인 | 9/28 "자동 크롭 + 사용자 확인"(face-crop-c5 decision.md) | C6 26장: 팔·손 없음 15/17, 손상 0.06, 과잉 거부 2/26 | 없음 | **정식: outputs/face-crop-c5-20260928/crop_c5.py + crop_c6.py (methods_c5.md·methods_c6.md)**. e2e face.py·retarget3 face_crop.py 는 간이판(채택 아님) | 새 모듈. 캐릭터 입력 분석 단계(FLOW 3) 옆. 필요: 머리 검출(dghs-imgutils 0.19, 별도 venv `outputs/pose-identity-20260927/identity-v2/.venv`), DWPose 몸 18점 + 얼굴 68점, isnet 전경(scripts/body_comparison_runner.py:234) | 운영 venv 에 imgutils·DWPose 없음 → 설치 승인 필요. ordinary-female 류(팔이 머리 뒤)는 거부됨 | 이번 구현 (설치 승인 전제) |
| 골격 정규화·검사(K1~K7)·경고 처리 | "골격 검사 경고 + 사용자 선택 확정", K1·K2 다른 이미지 요청 + 설정값(9/30), K7 범위 확정, 추측 관절 = 제외하지 않고 경고 지표 | 정규화: 붕괴 10→4/30. K7 18장 중 의도대로 1. 처음 보는 자세 7장 판정 | 정규화 없음. 운영은 `pose_estimation.prepare_pose_control_input` 의 크기 맞춤만 | outputs/pose-newimg-20260929/prepare_controls.py, leg-overlap-20260929/measure.py, e2e-20261001/analyze.py (`process_pose`, `check`) | genai_lab/pose_estimation.py:104 `execute_pose_reference_estimation`, :255 `prepare_pose_control_input`; scripts/pose_reference_runner.py:34 `create_pose_preview_and_coordinates`; 검사 문턱 outputs/pose-identity-20260927/skeleton-check/rule-values.json | 받침 물체(의자 등) 자동 검출 없음. 선화 K2 거부 | 이번 구현 |
| 얼굴 방향 태거 조건 (경고·유사도 안내·조건부 IP) | "③ 안내 넣고…", "조건부 적용으로 진행" | 18장: 놓침 0, 오경보 6(선화 3) | 없음 | outputs/input-direction-20260930/tags.py, e2e analyze.py | genai_lab/clothing_analysis.py:56 `WdTagSession` 재사용 | 오경보(정면인데 경고) 비용은 안내 문구 | 이번 구현 |
| 사용자 지정 성별 계약 | 기존 제품 계약(FLOW 4·reference_tag_policy). 시나리오별 성별 지정 10/1 | gender-contract GC1·GC2 | **운영 경로엔 있음, 1회 생성 시험 경로엔 없었음** | gender-contract-20261001/make_plan.py (방식 재현만) | genai_lab/character_preferences.py:14 `load_character_gender`; reference_tag_policy.py:45·69; clothing_reference_generation.py:12 `BASE_GENDER_CONDITION_TAGS`, :58 `prepare_design_reference_request`, :105 `gender_guard` | 남성 지정 시 의상 재현 하락(34→23) | **이번 구현 (출시 차단 항목, 5절)** |
| 의상 승인 태그 + 사용자 수정 단계 | 제품 계약 FLOW 5단계(기존). 1회 생성 경로 연결은 채택 근거 없음 | garment-fidelity T: 종류·주색 1→14/20. 세부 5/20 미해결 | 운영 GUI 에 태그 승인 있음(Base 경로) | e2e analyze.py(상위 8 선택), garment-fidelity prompts.json | genai_lab/garment_detail_analysis.py:95 `analyze_garment_details`, :23 `detail_group`; clothing_reference_generation.py:58 | 분석기 오인(마네킹 다리→thighhighs, 색 오독). 세부 동일성 불가 | 이번 구현(기존 승인 화면을 1회 경로에 연결) + 세부 재현은 이후 연구 |
| 프롬프트 조립·토큰 처리 | 덮지 않는 부위 규칙·주머니 규칙·청크(통합 IP 에서 함께 채택) | integration plan 120개 문자열, prompt-chunk EQ(1청크 SHA 동일) | 운영은 `build_reference_prompt`(다른 순서·꼬리) | outputs/integration-20260929/make_plan.py (`nouns`, `uncovered`, `has_pockets`), run_int.py (`pieces`, `ids_for`, `encode_pcs`) | genai_lab/reference_prompt_budget.py:36 `build_reference_prompt` | 의상 명사 목록이 시험마다 늘었다(e2e 에서 sweater·skirt 등 추가) — 제품 어휘 표로 통합 필요 | 이번 구현 |
| 부정 프롬프트 최종본 | 노출 예방 "1번 채택", 배경 예방 "1번 채택" | negative-prevent2·bg-prevent(BP1 경계 미달·BP2 성립) | 없음(운영 기본 부정은 configs/animagine.yaml:16) | outputs/bg-prevent-20260930/plan.json[0].negative | configs/animagine.yaml:16 근처, 성별어는 템플릿에서 빼고 성별 계약 함수가 붙임 | 배경 잡티 감소 효과는 기준 경계(11→6) | 이번 구현 |
| T2I-Adapter 골격 + 얼굴 IP 일정 | 세기 1.2 "기본값 확정", 조건부 IP "조건부 적용" | integration IP, final-int(조건부 IP 통합 확인), SHA 재현 다수 | **없음** — 운영은 ControlNet(xinsir openpose, 0.65) + img2img Base | outputs/integration-20260929/run_int.py, ip-timing-20260930/run_int.py(`ip_early`) — UNet forward pre-hook 주입 | 새 생성 경로. 모델: D:/genai-cache/models/t2i-adapter-openpose-sdxl-1.0, h94/IP-Adapter ip-adapter-plus-face_sdxl_vit-h + image_encoder | 하이앵글: 동일성과 몸 안정 교환(face-ip-high). 무릎 자세 초기 IP 붕괴(ip-timing-int) → 조건부로 회피 | 이번 구현 (옵션, 기본 OFF) |
| 1회 생성 경로 | D-074 방향 결정(후보, 운영 미반영), 편집 2단계 대신 "여러 seed 생성 후 선택" | e2e E1~E5, integration IP | 없음 | outputs/*/run_int.py | genai_lab/generation_orchestrator.py:249 근처(`reference_pose` 인자와 같은 방식의 명시적 옵션) | 운영 기본값 전환·FLOW 갱신은 별도 결정 | 이번 구현 (옵션, 기본 OFF) |
| 결과 4장 제공·사용자 선택 | "추천대로 4장" | integration: 요청 30/30 성공 | 없음 | integration run 구조(요청당 seed 4) | GUI 8단계(결과 비교·승인, FLOW 8) | 장당 약 22초 × 4 | 이번 구현 |
| 노출 검사와 결과 숨김 | "노출 실패는 숨기는 걸로" | **단독 기준 미달**: 게이트 v4 놓침 26/62(exposure-miss Y3). 부정 프롬프트와 함께 통합에서 보이는 노출 0/120 | 없음 | outputs/integration-20260929/gate.py, e2e post.py(규칙 자동 생성) | 새 모듈. WdTagSession 재사용 | 규칙(승인 태그·몸통 덮음·핵심 태그)을 의상 태그에서 기계적으로 만들면 깨짐(e2e S4 과다 숨김, garment-fidelity midriff 버그). "검증 완료" 아님 | 이번 구현(숨김 장치) + 사용자 결정(규칙 생성 방식·단독 안전 기준 미달 수용 여부) |
| 배경 분리(isnet) | **채택 근거 없음** (bg-artifact BA3: 분리로 해결 안 됨 → 부정 프롬프트 채택) | bg-artifact | 운영에 함수 있음 | — | scripts/body_comparison_runner.py:234 `extract_anime_character_foreground_mask`, configs/animagine.yaml:85 | 분리 실패 시 잔여 잡티 | 사용자 결정 대기 |
| 마른 체형 규칙 | 9/30 "PS 방식으로 채택" | slim-body SB1. **이후** slim-onoff 자세 28→20, slim-cost NS 기준 미달 | 없음 | slim-body plan, final-int make_plan.py `slim_prompt` | 프롬프트 조립 모듈 | 자세·동일성 저하 | 사용자 결정 대기 (PS 유지 / NS / 끔) |
| 리타게팅 | 시험 결과 불필요(RX2) | retarget3 | — | — | — | — | 구현 안 함 |

### 시험 코드를 그대로 복사하면 안 되는 부분
- **(2026-10-01 정정) 얼굴 크롭 C6 정식 정의**: 머리 상자 선택(코 포함 상자 중 신뢰도 최고, 없으면 전체 중 신뢰도 최고) → 확장(좌우·위 10%, 아래 없음) → isnet 전경 ∩ 확장 상자 → DWPose 팔 띠 제거(얼굴 68점 볼록 껍질 보호) → 턱 아래 제거 → 마스크 경계 상자 + 5% 여백, 흰 배경 정사각 → 팔 근접 시 거부. 상세는 face-crop-c5-20260928/methods_c5.md·methods_c6.md. e2e/retarget3 의 "가장 큰 상자 +10% 사각 크롭"은 시험 간이판이며, 이전 구현 프롬프트가 이를 C6 로 잘못 적었다.
- `run_int.py` 의 UNet forward pre-hook 잔차 주입과 `scheduler.step` 래퍼(x0 미리보기용): 시험 관측 장치다. 제품은 `StableDiffusionXLAdapterPipeline`(pose-edit-compare 에서 SHA 동일 확인) 또는 같은 계약의 명시 함수로.
- `D:/genai-cache` 하드코딩, `HF_HUB_OFFLINE` 환경변수 설정, `os.chdir('D:/genai-cache/huggingface/easy-dwpose')`(DWPose 모델 위치 의존) → 설정 한 곳으로.
- 시험의 성별 처리(태거 1boy/1girl, 고정 "1boy" 부정) — 쓰면 안 됨. 제품 함수 재사용.
- 노출 게이트의 의상별 규칙을 태그에서 자동으로 만드는 코드(e2e post.py, garment-fidelity gate.py): 알려진 버그 경로.
- 의상 명사·덮지 않는 부위 목록: 시험마다 손으로 늘린 목록 — 제품 어휘 표로.
- 라벨·시트·블라인드 스크립트: 제품 대상 아님.
- DWPose 는 운영 venv(D:/genai-cache/venv)에 없고 catvton-venv 에만 있다(easy_dwpose). imgutils 는 identity-v2/.venv 에만 있다.

---

## 4. 사용자 결정 대기 목록

| 항목 | 선택지 | 근거 | 이번 구현에 미치는 영향 |
|---|---|---|---|
| 마른 체형 규칙 | PS 유지 / NS(skinny 제거) / 끔 | slim-onoff, slim-cost | 프롬프트 조립 모듈의 체형 규칙 값. 설정값으로 두면 구현은 막히지 않음 |
| 하이앵글 지원 수준 | (a) 제한 지원: 경고 + 4장 선택 / (b) 범위 밖 안내 / (c) 얼굴 참조 약화 기본값(동일성 포기) | face-ip-high 합격 없음, high-decomp | 경고 문구·입력 확인 화면. 실패 자동 검출 장치가 없어 (a)도 사용자 선택에 의존 |
| 노출 게이트 수용 | 단독 기준 미달(놓침 26/62)을 부정 프롬프트와의 조합으로 수용할지 | exposure-miss, integration | "노출 안전" 표시 문구와 숨김 정책 |
| 노출 게이트 규칙 생성 방식 | 의상 승인 화면에서 사용자가 몸통 덮음·핵심 부품 확인 / 자동 | e2e·garment-fidelity 버그 | 게이트 입력 구조 |
| 배경 분리 사용 | 결과에 isnet 흰 배경 합성 / 안 함 | bg-artifact BA3 | 결과 처리 단계 |
| 의상 수정 단계 강제 여부 | 1회 경로에서 태그 확인을 필수로 / 선택 | garment-fidelity | GUI 흐름 |
| imgutils 설치 | 운영 venv 에 dghs-imgutils 설치 승인 | 얼굴 크롭 | 미승인 시 얼굴 크롭 구현 차단 |

---

## 5. 이후 연구 및 출시 차단 목록

### 출시 차단
| 문제 | 이유 | 처리 |
|---|---|---|
| 1회 생성 경로의 성별 계약 미적용 | 기존 제품 계약 위반(사용자 지정 성별 무시 → 남성 지정 캐릭터 29/36 여성) | 이번 구현에 포함(제품 함수 재사용) + 3성별 단위 테스트 |
| 사용 금지 파일 | 143960292_19(이름 무관), 참조 의상 폴더의 0e0f64558c7c6f84b97b43b5b60e693e.jpg(미성년 사진) | 시험·검증·배포 레시피 어디에도 넣지 않음 |

### 이후 연구 (이번 구현에서 제외)
| 항목 | 지원 제한으로 처리 가능? | 사용자 확인 필요? | 기능 비활성화 필요? | 출시 차단? |
|---|---|---|---|---|
| 의상 세부 디자인·색 배치 완전 재현 (색 5~6/20) | 예 — "의상은 종류·주색 수준" 안내 | 결과 승인 단계에서 | 아니오 | 아니오 |
| 의상 이미지 시각 조건(두 번째 IP-Adapter) | — (8GB 에서 실행 불가) | — | 해당 없음(구현 안 함) | 아니오 |
| 하이앵글·로우앵글 실패 원인 규명(깊이 정보 등) | 예 — 하이앵글 제한 지원/범위 밖 | 예(4절) | 아니오 | 아니오 |
| 하이앵글 몸 구성·머리 잘림 자동 검출 | 아니오(현재 장치 없음) | 예 | 아니오 | 아니오 |
| 받침 물체·선화·총 정면 편향 | 예 — 범위 밖 안내 | 입력 확인 | 아니오 | 아니오 |
| "외형 태그 적음 + 원근 자세" 경고 | 근거 부족(2-3 주의 2) | — | — | 아니오 |
| 남성 지정 시 의상 재현 하락 | 예 — 안내 | 결과 승인 | 아니오 | 아니오 |
| 미검증 입력 일반화(캐릭터 다양성) | 예 — 지원 범위 명시 | — | 아니오 | 아니오 |
| 신규 모델·추가 어댑터 탐색, 깊이 어댑터(다운로드 필요) | — | 다운로드 승인 | — | 아니오 |

---

## 6. 구현 검증 사례와 완료 조건

검증은 아래 네 범주로 한정한다. 최적화 실험을 끼워 넣지 않는다.

1. **옵션 OFF 바이트 동일**: 새 경로 옵션을 끈 상태에서 기존 Base·Final 출력이 구현 전과 바이트 동일(BACKLOG 66). 기존 회귀 사례를 그대로 사용.
2. **채택 구성의 제품 재현**: 아래 사례를 제품 코드로 생성해 시험 raw SHA 와 일치(같은 PC·라이브러리).
   | 사례 | 입력 | 기대 SHA |
   |---|---|---|
   | final-int F_original_ordinary-female_KNEEL_209210101 | final-int-20261001/plan.json 해당 항목 | final-int-20261001/cases/…/run.json `raw_sha256` (slim-onoff 재현에서 일치 확인됨) |
   | integration original_ordinary-female_KNEEL_209210101 (조건부 IP 0.0 자세) | integration-20260929/plan.json | integration-20260929/cases/…/run.json |
   | final-int F_original_raccoon_HIGH_209210101 (조건부 IP 0.5 자세) — 단 성별 계약 적용 전 구성이므로 성별 OFF 비교용 | final-int plan | final-int run.json |
   | gender-contract U_original_raccoon_KNEEL_209210101 (성별 계약 적용) | gender-contract-20261001/plan.json | gender-contract-20261001/cases/…/run.json |
   프롬프트 문자열은 integration plan 120개·final-int plan 120개와 일치(성별 계약으로 바뀌는 항목은 gender-contract plan 기준).
   **(2026-10-01 정정)** 성별 계약 적용 후에는 1~3번 사례의 프롬프트가 바뀐다. SHA 재현은 시험 문자열을 생성 모듈에 그대로 주입해 확인하고, 성별·조립은 문자열 일치로 따로 확인한다(CODEX_IMPLEMENTATION_PROMPT.md 4절).
   메모리: 최대 reserved ≤ 6.5 GiB(시험 6.41~6.42).
3. **입력·결과 처리 연결**: K1·K2 거부(정책 설정 변경 시 처리만 바뀜 — 단위 테스트), K3~K7·추측 관절·얼굴 방향 경고와 "진행/다른 이미지" 선택, 얼굴 크롭 C6 거부·미리보기, 성별 3종(남/여/지정 안 함)의 긍정 첫 태그·부정 첫 태그가 `prepare_design_reference_request` 와 같음, 노출 게이트 숨김 "N장 제외됨" 표시. 검증 입력: e2e-20261001 analysis.json 의 5 시나리오 + 입력 검사 시연 3장(U1 경고, U2·U3 K2 거부).
4. **기존 테스트·회귀**: 기존 tests/ 전체 + 변경 모듈 단위 테스트(tests/test_reference_pose.py 같은 형식).

완료 조건: 1~4 통과 보고. 시험 수치(동일성·붕괴 등)를 다시 측정하는 것은 완료 조건이 아니다(시험은 이미 끝났다).

---

## 7. 환경·모델 경로·미커밋 변경·git status

- PC: mydesk, RTX 4060 8GB(VRAM 8188 MiB), Windows 11. 저장소 Z:\genai-lab (= \\192.168.0.109\win_g\genai-lab).
- Python 환경:
  - 생성: D:/genai-cache/venv (diffusers, transformers, onnxruntime) — imgutils·easy_dwpose 없음
  - DWPose: D:/genai-cache/catvton-venv (easy_dwpose)
  - 얼굴 크롭: outputs/pose-identity-20260927/identity-v2/.venv (dghs-imgutils 0.19.0, Python 3.10.6)
- 모델 캐시: D:/genai-cache 는 **G: NVMe 로 가는 정션**이다(D: HDD Predictive Failure, 원본은 .old-failing). 경로 상수는 설정 한 곳에서 바꿀 수 있게.
  - Animagine: D:/genai-cache/huggingface/models--cagliostrolab--animagine-xl-3.1 (snapshot 483f0c32…)
  - T2I-Adapter openpose SDXL: D:/genai-cache/models/t2i-adapter-openpose-sdxl-1.0
  - IP-Adapter: models--h94--IP-Adapter (sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors, ip-adapter-plus_sdxl_vit-h.safetensors, models/image_encoder)
  - WD 태거: models--SmilingWolf--wd-vit-tagger-v3 (임계 0.35)
  - DWPose: D:/genai-cache/huggingface/easy-dwpose (yolox_l.onnx, dw-ll_ucoco_384.onnx)
  - isnet-anime: configs/animagine.yaml:85 `foreground_model_id`
  - 애니 머리 검출: identity-v2/model-cache (head_detect_v2.0_s)
  - 스케줄러 설정: outputs/pose-identity-20260927/exposure-v2/preflight-data.json `A.scheduler_config`
- git (2026-10-01 확인):
  - 현재 브랜치 codex/sdxl-local-garment-strength-20260923, HEAD 8283721. origin 대비 미푸시 커밋 2개(8283721 D-074 기록, 5372d1b pose ControlNet opt-in).
  - 추적 안 된 항목: `afe_load(p.read_text())`(이름이 이상한 파일), `approved-bases/`, `inputs/training_candidates/` — 누가 만든 것인지 확인 필요, 이 인수인계와 무관.
  - 시험 요약 브랜치: claude/pose-tests-20261001 (5636e7f, origin/main 기준, outputs/ 문서 78개만 강제 추가). 오늘 이후 시험(gender-contract 이후)은 그 브랜치에 없다.
  - 모든 시험에서 운영 파일 327개 해시·git status 전후 동일.
- 이 문서 작성 중 운영 소스·FLOW·DECISIONS·BACKLOG 수정, 커밋·푸시, 다운로드·설치 없음.
