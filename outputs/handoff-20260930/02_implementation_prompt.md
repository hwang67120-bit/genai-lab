> **2026-10-01: 이 초안은 `outputs/implementation-handoff-20261001/` (README.md, CODEX_IMPLEMENTATION_PROMPT.md) 로 대체됐다. 이후 시험(성별 계약, 마른 체형 자세 저하, 하이앵글 얼굴 참조 교환 등)이 반영되지 않은 부분이 있으니 그쪽을 기준으로 한다.**

# Codex 프롬프트 2 — 1회 생성 구조 운영 구현 (단계별)

(프롬프트 1 로 DECISIONS·BACKLOG 반영이 끝난 뒤 사용. 작성: Claude, 2026-09-30. 초안.)

---

## 목표
D-074 후보 + D-075~D-078 로 확정한 구성을 **제품 코드에 옵션으로** 넣는다. 현재 운영 경로(Animagine Base → 2차 정제, FLOW.md)는 그대로 두고, 새 경로는 **명시적 옵션, 기본 OFF**. OFF 에서 기존 Base·Final 출력이 **바이트 동일**해야 한다(BACKLOG 66).

## 참조 구현 (시험 코드, 그대로 복사하지 말고 구조를 따른다)
| 기능 | 파일 (outputs/…) | 비고 |
|---|---|---|
| 골격 정규화·검사 K1~K6·제어 이미지(카드 색 반전) | pose-norm-20260928/prepare_controls.py, pose-newimg-20260929/prepare_controls.py | 원본 DWPose → bbox 정규화 → 재검출 |
| K7 다리 겹침 | leg-overlap-20260929/measure.py | max(dK, dA) < 0.15T |
| 자세 태그(WD 태거, allow 목록) | pose-range-20260928/pose_tags.json(allow), pose-newimg-20260929/pose_tags.py | 방향 태그는 자동 채택 안 함 |
| 얼굴 방향 태거 조건 | input-direction-20260930/tags.py | looking_at_viewer<0.35 or profile/from_side/from_behind/from_above>=0.35 |
| 프롬프트 조립·덮지 않는 부위·주머니 규칙 | integration-20260929/make_plan.py (nouns/uncovered/has_pockets) | 의상 명사 목록은 제품 어휘 표와 통합 필요 |
| 청크 인코딩 | integration-20260929/run_int.py (pieces/ids_for/encode_pcs) | prompt-chunk EQ 로 1청크 SHA 동일 검증됨 |
| 생성 (T2I 잔차 훅 주입, IP 일정, 28단계) | integration-20260929/run_int.py, ip-timing-20260930/run_int.py(ip_early) | StableDiffusionXLAdapterPipeline 로도 같은 결과(pose-edit-compare 검증 V_ SHA 일치) |
| 노출 게이트 v4 | integration-20260929/gate.py | A·B·C + D, 의상별 승인 태그·몸통 덮음 |
| 배경 분리 | scripts/body_comparison_runner.py extract_anime_character_foreground_mask | isnet-anime |
| 마른 체형 규칙 | slim-body-20260930/criteria.md, retarget3-20260930 (어깨폭/몸통) | 캐릭터 이미지 DWPose |
| 얼굴 자동 크롭 C6 | face-crop-c5-20260928/crop_c6.py, decision.md | 팔 근처면 거부 |

## 단계 (단계마다 끝나면 멈추고 보고, 커밋은 사용자 확인 후)

### 1. 입력 전처리 모듈
- 자세 이미지: DWPose → 정규화 → 재검출 → 검사(K1~K7, 추측 관절 수, 방향 태거 조건) → 제어 이미지.
- 결과 객체: 제어 이미지, 관절 좌표, 검사 결과(검사별 걸림 여부·값·사유), 겹친 골격 그림, 자세 태그, 얼굴 정면 여부.
- 검사 처리 정책은 **설정 한 곳**에 검사별로 둔다: `reject`(다른 이미지 요청) / `warn`(경고+사용자 선택) / `pass`. 기본값 K1·K2 = reject, K3~K7·추측 관절·방향 = warn. 문턱값(K7 0.15T, 추측 관절 신뢰도 0.30~0.50·2개, 태거 0.35)도 같은 설정에 둔다. 전처리 모듈은 판정값만 내고, reject/warn 결정은 정책을 읽어 한다(서비스 단계에서 코드 수정 없이 조정 가능하게, D-077).
- 캐릭터 이미지: 태그(외형·체형·고정), 얼굴 C6 크롭(+미리보기), 어깨폭/몸통 비율(마른 체형 판정).
- 검증: pose-newimg·leg-overlap·input-direction 의 18장에서 시험 결과와 같은 판정이 나오는지(골격 검사 결과, K7 값, 방향 조건). 제어 이미지는 pose-norm control_*.png 와 SHA 비교. 정책 설정을 바꾸면(예: K2 → warn) 코드 수정 없이 처리만 바뀌는지 단위 테스트.

### 2. 프롬프트 조립 모듈
- 순서: [사용자 지정 성별 태그 + 보조 태그(제품 BASE_GENDER_CONDITION_TAGS)] + 의상 태그 + 덮지 않는 부위 + 외형·체형·고정(+마른 체형 규칙) + 자세 태그(주머니 규칙) + 꼬리(solo, full body, white background, simple background, coherent anatomy, best quality).
- 부정: D-075 최종본(성별어 없는 템플릿) + 사용자 지정 성별이면 맨 앞에 반대 성별 차단(제품 clothing_reference_generation.py 와 같은 함수 재사용). 성별 출처는 character_preferences 만, 태거 성별은 쓰지 않는다.
- 검증: 지정 남성·여성·지정 안 함 각각에서 긍정 첫 태그와 부정 첫 태그가 제품 경로(prepare_design_reference_request)와 같은지 단위 테스트.
- 75토큰 초과 시 고정 청크. 조립 결과·토큰 수·청크 수를 기록.
- 검증: integration-20260929/plan.json 의 120개 positive/negative 를 같은 입력으로 다시 만들어 문자열 일치.

### 3. 생성 모듈 (옵션, 기본 OFF)
- Animagine XL 3.1, T2I-Adapter openpose SDXL(revision f909988…), ip-adapter-plus-face_sdxl_vit-h + ViT-H, 736×1232, 28단계, CFG 5.5, EulerAncestral(exposure-v2 preflight scheduler_config), enable_model_cpu_offload.
- 골격 세기 1.2, 0~10단계. IP: 기본 0.0→11단계 0.9 / 얼굴 비정면이면 0.5→11단계 0.9.
- 요청당 seed 4장 순차, 장마다 스케줄러·IP scale·훅 상태 초기화, 한 장씩 콜백으로 전달.
- 검증 (필수):
  - integration-20260929 의 seed 209210101 몇 장을 같은 입력으로 생성해 **raw SHA 일치**(같은 PC·라이브러리 기준). 불일치면 차이 원인부터 보고.
  - 옵션 OFF 에서 기존 Base·Final 바이트 동일.
  - 최대 reserved <= 6.5 GiB, 장당 시간 기록(시험: 20.9초, RTX 4060).

### 4. 결과 처리
- 노출 게이트 v4 → 실패 장 숨김, "N장 제외됨". isnet 분리 + 흰 배경 합성. 각 장 seed·게이트 결과·선택 여부 저장. "다른 seed로 4장 더"는 seed 만 바꿔 재호출.

### 5. GUI 연결
- 입력 확인 화면: 겹친 골격 그림, 경고 사유, "진행 / 다른 이미지"(reject 검사는 "다른 이미지"만, 정책 설정에 따름), 범위 밖 안내, "캐릭터 유사도가 낮을 수 있음", 얼굴 크롭 미리보기 확인.
- 결과 화면: 4장 순차 표시, 제외 수, 선택.

### 6. 재검증 (통합)
- integration-20260929 과 같은 30요청(의상 3 × 자세 5 × 캐릭터 2)을 **제품 코드로** 실행하고 같은 기준(IN1~IN4)으로 판정. 사전 등록·라벨 잠금 방식은 기존 시험과 같이.
- 조건부 IP 일정은 여기서 처음으로 통합 검증된다(D-075 의 남은 확인 항목).

## 주의
- 사용 금지 파일: 143960292_19(이름 무관 모두), 참조 의상 폴더의 0e0f64558c7c6f84b97b43b5b60e693e.jpg(미성년 사진).
- 모델 경로 D:/genai-cache 는 현재 G:\genai-cache 로 가는 정션이다(D: 디스크 고장). 경로 상수를 새로 만들 때는 설정 한 곳에서 바꿀 수 있게.
- 다운로드는 사용자 승인 후.
- 이 구현은 D-074 운영 반영 조건(노출 게이트 합격·자동 크롭·골격 검사·정체성 판정·OFF 바이트 동일) 충족을 위한 것이며, 운영 기본값 전환과 FLOW.md 갱신은 별도 결정으로.
