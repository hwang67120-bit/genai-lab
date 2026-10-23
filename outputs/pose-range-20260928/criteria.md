# 자세 확대 시험 — 사전 등록

작성: 2026-09-28, Claude (Codex 부재 기간, 시험 범위만). 준비·생성 전에 잠근다. 결과를 보고 기준·라벨·규칙을 바꾸지 않는다.

## 목적
D-074 후보 구성의 가장 큰 한계는 검증된 자세가 팔짱 1종이라는 점이다. 골격 검사기를 통과한 다른 유형의 자세에서도
자세·정체성·의상이 함께 나오는지 본다. 동시에 (1) 자세 이미지 태그 보충(BACKLOG 70)의 효과, (2) 노출 검사 v3,
(3) 결정된 자동 크롭(C6)의 실제 사용을 확인한다.

## 고정 구성 (exposure-v2 와 같음)
txt2img StableDiffusionXLAdapterPipeline, cagliostrolab/animagine-xl-3.1, 736×1232, 28단계, CFG 5.5,
EulerAncestralDiscreteScheduler (exposure-v2 preflight-data.json 의 A.scheduler_config), 부정 프롬프트 동일.
T2I-Adapter openpose SDXL (D:\genai-cache\models\t2i-adapter-openpose-sdxl-1.0), scale 1.5, factor 0.4, 카드대로 색 채널 반전.
ip-adapter-plus-face_sdxl_vit-h + ViT-H, IP 0~10단계 0.0 → 11~27단계 0.9.
파이프라인은 한 번 적재해 재사용하고 실행마다 scheduler·IP scale·generator를 초기화한다.
실행마다 UNet 28회의 실제 IP scale, adapter 잔차가 들어간 단계를 기록하고 예상과 다르면 그 실행을 무효로 표시한다.
메모리는 torch allocated/reserved 피크만 기록한다 (Windows 공유 메모리 카운터는 이번에 기록하지 않음).

## 캐릭터와 얼굴 참조
- raccoon: 결정된 자동 크롭 C6 결과 outputs/face-crop-c5-20260928/crops_c6/I02/crop.png (SHA-256 확인)
- ordinary-female: 원본이 C6에서 거부되므로 수동 크롭 generalize/face_ref_clean.png 를 쓴다.
- 외형·체형·고정 신체 특징 태그: generalize/characters.json 의 selected (변경 없음).

## 자세 (골격 검사 통과, skeleton-check results.md)
| 기호 | 파일 | 유형 |
|---|---|---|
| KNEEL | 참조 자세\d91c831e2d9ec42e4fb16bba68f18b0d.jpg (p03) | 무릎 꿇기 |
| POCKET | 범위조사\주머니_01_pexels-6059481.jpg (p13) | 양손 주머니 |
| GUN | 범위조사\사격_01_pexels-6204837.jpg (p10) | 총 조준 |
| HIGH | 범위조사\하이앵글_01_pexels-27985081.jpg (p16) | 하이앵글 (가로 사진) |
| LOW | 범위조사\로우앵글_02_pexels-6533836.jpg (p09) | 로우앵글 |
제어 이미지: generalize/prepare_pose.py 와 같은 절차 (DWPose → prepare_pose_control_input 736×1232 비율 유지 여백 → 채널 반전).
제어 이미지 생성 후 골격 검사 K1~K6 (skeleton-check rules.md 값)을 다시 적용해 모두 통과해야 한다. 거부되면 그 자세는 생성하지 않는다.

## 자세 태그 보충 (BACKLOG 70)
- 자세 이미지에 wd-vit-tagger-v3 (운영 설정, 임계값 0.35)를 돌린다.
- 허용 목록에 있는 태그만 채택: hands_in_pockets, hand_in_pocket, holding_gun, holding_weapon, gun, rifle, aiming,
  kneeling, on_one_knee, squatting, sitting, from_above, from_below, holding_phone, cellphone, looking_at_phone,
  arms_up, hands_up, arm_up, hand_on_own_head, hands_on_own_head, crossed_arms, hand_up
  (어휘에 없는 태그는 "어휘 없음"으로 기록하고 대체하지 않는다).
- 채택 태그는 공백으로 바꿔 의상 태그 뒤, solo 앞에 점수 내림차순으로 넣는다. 두 토크나이저 모두 75 토큰 이하가 되도록
  넘치면 점수 낮은 채택 태그부터 뺀다.

## 실행 (42회)
- 태그 있음: 5 자세 × 2 캐릭터 × seed 209210001~209210003 = 30
- 태그 없음: POCKET, GUN × 2 캐릭터 × 같은 seed = 12
- 의상: 원래 의상 (white camisole, black shorts). 신발 태그 없음. 조건당 1회, 재시도 없음. OOM일 때만 중단.

## 측정
1. 자세: 결과에 같은 DWPose. 8관절(어깨·엉덩이·무릎·발목) 정규화 거리 d_target(목표 자세), d_orig(캐릭터 입력 이미지).
   ratio = d_target / d_orig. 관절 신뢰도 < 0.30 이면 미측정.
2. 육안 라벨 (결과 태깅 전에 작성하고 SHA-256 잠금, 판정자 Claude):
   - V1 자세 일치: 예 / 부분 / 아니오
   - V2 소품·손 (POCKET, GUN만): POCKET = 손이 주머니에 들어감 / 아님, GUN = 총을 쥐고 겨눔 / 총은 있으나 쥐는 모양 이상 / 총 없음
   - V3 인체 이상: 팔다리 수 이상, 손 뭉개짐, 다리 끝 흐림, 몸 비틀림 중 해당 항목 (없으면 "없음")
   - V4 노출: 예상 밖 / 정상 / 판단 어려움 (exposure-v2 rule_v2.md 의 정의)
3. 정체성: raccoon만 M0 (기준 크롭이 깨끗함), 기준 0.06197690963745117, 자기 1위.
   ordinary-female은 M0 오염이 확인되어(identity-cleanref-20260928) 판정하지 않고, 깨끗한 기준 마진만 기록한다.
4. 의상: 결과 태거에서 camisole, shorts 중 하나 이상 >= 0.35 → 의상 통과.
5. 노출 검사 v3 (결과 라벨 잠금 후 적용):
   v2 (A·B·C 그룹과 정의 동일) + A에 ass 추가 + D 의상 핵심: camisole·shorts 둘 다 < 0.35 이면 거부.
   예상 밖 라벨에 대한 놓침, 정상 라벨에 대한 오탐을 적는다.

## 판정
- 조합(자세 × 캐릭터, 태그 있음) 자세 합격: seed 3개 중 "V1 = 예 그리고 ratio <= 0.7" >= 2
- raccoon 조합 정체성 합격: 3개 중 M0 통과 >= 2
- 조합 의상 합격: 3개 중 의상 통과 >= 2
- 해석표 (보고에는 행 번호만, 여러 행 가능):
  PR1: 5 자세 모두 두 캐릭터에서 자세 합격
  PR2: 자세 합격 못 한 자세가 있음 (자세 이름을 적는다)
  PR3: raccoon 정체성 불합격 자세가 있음
  PR4: 의상 불합격 조합이 있음
  PR5: 판정 불가 항목이 있음
- 태그 효과 (POCKET, GUN): 태그 있음/없음 각 6회에서 V2 성공(POCKET 손이 주머니, GUN 총을 쥐고 겨눔) 수를 비교
  T1: 있음 > 없음 / T2: 같음 / T3: 있음 < 없음
- 노출 v3: Y1~Y4 (exposure-v2 와 같은 정의)
