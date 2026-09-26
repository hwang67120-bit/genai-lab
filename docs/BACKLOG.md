# BACKLOG

단계 구분은 우선순위가 아니라 선행 조건이다.
배포 방식은 C(레시피 배포)로 확정됐다 — 병합 가중치를 배포하지 않고 앱이 원본을 내려받아 사용자 PC에서 병합한다.

떠오른 항목은 해당 단계에 한 줄 추가하고 하던 작업을 계속한다.
본선이 아닌 것을 실행하기 전에 여기 적는다.
항목이 닫히면 체크하고, 결정이면 docs/DECISIONS.md로 옮긴다.
"근거: 미확인"은 아직 확인하지 않았다는 뜻이며, 없다는 뜻이 아니다.
최초 작성: 2026-09-24

작성 범위는 목록 고정과 읽기 전용 확인이다. '다음'은 이번에 수행한 작업이 아니다. 커밋·백업·매핑·모델 다운로드·생성·병합·삭제는 실행하지 않았다. 원문 수치와 상태는 지우지 않고 정정을 병기했다. 확인 표시 없는 실험 수치·미구현 주장은 제공된 원문에서 전재했으며 재측정하지 않았다.

## 현재 본선

- [ ] ①-3 모델 단독 생성 비교 → 병합 가치 판별
근거: docs/MERGE_CANDIDATES.md 기준, C 시나리오로 Q1 재확인 후 진행. 단독 생성 미착수.
다음: 항목12에서 C 시나리오 권리를 확인한 후 항목11을 진행한다.

## A. 지금 (본선을 막지 않음 · 비용 없음)

- [ ] 1. **미커밋 변경 커밋**
근거: 소스 8개 수정(`generation_orchestrator`, `generation_replay`, `generator`, `model`, `part_error_correction`, `selected_garment_correction`, `visual_reference`, `run.py`) + 신규 `genai_lab/provenance.py`, `tests/test_provenance.py`가 미커밋. 단일 드라이브, 백업 없음. 확인: 소스 8개 수정 외 신규 provenance.py·test_provenance.py·MERGE_CANDIDATES.md 등이 미추적. 별도 백업 부재·단일 물리 드라이브 여부는 미확인.
다음: 커밋.

- [ ] 2. **outputs 백업**
근거: 2.4 GB, git 추적 1개 파일, `\\192.168.0.109\win_g`(원격 PC의 D 드라이브 공유) 단일 위치. 정정: 원문 2.4 GB·추적 1개 → 현재 outputs 12,009파일, 2,478,270,461바이트(2.478 GB / 2.308 GiB), Git 추적 24파일. 전체 stat 및 git ls-files outputs 기준. 원격 공유의 실제 드라이브/외부 백업은 미확인.
다음: 별도 매체로 1차 복사.

- [ ] 3. **드라이브 매핑 영구화**
근거: Z: 매핑이 재부팅 시 소실. 근거: 미확인 — 재부팅 후 매핑 소실은 이번에 재현하지 않음. 명령은 기록만 하고 실행하지 않음.
다음: `net use Z: \\192.168.0.109\win_g /persistent:yes`

- [ ] 4. **금지 이미지 잔존 정리**
근거: `143960292_19.jpg`는 사용자가 사용 금지 지정. 원본은 제거됐으나 파생 산출물에 참조 잔존. 확인: 사전 확인 4의 텍스트 잔존 15파일. inputs/outputs 내 파일명에 지정 문자열 포함 0개. 익명 파생 이미지의 시각적 잔존 여부까지 확인한 것은 아님. 제외 사유는 user_excluded_unsuitable로만 기록.
다음: 사전 확인 4의 목록을 사용자에게 보고하고 방침을 받는다. **임의 삭제 금지** — 과거 측정의 재현성이 걸린다.

- [x] 5. **provenance 바이트 동일성 검증**
근거: 계측은 구현됐으나 완료된 실제 실행이 없다. 기존 3개 `run_provenance.json`은 pytest 픽스처(`completed: false`, `loaded: {}`, style 참조가 pytest 임시경로). 기준 해시는 `outputs/run-provenance-validation-20260924/baseline.json`에 있음 — Base `cd1a11ff…`, Final `c10bf6a3…`, commit `2cf0f82`. 정정·완료: 원문 미완료는 과거 상태. outputs/run-provenance-validation-20260924/verification.json은 completed, generation_attempts=1, both_byte_identical=true. Base cd1a11ffa5664d37106758e544ca783d0950f868dcebe0ae39bd04f2f5d9d3e0 / Final c10bf6a3aedcfae17e289063d8bc1b83145672e2e93dd138b8fa18d53d275e81 모두 기준과 일치. pytest-full-final.txt는 1232 passed / 3 xfailed.
다음: 완료 증거를 항목9의 GUI 경로 비교 기준으로 사용한다.

- [ ] 6. **마스크 밖 변화율 × strength 재집계**
근거: 의도한 편집이 119px(0.0116%)인 muscular-male에서 이미지의 55.63%가 바뀌었고 그중 55.62%가 마스크 밖. muscular-female은 밖이 98.22%. ordinary-female 65.54%, raccoon 10.58%. strength 0.95 실험 데이터 존재.
다음: 0.90과 0.95의 마스크 밖 변화율만 비교. **새 생성 불필요.**

- [ ] 7. **결함 발생률 측정**
근거: 목표 사양의 "육손·꺾인 허리·3개 이상 다리·뭉개짐"에 측정 수단 없음. 검출기도 없음.
다음: 기존 `outputs/` 이미지 육안 집계. 열 = 캐릭터·의상·손가락·허리·다리·뭉개짐.

- [ ] 8. **지표 검증 (CLIP vs 사람)**
근거: 모든 판단이 CLIP 기반인데 검증된 적 없음. 다른 캐릭터가 0.8418(절대값 바닥 존재). 1단계 순위와 파트 Borda가 15/15 불일치. 의상 CLIP 1위 4/10인데 육안 "예"는 0건이고, 4파트 여성은 1위·마진 +0.0380인데 육안 실패. 페라리는 4/19·6/19위인데 육안 전달 확인.
다음: 기준 1장 + 출력 3장 3지선다, 정체성 6문항 + 의상 카테고리 4문항, 10명. 여유 0.0315였던 ordinary-female × 흑백7파트 쌍을 포함. 의상 문항은 충실도가 아니라 **카테고리 일치**로 묻는다.

- [ ] 9. **GUI 최초 사용 세션 겸 GUI 경로 provenance**
근거: 지금까지 모든 측정이 `scripts/` 경로. 자세 계통은 `generator.py:265`가 `approved_pose_estimation is not None`으로 막고 있고 그 인자는 `gui_main.py:3311`에서만 전달됨. GUI 경로가 다른 코드 경로일 수 있음.
다음: 코드를 열지 않은 상태로 GUI만 사용해 생성. 막힌 지점·소요 시간 기록, 세션 중 수정 금지. 동시에 `run_provenance.json`을 남겨 5의 scripts 실행과 read 키 차집합·출력 해시를 비교.

- [x] 10. **의상 카테고리 재판정 완료**
근거: 체계 검증에서 중단(축 4가 비키니/바디슈트를 보장 구분 못 함). 완화 기준("4축 중 하나라도 갈리면 통과") 정정 지시 전달됨. 정정·완료: outputs/garment-category-rejudge-20260924/README.md:151 이하에 정정 후 재개·3패스 각 10행·집계 완료 기록 존재. 무정보 4/2/2/2, 판정불가 2/6/4/6, 유효 N 4/2/4/2. 표4는 전축일치2·일부3·불일치1·보류4. 일반 카테고리 성능 판단 불가. 원래 중단 기록 보존됨.
다음: 완료 보고서의 무정보·판정불가 한계를 항목8의 평가 설계에 반영한다.

## B. 모델 (본선)

- [ ] 11. **①-3 단독 생성 비교**
근거: 단축 후보 Animagine XL 4.0
다음: Base 생성만, 중립 프롬프트 통일, 기준 3.1 포함, contact-sheet 무라벨.

- [ ] 12. **Illustrious·NoobAI의 Q1 재확인**
근거: C에서는 Q2 미적용. 둘의 Q1이 `미확인`. Illustrious는 "공개하지 않는 개인 병합의 예외 불명확"이 기록됨 — C가 바로 그 경우. 이번 작성에서는 라이선스를 재판정하지 않았다. MERGE_CANDIDATES.md의 기존 Q1 미확인 상태 유지.
다음: 약관 재확인. 불명이면 `미확인` 유지.

- [ ] 13. **병합 실행 + 비교**
근거: 11에서 구분되는 후보가 나온 뒤에만 의미.
다음: 항목11 결과와 사용자 선택 후 병합 비교를 설계한다.
- 선행: 11

- [ ] 14. **마스크 밖 재생성 (기전 가설)**
근거: UNet 4채널이라 Diffusers가 진짜 인페인팅 대신 잠재 혼합으로 대체(`latents = mask × latents + (1−mask) × noised_original`). strength 0.90이면 `noised_original`이 거의 순수 노이즈라 마스크 밖도 사실상 재생성. **구성은 확인, 인과는 미확인.** 원문은 기전 가설로 보존. strength를 원본 잠재 잔존 비율로 환산하거나 마스크 밖 재생성을 확정하는 근거는 이번에 확인하지 않음. 구성과 인과를 분리.
다음: 항목 6의 결과로 판정.
- 선행: 6

- [ ] 15. **D-065 결정**
근거: 실행 경로가 D-064 지정과 3개 불일치(모델 `animagine` vs `inpainting-0.1`, 어댑터 기본 vs Plus, Human-Agnostic 미적용). 이미지 투영 출력 `[1,4,2048]` = 4토큰. `garment_inpaint` 설정 26키 전부 read=0. 정정: docs/DECISIONS.md 최고 번호 D-072(:5), D-065(:65)는 이미 “기준 캐릭터 교체 의상 선택과 중립색 잔여 진단”이다. 원문 D-065는 대화상 경로 선택 호칭으로 저장된 결정과 다름. 감사 근거 outputs/refinement-path-audit-20260924/. 완료 provenance 투영은 [2,4,2048], 이전 정적검사 [1,4,2048]과 토큰4는 같고 배치만 다름. garment_inpaint 원문26키 → 현재 선언25키/read0/unread25.
다음: A(복귀)/B(유지·한계 문서화)/C(어댑터만) 택1. 사용자 결정.
- 선행: 8

- [ ] 16. **의상 경로 비교 실험**
근거: 4채널·4토큰이 의상 실패 원인이라는 주장은 구성만 확인, 인과 미확인. 판정 기준이 없다 — 의상 CLIP 순위는 육안과 불일치, 색 지표는 상한 0.5241 < 임계 0.55, 육안은 독립 평가 아님.
다음: 판정 기준 확정 후 A/B 1쌍.
- 선행: 8, 15

- [ ] 17. **원본 의상 잔존**
근거: 10/10에서 관찰. 출력 의상 구조가 목표가 아닌 원본을 따라감.
다음: 14·16의 결과로 판단.

- [ ] 18. **자세 계통 미연결**
근거: `pose_estimation.py` 365줄 + `pose_fallback.py` 423줄 + `character_target_landmarks.py` 428줄 + `pose_reference.py` 145줄 + `original_body_pose.py` 98줄 = 1,459줄. provenance 결과 `pose_control.*` 6키는 read, `pose_reference_estimation.*` 7키와 `pose_fallback.*` 8키는 unread. 스크립트 경로에는 제어 이미지(`control_map_image`)를 공급할 경로가 없음. 정정: 5파일 합 1,459줄은 확인. 완료 provenance에서는 pose_control 선언6 중 enabled 1키 read·5키 unread(원문 read6). pose_reference_estimation unread7 동일, pose_fallback unread9(원문8). scripts 실행 관측이며 GUI 전체 미연결 확정 아님.
다음: 9의 GUI 실행에서 unread 여부 확인.

- [ ] 19. **자세 결과 정책이 관측 전용**
근거: `pose_result_policy.mode: observe_only`, `block_on_pose_mismatch: false`. 자세 불일치를 기록만 하고 통과시킴.
다음: 46과 함께 판단.

- [ ] 20. **카테고리 자산 부재**
근거: 목표에 "여학생 교복 ≠ 남학생 교복", "비키니 ≠ 바디슈트"가 있으나 수영복·교복 참조 자산이 없어 시험 불가.
다음: 카테고리별 참조 의상 확보.

- [ ] 21. **의상 어휘 누락**
근거: `garment_edit_plan.py:141` `dress_terms = {dress, gown, bodysuit, leotard, swimsuit}` — `bikini`·`jumpsuit` 없음.
다음: 20 확보 후 실제 오적용이 나는지 확인. **선제 수정 금지** — 고치면 원래 문제가 있었는지 알 수 없게 된다.
- 선행: 20

- [ ] 22. **체형 측정 수단 부재**
근거: 목표 "인체 정확도"에 측정 수단 없음. WD14 breasts·muscular는 "무엇이 보이는가"이지 체형이 아님.
다음: 실루엣 폭 프로파일 지표 + 보정 스위트(가로 1.1배 → 1.10 등 정답을 계산으로 아는 케이스). **값이 나빴을 때의 행동을 함께 정의할 것** — 안 그러면 재기만 하는 것이 하나 더 는다.

- [ ] 23. **자세 측정 수단 부재**
근거: 목표 "자세 80%". 현 지표(CLIP·색·WD14·마스크 픽셀) 중 자세를 보는 것 없음.
다음: 방법 후보 조사만.

- [ ] 24. **muscular-male 마스크 퇴화**
근거: 선택 편집 119px(0.0116%). 불변식 도입 후 과거 19회 중 2회 차단, 둘 다 muscular-male.
다음: 원인 조사.

- [ ] 25. **strength 0.95 참조 분석 실패 1건**
근거: 1건이 참조 분석 단계에서 실패. strength는 0.90 복귀 완료(바이트 동일 확인).
다음: 실패 로그 확인.

- [ ] 26. **색 지표 상한 미달**
근거: 유채색 의상 색 유사도 상한 0.5241, 임계 0.55. 구조적 통과 불가.
다음: 임계 재설정 또는 지표 교체.

- [ ] 27. **꼬리 보호 실효**
근거: `hair_error_correction.py:160-170` `part_queries` 5개에 꼬리 없음. `output_coordinate_control.py:119`에 `output_tail` 조건부 보호 존재. 보존 여부 미검증. 위치 확인: hair_error_correction.py:160-173의 쿼리5종에 꼬리 없음. output_coordinate_control.py:122-123에 garment 대상 output_tail 조건부 처리 존재(원문 :119).
다음: 라쿤 승인 태그에 `tail`이 있는지부터 확인.

- [ ] 28. **`missing_ratio` 불변식**
근거: 불변식 2개(카테고리 미해결, 보호 > 요청) 도입 완료. 이 건은 미도입.
다음: 필요 여부 재검토.

- [ ] 51. **CatVTON 재평가**
근거: 애니 캐릭터 품질은 실패로 확인된 것이 아니라 미측정이다. D-023(docs/DECISIONS.md:344)은 "실제 후보 한 장을 사용자가 시험하기 전에는 적합 판정을 내리지 않습니다"라고 했으나, D-050(:598)의 약 124초 실행은 승인 영역 안 변경을 측정하지 않았고 원시 출력이 삭제돼 모델 무변경과 후처리 제거를 구분할 수 없었다. D-050 개선 후 검증(:604)은 GPU CatVTON 추론 0회이며, 이후 TPS·2D Inpaint 전환 결정들의 검증에도 CatVTON 품질 재측정 근거는 없다(D-051 :612, D-057 :669, D-058 :678). 이는 과거 전체 CatVTON 실행이 0회라는 뜻이 아니다.
D-051(:608–609)은 조건 잠재 벡터만으로 정확한 의상 좌표 대응을 기대했던 한계를 적고 TPS로 전환했다. D-057(:665–667)·D-058(:675)에서 2D Inpaint 경로로 전환됐으며 기존 CatVTON 코드는 "비교·복구 자산으로 보존"하기로 했다. 현재 configs/animagine.yaml:126에는 "CatVTON 의상 합성 실행 설정은 제거됨"이 명시돼 있다.
라이선스 판단(D-023, :345, CC BY-NC-SA 4.0)은 비상업·무료 배포와 배포 방식 C(레시피 배포) 결정 이전 기준이다. 이번 항목에서는 기존 판단을 인용하며 현재 Q1·Q2 허용 여부를 새로 확정하지 않는다. 착용 전용 모델은 옷·피부 역할 배정을 학습하므로 범용 인페인팅에 없는 지시 층을 모델이 내장한다는 가설이 있다(미검증).
사전 확인(2026-09-26): CatVTON 문자열 잔존 경로는 genai_lab/body_comparison.py, genai_lab/catvton_preflight.py, genai_lab/clothing.py, genai_lab/clothing_reference.py, genai_lab/generator.py, genai_lab/guardrails.py, genai_lab/try_on_metrics.py, scripts/body_comparison_runner.py, scripts/catvton_preflight_runner.py, scripts/pose_reference_runner.py, scripts/run_today_reference_edit.py; outputs/retired-code-20260908-kk98xxqg.tar.gz 목록에는 이름에 catvton이 포함된 파일 0개(다른 이름의 파일 내용에 포함됐는지는 미확인, 압축 해제·내용 열람 없음); D:/genai-cache/catvton-venv 및 Scripts/python.exe 존재, 현행 pose_reference_estimation.python_executable(configs/animagine.yaml:88–89)도 이 환경을 참조한다. 경로·환경 잔존은 CatVTON 추론 연결 또는 실행 가능성의 증명이 아니다.
다음: 라이선스를 Q1(로컬 사용)·Q2(배포)로 나눠 재검토한다. 통과하면 기존 A/B 케이스 1건으로 원시 출력을 보존해 실행한다.
- 선행: IP-Adapter 의상 조건 마스크 A/B 결과. B 영역 잔존이 줄지 않으면 이 항목의 우선순위를 올린다. 완료 기록 outputs/ip-mask-ab-20260926/README.md에서 육안 잔존 감소는 0/2건이므로 우선순위 상향 검토 근거가 생겼다(P1 방향 일치 1/2, 별도 수치 반증 조건에는 해당 없음).

## C. 배포 — 방식 C 확정 (레시피 배포)

- [ ] 29. **병합 결정성**
근거: C는 사용자 PC에서 병합. torch 버전·연산 정밀도 차이로 사용자별 가중치가 달라질 수 있음. 이 프로젝트의 모든 측정이 바이트 결정성 위에 서 있음.
다음: 기준 SHA-256 공개 + 앱이 병합 후 대조하는 설계.
- 선행: 13

- [ ] 30. **디스크 정점**
근거: 단축 후보 3개 20.815 GB + 병합 결과 약 6.5 GB ≈ 27 GB 정점. 밝히려던 최소 사양 25 GB를 넘음. 병합 후 원본 삭제 시 약 7 GB. D: 가용 706.875 GB(개발 환경 기준). 원문 약27 GB는 계획 추정치. 후보3개 합은 20,814,692,952바이트(20.815 GB /19.385 GiB). 최종 레시피에 기준3.1도 포함하면 그 파일도 정점에 더해야 한다. GB/GiB와 실제 병합 출력 크기 미확정이므로 실제 정점으로 확정하지 않는다. 706.875 GB는 이전 조사값으로 보존; 이번에 재측정하지 않음.
다음: 정점 기준으로 사양표 수정 또는 순차 병합으로 정점 낮추기.

- [ ] 31. **사용자 PC 병합 RAM**
근거: 체크포인트를 올려 섞는 작업. 16 GB 목표에 영향. 미측정.
다음: 병합 시 RSS 측정.
- 선행: 13

- [ ] 32. **라이선스 전달 UI**
근거: C에서도 원본 라이선스의 사용 제한 조항을 사용자에게 전달할 의무는 유지됨(R·F 양쪽).
다음: 최초 실행 화면 설계에 포함.

- [ ] 33. **RAM 회수**
근거: 최고점 15.971 GiB, 생성 후 분석·게이트 단계에서 발생. 2.839 GiB 미귀속. `del`+`gc.collect()`의 RSS 반환 효과 **미측정**. "7 GiB 회수"는 근거 없는 표현.
다음: RSS 실제 반환량과 재적재 비용만 측정.

- [ ] 34. **해상도 상한 확정**
근거: 단조 아님. L2(1.511 MP)가 L3(1.917 MP)보다 정체성(0.1208 vs 0.0978)·의상(+0.2109 vs +0.1309) 우위, 더 빠르고 시원함. 공유 메모리 스필은 L3부터(1,228/1,548 MiB vs 기준 78 MiB). L4는 9.342 GiB에서 중단.
다음: 1.36~1.51 MP 구간에서 확정.

- [ ] 35. **시스템 RAM 하한**
근거: 개발 환경 32GB, 목표 최소 16GB, 검증 불가.
다음: 16GB 환경 확보 또는 제한 명시.

- [ ] 36. **VRAM 8GB 목표**
근거: raccoon 8.012 GiB로 초과(P8 FALSE). 자세 ControlNet을 켜면 더 늘어남.
다음: 34 확정과 함께 재측정.

- [ ] 37. **GroundingDINO / SAM2 fp32 → fp16**
근거: 미확인. 근거: 미확인 — 이번에 보조 모델 로드·dtype 계측을 하지 않음.
다음: 현재 dtype 확인.

- [ ] 38. **WD14 상주 여부**
근거: 미확인. 근거: 미확인 — 이번에 WD14 적재·해제를 계측하지 않음.
다음: 적재·해제 시점 확인.

- [ ] 39. **미사용 ControlNet 제거**
근거: 배포 다운로드 목록에 미사용 ControlNet 포함 예정.
다음: 목록 확정 시 제외.

- [ ] 40. **인스톨러**
근거: 없음. 설치형, 비상업·무료, 로컬 실행(참조 이미지가 PC를 벗어나지 않음).
다음: 착수 전.

- [ ] 41. **최초 실행 UX**
근거: C에서 원본 모델 20 GB 내려받기 + 병합이 첫 실행에 들어감. 진행 표시 설계 없음.
다음: 40과 함께.

- [ ] 42. **대기시간 수용도 / 설치 의향 조사**
근거: 1회 생성 7~8분. 앱·인스톨러 부재 상태에서 물으면 가정에 대한 가정.
다음: 40 이후로 보류.

- [ ] 43. **관찰 세션 (3인)**
근거: 앱 필요.
다음: 40 이후.

## D. 정리 (아무 때나)

- [ ] 44. **미사용 코드 제거**
근거: 사전 확인 5 결과를 적는다. provenance의 `unread` 목록이 자료가 되지만, 현재 값(542 선언 정정: 원문542/106/436은 중단 기록. 완료 provenance는 declared542/read425/unread190, tracking_available=true. read에는 런타임 추가 키가 있어 단순 차감과 다르다. 사전 확인5 후보 위치 참조. unread는 이번 실행의 미접근 목록이며 사산 코드 확정 목록 아님.
다음: 완료 실행 unread와 사전 확인5 후보의 실제 호출을 대조해 제거 후보만 확정한다.
- 선행: 5

- [ ] 45. **게이트 집행 / 강등 / 제거**
근거: 4건 중 3건이 `semantic RETRY`인데 같은 보고서에 `No case was retried`. muscular-female의 `structure REVIEW`도 무반응. `pose_result_policy`도 `observe_only`. 추가 확인: 완료 실행 Base semantic PASS/none/retry0, structure REVIEW/none/retry0. 과거 semantic RETRY3건과 혼합하지 않는다. outputs/run-provenance-validation-20260924/README.md:535-560.
다음: 게이트별로 집행·강등·제거 중 택1. **현재처럼 "RETRY라고 적고 통과"가 제일 나쁘다** — 표를 읽는 사람이 무언가 작동한다고 믿게 된다.

- [ ] 46. **읽힘 ≠ 사용됨 보완**
근거: provenance의 `semantics`가 명시 — "Observed dict value accesses, not causal usage". `pose_control` 6키가 읽혀도 ControlNet이 붙었다는 뜻이 아님. `loaded` 항목이 메우지만 현재 전부 비어 있음. 정정: 완료 기록 loaded는 비어 있지 않다. StableDiffusionXLInpaintPipeline, Animagine3.1, UNet4, ip-adapter_sdxl.bin, hidden1664, ImageProjection [2,4,2048], scale0.8, EulerAncestralDiscreteScheduler, LoRA[]/ControlNet[]. 읽힘과 실효가 다르다는 한계는 유지.
다음: 항목9의 GUI 실행에서 read와 loaded를 함께 대조한다.
- 선행: 5

- [ ] 47. **봉인 계약 분리**
근거: Base 단계와 정제 단계가 같은 계약 공유.
다음: 분리 필요성 검토.

- [ ] 48. **문서-설정 불일치**
근거: 사전 확인 2 결과를 적는다. 확인: docs/REFINEMENT_EXECUTION_CONTRACT.md:53은 기본 flux_whole_image, configs/animagine.yaml:619-621은 sdxl_local. 불일치. 이번에 수정하지 않음.
다음: 불일치 시 문서를 실제 값에 맞춘다.

- [ ] 49. **`reference_regions.py:15-16` 무효 선언**
근거: `"base_output"`의 `use_target_garment_tags: True`가 마스크에 영향 없음. 태그가 다른 실행 간 `garment.png` 해시 동일로 확인. 확인: reference_regions.py:16 선언과 :125/:175 소비 코드 존재. 특정 태그 변경에서 마스크 해시가 같았다는 관측을 선언 자체 미접근으로 확대하지 않는다.
다음: 선언 제거 또는 실제 연결.

- [ ] 50. **`garment_edit_plan.py:222` `target |= source`**
근거: 이 한 줄로 밴드 기하(`:179-201`, `row(0.16)`~`row(0.97)`)가 사실상 무효화됨. 확인: garment_edit_plan.py:222에 target |= source 존재. source 밖으로 확장되는 밴드까지 전부 무효라고 합집합 한 줄만으로 확정하지 않는다.
다음: 의도된 동작인지 확인.

## E. 미해결 질문 (담당 없음)

- [ ] E-1. Base 단계가 의상 참조를 사용하지 않음(233초 재생성, 바이트 동일 확인). 의도인지 결함인지 미정.
근거: 제공된 과제의 미해결 질문 전재. 새 측정하지 않음.
다음: Base의 의상 비의존성이 의도된 계약인지 확인한다.

- [ ] E-2. 정제가 정체성 여유를 15~78% 감소시킴(R4). 허용 범위 미정.
근거: 제공된 과제의 미해결 질문 전재. 새 측정하지 않음.
다음: 정체성 마진 감소 허용 범위를 사용자와 정한다.

- [ ] E-3. 전체 파이프라인은 바이트 결정적(Base·Final, 15시간 간격 및 strength 복귀 전후로 확인).
근거: 제공된 과제의 미해결 질문 전재. 새 측정하지 않음.
다음: 다른 실행 환경의 결정성 확인 범위를 정한다.

- [ ] E-4. 캐릭터 검색 1위 9/10, 정체성 마진 ≥0.05 7/10, 의상 검색 1위 4/10, 육안 충실도 예 0·부분 7·아니오 3, 원본 의상 잔존 10/10.
근거: 제공된 과제의 미해결 질문 전재. 새 측정하지 않음.
다음: 항목8에서 지표와 육안 판정의 불일치를 확인한다.

- [ ] E-5. CLIP 절대 코사인에 약 0.84의 바닥이 존재(다른 캐릭터가 0.8418).
근거: 제공된 과제의 미해결 질문 전재. 새 측정하지 않음.
다음: 항목8의 인간 평가와 절대 코사인 해석을 대조한다.

## 사전 확인 1~5

1. BACKLOG.md는 작성 직전 없었다. 신규 생성하며 덮어쓰지 않았다.
2. REFINEMENT_EXECUTION_CONTRACT.md 존재. :53의 기본 flux_whole_image와 configs/animagine.yaml:621의 sdxl_local 불일치.
3. 결정 번호 최댓값 D-072(:5). 파일 마지막 결정 절 D-064(:708). D-065(:65)는 다른 주제로 이미 존재하며 대화상 D-065와 구분한다.
4. inputs/outputs에서 rg -l --hidden --no-ignore -F로 지정 문자열을 검색했다. 내용에 남은 파일 15개, 파일명에 포함된 파일 0개. 익명 파생 이미지의 시각적 잔존까지 배제하지 않는다. 삭제·변경·이미지 열람·업로드하지 않았다. 제외 사유: user_excluded_unsuitable.

- outputs/garment-range-survey-20260923/complete_report.py
- outputs/garment-range-survey-20260923/measure_identity.py
- outputs/garment-strength-095-20260923/measure_identity_tail.py
- outputs/identity-rerank-20260923/README.md
- outputs/identity-rerank-20260923/audit_excluded_history.py
- outputs/identity-rerank-20260923/gallery-active.json
- outputs/identity-rerank-20260923/gallery.json
- outputs/identity-rerank-20260923/masks.json
- outputs/identity-rerank-20260923/matrices.json
- outputs/identity-rerank-rescore-20260923/README.md
- outputs/identity-rerank-rescore-20260923/labels.json
- outputs/identity-rerank-rescore-20260923/rescore.py
- outputs/identity-rerank-rescore-20260923/write_reports.py
- outputs/pre-finetune-gpu-20260920/input_analysis.json
- outputs/resolution-limit-sweep-20260924/measure_quality.py

5. 미사용·무효 가능성 후보(삭제 판단 아님):

| 후보 | 위치 | 확인 범위 |
|---|---|---|
| garment_inpaint 러너 | scripts/garment_inpaint_runner.py; configs/animagine.yaml:130-145 | genai_lab/·scripts/ .py 검색에서 러너 이름 호출 참조 0; 완료 실행25키 unread. 외부 실행 경로 전체가 없다는 증거는 아님. |
| use_target_garment_tags | genai_lab/reference_regions.py:16, :125, :175 | 선언과 소비 코드 존재. 과거 결과 해시 동일만으로 미호출을 증명하지 않음. |
| guidance_trace | genai_lab/final_candidate_review.py:113-114, :206 | genai_lab/·scripts/ .py 문자열 검색에서 소비처만 발견. 다른 이름·외부 생산 경로 미확인. |
| ip_adapter_reference_image / ip_adapter_image | genai_lab/generator.py:183, :365-368 | visual_inputs 경로에서 pop 후 visual_condition으로 대체. visual_reference.py:788-816이 이미지 임베딩을 준비하므로 IP-Adapter 전체 미작동으로 단정하지 않음. |

## 정정 및 미확인 요약

- 항목5·10은 원래 미완료 서술을 보존하고 완료 증거를 병기해 체크했다.
- outputs 크기/추적파일, D-065 번호 충돌, 설정 키 개수, pose 키 접근, 실제 loaded 내용은 해당 항목에서 원문과 나란히 정정했다.
- 항목14·49·50의 원문 기전 해석은 확인된 코드와 구분했다. 이번에 인과를 확정하지 않았다.
- 직접적인 '근거: 미확인' 항목은 3(재부팅 매핑), 37(보조 모델 dtype), 38(WD14 상주). 추가 미확인은 1·2의 외부 백업/물리매체, 12의 C 시나리오 Q1, 30·31의 실제 병합 정점/RAM, 35의 16GB 환경, 44의 전체 경로 미사용 여부다.
- 나머지 전재 수치를 이번에 재측정한 것처럼 해석하지 않는다. 목록 밖 실행 항목은 추가하지 않았다.

## 작성 검증

- 현재 본선1 + A10 + B18 + C15 + D7 + E5 = 56항목. 번호1~50 보존.
- 각 항목에 제목·근거·다음3줄. 원문 선행 조건은 별도 줄로 보존.
- 추가 파일은 docs/BACKLOG.md 하나. 기존 provenance 변경·라이선스 문서·산출물·소스·설정·계약·테스트를 수정하지 않았다.
- 기존 추적 파일과 보존 대상 미추적 파일 총 349개의 작업 전후 SHA-256 동일. 이번 Git 상태 추가분은 BACKLOG.md 하나.

### Git 상태 전문

```text
 M genai_lab/generation_orchestrator.py
 M genai_lab/generation_replay.py
 M genai_lab/generator.py
 M genai_lab/model.py
 M genai_lab/part_error_correction.py
 M genai_lab/selected_garment_correction.py
 M genai_lab/visual_reference.py
 M run.py
?? afe_load(p.read_text())
?? approved-bases/
?? docs/BACKLOG.md
?? docs/MERGE_CANDIDATES.md
?? genai_lab/provenance.py
?? inputs/training_candidates/
?? tests/test_provenance.py
```
