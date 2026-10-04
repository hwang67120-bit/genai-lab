# Claude 전달 / Codex 인수인계 — 자세 편집의 캐릭터·의상 동일성 시험

작성: 2026-10-03. 이번 요청은 시험 준비 문서 작성이며, 작성 중 생성은 실행하지 않는다.
실행자에게 이 문서를 전달하면 아래 최대 6회 시험을 따른다. 기존 QWEN_POSE_TEST_PROMPT.md의 6회를 추가로 실행하지 않는다.

## 1. 질문과 범위

의상 시험에서는 승인 의복 태그와 노출 지시를 실제 프롬프트에 넣자 배 표현이 개선됐다. 이것은 의상 디자인 전체 보존의 증거는 아니다. 자세 편집에도 같은 원리인 '유지할 특징을 구체적으로 전달'을 적용해 확인한다.
질문: 일반적인 보존 지시보다, 완성 이미지의 구체적 보존 설명이 캐릭터·의상 동일성을 높이는가? 세부 참조를 더하면 추가 개선이 있는가? 자세 변경도 동시에 유지되는가?

흐름: 캐릭터·의상·그림체가 완성된 승인 이미지 → Qwen 자세 편집 → 보존·자세 확인 → 최종 후보 1장.
편집 후 SDXL 재생성, 의상 재교체, 얼굴 복원, 자동 보정 반복은 넣지 않는다.

- 운영 소스·설정·기존 결과·잠금 문서 수정 금지. 커밋·푸시 금지.
- 새 코드·결과는 outputs/qwen-pose-identity-20261003/ 아래만. 코드 작성은 Linux, 실행·문서·결과 저장은 연결된 Windows 환경에서 한다.
- 기존 승인된 Qwen 시험 환경·캐시만 사용. 패키지 설치·모델 다운로드·유료 API·외부 업로드 없음. 필요하면 생성 전 멈추고 필요한 내용을 보고한다.
- 프로젝트 사용 금지 자료와 파생본은 열거나 사용하지 않는다. approved-bases/, inputs/training_candidates/는 수정하지 않는다.
- 조건당 1회, 최대 6회. 재시도·추가 seed·결과를 보고 문장이나 참조를 고치는 행위 금지.

## 2. 먼저 읽고 확인할 근거

- docs/IMAGE_GENERATION_FLOW.md
- outputs/text-ab-20260926/README.md
- outputs/qwen-smoke-20261002/results.md의 T7~T12 및 마지막 정정 절
- 같은 폴더의 run_smoke.py, prompt_T12.txt와 실제 T12 run.json·입력 기록
- 기존 시작 이미지·자세 성공 기록을 추적할 outputs/pose-choice-20261001/, outputs/final-int-20261001/

T12는 일부 장식과 두 꼬리가 돌아왔지만 세부 모양과 손 동작이 완전히 맞지 않았다. 기존 개발 사례로 기록하며 새 시험의 성공 정답으로 쓰지 않는다. 크기 변경 효과의 기전도 확정하지 않는다.

## 3. 입력 확정 — 새 결과를 보기 전에

캐릭터 2명: raccoon, ordinary-female. 목표는 기존 POCKET의 정면 몸 골격 1개만 쓴다.

- raccoon 시작 후보: outputs/pose-choice-20261001/cases/N0_NONE_raccoon_209211001/raw-Base.png. T12와 같은 이미지인지 전체 SHA-256으로 확인한다.
- ordinary-female: 기존 승인된 의상 적용 완료 이미지 중 목표와 자세가 다른 1장. 기록상 정체성·의상 승인, 붕괴·초과 노출 없음인 후보를 사용한다. 여러 개면 seed 오름차순, 실행 ID 사전순으로 선택한다.
- 이미 승인된 이미지·특징은 재승인을 요구하지 않는다. 승인 근거가 없거나 서로 충돌하면 목록을 보여 주고 입력 확인을 받은 뒤 잠근다. 자동으로 통과 처리하지 않는다.
- 캐릭터마다 A/B/C의 시작 이미지는 같은 파일이다. 원래 캐릭터 입력이나 의상 보드는 참고 자료이고, 보존의 정답은 이 완성 이미지다.
- 목표 골격: outputs/pose-norm-20260928/control_POCKET_before.png. 기존 실제 성공 결과와 경로·해시를 확인한다. T2I용 RGB→BGR 반전 후 이미지를 그대로 쓰지 않는다. 정규화·골격 렌더링 규칙은 바꾸지 않는다.
- 목표 사진은 검토용이며 Qwen 입력에는 넣지 않는다.
- 손 목표는 기존 사용자 결정인 '양팔을 내리고 손을 골반 옆, 반바지 옆선 가까이에 둠'으로 고정한다. 주머니 넣기 성공이라고 부르지 않는다. 목표 앞뒤 방향은 정면이다.
- 이번 대상은 T12에서 사용한 자세와 seed를 포함한 개발 검증이다. 새로운 자세·seed에 대한 일반화 시험이라고 부르지 않는다.

## 4. 보존 명세와 세부 참조

preservation.json에 캐릭터별 항목을 적는다:
ID / 종류 / 승인 이미지에서 보이는 사실 / 출처 / 확인 상태 / 검토 위치.
종류: 얼굴·머리·눈·체형, 의상 구성·색·목선·끈·소매·밑단·무늬, 귀·꼬리·기존 장식, 그림체.

- 기존 승인 태그는 후보로 재사용한다. 이미지에 없는 특징·성별·체형·노출을 새로 추가하지 않는다. 실제 사용자 성별 저장소는 변경하지 않는다.
- 의상 종류만 쓰지 말고 구체적 디자인을 적는다. 예: 흰 상의뿐 아니라 캐미솔의 끈 색과 목선, 반바지의 밑단과 옆선.
- 보이는 꼬리 개수·색·줄무늬를 따로 기록한다. 종 이름으로 실제 모양을 덮어쓰지 않는다.
- 원본 팔 자세·표정 변화 요구·불필요한 소품·원하는 몸매를 보존 목록에 섞지 않는다. 체형 보존은 기준 이미지와의 비교다.
- 가려져 확인할 수 없는 부분은 추정해 채우지 않는다. 자세 때문에 생기는 자연스러운 주름·가림·보이는 면 변화는 허용한다.
- 그림체는 기준 이미지 참조로 지정하며 새 작가명·화풍 태그를 만들지 않는다.

C의 그림 3: 시작 이미지에서 자른 최대 3개 세부 부위를 한 이미지로 배치한다. 새로 그리거나 보정하지 않는다.
선택은 생성 전, 사라지기 쉬운 작은 장식·주요 무늬·의상 경계를 기준으로 한다. 두 캐릭터의 부위가 같을 필요는 없다.
raccoon은 T12 세부 참조를 우선 검토한다. 원래 팔·몸 자세가 얼마나 포함되는지 기록한다. 다른 크롭이 필요하면 좌표·이유를 생성 전에 확정한다. ordinary-female도 같은 원칙을 적용한다.
좌표·배치·원본 및 참조 SHA-256을 detail_regions.json에 저장한다. 결과를 보고 크롭을 바꾸지 않는다.

## 5. 조건 — 캐릭터 2 × 3조건 = 최대 6회

| 조건 | 그림 입력 | 긍정 프롬프트 |
|---|---|---|
| A 일반 지시 | 1 시작 이미지 + 2 목표 골격 | 공통 문장만 |
| B 구체적 지시 | A와 동일 | A + 승인된 보존 사실 문단 |
| C 세부 참조 | A + 3 세부 참조 | B + 그림 3의 역할 문장 |

A→B는 보존 설명 추가만 바뀐다. B→C는 세부 참조와 그 역할 문장이 함께 바뀌므로 '세부 참조 방식의 효과'로만 해석한다. 이미지 단독 효과라고 하지 않는다.
A는 아래 새로운 대조 조건이다. T7·T11·T12의 문장과 완전히 같지 않으므로 과거 결과로 대체하지 않는다. 추가 재현 실행은 없다.

공통 문장(A 전체, B/C의 동일한 접두부):
Picture 1 is the approved character and outfit to preserve. Picture 2 is an OpenPose skeleton used only for the target body pose. Redraw the character from Picture 1 in the front-facing body pose of Picture 2. Keep the same character identity, hairstyle, body proportions, outfit design, garment colors, patterns, existing accessories, and illustration style as Picture 1. Adapt folds and occlusion naturally to the new pose. Both arms hang down at the sides of the body, and each hand rests lightly beside the hip, next to the side seam of the shorts. Keep a plain white background and the entire character in frame. Do not add or remove garments, accessories, or body parts, and do not increase exposure beyond the outfit in Picture 1.

B 추가 문장: preservation.json의 승인 사실만 영어 자연문으로 옮겨 공통 문장 뒤에 항목 순서대로 붙인다. SDXL 태그 나열·75토큰 분할·마스크 규칙은 이식하지 않는다.

C 추가 문장:
Picture 3 contains detail crops from Picture 1, not a different character or a new pose reference. Use it to preserve the shapes, colors, patterns, and existing details specified above. Follow Picture 2 and the pose instruction for the body pose.

실행 전에 A/B/C 전문을 prompts.json에 확정하고, 부정·중복에 따른 충돌, 그림 번호, 문장과 보이는 의상의 일치를 확인한다. 자동 LLM 호출은 하지 않는다.

## 6. 고정 실행 설정과 사전 등록

T12의 실제 기록과 대조해 고정한다:
QwenImageEditPlusPipeline, 동일 모델 revision·GGUF Q4_K_M·텍스트 인코더 nf4·VAE/dtype,
동일 스케줄러·offload group_block1·zero_cond_t, 40 steps, true_cfg_scale=4.0,
guidance_scale=1.0, negative_prompt=' ', seed=209212001, num_images_per_prompt=1.
양자화·해상도·정밀도를 이번 시험 변수로 바꾸지 않는다.

출력 크기는 800×1312로 명시한다. 그림 3의 비율로 자동 결정되게 두지 않는다. 두 시작 이미지와 골격의 크기·VL/VAE 내부 변환 크기 및 실제 비율 변화를 기록한다. 다른 크기 입력 때문에 별도 전처리가 필요하면 생성 전 확정하고 캐릭터 내 A/B/C에 동일하게 적용한다.
기존 설정 재현에 필요한 파일·환경을 확인할 수 없으면 생성 전에 멈춘다. 알 수 없는 offload 이름을 다른 방식으로 자동 대체하지 않는다.

생성 전에 preflight.md, preservation.json, detail_regions.json, manifest.json, prompts.json, run-plan.json, criteria.md 및 시험 스크립트를 확정한다. sha256.txt에 해시를 기록하고 이후 수정하지 않는다. 자기 자신 해시는 넣지 않는다.
criteria.md에는 아래 판정·비교·종료 규칙을 그대로 포함한다.

## 7. 실행·기록·중단

순서: raccoon A/B/C → ordinary-female A/B/C. 첫 장도 6회에 포함한다.
이전 기록은 1장 약 30~43분이었다. 6회는 대략 3~4.5시간에 적재·측정 시간이 더해질 수 있으며 완료 시간 보장은 아니다.

- 각 호출 전에 실행 ID·입력/문장/설정 해시를 기록하고, 완료·오류·부분 결과를 즉시 저장한다.
- 실제 모델 입력 그림 전부(1~3 각각의 VL·VAE), 전달 프롬프트, raw.png·SHA, 호출 인자, 환경·모델 해시, 출력 크기, 시간·단계 수를 저장한다.
- 메모리는 allocated/reserved와 전용/공유 사용량을 구분한다. 기존 계측을 재사용하고 미계측은 미기록으로 쓴다. reserved만으로 공유 사용량을 확정하지 않는다. SDXL의 6.5 GiB 한도를 이 시험에 적용하지 않는다.
- OOM, 호출 오류, 잠금 해시·설정 불일치, 사용자 중단이면 즉시 멈춘다. 재시도하지 않는다. 완료했지만 품질이 나쁜 결과는 그대로 남기고 나머지 잠금 조건을 진행한다.
- 운영 게이트는 미실행으로 기록한다. 원시 결과를 덮어쓰지 않는다. 목표를 넘는 민감 부위 노출은 비교 시트에 복제하지 않고 경로와 사실만 기록한다.

## 8. 판정 — 자세 성공과 보존 성공을 분리

먼저 입력·실행 기록과 수치를 정리하고, 조건명을 가린 무작위 비교 시트로 육안 판정한 뒤 조건표와 대조한다. 에이전트는 입력 준비 이력이 있으므로 완전 블라인드가 아니며, 최종 판정은 사용자 검토 전까지 잠정이다.

각 조건에 기록:
- 자세: 몸 방향·팔·다리 / 손 목표 / 시점을 각각 예·부분·아니오·판단 어려움으로 기록. 원래 자세 유지나 무변화는 성공 아님.
- 정체성: 같은 캐릭터인지, 얼굴·머리·체형 변화 근거.
- 보존 명세 항목별: 유지 / 변형 / 소실·추가 / 가림으로 확인 불가. 그림체 변화 별도 기록.
- 의상: 종류 일치와 세부 디자인 일치를 분리. 줄무늬·끈·장식 변경을 사후에 허용 오차로 빼지 않는다.
- 구조·오염·초과 노출: 발생 여부와 위치. 가림을 소실로 단정하지 않는다.

동시 충족: 자세·손·시점·정체성·의상 디자인·그림체가 모두 '예', 검증 가능한 보존 항목의 변형·소실·추가 없음, 구조 결함·오염·초과 노출 없음. 확인 불가가 남은 조건은 완전 보존 확인으로 세지 않는다. 부분 개선은 별도로 적는다.

보조 골격 거리 계산 시 동일 캐릭터의 시작·목표·A/B/C 모두에서 신뢰도 >=0.30인 공통 몸 관절만 사용한다. 기존 목 원점·목~골반 중점 정규화를 유지하고 코드 위치, 관절 ID·점수·개수와 누락을 기록한다. 정규화 필수점 미검출·몸통 길이 0이면 미측정. 다른 관절 집합 거리끼리 개선이라고 비교하지 않는다. 이 값만으로 손 동작·원근·동일성을 판정하지 않는다.
기존 얼굴 유사도를 쓰면 시점에 민감한 참고값으로만 적는다. 새 합격 임계나 전체 픽셀 차이를 동일성 근거로 만들지 않는다.

비교표: 캐릭터별 A→B, B→C에서 좋아진 항목 / 나빠진 항목 / 그대로 / 확인 불가를 모두 적는다.
- B에서 보존이 개선되고 자세가 악화되지 않음: 구체적 보존 설명을 구현 후보로.
- C가 B보다 추가 개선되고 자세가 악화되지 않음: 세부 참조 전달을 추가 구현 후보로.
- 보존은 좋아졌으나 자세가 악화됨: 교환 관계, 동시 해결로 세지 않음.
- 개선 없음 또는 두 캐릭터 결과가 갈림: 항목별·캐릭터별로 보고. 원인 확정이나 모델 전체의 불가능 선언 없음.
후보는 제품 통과가 아니다. 판단 어려움은 통과로 세지 않는다. 캐릭터 2 × seed 1이며 일반화하지 않는다.

## 9. 산출물·다음 실행자 인수인계·종료

outputs/qwen-pose-identity-20261003/ 아래:
preflight.md, preservation.json, detail_regions.json, manifest.json, prompts.json,
run-plan.json, criteria.md, sha256.txt, 시험 스크립트,
cases/<character>/<A|B|C>/{run.json,raw.png,model_inputs/},
results.md, metrics.json, comparison.png, blind_sheet.png, blind_key.json, user_labels.md,
resume.md.

resume.md는 매 호출 후 갱신한다: 실행 호스트·환경·명령, 잠금 해시, 완료/미실행/실패 ID,
현재 프로세스·세션, 오류, 다음 허용 ID, 남은 호출 수. 다른 실행자가 이어받을 때 기존 프로세스가 살아 있는지 먼저 확인해 중복 실행하지 않는다. 실패·중단된 셀은 자동 재시도하지 않는다.
내일 Codex가 이어받더라도 잠금 내용을 바꾸지 않는다. 기존 Codex GPU 금지 역할은 자동 해제되지 않는다. 별도 실행 지시 전까지 결과 검토·준비만 한다. 자동 예약 실행은 하지 않는다.

최종 보고:
1. 시작 이미지·목표 골격·세부 참조 경로와 SHA, 승인 근거, 기존 T12와의 차이.
2. 캐릭터별 보존 명세와 A/B/C 실제 프롬프트 전문, 사전 등록 해시.
3. 실행 수/최대 6, 오류·미실행, 원시 결과 해시·시간·메모리.
4. 자세/손/시점/정체성/의상 종류/의상 디자인/그림체/오염 표와 항목별 보존 변화.
5. A→B·B→C 비교, 동시 충족 건수, 남은 결함, 사용자 검토 시트 링크.
6. 구현 후보가 있다면 '보존 명세·프롬프트 조립·세부 참조·검토표' 중 필요한 것만 제안. 운영 구현은 하지 않는다.
7. git status, 다음 실행자가 이어받을 위치.
여기서 종료한다. 추가 시험을 자동으로 시작하지 않는다.

## 참고

- https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py
공식 자료는 호출·입력 방식의 근거이며 이 프로젝트의 보존 성공 근거가 아니다.
