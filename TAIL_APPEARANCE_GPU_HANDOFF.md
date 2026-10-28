# 꼬리 외형 확인 기능 — Claude 검증 인수인계

작성: 2026-10-04 Codex. 사용자 승인 설계: [TAIL_APPEARANCE_IMPLEMENTATION_DESIGN.md](TAIL_APPEARANCE_IMPLEMENTATION_DESIGN.md).
GPU 실행 0회. 새 모델·패키지 설치 없음. 자세·Qwen·비키니·C6·부정 프롬프트·개수 문구 변경 없음.
이번 변경은 꼬리 외형의 전달과 사용자 검토 기능이며 개수·색 보존을 보장하지 않는다.

## 1. 실제 연결

- 입력 확인창에서 fixed 그룹의 꼬리·귀 태그와 원문을 표시한다.
- 꼬리 문구는 영어 쉼표 구분, 기본 빈칸·미확인. 수정하면 확인 해제.
- 검토 후 보강: 기존 excluded_garment_tag를 재사용하고, 꼬리·귀 입력에만 노출 어휘 검사(APPENDAGE_EXPOSURE_TAGS)를 추가한다. 대소문자·밑줄·공백을 정규화한다. 기존 의상 승인이나 출력 노출 게이트의 정책을 바꾸지 않는다. 등록 태그 검사이며 자유로운 자연어의 모든 우회를 막는 분류기는 아니다.
- 귀 문구는 표시하되 기본 비활성. OnePassPromptSettings.enable_ear_override=False.
- 확인된 꼬리 문구만 fixed 자리의 일반 tail과 *_tail을 대체한다.
  종 이름 오인을 자동 수정하지 않고 사용자 문구를 전달한다.
- 문구 미확인은 기존 positive/negative/토큰 계획을 유지한다.
- 꼬리·귀가 감지된 캐릭터는 결과 확인 때 추가 체크가 필수다.
  화면뿐 아니라 StudioResults.approve도 이를 검사한다.
- PNG 분석은 실제 투명 픽셀이 있을 때만 흰 배경에 합성한다.
  불투명 RGB·RGBA는 기존 사본 생성 경로를 유지한다. 원본 파일은 변경하지 않는다.
  구현 근거: [Pillow alpha_composite](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.alpha_composite).

## 2. CPU·GPU 책임 구분

Codex의 회귀는 가짜 생성 백엔드와 임시 QSettings를 쓴다.
실제 Qt 버튼 → 확인 데이터 → 요청 → 원시 기록 → 결과 승인·내보내기까지 검사한다.
GPU 출력 SHA 재현을 CPU 통과로 주장하지 않는다.
811cc108 기준 테스트 결과·SHA는 outputs/tail-appearance-implementation-20261004/results.md 및 file-hashes.json에 있다. 이후 입력 제한 보강 결과는 같은 폴더의 review-fix.md, pytest-review-fix.txt, review-fix-hashes.json을 따른다.

사용자의 “리팩토링한것도 포함 해서” 승인에 따라 기능과 리팩터링은 811cc108에 함께 커밋·푸시됐다. 구현 시작 직전 사본:
outputs/tail-appearance-implementation-20261004/before/.
811cc108의 변경에는 리팩터링도 포함된다. 해당 기능 구현만 구분해 검토하려면 before 사본과 비교한다.

## 3. 시험 D와 차이 — 승인된 동작

과거 outputs/tail-dup-20261004/plan.json의 D는 animal ears도 삭제했다.
이번 꼬리 기능은 귀를 유지하므로 다음처럼 된다.

- 시험 D: light blue tail, striped tail, raccoon ears
- 제품: animal ears, light blue tail, striped tail, raccoon ears

test_trial_d_difference_is_only_preserved_animal_ears가 이 차이를 고정한다.
D의 원시 이미지 SHA 재현을 요구하지 않는다.
이번 문구 없음 조건의 기존 이미지 SHA 재현과 분리한다.

## 4. 검증 입력과 대조

생성 전에 사용자가 대상·문구를 확인해야 한다. 자동 실행하지 않는다.
대상은 라쿤 + 사용자가 선택한 1~2명. 문구 후보는 승인 전 확정값이 아니다.

- 라쿤 후보: light blue tail, striped tail
- 악어 후보: grey tail, scaly tail
- 카멜레온 후보: teal tail, spiral tail

문구 없음/있음 × 같은 seed 4장을 짝지어 종류·개수·색을 각각 판정한다.
모델·의상·성별·크롭·해상도·IP 일정·seed·부정은 두 조건에서 동일하게 한다.
문구 추가로 토큰·청크 수가 바뀌면 반드시 별도로 보고한다.
추가 시도·자동 재생성으로 실패 결과를 대체하지 않는다.

### 투명 배경 변경을 문구 효과와 섞지 않기

1. 불투명 캐릭터: 문구 없음 기준의 사본·크롭·프롬프트·원시 SHA를 기존과 비교한다.
2. 투명 캐릭터: 새 흰 배경 사본과 크롭을 CPU에서 확인한다.
3. 새 사본·크롭을 문구 없음/있음 양쪽이 공유하도록 고정한다.
4. 기존 검정/잔상 사본에서 나온 이미지와 새 흰 배경 이미지의 SHA는 같아야 하는 대상이 아니다.
   변경 이유를 '투명 입력 배경 전처리'로 기록한다.
5. 새 배경에서 크롭이 실패하면 기존 크롭으로 조용히 교체하지 않는다.

## 5. 호출 진입점

GUI:
캐릭터·의상 선택 → 이미지 만들기 → 꼬리 문구 입력·확인 → 결과 4장 → 꼬리·귀 포함 결과 확인 → 저장.

검증 스크립트에서 사용할 기존 함수:
genai_lab.studio_generation.build_request(..., appearance=AppearanceOverrides(...)).
AppearanceOverrides와 PartAppearance는 genai_lab.onepass_prompt에서 가져온다.
예: appearance=AppearanceOverrides(tail=PartAppearance("light blue tail, striped tail", confirmed=True)).
문구 없음은 AppearanceOverrides() 또는 인자를 생략한다.

분석 자료는 기존 analyze_inputs의 반환값을 사용한다.
gender는 사용자 확인값, preferences는 검증용 임시 QSettings, seeds는 지정한 4개를 명시한다.
실제 사용자 저장소를 시험용 값으로 덮어쓰지 않는다.
새 부위를 임의 주입하거나 기록의 프롬프트를 수동 조작하지 않는다.

## 6. 확인할 기록

- inputs/references.json: 원본 SHA, preprocessing.alpha_composited, background_rgb, analysis_sha256.
- inputs/approval.json: 초안·확인·실제 적용·원래 태그·제거 이유, 부위 검토 필요 여부.
- generation의 각 후보 run.json:
  prompt.positive/negative, prompt.encoders의 실제 토큰 계획,
  prompt.rules.appendage_appearance,
  appearance_token_counts_before/after, appearance_chunks_before/after,
  기존 IP 일정·원시 SHA·valid 기록.
- generation/user-review.json: appendage_review_required, checks.appendages, reviewer=user.
  자동 검사 통과가 아니라 사용자 확인이다.

## 7. GUI 검토 항목

- 꼬리 없는 캐릭터에는 꼬리 칸이 없는가.
- snake_tail 등 자동 태그가 그대로 보여 오인을 확인할 수 있는가.
- 문구 빈칸/미확인에서는 기존 태그를 유지하는가.
- 문구 수정 후 다시 확인해야 적용되는가.
- 귀 칸이 비활성이고 적용 안 됨이 표시되는가.
- 작은 창에서도 입력 항목은 스크롤되고 생성·뒤로 버튼이 접근 가능한가.
- 추가 확인을 하지 않으면 결과 승인이 막히는가.
- 후보 변경 후 모든 확인을 다시 해야 하는가.
- 투명 입력을 흰 배경 사본으로 분석하고 원본은 보존하는가.

## 8. 남은 한계

꼬리 검출 자체가 빠진 캐릭터는 이번 자동 태그 기반 항목이 나오지 않을 수 있다.
색·무늬를 모델이 실제로 지킬지는 Claude GPU 확인 전까지 미검증이다.
자동 꼬리 개수 판정·오류 제거는 없다.
Qwen 보존 명세 R10/R11의 '꼬리 2개'는 후속 검토 대상으로 남기고 이번에 변경하지 않았다.
