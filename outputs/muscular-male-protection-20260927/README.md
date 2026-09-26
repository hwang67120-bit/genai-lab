# muscular-male 보호 마스크 분해 — 저장 자료 감사 (2026-09-27)

**보호 영역을 키운 주 구성 요소는 `output_hair`다.** 머리 질의가 전신 박스를 채택했고, SAM2가 그 박스에서 몸 실루엣 498,137px를 분할했다. 정상 크기의 머리 후보들과 합친 머리 마스크는 498,176px, 5×5 최대 필터 적용 후 514,975px다. 이 머리 보호 단독이 실제 effective_protection 전체와 픽셀 단위로 같다. `output_face`에도 복부 51,806px 오검출이 있지만 머리 보호에 완전히 포함되어 보호 합집합을 추가로 늘리지는 않는다.

**119px 사례와 같은 원인인가: 예 — 이 사례들의 저장된 마스크 생성·합성 경로 범위에서.** 과거 v2와 이번 세 의상에서 Base 파일 SHA-256, raw hair/face 픽셀, 합성 보호 픽셀이 모두 같다. 어떤 영상 특징 때문에 검출기가 전신을 hair로 골랐는지는 미확정이며 다른 남성 캐릭터 전체로 일반화하지 않는다. 원인을 검출 모델 자체의 일반적 성질로 확장하지 않는다.

이미지 생성 0회, 모델 적재 0회, 검출·분할 재실행 0회. `C:/Users/user/AppData/Local/Temp/genai-parts-*/parts.json`과 후보 SAM2 마스크가 살아 있어 기존 자료만 읽었다. 새로 수행한 것은 PNG 이진 마스크 통계·합집합/기존 편집 계획 계산·Base 위 표시다. 원본 및 운영 소스·설정·기존 보고서는 수정하지 않았다.

## 1. 대상과 증거 연결

일반화 실행은 모두 OFF를 주 대상으로 삼았다. 부분·최소 ON도 같은 보호/요청 수치로 차단된 기록이 있으며, 이번 표는 사용자 지정 3의상 + v2 + 대조군의 5건이다. 임시 `parts.json`에는 run_id가 없어 실행 시각 구간, 이미지 크기, 저장 마스크와의 일치로 연결했다. 과거 v2 임시 기록은 2026-09-23 02:03:46이며, 실행 경로 `20260923-015810-98169b12`와 저장 구성 마스크에 일치한다.

| 대상 | Base SHA-256 | 당시 parts.json | 시각 연결 / 저장본 대조 |
|---|---|---|---|
| male-ferrari | `2ff31ed03cb8eae2b216b465800a094701a6c8030387e592579f83d814a652c2` | `C:\Users\user\AppData\Local\Temp\genai-parts-nz9mdn_1\parts.json` | 2026-09-27T02:03:53.917854 ~ 2026-09-27T02:11:33.742430 안에 기록 |
| male-partial | `2ff31ed03cb8eae2b216b465800a094701a6c8030387e592579f83d814a652c2` | `C:\Users\user\AppData\Local\Temp\genai-parts-oeaf38qk\parts.json` | 2026-09-27T02:20:09.489784 ~ 2026-09-27T02:27:26.780378 안에 기록 |
| male-minimal | `2ff31ed03cb8eae2b216b465800a094701a6c8030387e592579f83d814a652c2` | `C:\Users\user\AppData\Local\Temp\genai-parts-ohzm1j_5\parts.json` | 2026-09-27T02:35:22.685123 ~ 2026-09-27T02:42:42.390303 안에 기록 |
| male-v2 | `2ff31ed03cb8eae2b216b465800a094701a6c8030387e592579f83d814a652c2` | `C:\Users\user\AppData\Local\Temp\genai-parts-3fw0ur67\parts.json` | 과거 실행 시각 및 저장 구성 마스크 일치 |
| ordinary-ferrari | `cd1a11ffa5664d37106758e544ca783d0950f868dcebe0ae39bd04f2f5d9d3e0` | `C:\Users\user\AppData\Local\Temp\genai-parts-724sww9u\parts.json` | 2026-09-26T23:39:45.509655 ~ 2026-09-26T23:47:55.901153 안에 기록 |

전체 실제 파일 경로와 SHA-256은 `components.json → cases[].source_sha256`에 보존했다. 임시 디렉터리가 정리돼도 검출 박스·점수·후보 마스크 픽셀/bbox·기록 내용은 JSON 안에 남는다. 원시 PNG를 별도로 복사하지는 않았다.

## 2. 보호 구성 요소 표

raw는 SAM2 채택 후보 합집합이며, expanded는 기존 `MaxFilter(5)` 뒤의 실제 보호 기여 마스크다. bbox는 `[x0,y0,x1,y1)` 우측·하단 제외 픽셀 좌표다. 비율 분모는 **해당 Base에 저장된 output-region foreground**(MM 495,377px, OF 153,848px)이며 캔버스가 아니다. 검출 마스크가 이 전경 밖에도 있으면 비율이 1을 넘을 수 있다. `parts.json.area_validity.foreground_area_ratio`는 이 실행에서 실제로 canvas를 분모로 썼으므로 그 숫자를 전경 비율로 전사하지 않았다.

보호용 DINO/SAM2와 output-region 인체 파서는 서로 다른 분석이다. Base 인체 파서의 hair=9,984px를 의상 교체 보호용 hair=498,176px와 혼동하지 않았다. 정제 재검출의 임시 전경 파일 자체는 삭제돼 있지만 저장된 Base 전경으로 완료 3건의 계획 마스크 10종을 모두 픽셀 일치 재구성했다(5절).

| 대상 | 구성 요소 | raw px | raw bbox | raw/전경 | 확장 px | 확장 bbox | 확장/전경 | 채택·역할 |
|---|---|---:|---|---:|---:|---|---:|---|
| male-ferrari | output_hair | 498,176 | [16, 16, 727, 1386] | 1.005650 | 514,975 | [14, 14, 729, 1388] | 1.039562 | hard |
| male-ferrari | output_face | 63,209 | [225, 81, 511, 611] | 0.127598 | 66,910 | [223, 79, 513, 613] | 0.135069 | hard |
| male-ferrari | output_human_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-ferrari | output_animal_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-ferrari | output_hair_accessory | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-ferrari | output_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| male-ferrari | output_tail | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| male-partial | output_hair | 498,176 | [16, 16, 727, 1386] | 1.005650 | 514,975 | [14, 14, 729, 1388] | 1.039562 | hard |
| male-partial | output_face | 63,209 | [225, 81, 511, 611] | 0.127598 | 66,910 | [223, 79, 513, 613] | 0.135069 | hard |
| male-partial | output_human_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-partial | output_animal_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-partial | output_hair_accessory | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-partial | output_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| male-partial | output_tail | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| male-minimal | output_hair | 498,176 | [16, 16, 727, 1386] | 1.005650 | 514,975 | [14, 14, 729, 1388] | 1.039562 | hard |
| male-minimal | output_face | 63,209 | [225, 81, 511, 611] | 0.127598 | 66,910 | [223, 79, 513, 613] | 0.135069 | hard |
| male-minimal | output_human_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-minimal | output_animal_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-minimal | output_hair_accessory | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-minimal | output_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| male-minimal | output_tail | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| male-v2 | output_hair | 498,176 | [16, 16, 727, 1386] | 1.005650 | 514,975 | [14, 14, 729, 1388] | 1.039562 | hard |
| male-v2 | output_face | 63,209 | [225, 81, 511, 611] | 0.127598 | 66,910 | [223, 79, 513, 613] | 0.135069 | hard |
| male-v2 | output_human_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-v2 | output_animal_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-v2 | output_hair_accessory | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| male-v2 | output_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| male-v2 | output_tail | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| ordinary-ferrari | output_hair | 18,750 | [232, 151, 447, 333] | 0.121874 | 22,015 | [230, 149, 449, 335] | 0.143096 | hard |
| ordinary-ferrari | output_face | 5,538 | [321, 231, 411, 334] | 0.035997 | 6,613 | [319, 229, 413, 336] | 0.042984 | hard |
| ordinary-ferrari | output_human_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| ordinary-ferrari | output_animal_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 미채택 |
| ordinary-ferrari | output_hair_accessory | 15,352 | [282, 151, 447, 316] | 0.099787 | 18,007 | [280, 149, 449, 318] | 0.117044 | hard |
| ordinary-ferrari | output_ears | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |
| ordinary-ferrari | output_tail | 0 | None | 0.000000 | 0 | None | 0.000000 | 질의 없음 |

### 겹침과 실제 기여

| 대상 | 구성 쌍 | raw 겹침 px | 확장 후 겹침 px |
|---|---|---:|---:|
| male-ferrari | output_hair:output_face | 63,209 | 66,910 |
| male-partial | output_hair:output_face | 63,209 | 66,910 |
| male-minimal | output_hair:output_face | 63,209 | 66,910 |
| male-v2 | output_hair:output_face | 63,209 | 66,910 |
| ordinary-ferrari | output_hair:output_face | 2,656 | 4,783 |
| ordinary-ferrari | output_hair:output_hair_accessory | 15,352 | 18,007 |
| ordinary-ferrari | output_face:output_hair_accessory | 394 | 2,149 |

표에 없는 쌍의 겹침은 0이며 전체 21쌍/건은 JSON에 있다. MM은 확장 얼굴 66,910px 전부가 확장 머리 안에 들어가고, 머리만의 보호 기여는 448,065px다. 머리 보호 하나가 합집합 514,975px와 같다. OF의 hair_accessory는 15,352px(확장 18,007px)지만 머리 안에 완전히 포함돼 추가 보호 면적은 0이다. OF에서도 머리 마스크에 팔 위 손 일부가 포함되는 작은 오검출이 보이나 전신으로 퍼지는 MM과 규모가 다르다.

## 3. GroundingDINO → SAM2 후보 내역

설정의 모델 ID는 `IDEA-Research/grounding-dino-tiny`, `facebook/sam2.1-hiera-tiny`. 이 경로는 `TransformersPartsBackend._load()`에서 CPU 적재하며 로컬 캐시만 사용한다. 이번에 모델을 적재하지 않았으므로 현재 설치 버전을 과거 실행 버전이라고 기재하지 않는다. 해당 시점의 패키지 버전은 본 감사에서 확인한 parts.json에 기록 없음.

질의별 박스는 저장된 실수 xyxy를 그대로 JSON에 보존하고 아래는 소수 3자리 표시다. score는 DINO 검출 점수, quality는 SAM2 mask_quality_score이며 의미상 정확도의 확률로 해석하지 않는다. MM 4건의 후보 마스크 픽셀 해시는 동일하다.

| 대상 | 질의·후보 | DINO box | score | SAM2 quality | SAM2 px | 마스크 bbox | 최종 상태 |
|---|---|---|---:|---:|---:|---|---|
| male-ferrari | hair. [0] | [16.393, 16.601, 725.218, 1384.158] | 0.498629 | 0.936204 | 498,137 | [16, 16, 727, 1386] | detected / accepted_mask_available |
| male-ferrari | hair. [1] | [299.676, 12.558, 443.582, 129.91] | 0.374838 | 0.939641 | 8,554 | [304, 17, 441, 129] | detected / accepted_mask_available |
| male-ferrari | hair. [2] | [299.007, 12.631, 444.005, 190.04] | 0.331325 | 0.924171 | 9,937 | [303, 17, 441, 190] | detected / accepted_mask_available |
| male-ferrari | face. [0] | [318.993, 79.932, 423.544, 222.791] | 0.766407 | 0.953995 | 11,403 | [319, 81, 424, 221] | detected / accepted_mask_available |
| male-ferrari | face. [1] | [238.144, 362.459, 504.047, 608.913] | 0.312998 | 0.916914 | 51,806 | [225, 366, 511, 611] | detected / accepted_mask_available |
| male-ferrari | human ears. person ears. [0] | [18.894, 15.228, 722.367, 1382.985] | 0.438763 | 0.964107 | 495,600 | [16, 16, 726, 1386] | uncertain / small_part_area_upper_bound_exceeded |
| male-ferrari | human ears. person ears. [1] | [425.268, 101.958, 444.776, 150.05] | 0.347583 | 0.882277 | 461 | [428, 104, 443, 150] | uncertain / small_part_area_upper_bound_exceeded |
| male-ferrari | human ears. person ears. [2] | [297.823, 99.082, 317.883, 150.016] | 0.361411 | 0.876795 | 497 | [300, 101, 316, 146] | uncertain / small_part_area_upper_bound_exceeded |
| male-ferrari | human ears. person ears. [3] | [297.152, 95.444, 444.931, 153.988] | 0.400066 | 0.497837 | 6,311 | [301, 84, 441, 161] | uncertain / small_part_area_upper_bound_exceeded |
| male-ferrari | animal ears. [0] | [18.449, 216.529, 724.46, 817.614] | 0.366450 | 0.941302 | 238,245 | [20, 184, 724, 808] | uncertain / small_part_area_upper_bound_exceeded |
| male-ferrari | animal ears. [1] | [17.607, 203.601, 723.161, 1384.219] | 0.326265 | 0.944046 | 431,238 | [18, 181, 724, 1385] | uncertain / small_part_area_upper_bound_exceeded |
| male-ferrari | animal ears. [2] | [17.32, 19.342, 722.798, 1383.2] | 0.329937 | 0.956206 | 496,897 | [16, 16, 726, 1386] | uncertain / small_part_area_upper_bound_exceeded |
| male-ferrari | hair ornament. hair accessory. hair clip. | 없음 | — | — | 0 | 없음 | uncertain / no_candidate_above_threshold |
| male-partial | hair. [0] | [16.393, 16.601, 725.218, 1384.158] | 0.498629 | 0.936204 | 498,137 | [16, 16, 727, 1386] | detected / accepted_mask_available |
| male-partial | hair. [1] | [299.676, 12.558, 443.582, 129.91] | 0.374838 | 0.939641 | 8,554 | [304, 17, 441, 129] | detected / accepted_mask_available |
| male-partial | hair. [2] | [299.007, 12.631, 444.005, 190.04] | 0.331325 | 0.924171 | 9,937 | [303, 17, 441, 190] | detected / accepted_mask_available |
| male-partial | face. [0] | [318.993, 79.932, 423.544, 222.791] | 0.766407 | 0.953995 | 11,403 | [319, 81, 424, 221] | detected / accepted_mask_available |
| male-partial | face. [1] | [238.144, 362.459, 504.047, 608.913] | 0.312998 | 0.916914 | 51,806 | [225, 366, 511, 611] | detected / accepted_mask_available |
| male-partial | human ears. person ears. [0] | [18.894, 15.228, 722.367, 1382.985] | 0.438763 | 0.964107 | 495,600 | [16, 16, 726, 1386] | uncertain / small_part_area_upper_bound_exceeded |
| male-partial | human ears. person ears. [1] | [425.268, 101.958, 444.776, 150.05] | 0.347583 | 0.882277 | 461 | [428, 104, 443, 150] | uncertain / small_part_area_upper_bound_exceeded |
| male-partial | human ears. person ears. [2] | [297.823, 99.082, 317.883, 150.016] | 0.361411 | 0.876795 | 497 | [300, 101, 316, 146] | uncertain / small_part_area_upper_bound_exceeded |
| male-partial | human ears. person ears. [3] | [297.152, 95.444, 444.931, 153.988] | 0.400066 | 0.497837 | 6,311 | [301, 84, 441, 161] | uncertain / small_part_area_upper_bound_exceeded |
| male-partial | animal ears. [0] | [18.449, 216.529, 724.46, 817.614] | 0.366450 | 0.941302 | 238,245 | [20, 184, 724, 808] | uncertain / small_part_area_upper_bound_exceeded |
| male-partial | animal ears. [1] | [17.607, 203.601, 723.161, 1384.219] | 0.326265 | 0.944046 | 431,238 | [18, 181, 724, 1385] | uncertain / small_part_area_upper_bound_exceeded |
| male-partial | animal ears. [2] | [17.32, 19.342, 722.798, 1383.2] | 0.329937 | 0.956206 | 496,897 | [16, 16, 726, 1386] | uncertain / small_part_area_upper_bound_exceeded |
| male-partial | hair ornament. hair accessory. hair clip. | 없음 | — | — | 0 | 없음 | uncertain / no_candidate_above_threshold |
| male-minimal | hair. [0] | [16.393, 16.601, 725.218, 1384.158] | 0.498629 | 0.936204 | 498,137 | [16, 16, 727, 1386] | detected / accepted_mask_available |
| male-minimal | hair. [1] | [299.676, 12.558, 443.582, 129.91] | 0.374838 | 0.939641 | 8,554 | [304, 17, 441, 129] | detected / accepted_mask_available |
| male-minimal | hair. [2] | [299.007, 12.631, 444.005, 190.04] | 0.331325 | 0.924171 | 9,937 | [303, 17, 441, 190] | detected / accepted_mask_available |
| male-minimal | face. [0] | [318.993, 79.932, 423.544, 222.791] | 0.766407 | 0.953995 | 11,403 | [319, 81, 424, 221] | detected / accepted_mask_available |
| male-minimal | face. [1] | [238.144, 362.459, 504.047, 608.913] | 0.312998 | 0.916914 | 51,806 | [225, 366, 511, 611] | detected / accepted_mask_available |
| male-minimal | human ears. person ears. [0] | [18.894, 15.228, 722.367, 1382.985] | 0.438763 | 0.964107 | 495,600 | [16, 16, 726, 1386] | uncertain / small_part_area_upper_bound_exceeded |
| male-minimal | human ears. person ears. [1] | [425.268, 101.958, 444.776, 150.05] | 0.347583 | 0.882277 | 461 | [428, 104, 443, 150] | uncertain / small_part_area_upper_bound_exceeded |
| male-minimal | human ears. person ears. [2] | [297.823, 99.082, 317.883, 150.016] | 0.361411 | 0.876795 | 497 | [300, 101, 316, 146] | uncertain / small_part_area_upper_bound_exceeded |
| male-minimal | human ears. person ears. [3] | [297.152, 95.444, 444.931, 153.988] | 0.400066 | 0.497837 | 6,311 | [301, 84, 441, 161] | uncertain / small_part_area_upper_bound_exceeded |
| male-minimal | animal ears. [0] | [18.449, 216.529, 724.46, 817.614] | 0.366450 | 0.941302 | 238,245 | [20, 184, 724, 808] | uncertain / small_part_area_upper_bound_exceeded |
| male-minimal | animal ears. [1] | [17.607, 203.601, 723.161, 1384.219] | 0.326265 | 0.944046 | 431,238 | [18, 181, 724, 1385] | uncertain / small_part_area_upper_bound_exceeded |
| male-minimal | animal ears. [2] | [17.32, 19.342, 722.798, 1383.2] | 0.329937 | 0.956206 | 496,897 | [16, 16, 726, 1386] | uncertain / small_part_area_upper_bound_exceeded |
| male-minimal | hair ornament. hair accessory. hair clip. | 없음 | — | — | 0 | 없음 | uncertain / no_candidate_above_threshold |
| male-v2 | hair. [0] | [16.393, 16.601, 725.218, 1384.158] | 0.498629 | 0.936204 | 498,137 | [16, 16, 727, 1386] | detected / accepted_mask_available |
| male-v2 | hair. [1] | [299.676, 12.558, 443.582, 129.91] | 0.374838 | 0.939641 | 8,554 | [304, 17, 441, 129] | detected / accepted_mask_available |
| male-v2 | hair. [2] | [299.007, 12.631, 444.005, 190.04] | 0.331325 | 0.924171 | 9,937 | [303, 17, 441, 190] | detected / accepted_mask_available |
| male-v2 | face. [0] | [318.993, 79.932, 423.544, 222.791] | 0.766407 | 0.953995 | 11,403 | [319, 81, 424, 221] | detected / accepted_mask_available |
| male-v2 | face. [1] | [238.144, 362.459, 504.047, 608.913] | 0.312998 | 0.916914 | 51,806 | [225, 366, 511, 611] | detected / accepted_mask_available |
| male-v2 | human ears. person ears. [0] | [18.894, 15.228, 722.367, 1382.985] | 0.438763 | 0.964107 | 495,600 | [16, 16, 726, 1386] | uncertain / small_part_area_upper_bound_exceeded |
| male-v2 | human ears. person ears. [1] | [425.268, 101.958, 444.776, 150.05] | 0.347583 | 0.882277 | 461 | [428, 104, 443, 150] | uncertain / small_part_area_upper_bound_exceeded |
| male-v2 | human ears. person ears. [2] | [297.823, 99.082, 317.883, 150.016] | 0.361411 | 0.876795 | 497 | [300, 101, 316, 146] | uncertain / small_part_area_upper_bound_exceeded |
| male-v2 | human ears. person ears. [3] | [297.152, 95.444, 444.931, 153.988] | 0.400066 | 0.497837 | 6,311 | [301, 84, 441, 161] | uncertain / small_part_area_upper_bound_exceeded |
| male-v2 | animal ears. [0] | [18.449, 216.529, 724.46, 817.614] | 0.366450 | 0.941302 | 238,245 | [20, 184, 724, 808] | uncertain / small_part_area_upper_bound_exceeded |
| male-v2 | animal ears. [1] | [17.607, 203.601, 723.161, 1384.219] | 0.326265 | 0.944046 | 431,238 | [18, 181, 724, 1385] | uncertain / small_part_area_upper_bound_exceeded |
| male-v2 | animal ears. [2] | [17.32, 19.342, 722.798, 1383.2] | 0.329937 | 0.956206 | 496,897 | [16, 16, 726, 1386] | uncertain / small_part_area_upper_bound_exceeded |
| male-v2 | hair ornament. hair accessory. hair clip. | 없음 | — | — | 0 | 없음 | uncertain / no_candidate_above_threshold |
| ordinary-ferrari | hair. [0] | [279.782, 143.718, 449.989, 324.239] | 0.498975 | 0.935942 | 15,398 | [282, 151, 447, 316] | detected / accepted_mask_available |
| ordinary-ferrari | hair. [1] | [234.267, 143.663, 450.237, 329.698] | 0.367958 | 0.868888 | 18,726 | [232, 151, 447, 333] | detected / accepted_mask_available |
| ordinary-ferrari | face. [0] | [317.038, 221.213, 410.442, 327.893] | 0.673287 | 0.928427 | 5,538 | [321, 231, 411, 334] | detected / accepted_mask_available |
| ordinary-ferrari | human ears. person ears. [0] | [176.553, 33.825, 486.769, 1194.828] | 0.458767 | 0.987724 | 150,243 | [179, 37, 482, 1195] | uncertain / small_part_area_upper_bound_exceeded |
| ordinary-ferrari | human ears. person ears. [1] | [386.669, 156.326, 438.317, 196.425] | 0.310503 | 0.897252 | 809 | [389, 155, 437, 198] | uncertain / small_part_area_upper_bound_exceeded |
| ordinary-ferrari | human ears. person ears. [2] | [292.876, 257.554, 327.526, 300.899] | 0.345468 | 0.878756 | 625 | [296, 260, 325, 301] | uncertain / small_part_area_upper_bound_exceeded |
| ordinary-ferrari | human ears. person ears. [3] | [292.615, 257.496, 333.128, 337.1] | 0.307878 | 0.836619 | 1,244 | [295, 261, 333, 337] | uncertain / small_part_area_upper_bound_exceeded |
| ordinary-ferrari | animal ears. | 없음 | — | — | 0 | 없음 | uncertain / no_candidate_above_threshold |
| ordinary-ferrari | hair ornament. hair accessory. hair clip. [0] | [278.29, 142.422, 451.819, 323.046] | 0.420426 | 0.935814 | 15,352 | [282, 151, 447, 316] | detected / accepted_mask_available |

MM의 핵심은 다음 세 지점이다.

- `hair.` 후보 0: 전신 박스 `[16.393,16.601,725.218,1384.158]`, score 0.498629, quality 0.936204, 분할 498,137px. 직사각형 전체를 채운 것이 아니라 그 안의 인물 실루엣을 따라간다. 따라서 ‘박스 전체를 기계적으로 칠했다’고 설명하지 않는다.
- hair 후보 1·2는 머리 근처이며 8,554px·9,937px. 전신 후보와 함께 OR되어 정상 작은 후보가 큰 후보를 제거하지 못한다.
- `face.` 후보 1: 복부 박스 `[238.144,362.459,504.047,608.913]`, score 0.312998, quality 0.916914, 분할 51,806px. 얼굴 후보 0은 11,403px다. 복부를 얼굴로 잘못 받아들인 것이 별도로 보인다.

사람 귀·동물 귀에도 거대 후보는 있으나 `small_part_area_upper_bound_exceeded`로 미채택되어 이번 effective_protection 기여는 0이다. 머리·얼굴은 같은 면적 상한 대상에 없다. hair 원시 합집합의 bbox는 **711×1,370**(확장 후 715×1,374)으로 측정됐다. 이전 설명의 **686×1326은 별도 평가용 Final hair 마스크(16,703px)**의 bbox다. `outputs/sdxl-local-garment-strength-v2-20260923/summary.json → cases.muscular-male.regions.hair`와 `cases/muscular-male/measurement/output_regions/candidate_1_final/regions.json`에서 확인했다. 의상 교체 입력의 보호용 hair(498,176px)와 다른 마스크이므로 그 bbox를 보호 마스크 치수로 옮겨 쓰지 않았다.

## 4. 보호 합성 코드 근거

| 위치 | 실제 코드/동작 | 이번 자료에 해당하는 의미 |
|---|---|---|
| `genai_lab/hair_error_correction.py:144–171` | `box_threshold=.30`, `mask_threshold=.75`; `hair.`, `face.`, 사람귀·동물귀·머리장식 질의 | tail 질의는 이 분석기에 없음 |
| `genai_lab/extra_parts_analysis.py:31–48` | `local_files_only=True`, CPU 모델 적재 | 이번 감사에서는 호출하지 않음 |
| `genai_lab/extra_parts_analysis.py:359–383` | 후보 boxes/scores 및 SAM2 후보 PNG 저장 | 임시 저장 자료를 발견해 재검출 불필요 |
| `genai_lab/extra_parts_analysis.py:384–387` | `mask_scores >= self.mask_threshold` | quality만으로 후보를 고르는 reviewer 없는 경로 |
| `genai_lab/extra_parts_analysis.py:443` | `combined = candidates.any(axis=0)` | 큰 전신 hair 후보와 작은 hair 후보가 합쳐짐 |
| `genai_lab/extra_parts_analysis.py:133–138,479–521` | 귀·꼬리·머리장식 면적 상한; hair/face 상한 없음 | 거대 귀는 거부, 거대 hair/face는 남음 |
| `genai_lab/output_coordinate_control.py:97–117` | face·animal_ears·ears·hair_accessory, garment이면 hair 추가; `MaxFilter(5)` 후 OR | 선택된 MM raw hair에 16,799px 증가. 이미 raw 단계부터 전신임 |
| `genai_lab/output_coordinate_control.py:119–125` | output_tail 존재 시 조건부 보호로 OR | 조건부 꼬리 경로는 있으나 이 5건에는 채택 꼬리 없음 |
| `genai_lab/output_coordinate_control.py:127–130` | hard의 5×5 ellipse dilation에서 hard를 뺀 boundary | 이 영역을 hard 보호로 한 번 더 합치지 않음; soft 감쇠용 |
| `genai_lab/output_coordinate_control.py:134–154` | output_human_ears 진단용; hard_parts/conditional_parts 기록 | human_ears를 이름만 보고 보호에 더하지 않음 |
| `genai_lab/garment_edit_plan.py:281–287` | requested=source\|target\|add; domain=foreground\|growth; `effective_protection = hard_keep \| (conditional & ~release)` | 사용자 add/exclude/release 입력은 현 호출에 없음 |
| `genai_lab/garment_edit_plan.py:301–305` | `soft *= 1.0 - boundary * soft_boundary_strength`; `soft *= hard_edit` | 경계 감쇠 0.5와 내부 페더로 남은 얇은 편집 조각의 W 감소 |
| `genai_lab/selected_garment_correction.py:344–358` | 보호 분석과 계획 생성 | 동일 Base 좌표에서 수행 |
| `genai_lab/selected_garment_correction.py:360–368` | hard_protection_pixels >= requested_pixels이면 예외 | 부분·최소는 여기서 차단 |
| `genai_lab/selected_garment_correction.py:372–402` | report·plan·진단 마스크 저장 | 예외 뒤이므로 차단 건의 effective PNG는 없음 |

## 5. 저장 마스크 대조 및 요청 영역

| 대상 | 자신의 저장 effective와 재합성 | 차이 px | 저장 계획 10종 재계산 | 비고 |
|---|---|---:|---|---|
| male-ferrari | 동일 | 0 | 10/10 픽셀 동일 | 당시 query 마스크로 직접 재합성 |
| male-partial | 자신의 effective 저장본 없음 | 해당 없음 | 예외 기록의 요청/보호 수치와 일치 | 이 건의 당시 query 마스크를 합친 결과는 Ferrari 저장 effective와 픽셀 동일(차이 0) |
| male-minimal | 자신의 effective 저장본 없음 | 해당 없음 | 예외 기록의 요청/보호 수치와 일치 | 이 건의 당시 query 마스크를 합친 결과는 Ferrari 저장 effective와 픽셀 동일(차이 0) |
| male-v2 | 동일 | 0 | 10/10 픽셀 동일 | 당시 query 마스크로 직접 재합성 |
| ordinary-ferrari | 동일 | 0 | 10/10 픽셀 동일 | 당시 query 마스크로 직접 재합성 |

**차단 건은 자신의 저장 effective와 비교했다고 주장하지 않는다.** 파일이 저장되기 전에 예외가 발생했다. 그 건의 당시 임시 query 마스크가 남아 있고, 완료 Ferrari의 구성 마스크/보호와 정확히 같으며 오류 기록의 514,975px와 일치한다. 이것은 새 검출로 추정한 분해가 아니다. 요청·충돌·W 등의 차단 건 추가 수치는 저장된 source와 같은 Base 전경·승인 태그로 기존 계획 함수를 오프라인 계산한 값이며 실제 Inpaint 입력/출력이 아니다.

완료 3건은 `source_garment_removal / target_garment_coverage / garment_growth_envelope / requested_edit / hard_edit_domain / hard_protection / conditional_protection / effective_protection / mask_conflict / soft_guidance` 10종 모두 저장본과 차이 0이다. 다른 foreground로 바꿔 결과를 맞춘 적은 없다.

| 대상 | 요청 px | 보호 전체 px | 요청∩보호(도메인 제한 전) | 요청 중 도메인 밖 px | 실제 도메인 내 보호 충돌 px | hard edit px | W>0 px | W max | 불변식·출력 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| male-ferrari | 523,471 | 514,975 | 492,453 | 7,283 | 486,049 | 30,139 | 89 | 4 | 통과·완료 |
| male-partial | 511,625 | 514,975 | 490,881 | 10,000 | 481,807 | 19,818 | 89 | 4 | 정제 전 차단(뒤쪽 값은 오프라인) |
| male-minimal | 489,874 | 514,975 | 488,635 | 14,845 | 474,910 | 119 | 25 | 64 | 정제 전 차단(뒤쪽 값은 오프라인) |
| male-v2 | 490,735 | 514,975 | 489,496 | 14,845 | 475,771 | 119 | 25 | 64 | 불변식 도입 전 완료 |
| ordinary-ferrari | 182,817 | 23,845 | 23,316 | 3,982 | 22,662 | 156,173 | 149,930 | 254 | 통과·완료 |

Ferrari는 보호 514,975 < 요청 523,471로 **8,496px 차이**여서 불변식을 통과한다. 그러나 이 불변식은 전체 보호 면적과 전체 요청 면적의 비교이지 ‘쓸 만한 편집 영역이 남았는가’ 검사가 아니다. 도메인 내 충돌을 빼면 hard 30,139px이나 W>0은 89px, 최댓값 4다. 반면 old v2는 hard 119px·W>0 25px·max64였으며 당시에는 이후 추가된 불변식에 막히지 않고 실행됐다. 이번 최소 의상을 오프라인으로 계산해도 hard 119px가 나온다.

## 6. 판단과 시각 확인

- **보호 영역을 키운 주 구성 요소:** output_hair. 채택된 전신 후보가 핵심이며 팽창은 498,176→514,975px로 추가 확대했을 뿐이다.
- **그 구성 요소가 커진 이유 후보:** ‘검출 박스 과대’와 ‘그 박스에서 인물 전체 실루엣을 분할하여 hair로 수용’이 저장 박스·마스크로 뒷받침된다. 박스 사각형 전체 채움, 팽창만의 문제라는 설명은 자료와 다르다. 검출기 내부가 왜 그 의미를 선택했는지는 미확정.
- **얼굴 문제도 있음:** 복부 후보가 face로 채택됐지만 현재 보호 합집합의 추가 면적은 0. hair만 고쳤을 때 이 face 오검출이 어떻게 남을지는 이번에 수정/생성하지 않아 결과를 주장하지 않는다.
- **119px 사례와 같은 원인인가:** 예. 같은 Base·같은 raw 구성 마스크·같은 보호 합집합이다. 의상별 요청 영역만 달라져 ‘통과하지만 거의 편집 안 됨’과 ‘정제 전 차단’으로 갈린다. 백로그 24번 `docs/BACKLOG.md:121–122`의 119px 퇴화와 연결된다. 백로그는 수정하지 않았다.

각 건 디렉터리의 `components-overlay.png`는 Base 위 확장 보호를 색으로 표시한다(머리 빨강, 얼굴 파랑, 머리장식 노랑). 겹침은 겹쳐 칠해진다. `detector-boxes.png`는 모든 저장 질의 박스와 점수를 표시한다. 혼잡한 전체 박스 그림과 별도로 `output_hair-candidate_*.png`, `output_face-candidate_*.png`에 각 SAM2 후보와 해당 검출 박스를 표시했다. **Final 초과 노출 이미지의 복사본이 아니라 사용자 지정 Part 2의 Base 진단 오버레이다.**

| 대상 | 보호 오버레이 | 전체 검출 박스 |
|---|---|---|
| male-ferrari | [components-overlay.png](cases/male-ferrari/components-overlay.png) | [detector-boxes.png](cases/male-ferrari/detector-boxes.png) |
| male-partial | [components-overlay.png](cases/male-partial/components-overlay.png) | [detector-boxes.png](cases/male-partial/detector-boxes.png) |
| male-minimal | [components-overlay.png](cases/male-minimal/components-overlay.png) | [detector-boxes.png](cases/male-minimal/detector-boxes.png) |
| male-v2 | [components-overlay.png](cases/male-v2/components-overlay.png) | [detector-boxes.png](cases/male-v2/detector-boxes.png) |
| ordinary-ferrari | [components-overlay.png](cases/ordinary-ferrari/components-overlay.png) | [detector-boxes.png](cases/ordinary-ferrari/detector-boxes.png) |

## 7. 한계 및 검증

대상 5건은 서로 독립 캐릭터 5명이 아니다. muscular-male 4건은 같은 Base를 사용한다. 임시 기록 연결에 run_id가 없다는 한계를 위에 명시했다. 실제 foreground와 마스크가 서로 다른 분석기로 나와 비율 1 초과가 생기며 그 값을 억지로 1에 잘라 쓰지 않았다. 보호 마스크의 기계적 합성은 픽셀 대조로 확인했지만 검출의 의미 정확성 해석은 Codex 육안이다.

기존 정책을 고치거나 보호를 제거하지 않았다. 모델 재실행·생성·설정 조정·운영 테스트 실행도 하지 않았다. 새 계측 스크립트는 이 출력 디렉터리에만 작성했고 Linux에서 작성, Windows의 기존 Python으로 배열 측정을 실행했다. `components.json`에 원시 수치·박스·점수·출처 해시와 일치 검증을 남겼다.

시작 시 기존 변경: `M docs/BACKLOG.md`, `?? afe_load(p.read_text())`, `?? approved-bases/`, `?? inputs/training_candidates/`. 이번 변경과 섞어 커밋하지 않았고 커밋 자체를 하지 않았다.

## 종료 검증 / git status

```text
 M docs/BACKLOG.md
?? afe_load(p.read_text())
?? approved-bases/
?? inputs/training_candidates/
```

시작 시 상태와 같다. BACKLOG의 기존 8줄 추가는 그대로 두었으며 이번 작업에서 수정하지 않았다. 새 산출물 디렉터리는 outputs 무시 규칙 때문에 보통 git status에 나오지 않는다.

```text
!! outputs/exposure-audit-20260927/
!! outputs/muscular-male-protection-20260927/
```

일반화 사전 기록의 운영 소스 123개 SHA-256을 다시 대조해 변경 0개, configs/animagine.yaml 해시 동일을 확인했다. 생성 0회·모델 적재 0회·검출 재실행 0회. 커밋하지 않았다.
