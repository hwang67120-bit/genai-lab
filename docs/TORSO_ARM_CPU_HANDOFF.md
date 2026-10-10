# 몸통 폭 팔 경계 — CPU 구현 인수인계

CPU 계산을 준비했다. 실제 See-through 추론·GPU 재현·GUI 연결은 아직 하지 않았다. 이번 변경은 측정 기능이며 생성 조건이나 체형을 자동으로 바꾸지 않는다.

## 입력에서 결과까지

호출 시작점은 genai_lab/identity_measurement.py의 measure_pixels다. 기존 호출은 그대로 쓰며 기본 use_arm_parts=False다.

팔 분할 결과가 준비됐을 때만 use_arm_parts=True와 handwear_mask를 함께 전달한다. 마스크는 측정 이미지와 같은 좌표·크기여야 한다. 자동 리사이즈나 좌표 변환은 없다.

1. prepare_arm_region: handwear를 전경 안으로 제한한다. 목·골반 중점을 팔로 분류했다면 사용 불가로 기록한다.
2. torso_axis: 목·골반을 기준으로 가슴·허리·골반의 측정 중심과 방향을 구한다.
3. trace_torso_ray: 시제품과 같은 0.5픽셀 간격으로 걷는다. 첫 팔 경계를 끝점으로 잡되, 실루엣 전체 경로도 확인해 다른 쪽의 누락된 팔 띠를 찾는다.
4. measure_torso_with_arms: 폭과 경계 종류, 사용 가능 여부, 검토선을 반환한다. 팔 경계가 없는데 팔 띠가 닿으면 측정 불가다.
5. measure_body와 draw_overlay: 몸통 세 항목에만 새 결과를 넣고 팔 마스크를 확인 그림에 반투명으로 표시한다. 팔다리·머리색 계산은 유지한다.

## 상태와 기록

| 입력 상태 | CPU 결과 |
|---|---|
| 새 모드 꺼짐 | 기존 측정과 기록·그림 그대로 |
| 새 모드 켜짐, 마스크 없음 | arm_parts_pending, status=pending. 기존 숫자로 돌아가지 않음 |
| 사용 가능한 마스크, 양쪽 필요한 경계 확인 | status=measured, w_px·sil_w_px·boundary·ends 기록 |
| 팔 띠에 닿지만 해당 방향 분할 경계 없음 | arm_touch, status=unmeasurable. 비교 점수에서 제외 |
| 몸통 중심을 팔로 분류 | arm_usable=False, 이유 기록. 겹친 팔을 잘라낸 척하지 않음 |
| 크기·형식이 잘못된 마스크 | 오류로 반환. 자동 변환하지 않음 |

팔 영역이 사용 가능하면 기존 arm_merge 문턱을 적용하지 않는다. 사용 불가이며 팔 띠 미해결도 없는 경우에는 기존 arm_merge 보호를 유지한다. 색 표본은 팔 경계 안쪽만 사용한다.

새 모드를 켠 경우 arm_region에 available·usable·reason·pixels를 기록한다. 분할 모델의 경로·revision·코드/가중치 SHA·추론 시간은 실제 모델 연결 단계에서 추가할 항목이다. 지금 추론하지 않았으므로 가상의 값은 넣지 않았다.

## 검증 근거

검사 결과는 outputs/torso-arm-cpu-20261010/에 있다.

- tests/test_torso_arm_measurement.py: 정상·양쪽/한쪽 경계·중점 오분할·전경 틈·대기·잘못된 마스크·비교 제외·기본 동작 불변을 CPU로 검사한다.
- 새 규칙·기존 측정·화면 대역 검사 27개 통과(32.65초, pytest-01.log). 직접 호출부 회귀 검사 3개도 통과(28.52초, pytest-direct-callers-local.log). 합계 30개이며 전체 프로젝트 검사 수가 아니다. 직접 호출부의 첫 제한 환경 실행은 Qt 경로 조회가 WinError 65로 막혀, 기존 Windows 환경에서 재실행했다.
- 기본 모드의 합성 입력 두 조건에서 측정 JSON·확인 PNG·머리색 배열 SHA가 수정 전과 같다. 기준은 legacy-baseline.json이다.
- 시제품 run.py에서 torso 함수만 AST로 추출했다. 모델 로드 등 파일의 다른 코드는 실행하지 않았다. 합성 6조건 × 몸통 3항목 = 18건의 폭·끝점·판정·중심/방향이 일치했다(prototype-geometry-comparison.json).
- 이 18건은 합성 계산 검사다. 실제 이미지 36건의 모델 분할과 폭 재현을 대신하지 않는다.
- 헤어와 GUI 실행 관련 파일 10개의 수정 전후 SHA가 같다. 실제 모델·GPU 실행, 전체 프로젝트 테스트와 실제 GUI 생성은 이번에 실행하지 않았다.

## 이후 연결 작업

현재 identity_measure_worker는 기존 CPU 방식 그대로다. 새 옵션을 자동으로 켜거나 GUI에 노출하지 않았다. 헤어 코드·모델 설정은 변경하지 않았다.

Claude의 헤어 GPU 시험 종료 후, 별도 프로세스에서 로컬 See-through를 1회 실행해 같은 이미지 좌표의 handwear를 전달한다. GPU가 없거나 사용 중이면 새 모드는 부위 분할 대기로 유지하고 재측정한다. 잘못된 크기의 마스크를 결과 좌표로 자동 옮기지 않는다.

실제 연결 검증은 outputs/see-through-arm-measure-20261009/results.json의 허용된 36건 status·폭(px) 대조이며, 기존 제외 자료 정책을 지킨다. 모델·코드 SHA·추론 시간과 GPU 작업 겹침 방지도 그 단계에서 확인한다.

다음 작업: 현재 GPU 시험이 끝난 뒤 실제 분할 모델 연결과 재현을 진행한다. 지금 사용자 추가 확인은 필요 없다.