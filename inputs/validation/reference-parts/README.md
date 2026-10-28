# 동물귀·헤어 장신구·꼬리 참조 분석 검증 자료

이 폴더는 생성 결과와 학습 자료에서 분리한 **고정 검증 세트**입니다.
검증기는 이미지를 생성하거나 임계값을 자동 변경하지 않습니다. 검출된 진단 마스크도 생성 조건이나 보호 마스크에 연결하지 않습니다.

## 폴더 규칙

- 원본 이미지와 사람이 작성한 정답 마스크만 둡니다.
- `outputs/`, `inputs/training_candidates/`, 승인 결과를 복사해 넣지 않습니다.
- manifest가 가리키는 모든 파일은 이 폴더 안에 있어야 합니다.
- 동일 이미지를 학습과 검증에 함께 사용하지 않습니다.

## 정답 상태

- `present`: 부위가 명확히 있으며 이진 정답 마스크가 필요합니다.
- `absent`: 부위가 명확히 없습니다. 마스크를 지정하지 않습니다.
- `ambiguous`: 가려짐·장식 혼동 등으로 사람이 확정하기 어렵습니다. 정확도 분모에서 분리하되 결과에는 남깁니다.

`no_detection`은 모델의 관측 결과일 뿐 `absent` 정답으로 자동 승격하지 않습니다.

## manifest 형식

`labels.example.jsonl`을 복사해 실제 상대 경로로 수정합니다.
각 사례는 `animal_ears`, `hair_accessory`, `tail` 정답을 모두 가져야 합니다.
`context_masks`를 제공한다면 `garment`와 `hair`를 함께 제공합니다.
`protection_masks`에는 얼굴과 사람 형태 귀처럼 침범률을 따로 측정할 영역을 넣을 수 있습니다.

동물귀와 헤어 장신구가 같은 위치에 있어도 정답 마스크를 각각 저장합니다. 두 마스크를 서로 빼거나 하나로 합치지 않습니다.

## 실행

```powershell
& "D:\genai-cache\venv\Scripts\python.exe" scripts\validate_reference_parts.py `
  --manifest inputs\validation\reference-parts\labels.jsonl `
  --config configs\animagine.yaml
```

출력은 기본적으로 `outputs/reference-analysis-validation/<시각>/`에 생성됩니다.

- `metrics.json`: 세 부위의 검출 정밀도·재현율·오검출률·불확실률·마스크 IoU/Dice
- `predictions.jsonl`: 사례별 원시 판정, 사유, 동물귀·장신구 교차 진단
- `cases.csv`: 부위별 표 계산용 결과
- `cross-review.csv`: 동물귀·장신구 중첩 상태와 중첩률
- `spatial-diagnostics.csv`: 머리 ROI, 꼬리 신체 연결, 교차 클래스 충돌 진단
- `overlays/`: 정답 초록, 예측 빨강, 겹침 노랑
- `cross-review/`: 동물귀와 장신구 예측 마스크의 중첩 영역
- `spatial-diagnostics/`: 사례별 머리 ROI와 꼬리 연결 후보 영역
- `config-snapshot.json`: 실행 당시 분석 설정

`overlap_ambiguous`는 두 후보가 겹친다는 관측값입니다. 검증기는 승자를 선택하거나 마스크를 변경하지 않으며 자동 조건 주입도 실행하지 않습니다.
