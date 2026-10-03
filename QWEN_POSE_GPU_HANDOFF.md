# Claude GPU 검증 인수인계 — Qwen 보존 설명 B 구현

작성 2026-10-03. **Codex GPU 실행 0회, 모델 적재 0회, 설치·다운로드 0회. 커밋·푸시 전 검토용 작업본.**
사용자 승인 범위는 B만이며, 기존 Qwen 환경을 별도 프로세스로 쓴다. 이 문서는 검증 절차이고 지금 자동 실행하는 배치가 아니다. GPU 실행은 사용자와 Claude의 실행 권한에 따라 시작한다.

## 1. 코드와 자료 확인

- 구현 안내: docs/QWEN_POSE_EDIT.md
- CPU 보고: outputs/qwen-pose-implementation-20261003/results.md, cpu-evidence.json, changed-files.sha256.json
- 실제 시험: outputs/qwen-pose-identity-20261003/{run-plan.json,preservation.json,prompts.json,cases/<char>/B/run.json}
- 검증 대상은 B 두 건뿐이다. C·다른 seed·다른 오프로드로 자동 확대하지 않는다.
- 패키지 설치·모델 다운로드·운영 환경 변경 없음. 버전이 다르면 중단하고 보고한다.

## 2. 설정 준비 — CPU만

저장소 루트에서 PowerShell로 실행한다. 아래 경로는 이 호스트의 **기존 시험 기록에서 가져온 예**다. 제품 코드의 기본값이 아니다. 드라이브가 다르면 실제 기존 캐시를 확인해 변수만 바꾼다.

```powershell
$qwenPython = 'G:\genai-cache\qwen-test-venv\Scripts\python.exe'
$qwenModel = 'G:\genai-cache\models\Qwen-Image-Edit-2511'
$qwenGguf = 'G:\genai-cache\models\Qwen-Image-Edit-2511-GGUF\qwen-image-edit-2511-Q4_K_M.gguf'
$lockDir = 'outputs/qwen-pose-gpu-verify-20261003/runtime'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $qwenPython -m scripts.prepare_qwen_pose_config --python $qwenPython --model-root $qwenModel --gguf $qwenGguf --output $lockDir
```

기존 시험 캐시인지 확인한 후 runtime.json과 model-files.json을 검토한다. 패키지 정확한 기대값은 torch2.9.1+cu128 / diffusers0.38.0 / transformers4.57.6 / bitsandbytes0.50.2 / gguf0.19.0이다. 로컬 잠금 생성은 원격 revision 검증을 대신하지 않는다.

새 결과 폴더만 사용한다. 생성하지 않는 요청 준비:

```powershell
$runtime = "$lockDir/runtime.json"
& $qwenPython -m scripts.qwen_pose_replay --runtime $runtime --character raccoon --output outputs/qwen-pose-gpu-verify-20261003/preflight-raccoon
& $qwenPython -m scripts.qwen_pose_replay --runtime $runtime --character ordinary-female --output outputs/qwen-pose-gpu-verify-20261003/preflight-ordinary
```

prepare_replay는 기존 사용자 확인 명세와 선택 시점·손 문장을 재현한다. 기준 이미지·골격 SHA 및 B 지시문 전문 일치를 확인하며, --execute 없이는 worker를 띄우지 않는다. 시험 분석 기록을 재사용하는 CLI이므로 새 자동 분석 품질의 검증은 아니다.

## 3. GPU 재현 — 각 1회, 순차 실행

다른 생성·분석 작업을 종료한 뒤 실행한다. 앞 명령이 실패하면 다음 명령을 실행하지 않는다. 오류·SHA 불일치 시 자동 재시도나 설정 조정 없이 원인 기록 후 멈춘다.

```powershell
& $qwenPython -m scripts.qwen_pose_replay --runtime $runtime --character raccoon --output outputs/qwen-pose-gpu-verify-20261003/raccoon-B --execute
# 종료 코드와 아래 기준을 확인한 뒤에만 다음 실행
& $qwenPython -m scripts.qwen_pose_replay --runtime $runtime --character ordinary-female --output outputs/qwen-pose-gpu-verify-20261003/ordinary-B --execute
```

| 캐릭터 | 비교할 기준 이미지 SHA | B 원시 결과 기대 SHA |
|---|---|---|
| raccoon | 2c2a47a5aaa3ab052db113eb5f8e382e34fdd985e1ca34d383f2dcbc68a794fc | c4df7f8f35e5276ad746f80dca05e58bb98eb19a98c4483e8ff04c1e4205580d |
| ordinary-female | 02cdab4694832442007c7c0c72969da97daff701162cbdb85f59ea581367c150 | b995c095a3b58602de5c735ddf2210671ae5016c608eb0edff21aa2484fda55a |

골격 원본 파일: outputs/pose-norm-20260928/control_POCKET_before.png
SHA: `929ec96738ffe6458ef2ffba32652d65a7e25f744733bb08d85ca66cd3574146`.
CPU helper 재저장본은 픽셀이 같아도 PNG 파일 SHA가 다르므로, 이 재현에는 위 **기존 파일**을 그대로 쓴다.

긍정 전문 SHA:
- raccoon: `71d705ea9ed72728b43274c15c37b9dba54160259dde7fe655e711a435d0530d`
- ordinary-female: `048c759cae0e2e4724bf59acc6df8269a75c8f95cbebbf11deb20ba17ed966fb`

검증 기록:
- 실제 입력 2장, C 참조 없음. VL·VAE 입력 시각화와 기존 B의 같은 단계 입력 비교.
- 명시 출력800×1312, 40단계, seed209212001, true_cfg4/guidance1, negative는 공백1개.
- group_block1 고정, zero_cond_t=True, TE 인코딩 후 해제.
- embedding 길이 검사: 설치 pipeline의 고정 system prefix를 뺀 전체 길이와 일치하는지. 불일치면 임의 절단하거나 검사를 우회하지 않는다.
- raw SHA, scheduler class/config, torch/diffusers/transformers 실제 버전 및 로드 위치.
- product736×1232: 내용736×1207, 위12/아래13 흰 여백. raw 불변.
- CPU 단위 테스트는 실제 GPU 메모리 해제를 증명하지 않는다. GUI에서 생성 자원 해제→분석 해제→사용자 확인→Qwen 순서와 잔여 allocated/reserved 차단도 확인한다.
- allocated/reserved와 시간은 run.json 기록. 전용·공유 메모리 표본은 현재 미구현으로 null이 정상이며, 필요하면 Claude 측 외부 측정값을 별도 파일에 기록한다.

## 4. 사용자 화면 확인 — 추가 생성 불필요

기존 완료 결과와 CPU 가짜 worker 테스트를 활용한다. 미확인·제외 항목이 전달되지 않는지, 분석 실패 시 직접 입력 가능 여부, 영어 문장 수정 후 실제 prompt 확인, 시점·손 기본값 공백, 명세·기준 이미지 변경 차단을 본다.

새 분석의 의상 상세·전체 머리 태그는 CPU WD를 사용한다. 눈·머리의 국소 마스크, 부위 색·장신구 저장 근거가 없으면 미확정이므로 사용자가 채운다. 사용자 검토표에서 전체 항목을 채워야 채택/미채택을 기록한다. 자동 판정이나 원본 덮어쓰기 없음.

## 5. 보고·중단

결과는 outputs/qwen-pose-gpu-verify-20261003/results.md에 저장한다. 각 B의 SHA 일치, 입력·프롬프트·호출값, 크기 변환, GPU 해제·메모리, 오류·미측정 항목을 적는다. 원 시험의 모든 항목 동시 충족0/6 한계는 그대로다.

사용자가 GPU 검증 결과를 확인하기 전에는 푸시하지 않는다. 버전 불일치·필수 캐시 누락이면 설치로 넘어가지 않고 별도 승인을 요청한다. 본 문서는 패키지 설치·새 모델 다운로드를 승인하지 않는다.
