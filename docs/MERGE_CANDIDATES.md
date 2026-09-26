# SDXL 병합 후보 라이선스 조사 — ①-2

이 문서는 라이선스 조사 기록이며 법률 자문이 아니다.
배포 형태(원본 각자 내려받기 / 병합 가중치 호스팅)에 따라
필요한 권리가 다르므로 Q1·Q2를 구분해 읽는다.
확인일 이후 라이선스가 변경될 수 있다.

확인일: **2026-09-24 (KST)**. 조사 대상은 아래 버전·파일로 고정했다. 모델 가중치 다운로드, 병합, 이미지 생성은 모두 **0회**다. 원격에서 읽은 것은 카드·약관·JSON 설정·파일 크기 메타데이터뿐이다.

## 판정 범위

- **Q1**: 사용자가 자신의 PC에서 병합해 사용하는 행위. 완전 비공개 사용의 허용 여부가 불명인 경우 따로 표시한다.
- **Q2**: Animagine XL 3.1과 병합한 가중치를 무료·비상업 목적으로 직접 호스팅·배포하는 행위. 무료라는 이유로 공지·공개·사용 제한 의무가 사라지지 않는다.
- 판단 칸은 `허용 / 불허 / 조건부 / 미확인`으로 작성했다. 조건부의 조건은 아래 각주에 있다. 비상업 조건 칸의 `허용`은 **비상업 사용/배포가 허용됨**을 뜻하며, 비상업만 허용한다는 뜻이 아니다.
- SDXL 계열 여부는 카드와 공개 설정을 대조했다. **가중치의 전체 키·shape 일치, 실제 병합 성공 및 화풍 효과는 미확인**이다. 다운로드 전 후보 적합성 확인과 병합 실증을 혼동하지 않는다.
- 모든 후보의 학습 이미지별 권리·동의 및 특정 생존 작가 포함 여부는 미확인이다. 이것만으로 제외하지 않았다. 모델명·카드에서 특정 생존 작가 개인의 화풍을 학습 대상으로 내세운 것으로 확인된 후보는 없었다. 이를 “생존 작가 작품을 학습하지 않았다”로 해석하지 않는다.
- 개발자·논문 저자·기여자 이름을 화풍 학습 대상 이름으로 단정하지 않았다. Kohaku는 다수 작가 태그의 학습을 명시하므로 특히 [ART]를 함께 읽어야 한다.

## 전체 조사표 — 병합 후보 5개

| 모델 | 저장소 | 아키텍처 | 라이선스 | Q1 로컬병합 | Q2 파생배포 | 비상업 조건 | 사용제한 전달의무 | 작가화풍 기반 | 출처 | 확인일 |
|---|---|---|---|---|---|---|---|---|---|---|
| Animagine XL 4.0 (원본판, 단축 후보) | cagliostrolab/animagine-xl-4.0 | SDXL 1.0 계열·4채널 [ARCH] | CreativeML Open RAIL++-M | 조건부 [R1] | 조건부 [R2] | 허용 [R3] | 조건부 [R2] | 미확인 [ART-A] | [카드 A4][A4]·[약관 R][R] | 2026-09-24 |
| Animagine XL 3.0 (단축 후보) | cagliostrolab/animagine-xl-3.0 | SDXL 1.0 계열·4채널 [ARCH] | CreativeML Open RAIL++-M | 조건부 [R1] | 조건부 [R2] | 허용 [R3] | 조건부 [R2] | 미확인 [ART-A] | [카드 A3][A3]·[약관 R][R] | 2026-09-24 |
| Kohaku XL Delta rev1 (단축 후보) | KBlueLeaf/Kohaku-XL-Delta | SDXL 1.0 계열·4채널 [ARCH] | Fair AI Public License 1.0-SD | 조건부 [F1] | 조건부 [F2]+[R2] | 허용 [F3] | 조건부 [F2]+[R2] | 미확인 [ART-K] | [카드 K][K]·[동봉 약관 F][F] | 2026-09-24 |
| Illustrious XL v0.1-GUIDED (추가 확인 후보) | OnomaAIResearch/Illustrious-xl-early-release-v0 | SDXL 1.0 계열·4채널 [ARCH] | FAIPL 1.0-SD + 카드 정책 + TERM_OF_USE | 미확인 [I1] | 조건부 [I2]+[F2]+[R2] | 조건부 [I2] | 조건부 [I2]+[F2]+[R2] | 미확인 [ART-I] | [카드 I][I]·[약관 F][F0]·[이용약관 IT][IT] | 2026-09-24 |
| NoobAI XL 1.1 (추가 확인 후보) | Laxhar/noobai-XL-1.1 | SDXL 1.0 계열·4채널·epsilon [ARCH] | FAIPL 1.0-SD + 카드의 비상업·공개 추가 조항 | 미확인 [N1] | 미확인 [N2] | 조건부 [N3] | 조건부 [N2]+[F2]+[R2] | 미확인 [ART-N] | [카드 N][N]·[상위 I][I]·[약관 F][F0] | 2026-09-24 |

### 기준 모델 — 동일한 열로 비교

| 모델 | 저장소 | 아키텍처 | 라이선스 | Q1 로컬병합 | Q2 파생배포 | 비상업 조건 | 사용제한 전달의무 | 작가화풍 기반 | 출처 | 확인일 |
|---|---|---|---|---|---|---|---|---|---|---|
| Animagine XL 3.1 (기준·로컬 캐시 있음) | cagliostrolab/animagine-xl-3.1 | SDXL 1.0 계열·4채널 [ARCH] | CreativeML Open RAIL++-M | 조건부 [R1] | 조건부 [R2] | 허용 [R3] | 조건부 [R2] | 미확인 [ART-A] | [카드 A31][A31]·[약관 R][R]·로컬 스냅샷 [CACHE] | 2026-09-24 |

### 조건과 짧은 원문 근거

각 인용은 15단어 이하이다. 같은 약관의 공통 문구는 여기에서 한 번만 인용한다.

**[R1] Open RAIL++-M — Q1.** §2의 수정·파생 권리와 §5의 사용 범위를 함께 적용한다. 원문 §5: “finetuning, updating, running, training, evaluating and/or reparametrizing the Model.” (9단어). 로컬 병합도 사용 제한(Attachment A)을 준수해야 한다. [원문][R]

**[R2] Open RAIL++-M — Q2.** §4: “reproduce and distribute copies of the Model or Derivatives of the Model” (12단어). 라이선스 사본, 변경 표시, 관련 권리 공지를 유지하고 §5 사용 제한을 수령인과의 구속력 있는 약정에 포함·고지해야 한다. §4는 다른 파생 라이선스도 허용하므로 **이 약관 자체에 동일 라이선스로 가중치를 공개해야 하는 share-alike 의무는 없다**. 상대 병합 모델의 별도 의무는 추가로 적용된다. [원문][R]

**[R3]** 비상업만으로 한정된 약관은 아니다. 현재 Animagine 3.0·3.1·4.0 카드도 수정·배포·개인 및 상업 사용을 허용한다고 명시한다. 과거 카드의 FAIPL 표기를 현재 버전에 적용하지 않았다. 이는 기존에 취득한 다른 리비전의 권리 관계까지 소급 판정한 것은 아니다. [A3][A3] / [A31][A31] / [A4][A4]

**[F1] FAIPL — Q1.** Definitions는 수정에 병합을 포함한다: “to combine a model with another model.” (7단어). Copyright 권한과 Prohibited Uses를 함께 적용한다. 외부 배포나 네트워크 제공이 없는 로컬 이용과 Notices가 발동하는 배포·네트워크 이용을 구분한다. [동봉 약관][F]

**[F2] FAIPL — Q2.** Notices: “also gets the text of this license along with the corresponding source code.” (13단어). 모델도 source code 정의에 포함된다. 수령인에게 약관과 대응 소스를 제공하고, 수정물에는 이 약관 또는 이 약관이 허용하는 모든 것을 허용하는 약관을 적용해야 한다. 수정 모델을 네트워크로 제공할 때에도 소스 제공 의무가 있다. Animagine 병합분에는 [R2]도 유지한다. 앱 전체의 공개 의무 범위는 앱과 모델의 결합 방식이 정해지지 않아 별도 미확인이다. [동봉 약관][F] / [발행자 원문][F0]

**[F3]** FAIPL 자체에는 비상업 한정이 없다. **내가 무료로 비상업 배포하는 것**과 **받는 사람에게 새로운 NC 제한을 부과하는 것**은 다르다. 후자는 FAIPL의 권한 유지 조항과 맞는지 별도 확인해야 하며, 여기서는 허용으로 확정하지 않는다.

**[I1] Illustrious — Q1의 미확인.** Training/Merging Policy: “You may fine-tune, merge, or train LoRA based on this model.” (11단어). 그러나 카드가 공개·레시피 제공을 함께 요구하며 비배포 개인 병합의 예외를 명확히 쓰지 않았다. 따라서 **아무에게도 공개하지 않고 내 PC에서만 쓰는 병합**은 확정하지 않았다. [카드][I]

**[I2] Illustrious — Q2의 조건.** 카드: “you must openly publish any derivative models and variants.” (9단어). 원 모델 참조, 데이터·병합 레시피 정보, 파생 모델 공개, FAIPL 및 이용약관을 준수하는 무료 공개 배포를 조건부로 분류했다. 카드의 폐쇄형 파생 모델 수익화 금지와 일반 수익화 비권장을 모든 비상업 배포의 금지로 확대하지 않았다. BASE의 research-only 설명과 GUIDED를 구별해 파일은 GUIDED로 지정했다. 제품별 이용약관 적용 범위는 여전히 확인 대상이다. [카드][I] / [TERM_OF_USE][IT]

**[N1·N2·N3] NoobAI — 단일 라이선스 이름만으로 판정할 수 없음.** 카드 §II: “We prohibit any form of commercialization” (6단어). §III: “Open source derivative models, merged models, LoRAs, and products.” (9단어). 파생물·제품 공개와 작업 세부 공유를 요구한다. 개인 비배포 병합에 이 공개 요구가 어디까지 적용되는지, 추가 NC 제한이 상속한 FAIPL의 권한 유지 조항과 어떻게 양립하는지는 미확인이다. 따라서 Q1·Q2 모두 미확인으로 보존했다. **NC 문구가 없어서 미확인인 것이 아니라, 실제로 추가 문구가 있어 그 결합 해석이 미확인**이다. 상업 사용을 허용한다고 해석하지 않는다. [카드][N]

### 설치형 앱의 두 배포 방식

여기서 원본 각자 다운로드는 사용자가 원 저작자 저장소에서 직접 받는 방식이다. 개발자 프록시·미러·설치 파일에 모델을 포함하는 경우는 이 전제와 다르다. 앱을 통해 사용자에게 사용하게 하는 경우의 약관 고지 등까지 없어지는 것은 아니다.

| 모델 | 원본 직접 다운로드 후 사용자 PC에서 병합 | 개발자가 병합 가중치 호스팅 | 차이 |
|---|---|---|---|
| Animagine XL 4.0 | 조건부 [R1] | 조건부 [R2] | 호스팅에는 사본·변경·공지·사용 제한 전달 추가 |
| Animagine XL 3.0 | 조건부 [R1] | 조건부 [R2] | 같은 구분 |
| Kohaku XL Delta rev1 | 조건부 [F1]+[R1] | 조건부 [F2]+[R2] | 배포 시 대응 소스·파생물 라이선스 의무 추가 |
| Illustrious XL v0.1-GUIDED | 미확인 [I1] | 조건부 [I2]+[F2]+[R2] | 비공개 로컬 예외 불명, 공개 배포 조건은 카드에 명시 |
| NoobAI XL 1.1 | 미확인 [N1] | 미확인 [N2] | 직접 다운로드만으로 추가 공개·NC 조건이 해소되지 않음 |
| Animagine XL 3.1 (기준) | 조건부 [R1] | 조건부 [R2] | 기준 모델에도 동일한 구분 적용 |

**Q1·Q2 판정값이 갈리는 것은 Illustrious**다. 다른 후보도 값이 같다고 의무가 같은 것은 아니다. FAIPL 모델의 재배포 의무를 Open RAIL++-M만으로 대체하지 않는다. 배포 방식은 이번에 선택하지 않았다.

## 단축 후보 확정 — 3개, 품질 순위 아님

1. **Animagine XL 4.0 원본판** — Open RAIL++-M의 병합·파생 배포 조건이 명시되어 있다. 카드는 애니 계열 SDXL 1.0 재학습, 연도별 스타일 태그를 설명한다. `animagine-xl-4.0.safetensors`로 고정하며 Opt 판과 섞지 않는다. [카드][A4]
2. **Animagine XL 3.0** — 같은 Open RAIL++-M 조건을 확인했다. 카드는 현대/빈티지 애니 표현을 위한 연대 태그를 명시한다. `animagine-xl-3.0.safetensors`로 고정한다. 기존 기준과 계열이 가까우므로 “화풍 변화가 클 것”이라고 예측하지 않는다. [카드][A3]
3. **Kohaku XL Delta rev1** — 병합을 명시적으로 다루는 FAIPL과 재배포 공개 조건을 확인했다. 카드는 SDXL 애니 모델 및 연대·다수 작가 태그를 설명한다. `kohaku-xl-delta-rev1.safetensors`를 선택하고 학습 전 `delta-base`를 선택하지 않는다. 다작가 학습 표시는 윤리 검토 항목으로 유지한다. [카드][K]

Illustrious·NoobAI는 조사표에서 삭제하지 않았다. 명시된 추가 조건에 대한 확인이 남아 이번 다운로드 단축 목록에 넣지 않았다. 이것은 품질 탈락 판정이 아니다. 세 단축 후보도 데이터 권리의 완전 검증이나 화풍 개선 성공을 뜻하지 않는다.

## 학습 자료와 작가 기준

| 표시 | 확인한 내용 | 남은 미확인 |
|---|---|---|
| [ART-A] Animagine 3.0·3.1·4.0 | 애니 데이터·개념/연대 태그 학습을 카드가 설명 | 전체 원본 목록, 작가별 동의·권리, 생존 작가 포함 여부 |
| [ART-K] Kohaku Delta | Danbooru2023 표본과 다수 작가 태그 학습을 명시; 특정 한 사람을 목표로 삼은 모델로는 표방하지 않음 | 개별 작가·생존 여부·동의. 카드의 KBlueLeaf 표기는 제작자/샘플 저자이므로 그것만으로 특정 타인 화풍 학습의 증거로 쓰지 않음 |
| [ART-I] Illustrious | Danbooru2023 기반 일러스트 모델, 다양한 스타일을 설명 | 작가별 목록·동의·학습 권리 |
| [ART-N] NoobAI | Danbooru와 e621 학습 및 artist 태그 형식 명시 | 작가별 목록·동의·학습 권리 |

이 표는 원 자료의 합법성을 보증하지 않는다. 카드에 데이터 출처가 있어도 개별 이미지의 학습 동의까지 확인된 것은 아니다. 사용자 방침에 따라 이 미확인 사항을 표시했으며, 미확인만을 이유로 모델을 임의 제외하지 않았다.

## 로컬 캐시를 먼저 확인한 결과 [CACHE]

검사 루트: `D:/genai-cache/huggingface` (하위 `hub` 포함).

- Animagine XL 3.1 스냅샷: `models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b/`.
- 로컬 `model_index.json`과 `unet/config.json`에서 SDXL·4채널 구조를 확인했다. 그러나 해당 스냅샷에는 README·LICENSE가 없었다. 따라서 라이선스의 1차 근거는 같은 리비전의 원격 카드와 카드가 직접 연결한 약관이다.
- 나머지 단축 후보 3개 및 Illustrious·NoobAI의 캐시 디렉터리는 없었다.
- 캐시 전체의 스냅샷에서 발견한 README/라이선스 이름 파일은 CatVTON README뿐이었다. 의상 합성 모델은 이번 일반 애니 SDXL 체크포인트 병합 후보가 아니므로 별도 라이선스 판정 대상으로 삼지 않았다.
- 캐시 존재만으로 사용할 수 있는 완전한 체크포인트가 다운로드돼 있다고 판정하지 않았다.

### 캐시에서 범위 밖으로 분류한 생성 모델 — 기록 유지

| 모델/자산 | 로컬 확인 근거 | 이번 후보에서 제외하는 기준 | 라이선스 판정 |
|---|---|---|---|
| FLUX.2 Klein 4B | `models--black-forest-labs--FLUX.2-klein-4B/.../model_index.json`: Flux2KleinPipeline | SDXL 1.0 아키텍처 아님 | 미확인 — 이 조사에서 약관 판정 생략 |
| Stable Diffusion v1.5 | `models--stable-diffusion-v1-5--stable-diffusion-v1-5/.../model_index.json`: StableDiffusionPipeline | SD 1.5로 아키텍처 불일치 | 미확인 — 이 조사에서 약관 판정 생략 |
| SDXL inpainting 0.1 | `models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/.../unet/config.json`: in_channels=9 | 일반 애니 화풍 체크포인트가 아니며 기준 4채널과 입력층 불일치 | 미확인 — 이 조사에서 약관 판정 생략 |

## 아키텍처 확인 [ARCH]

6개 조사 모델의 공개 `unet/config.json`을 읽었다. 공통 값은 `UNet2DConditionModel`, `in_channels=4`, `cross_attention_dim=2048`, `block_out_channels=[320,640,1280]`, `addition_embed_type=text_time`이다. 3.1·4.0·Kohaku·Illustrious·NoobAI의 공개 scheduler는 `prediction_type=epsilon`이었다. NoobAI의 다른 V-prediction 버전으로 일반화하지 않는다.

각 저장소의 `/raw/<아래 리비전>/unet/config.json` 및 `scheduler/scheduler_config.json`이 확인 경로다. 이는 설정 수준의 구조 확인이며, 단일 safetensors 파일의 모든 텐서를 대조한 결과가 아니다. 특히 GUIDED 등 파일 변형의 실제 텐서는 다음 단계 확인 대상이다.

## 디스크 요구량

파일 크기는 Hugging Face `https://huggingface.co/api/models/<저장소>?blobs=true`의 `siblings.size` 값이다. 가중치 본문은 요청하지 않았다. 각 저장소 전체가 아니라 **아래 단일 체크포인트 한 개씩**의 용량이다.

| 모델 | 고정 파일 | bytes | 대략 GB (10^9) | 대략 GiB (2^30) | 합계 구분 |
|---|---|---|---|---|---|
| Animagine XL 4.0 | animagine-xl-4.0.safetensors | 6,938,434,056 | 6.938 | 6.462 | 단축 후보 |
| Animagine XL 3.0 | animagine-xl-3.0.safetensors | 6,938,218,610 | 6.938 | 6.462 | 단축 후보 |
| Kohaku XL Delta rev1 | kohaku-xl-delta-rev1.safetensors | 6,938,040,286 | 6.938 | 6.462 | 단축 후보 |
| Illustrious v0.1-GUIDED | Illustrious-XL-v0.1-GUIDED.safetensors | 6,938,040,286 | 6.938 | 6.462 | 추가 확인 후보 |
| NoobAI XL 1.1 | NoobAI-XL-v1.1.safetensors | 7,105,349,958 | 7.105 | 6.617 | 추가 확인 후보 |
| Animagine XL 3.1 | animagine-xl-3.1.safetensors | 6,938,325,776 | 6.938 | 6.462 | 기준 모델, 신규 후보 합계 제외 |

- **단축 후보 3개 합계: 20,814,692,952 bytes ≈ 20.815 GB / 19.385 GiB.**
- 조사한 후보 5개 전부: **34,858,083,196 bytes ≈ 34.858 GB / 32.464 GiB.**
- `D:/genai-cache`가 있는 D: 가용 공간: **706,875,133,952 bytes ≈ 706.875 GB / 658.329 GiB**. Windows `Get-Volume -DriveLetter D`로 확인했다. 확인 이후 다른 작업으로 달라질 수 있다.
- 위 합계는 병합 중 임시 파일·병합 결과·Diffusers 변환본·캐시의 별도 복사본을 포함하지 않는다. 단일 FP16 병합 결과를 하나 더 보존한다면 대략 7 GB가 추가될 수 있으나 실제 병합 출력 크기는 미측정이다.
- 기존 Diffusers 형식 3.1 캐시가 있다고 단일 파일 체크포인트까지 캐시돼 있다고 가정하지 않았다. 필요한 저장 형식은 다음 다운로드 과제에서 선택한다.

## 근거 리비전과 문서 해시

라이선스가 바뀌더라도 이번에 무엇을 읽었는지 추적하도록 카드 리비전과 UTF-8 원문 SHA-256을 기록한다. 원문 전체를 저장한 별도 파일은 만들지 않았다. 아래 링크는 고정 리비전이다.

| 근거 | 저장소 리비전 | 읽은 파일 | 원문 SHA-256 |
|---|---|---|---|
| [A31][A31] | 483f0c322568ed13697ed01dd0be07204746d12b | README.md | 67dce7c2f0fb0503be0e79faf1008ebed30f853d654c53e637610e908fbee088 |
| [A4][A4] | 2b7c1b397761bf5bd3cc42e5b39ec99314a75a96 | README.md | 33c0532ffd945653223eea317756976a9af52faa2d6122b860dc56fe62e4cdc9 |
| [A3][A3] | d02fc0b0b01a6fbbe9bfca4db176c1f065663b2a | README.md | fc9cea0614be807b7a71c44411999a55bb473644c17cdbc444ce0dd19ea331d5 |
| [K][K] | ba7c1aeabc3cc6b3b01fd9688d6554d5940c0854 | README.md | d698f1d5ec9b532a37857d137ab2b09668ff567537da451edf4a42b77fd847e9 |
| [I][I] | dca0dac303e6dc4b0c31d8001bc685b89b5d0204 | README.md | 0205e7c1d93c5c4fd6a2045ebf9ae3cd333684074385130d5b9e5aef3e051b45 |
| [N][N] | 814a274af2b8097c0828819d561ec74c7d0c6cea | README.md | cf21e79ff853c46efa1dcfd809940caf7d21db0c45eefc05d67ddb8a891a9635 |
| [R][R] | 462165984030d82259a11f4367a4eed129e94a7b | LICENSE.md | 19b6998b569b53ac1fc2158a8a3202c8699a9a4605b47075715d9c96be7fb6d0 |

## 미확인 항목 — 이유 구분

| 항목 | 미확인 이유 | 조사에서 처리한 방법 |
|---|---|---|
| 학습 이미지별 권리·작가 동의·생존 여부 | 카드에 완전한 검증 자료가 없음 | 모든 후보에 표시, 이것만으로 제외하지 않음 |
| Illustrious의 비공개 개인 병합 | 카드 공개 요구에 개인 비배포 예외가 명확하지 않음 | Q1 미확인; 공개 배포 조건과 분리 |
| NoobAI의 비공개 개인 병합 및 라이선스 결합 | 추가 공개·NC 요구와 상속 FAIPL의 관계가 명확하지 않음 | Q1·Q2 미확인, 임의로 추가 조건 무시하지 않음 |
| FAIPL과 설치형 앱 코드 공개 범위 | 앱과 모델의 결합·배포 방식이 아직 결정되지 않음 | 모델 재배포 의무를 앱 전체 의무로 단정하지 않음 |
| 가중치 전체 텐서 호환성과 병합 후 품질 | 가중치 다운로드·병합·생성이 이번 범위 밖 | 공개 구조만 확인, 성공·품질 보장 안 함 |
| 후보의 로컬 LICENSE/README | 3.1 스냅샷에 없고 나머지는 캐시 없음 | 원격 카드가 연결한 약관으로 대조 |
| Animagine/NoobAI의 임의 LICENSE URL | 해당 이름 경로는 웹 도구에서 접근 실패; API 파일 목록에도 없음 | 접근 실패를 라이선스 불명으로 대체하지 않고 카드의 실제 라이선스 링크·본문 확인 |
| SD1.5·FLUX·전용 inpainting의 라이선스 | 아키텍처/용도 기준으로 범위 밖이라 추가 조사 생략 | 제외 자산 표에 미확인 및 이유 유지 |

핵심 카드·연결 약관은 원격 접근에 성공했다. 위에서 카드 자체의 내용 부족과 잘못된 파일 경로의 접근 실패를 구별했다.

## 변경 범위와 백로그

`docs/BACKLOG.md`는 존재하지 않아 항목 13 연결을 건너뛰었다. 백로그를 새로 만들지 않았다.

이번 작업은 **이 문서 1개 추가**다. 시작 시 실행 출처 기록 작업의 미커밋 소스 8개·신규 수집기/테스트 및 기존 미추적 경로가 이미 있었다. 그것들을 되돌리거나 이번 조사 변경으로 포함하지 않았다. 전체 git status가 깨끗하다고 보고하지 않는다. 작업 전후 기존 추적 파일 346개의 SHA-256과 상태를 대조해 기존 파일 변경 0개를 확인한다.

## 다음 행동

사용자가 배포 방식과 단축 후보를 선택한 뒤, 별도 다운로드 과제에서 해당 리비전·체크포인트 및 필요한 라이선스 고지 방식을 고정한다.

[A31]: https://huggingface.co/cagliostrolab/animagine-xl-3.1/blob/483f0c322568ed13697ed01dd0be07204746d12b/README.md
[A4]: https://huggingface.co/cagliostrolab/animagine-xl-4.0/blob/2b7c1b397761bf5bd3cc42e5b39ec99314a75a96/README.md
[A3]: https://huggingface.co/cagliostrolab/animagine-xl-3.0/blob/d02fc0b0b01a6fbbe9bfca4db176c1f065663b2a/README.md
[K]: https://huggingface.co/KBlueLeaf/Kohaku-XL-Delta/blob/ba7c1aeabc3cc6b3b01fd9688d6554d5940c0854/README.md
[I]: https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/blob/dca0dac303e6dc4b0c31d8001bc685b89b5d0204/README.md
[N]: https://huggingface.co/Laxhar/noobai-XL-1.1/blob/814a274af2b8097c0828819d561ec74c7d0c6cea/README.md
[R]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md
[F]: https://huggingface.co/KBlueLeaf/Kohaku-XL-Delta/blob/ba7c1aeabc3cc6b3b01fd9688d6554d5940c0854/LICENSE
[F0]: https://freedevproject.org/faipl-1.0-sd/
[IT]: https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/blob/dca0dac303e6dc4b0c31d8001bc685b89b5d0204/TERM_OF_USE
