# 헤어스타일 보존 데이터 흐름

머리카락 분석 후보를 생성 계약으로 곧바로 승격하지 않는다. 파이프라인은
ObservedHairEvidence, HairIntent, TargetHairPlan을 분리한다.

    참조 분석 후보
    → ObservedHairEvidence (관찰, 충돌, 미분류 기록)
    → 사용자 승인 태그
    → HairIntent
    → Pass 1 구조·외형 유도
    → 출력 얼굴·머리 좌표 재검출
    → TargetHairPlan
    → 검증 가능한 국소 항목만 Pass 2

## 단계 책임

- 길이, 묶은 머리, 전체 형태는 Pass 1 책임이다. 불일치는 Base 재생성
  권고로 기록하며 Pass 2가 전체 머리를 다시 만들지 않는다.
- 앞머리와 옆머리는 Pass 1에서 생성하고, 출력 좌표로 국소화할 수 있을 때만
  Pass 2 보정 후보가 된다.
- 색상과 질감은 머리 전체 범위의 국소 보정 후보가 될 수 있다.
- 미검출은 부재가 아니다. unresolved, occluded, unobserved는 자동 생성
  조건이나 실패 판정으로 승격하지 않는다.
- 참조 좌표는 출력 좌표로 복사하지 않는다. 출력 이미지에서 얼굴과 머리를
  다시 검출한 후 대상 좌표 계획을 만든다.
- 한 실행에서 시각 조건 방식은 하나만 사용한다. 현재 기본은 중립 배경으로
  격리한 머리 크롭을 사용하는 IP-Adapter이며 Reference-Only와 중복하지 않는다.

## 연구 근거

- Stable-Hair는 Hair Extractor와 Latent IdentityNet을 분리해 머리 전달과
  비머리 정체성 보존을 별도 경로로 처리한다.
  <https://arxiv.org/abs/2407.14078>
- HairFusion은 hair-agnostic 표현, 자세 정렬 cross-attention, 적응형 머리
  블렌딩으로 참조 자세와 대상 자세 차이를 처리한다.
  <https://arxiv.org/abs/2408.16450>
- HairFIT은 키포인트 기반 정렬과 의미 영역별 인페인팅을 분리한다.
  <https://arxiv.org/abs/2206.08585>
- IPAdapter-Instruct는 하나의 참조 이미지가 스타일·대상·구조 등 여러 의미로
  해석되는 모호성을 별도 instruction으로 다룬다.
  <https://arxiv.org/abs/2408.03209>
- Paint by Example은 강한 자기 참조 학습이 복사·붙여넣기 해법으로 퇴화할 수
  있어 content bottleneck과 증강을 사용한다.
  <https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Paint_by_Example_Exemplar-Based_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf>

이 연구들은 주로 실사 인물을 대상으로 한다. 애니메이션 캐릭터, 동물 귀,
아호게에 대한 성능은 프로젝트 검증 세트로 따로 측정한다.