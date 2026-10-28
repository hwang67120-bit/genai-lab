# FLUX 전체 이미지 정밀화

제품 자동 경로는 다음 순서를 사용한다.

    승인된 Animagine Base
    → FLUX.2 Klein 전체 이미지 후보 생성
    → hard safety 및 유사도 진단
    → hard safety 통과 결과를 사용자에게 표시
    → 실패하면 승인 Base와 Seed 고정

FLUX는 승인 Base와 격리된 의상 보드, Seed, 편집 지시를 받는다. 이미지
병합, hard paste, crop 붙이기, 마스크 합성과 다른 생성 모델 폴백은
사용하지 않는다.

Qwen-Image-Edit-2509와 OmniGen은 GPU 비교에서 제품 목표를 충족하지 못해
자동 경로에서 제거했다. 과거 실험 근거는
docs/REFERENCE_CONSISTENCY_POSTMORTEM.md에 보존한다.

공식 자료:

- FLUX.2 Klein: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
