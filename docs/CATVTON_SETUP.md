# CatVTON Windows 시험 환경

이 문서는 현재 PC에서 실제로 통과한 CatVTON 별도 실행 환경만 기록합니다. 메인 Animagine 가상 환경과 섞지 않습니다.

## 확인한 환경

- Windows, Python 3.10.6
- NVIDIA RTX 4060 8GB
- PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121
- Diffusers 0.35.2, Transformers 4.46.3
- NumPy 1.26.4, Pillow 10.3.0
- Matplotlib 3.10.8, truststore 0.10.4
- PEFT 미설치: 기본 CatVTON은 사용하지 않으며 공식 고정 Accelerate와 충돌함

## 공식 설치 파일을 그대로 쓰지 못한 이유

- 공식 README는 Python 3.9를 안내하지만 최신 개발판 Diffusers는 Python 3.10 이상을 요구했습니다.
- 최신 개발판 Diffusers와 고정된 Transformers가 서로 다른 Hugging Face Hub 버전을 요구했습니다.
- PyTorch 2.4 Windows 패키지는 이 PC에서 fbgemm.dll 의존성 오류가 발생했습니다.
- Matplotlib 3.9.1은 Windows 실행 파일 대신 소스 빌드를 시도해 C++ 컴파일러 오류가 발생했습니다.

따라서 CatVTON 코드는 바꾸지 않고 실행 환경만 위의 확인된 버전으로 고정했습니다.

## 실제 확인 결과

공식 사람 이미지와 상의 이미지로 작은 연결 시험과 실제 GUI 설정 시험을 각각 실행했습니다.

- RTX 4060 CUDA 인식: 통과
- SCHP와 DensePose 자동 마스크: 통과
- CatVTON 이미지 생성: 통과
- 의상 변경 허용 영역 밖 픽셀: 0
- 보호 합성 결과 크기: 768x1024
- 프로젝트 outputs 자동 저장: 없음

- 작은 연결 시험: 384x512, 10단계
- 실제 GUI 설정 시험: 576x1024, 30단계
- 실제 설정의 CatVTON 실행과 보호 합성 시간: 58.7초

## 입력 한계 시험

다음 입력도 모델은 오류 없이 이미지를 생성했습니다.

- 상의 이미지를 전신 의상으로 잘못 지정
- 의상 이미지의 오른쪽 일부를 잘라서 입력

CatVTON은 부족한 의상 범위나 앞·뒤 정보를 오류로 판정하지 않고 추측합니다. 따라서 다음 입력 확인 기능이 구현되기 전에는 결과를 사용자가 직접 검토해야 합니다.

- 선택한 의상 종류와 이미지 범위 일치
- 의상 전체가 잘리지 않고 보이는지 확인
- 앞면, 뒷면 또는 양면 정보 구분
- 목표 자세가 뒷모습일 때 뒷면 참조 존재 확인

## 공식 자료

- https://github.com/Zheng-Chong/CatVTON
- https://github.com/Zheng-Chong/CatVTON/blob/main/app.py
- https://github.com/Zheng-Chong/CatVTON/blob/main/model/cloth_masker.py
