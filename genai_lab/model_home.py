"""모델 폴더 한 곳(설치형 앱의 '최초 모델 준비' 결과, INSTALLER_PACKAGING_PREP.md).

GENAI_MODEL_HOME을 정하면 제품 경로의 모델 파일을 그 아래 정해진 배치에서 찾는다. 정하지 않으면 지금까지의 경로를 그대로 쓴다
(개발 PC 동작은 바뀌지 않는다). 실행 환경(Python) 경로는 여기서 다루지 않는다.
배치는 2026-10-23 CPU 실행 관찰로 제품 단계가 연 파일과, 코드에서 경로가 정해지는 Qwen 파일을 기준으로 한다. 완전한 목록이라고 확정하지 않는다.
"""
import os
from pathlib import Path

ENV = "GENAI_MODEL_HOME"
L2D_SAM_SNAPSHOT = "models--24yearsold--l2d_sam_iter2/snapshots/0b0608310fb7a89e32ecd7397147a249540cf1e7/checkpoint-18000.pt"

# 모델 폴더 안의 상대 위치
LAYOUT = {
    "qwen_model_root": "qwen/Qwen-Image-Edit-2511",
    "qwen_gguf": "qwen/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_K_M.gguf",
    "model_cache": "hf",                                   # WD 태거·anime-seg·l2d_sam(허깅페이스 캐시 구조 그대로)
    "pose_models": "dwpose/checkpoints",                    # dw-ll_ucoco_384.onnx, yolox_l.onnx
    "head_cache": "head-detection/hub",                     # deepghs/anime_head_detection(허깅페이스 캐시 구조)
    "head_semantic_repo": "tools/see-through",
    "head_semantic_checkpoint": "hf/" + L2D_SAM_SNAPSHOT,
    # 모드 B(그림체로 다시 그리기, STYLE_MODE_APP_DESIGN.md): base/realvis · extra/pose · extra/ip 배치. 모드 A만 쓰면 없어도 된다.
    "style_models": "style/sdxl",
    "style_reference": "style/references/target-02.png",
    "style_reference_pose": "style/references/target-02.pose.json",
    "style_lora": "style/lora/gnlstyle-v1-step500",          # D-204 학습한 스타일 LoRA(허가 자료 = 직접 짠 문장 + FLUX.2-klein-4B)
}


def model_home():
    """GENAI_MODEL_HOME → 절대 경로 폴더. 없으면 None(기존 경로 사용)."""
    value = os.environ.get(ENV, "").strip()
    if not value:
        return None
    home = Path(value)
    if not home.is_absolute() or not home.is_dir():
        raise ValueError(f"모델 폴더({ENV})는 이미 있는 절대 경로 폴더여야 합니다: {value}")
    return home


def model_path(key, legacy):
    """모델 폴더가 정해졌으면 배치 경로, 아니면 기존 경로. 배포판(GENAI_APP_HOME)은 기존(개발) 경로로 넘어가지 않는다(D-174):
    모델 폴더가 없으면 설치 폴더 안의 없는 자리를 가리켜 검사가 '없음'으로 막게 한다."""
    home = model_home()
    if home is not None:
        return home / LAYOUT[key]
    from genai_lab.app_home import app_home
    packaged = app_home()
    if packaged is not None:
        return packaged / "모델 폴더 미설정" / LAYOUT[key]
    return Path(legacy)


def resolve_in_home(value):
    """설정 파일의 상대 경로를 모델 폴더 기준으로 푼다. 절대 경로는 그대로."""
    path = Path(value)
    if path.is_absolute():
        return path
    home = model_home()
    if home is None:
        raise ValueError(f"상대 모델 경로를 쓰려면 {ENV}가 필요합니다: {value}")
    return home / path
