"""그림체 LoRA 선택지(D-159, 그림체 단계 맛보기 — 시험 전용, 제품 기본값·사용자 화면에 없음).

docs/STYLE_STAGE_DESIGN.md 3-3: 직접 학습 전에 공개된 Qwen 그림체 LoRA 하나를 빌려 효과 크기·동일성 섞임·기술 동작을 본다.
빌린 LoRA는 학습 자료가 공개돼 있지 않아 완성 기준 4(허가 자료로 학습)를 증명할 수 없다 → 제품 경로·설정에 연결하지 않는다.
SHA는 사용자 승인으로 내려받은 뒤 기록했다(2026-10-23, 저장소 LFS SHA와 일치). SHA가 없으면 쓸 수 없다(check_file이 거부).
"""
from pathlib import Path

STYLE_LORAS = {
    "anime_borrowed": dict(
        repo="prithivMLmods/Qwen-Image-Edit-2511-Anime", revision="977f8c2ed22c8581c8750370bfce34058b253598",
        folder="style-loras/Qwen-Image-Edit-2511-Anime", weight_name="Qwen-Image-Edit-2511-Anime-2000.safetensors",
        size=236117056, sha256="dc84f7662ed724ebfb9b8c18967eff0eabd5eaf17c2d206e2e09062db9ff8df9", license="apache-2.0", base_model="Qwen/Qwen-Image-Edit-2511",
        trigger="Transform into anime.", trial=True,
        note="맛보기 전용(S0.5). 학습 자료 미공개('ModelScope에서 학습'만 표기) — 제품에 쓰지 않는다"),
}
STRENGTH_RANGE = (0.0, 1.5)


def style_lora(name):
    """이름 → 명세. 빈 이름은 None(그림체 LoRA 없음 = 제품 기본)."""
    if not name:
        return None
    if name not in STYLE_LORAS:
        raise ValueError("알 수 없는 그림체 LoRA")
    return STYLE_LORAS[name]


def lora_path(name, model_root):
    """기준 모델 폴더와 같은 models 폴더 아래."""
    return Path(model_root).parent / style_lora(name)["folder"] / style_lora(name)["weight_name"]


def check_strength(strength):
    if type(strength) not in (int, float) or not STRENGTH_RANGE[0] < float(strength) <= STRENGTH_RANGE[1]:
        raise ValueError("그림체 LoRA 강도 오류(0 초과 1.5 이하)")
    return float(strength)


def check_file(name, model_root):
    """크기와 기록된 SHA가 있는지 확인(SHA 대조 자체는 작업 프로세스가 불러오기 직전에 한다)."""
    spec = style_lora(name)
    path = lora_path(name, model_root)
    if not spec["sha256"]:
        raise ValueError("그림체 LoRA SHA가 아직 기록되지 않았습니다(내려받은 뒤 기록).")
    if not path.is_file() or path.stat().st_size != spec["size"]:
        raise ValueError(f"그림체 LoRA 파일이 없거나 크기가 다릅니다: {path}")
    return path
