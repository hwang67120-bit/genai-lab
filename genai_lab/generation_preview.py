"""생성 과정 미리보기(2026-10-22 사용자 "결과 미리보기에는 이미지가 생성되는 과정만"): 단계마다 잠재값을 VAE 없이 대략 색으로 바꾼다.

변환은 우리 VAE(Qwen-Image-Edit-2511/vae)로 그림 30장을 인코딩해 맞춘 16×3 선형식 + 치우침(configs/latent_rgb_preview.json,
맞춤에 안 쓴 그림 상관 0.98). 잠재값은 파이프라인이 단계마다 다루는 정규화 공간((z-mean)/std, 2×2 묶음)을 그대로 받는다.
결과 그림을 바꾸지 않는다(콜백은 읽기만). 실패해도 생성은 계속한다 — 미리보기는 보조다.
파일은 단계마다 새 이름(preview-012.png)으로 쓴다. 같은 이름을 바꿔 쓰면 작업실이 읽는 순간 Windows가 교체를 거부했다
(요청 34: 9회 중 6회 'WinError 5 접근 거부'로 미리보기가 멈춤, D-157).
"""
import json
import os
from pathlib import Path

import numpy as np

FACTORS = Path(__file__).resolve().parents[1] / "configs" / "latent_rgb_preview.json"
PREVIEW_PATTERN = "preview-*.png"
KEEP = 3                        # 마지막 몇 장만 남긴다(지우기 실패는 무시 — 작업실이 읽는 중일 수 있다)


def load_factors(path=FACTORS):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    factors, bias = np.asarray(data["factors"], np.float32), np.asarray(data["bias"], np.float32)
    if factors.shape != (16, 3) or bias.shape != (3,):
        raise ValueError("미리보기 변환식 형식 오류")
    return factors, bias


def unpack(latents, height, width, vae_scale_factor=8):
    """(1, 묶음 수, 64) → (16, h, w). diffusers QwenImage `_unpack_latents`와 같은 배치(첫 장만)."""
    array = np.asarray(latents, np.float32)
    _, patches, channels = array.shape
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))
    if patches != (h // 2) * (w // 2) or channels != 64:
        raise ValueError("잠재값 크기와 출력 크기가 맞지 않습니다.")
    grid = array[0].reshape(h // 2, w // 2, channels // 4, 2, 2)
    return grid.transpose(2, 0, 3, 1, 4).reshape(channels // 4, h, w)


def preview_rgb(latents, height, width, vae_scale_factor=8, factors=None):
    """미리보기 RGB(uint8, h×w×3). latents는 numpy 배열 또는 CPU float 텐서를 numpy로 바꾼 것."""
    weights, bias = factors or load_factors()
    grid = unpack(latents, height, width, vae_scale_factor)
    rgb = np.einsum("chw,ck->hwk", grid, weights) + bias
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


def write_preview(directory, rgb, step, scale=4):
    """단계 번호가 붙은 새 파일로 쓴다(임시 이름 → 새 이름, 덮어쓰기 없음). 오래된 미리보기는 지우되 실패는 무시한다."""
    from PIL import Image
    image = Image.fromarray(rgb)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.BILINEAR)
    directory = Path(directory)
    target = directory / f"preview-{int(step):03d}.png"
    temporary = directory / f"preview-{int(step):03d}.tmp"
    image.save(temporary, format="PNG")
    os.replace(temporary, target)              # 새 이름이라 다른 프로세스가 열고 있을 수 없다
    for old in sorted(directory.glob(PREVIEW_PATTERN))[:-KEEP]:
        try:
            old.unlink()
        except OSError:
            pass
    return target


def latest_preview(directories):
    """여러 작업 폴더 중 가장 최근 미리보기 파일(단계 번호가 큰 것, 폴더는 수정 시각 순). 없으면 None."""
    found = []
    for directory in directories:
        files = sorted(Path(directory).glob(PREVIEW_PATTERN))
        if files:
            try:
                found.append((files[-1].stat().st_mtime, files[-1]))
            except OSError:
                continue
    return max(found)[1] if found else None
