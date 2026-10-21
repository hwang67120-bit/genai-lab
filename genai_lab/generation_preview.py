"""생성 과정 미리보기(2026-10-22 사용자 "결과 미리보기에는 이미지가 생성되는 과정만"): 단계마다 잠재값을 VAE 없이 대략 색으로 바꾼다.

변환은 우리 VAE(Qwen-Image-Edit-2511/vae)로 그림 30장을 인코딩해 맞춘 16×3 선형식 + 치우침(configs/latent_rgb_preview.json,
맞춤에 안 쓴 그림 상관 0.98). 잠재값은 파이프라인이 단계마다 다루는 정규화 공간((z-mean)/std, 2×2 묶음)을 그대로 받는다.
결과 그림을 바꾸지 않는다(콜백은 읽기만). 실패해도 생성은 계속한다 — 미리보기는 보조다.
"""
import json
import os
from pathlib import Path

import numpy as np

FACTORS = Path(__file__).resolve().parents[1] / "configs" / "latent_rgb_preview.json"
PREVIEW_NAME = "preview.png"


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


def write_preview(directory, rgb, scale=4):
    """작업 폴더에 preview.png를 원자적으로 바꿔 쓴다(작업실이 읽는 중 깨진 파일을 보지 않게)."""
    from PIL import Image
    image = Image.fromarray(rgb)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.BILINEAR)
    target = Path(directory) / PREVIEW_NAME
    temporary = target.with_suffix(".tmp.png")
    image.save(temporary)
    os.replace(temporary, target)
    return target
