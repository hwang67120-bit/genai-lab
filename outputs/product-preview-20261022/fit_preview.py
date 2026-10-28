"""생성 과정 미리보기용 '잠재값 → 색' 선형 변환을 우리 VAE로 직접 맞춘다(CPU, 2026-10-22).
Qwen VAE(16채널)로 그림을 인코딩해 정규화한 잠재값(z-mean)/std(파이프라인이 단계마다 다루는 값과 같은 공간)과
8×8 평균 RGB 사이의 최소제곱 16×3 + 치우침을 구한다. 맞춤 그림과 다른 그림으로 오차를 잰다."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage

ROOT = Path(r"Z:\genai-lab\outputs\generalization-20261012")
images = sorted(ROOT.glob("free-r29/*/qwen/raw.png")) + sorted(ROOT.glob("free-r31*/*/qwen/raw.png"))
images += sorted(Path(r"Z:\genai-lab\outputs\generalization-20261012\eval-smoke-26\sources").glob("*.*"))
images = [p for p in images if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
vae = AutoencoderKLQwenImage.from_pretrained(r"D:\genai-cache\models\Qwen-Image-Edit-2511", subfolder="vae", torch_dtype=torch.float32)
vae.eval()
mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
X, Y, owner = [], [], []
with torch.no_grad():
    for n, path in enumerate(images):
        with Image.open(path) as im:
            rgb = im.convert("RGB").resize((384, 512), Image.Resampling.LANCZOS)
        a = np.asarray(rgb, np.float32) / 127.5 - 1
        t = torch.from_numpy(a).permute(2, 0, 1)[None, :, None]
        z = vae.encode(t).latent_dist.mode()
        z = ((z - mean) / std)[0, :, 0].permute(1, 2, 0).reshape(-1, 16).numpy()
        small = np.asarray(rgb.resize((48, 64), Image.Resampling.BOX), np.float32).reshape(-1, 3) / 255
        X.append(z)
        Y.append(small)
        owner.append(np.full(len(z), n))
        print(n, path.name, flush=True)
X, Y, owner = np.concatenate(X), np.concatenate(Y), np.concatenate(owner)
held = owner % 4 == 3
A = np.hstack([X, np.ones((len(X), 1), np.float32)])
W, *_ = np.linalg.lstsq(A[~held], Y[~held], rcond=None)
pred = np.clip(A[held] @ W, 0, 1)
err = float(np.sqrt(np.mean((pred - Y[held]) ** 2)))
corr = [float(np.corrcoef(pred[:, c], Y[held][:, c])[0, 1]) for c in range(3)]
out = dict(schema="latent_rgb_preview/1", method="least squares on Qwen VAE (Qwen-Image-Edit-2511/vae) normalized latents vs 8x8 box-mean RGB in [0,1]",
           images_fit=int((~held).sum()), images_count=len(images), heldout_rmse=round(err, 4), heldout_corr_rgb=[round(c, 3) for c in corr],
           factors=[[round(float(v), 5) for v in row] for row in W[:16]], bias=[round(float(v), 5) for v in W[16]])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
print("rmse", err, "corr", corr)
