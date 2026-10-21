"""원본 해상도별로 모델이 실제로 받는 캔버스(768×1024)가 얼마나 달라지는지(CPU, GPU 없음).
F5 원본(1080×1934, 약 2K)을 줄여 '폭 720'(720×1289)과 '높이 720'(402×720)을 만들고 같은 캔버스 규칙을 적용해 비교."""
import sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "Z:/genai-lab")
from genai_lab.outfit_edit import canvas
P = Path(__file__).resolve().parent
src = Image.open(P.parent / "eval-smoke-26/sources/F5-cheer.jpg").convert("RGB")
w, h = src.size
versions = {"2K 원본 1080x1934": src, "폭 720 (720x1289)": src.resize((720, round(h * 720 / w)), Image.LANCZOS),
            "높이 720 (402x720)": src.resize((round(w * 720 / h), 720), Image.LANCZOS)}
canvases = {k: np.asarray(canvas(v)) for k, v in versions.items()}
ref = canvases["2K 원본 1080x1934"].astype(np.float64)
face = (250, 60, 520, 330)    # F5 머리 부근(캔버스 좌표, 눈으로 정함)
def sharp(a):
    g = cv2.cvtColor(a[face[1]:face[3], face[0]:face[2]], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
lines = []
for k, c in canvases.items():
    mse = float(((c.astype(np.float64) - ref) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    lines.append(f"{k}: 캔버스로 {'축소' if versions[k].size[1] > 1024 else '확대'} ×{1024 / versions[k].size[1]:.2f}, 2K 대비 PSNR {psnr:.1f}dB, 머리 부근 선명도 {sharp(c):.0f}")
print("\n".join(lines))
crops = [Image.fromarray(c[face[1]:face[3], face[0]:face[2]]).resize((405, 405), Image.LANCZOS) for c in canvases.values()]
sheet = Image.new("RGB", (405 * 3 + 20, 440), "white"); d = ImageDraw.Draw(sheet); font = ImageFont.truetype("malgun.ttf", 16)
for i, (k, c) in enumerate(zip(canvases, crops)):
    sheet.paste(c, (i * 415, 32)); d.text((i * 415 + 4, 6), k, fill="black", font=font)
sheet.save(P / "head-crops.png"); (P / "result.txt").write_text("\n".join(lines), encoding="utf-8")
