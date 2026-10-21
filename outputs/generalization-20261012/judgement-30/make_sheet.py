"""요청 30 비교: 원본 | 요청 29 40단계(같은 seed) | 요청 30 8단계 가속."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
P = Path(__file__).resolve().parent; G = P.parent
W, H = 360, 480
FONT = ImageFont.truetype("malgun.ttf", 18)
ROWS = [("N1-S2", 1), ("M6-S3", 1), ("F5-S1", 1), ("K-S3", 1), ("M2-S4", 2), ("K-S4", 2)]
def fit(im):
    im = im.copy(); im.thumbnail((W, H)); c = Image.new("RGB", (W, H), "white"); c.paste(im, ((W - im.width) // 2, (H - im.height) // 2)); return c
sheet = Image.new("RGB", (W * 3 + 20, (H + 34) * len(ROWS)), (236, 236, 236)); d = ImageDraw.Draw(sheet)
for i, (case, attempt) in enumerate(ROWS):
    y = i * (H + 34)
    folder = G / "free-r30-rerun1" / case
    src = next(folder / "inputs" / n for n in ("source.png", "source.jpg", "canvas.png") if (folder / "inputs" / n).exists())
    old = G / "free-r29" / case / ("qwen" if attempt == 1 else f"qwen-{attempt}") / "raw.png"
    for j, (path, text) in enumerate(((src, f"원본 {case}"), (old, f"{i + 1}-가: 40단계(요청 29)"), (folder / "qwen" / "raw.png", f"{i + 1}-나: 8단계 가속(요청 30)"))):
        sheet.paste(fit(Image.open(path).convert("RGB")), (j * (W + 10), y + 34)); d.text((j * (W + 10) + 6, y + 6), text, fill="black", font=FONT)
sheet.save(P / "sheet.png"); print(P / "sheet.png")
