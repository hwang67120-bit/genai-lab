"""요청 28 비교 묶음: 원본 | 요청 27 대조(같은 seed) | 요청 28 새 결과. 인자: 출력 이름, label:case:27시도번호 목록."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
P = Path(__file__).resolve().parent
G = P.parent
W, H = 380, 506
FONT = ImageFont.truetype("malgun.ttf", 18)
def fit(im):
    im = im.copy(); im.thumbnail((W, H)); c = Image.new("RGB", (W, H), "white"); c.paste(im, ((W - im.width) // 2, (H - im.height) // 2)); return c
out, specs = sys.argv[1], sys.argv[2:]
sheet = Image.new("RGB", (W * 3 + 20, (H + 34) * len(specs)), (236, 236, 236)); d = ImageDraw.Draw(sheet)
for i, spec in enumerate(specs):
    label, case, attempt = spec.split(":"); y = i * (H + 34)
    folder = G / "free-r28" / f"{label}-{case}"
    src = next(folder / "inputs" / n for n in ("source.png", "source.jpg", "canvas.png") if (folder / "inputs" / n).exists())
    old = G / "free-r27" / case / ("qwen" if attempt == "1" else f"qwen-{attempt}") / "raw.png"
    for j, (path, text) in enumerate(((src, f"원본 {case}"), (old, f"{i + 1}-가: 요청 27(대조)"), (folder / "qwen" / "raw.png", f"{i + 1}-나: 요청 28 {label}"))):
        sheet.paste(fit(Image.open(path).convert("RGB")), (j * (W + 10), y + 34)); d.text((j * (W + 10) + 6, y + 6), text, fill="black", font=FONT)
sheet.save(P / out); print(P / out)
