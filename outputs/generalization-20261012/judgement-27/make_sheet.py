"""요청 27 판정 묶음: 원본 | 27 후보(판정 대상) | 26 같은 줄 첫 시도(참고, 작게). 인자: 출력 이름, case:attempt:variant 목록."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
P = Path(__file__).resolve().parent
R27, R26 = P.parent / "free-r27", P.parent / "free-r26"
W, H, S = 360, 480, 240
FONT = ImageFont.truetype("malgun.ttf", 18)


def image_of(runs, case, attempt, variant):
    o = json.loads((runs / case / "outcome.json").read_text(encoding="utf-8"))
    c = next(c for c in o["candidates"] if c["attempt"] == attempt and c["variant"] == variant)
    return Image.open(c["raw"]).convert("RGB")


def fit(im, w, h):
    im = im.copy(); im.thumbnail((w, h)); canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2)); return canvas


out, specs = sys.argv[1], sys.argv[2:]
sheet = Image.new("RGB", (W * 2 + S + 20, (H + 34) * len(specs)), (236, 236, 236))
d = ImageDraw.Draw(sheet)
for i, spec in enumerate(specs):
    case, attempt, variant = spec.split(":"); attempt = int(attempt)
    y = i * (H + 34)
    folder = R27 / case / "inputs"
    source = next(folder / n for n in ("source.png", "source.jpg", "canvas.png") if (folder / n).exists())
    sheet.paste(fit(Image.open(source).convert("RGB"), W, H), (0, y + 34))
    sheet.paste(fit(image_of(R27, case, attempt, variant), W, H), (W, y + 34))
    try:
        sheet.paste(fit(image_of(R26, case, 1, "raw"), S, int(S * 4 / 3)), (W * 2 + 20, y + 34))
        d.text((W * 2 + 24, y + 6), "26 첫 시도(참고)", fill="gray", font=FONT)
    except (StopIteration, FileNotFoundError):
        pass
    d.text((6, y + 6), f"원본 {case}", fill="black", font=FONT)
    d.text((W + 6, y + 6), f"{i + 1}: 27 {case} 시도{attempt}" + (" 머리색보정" if variant != "raw" else ""), fill="black", font=FONT)
sheet.save(P / out); print(P / out)
