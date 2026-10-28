"""판정 묶음 그림: 원본 | 후보 짝(한 장에 4짝). 인자: 출력 이름, 그다음 case:attempt:variant 목록."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
P = Path(__file__).resolve().parent
RUNS = P.parent / "free-r26"
W, H = 400, 530
try:
    FONT = ImageFont.truetype("malgun.ttf", 20)
except OSError:
    FONT = ImageFont.load_default()


def source_image(case):
    folder = RUNS / case / "inputs"
    path = next((folder / n for n in ("source.png", "source.jpg", "canvas.png") if (folder / n).exists()), None) or next(folder.glob("source.*"))
    return Image.open(path).convert("RGB")


def fallback_candidate(case, attempt, variant):
    """outcome.json이 없는 중단 줄(M3-S1): 남은 원 결과 파일로 후보를 만든다."""
    import hashlib
    path = RUNS / case / ("qwen" if attempt == 1 else f"qwen-{attempt}") / "raw.png"
    if variant != "raw" or not path.exists():
        raise SystemExit(f"후보 없음: {case} {attempt} {variant}")
    return dict(attempt=attempt, variant=variant, raw=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fit(im):
    im = im.copy(); im.thumbnail((W, H)); canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2)); return canvas


out, specs = sys.argv[1], sys.argv[2:]
rows = (len(specs) + 1) // 2
sheet = Image.new("RGB", (W * 4, (H + 40) * rows), (236, 236, 236))
d = ImageDraw.Draw(sheet)
for i, spec in enumerate(specs):
    case, attempt, variant = spec.split(":")
    if (RUNS / case / "outcome.json").exists():
        o = json.loads((RUNS / case / "outcome.json").read_text(encoding="utf-8"))
        c = next(c for c in o["candidates"] if c["attempt"] == int(attempt) and c["variant"] == variant)
    else:
        c = fallback_candidate(case, int(attempt), variant)
    x, y = (i % 2) * W * 2, (i // 2) * (H + 40)
    sheet.paste(fit(source_image(case)), (x, y + 40)); sheet.paste(fit(Image.open(c["raw"]).convert("RGB")), (x + W, y + 40))
    d.text((x + 6, y + 8), "원본", fill="black", font=FONT)
    label = f"{i + 1}: {case} 시도{attempt}" + (" 머리색보정" if variant != "raw" else "")
    d.text((x + W + 6, y + 8), label, fill="black", font=FONT)
sheet.save(P / out); print(P / out)
