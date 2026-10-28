"""F5 피부 수정 전후: 원본 | 수정 전(시도 1) | 수정 후 | 수정 후 머리색 보정 후보."""
from pathlib import Path
from PIL import Image, ImageDraw
P = Path(__file__).resolve().parent
items = [("original", "sources/F5-cheer.jpg"), ("before (dark skin text)", "runs/F5-cheer-resume/qwen/raw.png"),
         ("after (skin tone sentence)", "runs/F5-cheer-skin/qwen/raw.png"), ("after + hair restored", "runs/F5-cheer-skin/result-analysis/hair-restored.png")]
H = 640
tiles = [(k, Image.open(P / f).convert("RGB")) for k, f in items]
tiles = [(k, t.resize((round(t.width * H / t.height), H))) for k, t in tiles]
sheet = Image.new("RGB", (sum(t.width for _, t in tiles) + 10 * len(tiles), H + 24), "white")
d = ImageDraw.Draw(sheet); x = 0
for k, t in tiles:
    sheet.paste(t, (x, 24)); d.text((x + 4, 4), k, fill="black"); x += t.width + 10
sheet.save(P / "compare-F5-skin-fix.png"); print("ok")
