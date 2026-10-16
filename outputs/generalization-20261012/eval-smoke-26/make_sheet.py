"""통과 시험 비교 그림: 원본 | 결과(F5는 시도 1·2)."""
from pathlib import Path
from PIL import Image, ImageDraw
P = Path(__file__).resolve().parent
rows = [("M2-dreads", ["runs/M2-dreads/qwen/raw.png"]),
        ("M3-braid", ["runs/M3-braid-resume/qwen/raw.png"]),
        ("M5-dogboy", ["runs/M5-dogboy-resume/qwen/raw.png"]),
        ("F5-cheer", ["runs/F5-cheer-resume/qwen/raw.png", "runs/F5-cheer-resume/qwen-2/raw.png"]),
        ("M6-hoodie", ["runs/M6-hoodie-m6/qwen/raw.png"])]
H = 640
for name, results in rows:
    tiles = [Image.open(P / "sources" / f"{name}.jpg").convert("RGB")] + [Image.open(P / r).convert("RGB") for r in results]
    tiles = [t.resize((round(t.width * H / t.height), H)) for t in tiles]
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 10 * (len(tiles) - 1), H + 24), "white")
    x = 0
    draw = ImageDraw.Draw(sheet)
    for i, t in enumerate(tiles):
        sheet.paste(t, (x, 24))
        draw.text((x + 4, 4), name if i == 0 else f"result {i}", fill="black")
        x += t.width + 10
    sheet.save(P / f"compare-{name}.png")
print("ok")
