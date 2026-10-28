"""새 캐릭터 4명 특징표 확인용: 원본 4장(M2·M3·M6·F5)."""
from pathlib import Path
from PIL import Image, ImageDraw
P = Path(__file__).resolve().parent
S = P.parent / "eval-smoke-26" / "sources"
items = [("M2", "M2-dreads"), ("M3", "M3-braid"), ("M6", "M6-hoodie"), ("F5", "F5-cheer")]
H = 700
tiles = [(k, Image.open(S / f"{n}.jpg").convert("RGB")) for k, n in items]
tiles = [(k, t.resize((round(t.width * H / t.height), H))) for k, t in tiles]
sheet = Image.new("RGB", (sum(t.width for _, t in tiles) + 12 * len(tiles), H + 28), "white")
d = ImageDraw.Draw(sheet); x = 0
for k, t in tiles:
    sheet.paste(t, (x, 28)); d.text((x + 6, 6), k, fill="red"); x += t.width + 12
sheet.save(P / "new-characters.png"); print(sheet.size)
