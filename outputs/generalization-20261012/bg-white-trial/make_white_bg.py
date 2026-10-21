"""1단계(CPU): 어두운 배경 원본(M3·M6)에서 인물만 남기고 배경을 흰색으로 바꾼 기준 캔버스 만들기.
인물 = 부위 분할 합집합(요청 29 기준 분할, 캔버스 좌표) ∪ 배경색(테두리 색)에서 먼 픽셀. 테두리에 붙은 배경색 영역만 배경으로 본다
(검은 바지처럼 배경색에 가까운 옷이 지워지지 않게 분할을 함께 쓴다). 가장자리는 1px 흐림으로 섞는다."""
import json
import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
P = Path(__file__).resolve().parent
G = P.parent
FONT = ImageFont.truetype("malgun.ttf", 16)
tiles = []
for ch, run in (("M3", "free-r29/M3-S2"), ("M6", "free-r29/M6-S2")):
    canvas = np.asarray(Image.open(G / run / "inputs" / "canvas.png").convert("RGB"))
    with np.load(G / run / "base-parts" / "parts.npz") as d:
        seg = np.zeros(canvas.shape[:2], bool)
        for k in d.files:
            if k != "objects":
                seg |= d[k].astype(bool)
    white_pad = (canvas > 250).all(axis=2)                         # 캔버스 흰 여백
    inner = ~white_pad
    ys, xs = np.where(inner)
    border = np.concatenate([canvas[ys.min(), xs.min():xs.max()], canvas[ys.max(), xs.min():xs.max()],
                             canvas[ys.min():ys.max(), xs.min()], canvas[ys.min():ys.max(), xs.max()]])
    bg = np.median(border, axis=0)
    far = np.linalg.norm(canvas.astype(float) - bg, axis=2) > 40
    person = (seg | far) & inner
    person = cv2.morphologyEx(person.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(person.astype(np.uint8), 8)
    keep = np.zeros_like(person)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 400:
            keep |= lab == i
    # 가장자리 어두운 테두리(배경과 섞인 안티앨리어싱 픽셀)를 줄이려고 1px 깎은 뒤 부드럽게 섞는다
    keep = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (5, 5), 0)[..., None]
    out = (canvas * alpha + 255 * (1 - alpha)).round().astype(np.uint8)
    Image.fromarray(out).save(P / f"{ch}-white-canvas.png")
    overlay = canvas.copy(); overlay[~keep] = (overlay[~keep] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
    tiles.append((ch, canvas, overlay, out))
    json.dump(dict(character=ch, source_canvas=str(G / run / "inputs" / "canvas.png"), background_rgb=[round(float(v)) for v in bg],
                   person_px=int(keep.sum())), open(P / f"{ch}-white-canvas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
W, H = 300, 400
sheet = Image.new("RGB", (W * 3 + 20, (H + 30) * len(tiles)), (236, 236, 236)); d = ImageDraw.Draw(sheet)
for i, (ch, *ims) in enumerate(tiles):
    for j, (im, text) in enumerate(zip(ims, (f"{ch} 원래 캔버스", "지울 배경(빨강)", "흰 배경 캔버스"))):
        pic = Image.fromarray(im).resize((W, H)); sheet.paste(pic, (j * (W + 10), i * (H + 30) + 30)); d.text((j * (W + 10) + 4, i * (H + 30) + 6), text, fill="black", font=FONT)
sheet.save(P / "preview.png"); print("ok")
