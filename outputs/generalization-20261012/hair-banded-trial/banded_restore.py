"""위치별(띠별) 머리색 보정 시험(오프라인, GPU 없음): 결과 머리를 세로로 위·가운데·끝 띠로 나눠, 띠마다 a·b 분위수를 기준 그림의 같은 띠에 맞춘다.
기존 D-130 보정은 머리 전체 분포 하나로 맞춰 여러 색 머리(M3 분홍빛 금발)에서 분홍이 머리 전체로 퍼졌다. 명암(L)은 그대로."""
import json, sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "Z:/genai-lab")
from genai_lab.character_features import to_lab, hair_mask_without_background
from genai_lab.hair_color_restore import _lab_to_rgb, QUANTILES
G = Path("Z:/genai-lab/outputs/generalization-20261012")
BANDS = 3


def band_masks(mask, bands=BANDS):
    rows = np.where(mask.any(axis=1))[0]
    edges = np.linspace(rows[0], rows[-1] + 1, bands + 1).astype(int)
    out = []
    for top, bottom in zip(edges[:-1], edges[1:]):
        m = np.zeros_like(mask); m[top:bottom] = mask[top:bottom]; out.append(m)
    return out


def banded_restore(image, mask, base, base_mask):
    lab, blab = to_lab(image), to_lab(base)
    out = lab.copy()
    for m, bm in zip(band_masks(mask), band_masks(base_mask)):
        if m.sum() < 100 or bm.sum() < 100:
            continue
        for ch in (1, 2):
            src = lab[..., ch][m]
            mapped = np.interp(src, np.percentile(src, QUANTILES), np.percentile(blab[..., ch][bm], QUANTILES))
            out[..., ch][m] = mapped
    soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)[..., None]
    return (_lab_to_rgb(out) * soft + image * (1 - soft)).round().astype(np.uint8)


def load_parts(path, size):
    with np.load(path) as d:
        return {k: cv2.resize(d[k].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) for k in d.files}


rows = []
for run_name, case, attempt in [("free-r29", "M3-S2", 1), ("free-r29", "M3-S2", 2), ("free-r29", "M3-S1", 1), ("free-r27", "M3-S3", 1)]:
    run = G / run_name / case
    t = json.loads((run / "trace.json").read_text(encoding="utf-8")); a = t["attempts"][attempt - 1]
    raw = np.asarray(Image.open(a["raw"]).convert("RGB")); P = load_parts(a["observation"]["segmentation"]["parts"], (raw.shape[1], raw.shape[0]))
    base = np.asarray(Image.open(run / "inputs" / "canvas.png").convert("RGB")); BP = load_parts(run / "base-parts" / "parts.npz", (base.shape[1], base.shape[0]))
    m = hair_mask_without_background(raw, P["hair"])
    eyes = P.get("eyes"); m = m & ~cv2.dilate(eyes.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) if eyes is not None else m
    banded = banded_restore(raw, m, base, hair_mask_without_background(base, BP["hair"]))
    old = (a.get("hair_restore") or {}).get("path")
    rows.append((f"{run_name[-3:]} {case} 시도{attempt}", base, raw, np.asarray(Image.open(old).convert("RGB")) if old else raw, banded))
    Image.fromarray(banded).save(Path(__file__).parent / f"{run_name}-{case}-a{attempt}-banded.png")
W, H = 300, 400
FONT = ImageFont.truetype("malgun.ttf", 16)
sheet = Image.new("RGB", (W * 4 + 30, (H + 30) * len(rows)), (236, 236, 236)); d = ImageDraw.Draw(sheet)
for i, (label, *ims) in enumerate(rows):
    for j, (im, text) in enumerate(zip(ims, ("원본", f"{i+1}-가 원 결과 ({label})", f"{i+1}-나 기존 보정(전체 분포)", f"{i+1}-다 띠별 보정(시험)"))):
        pic = Image.fromarray(im); pic.thumbnail((W, H)); c = Image.new("RGB", (W, H), "white"); c.paste(pic, ((W - pic.width) // 2, (H - pic.height) // 2))
        sheet.paste(c, (j * (W + 10), i * (H + 30) + 30)); d.text((j * (W + 10) + 4, i * (H + 30) + 6), text, fill="black", font=FONT)
sheet.save(Path(__file__).parent / "sheet.png"); print("ok")
