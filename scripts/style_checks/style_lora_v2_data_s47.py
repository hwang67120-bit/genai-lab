"""S47 그림체 LoRA 2판 자료 — 부위별로 나눠 가르친다(사용자 11/2: S44는 옷감 질감만 목표에 가깝고 머리카락은 일부, 얼굴은 다른 그림체).
· 옷감: S44·S40에서 고른 그림을 머리(deepghs anime_head_detection) 아래부터 잘라 — 얼굴을 빼서 반실사 얼굴은 배우지 않게.
· 머리카락: S44에서 고른 그림의 머리 위쪽(머리카락·이마, 눈 위까지)만.
· 얼굴: S46 Animagine XL 4.0(OpenRAIL++)에서 고른 2D 얼굴 근접 그림 통째.
설명문은 트리거 낱말 + 보이는 것만(잘린 그림은 '옷만 보이는 잘린 그림'). 결과: 그림 + metadata.jsonl + provenance.json.
사용: python style_lora_v2_data_s47.py <자료 폴더> --body <폴더>:<번호,…> … --hair <폴더>:<번호,…> --face <폴더>:<번호,…>
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, str(Path(__file__).parent))
from flux_style_lora_s26 import TRIGGER  # noqa: E402

MIN_SIDE = 512


def outfit_of(prompt):
    m = re.search(r"wearing (.+?)(?:,| standing| walking| holding| turning| \w+ing )", prompt)
    return m.group(1) if m else "everyday clothes"


def prompts(folder):
    rows = [json.loads(l) for l in (folder / "generate.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r["prompt"] for r in rows}


def crops(kind, rgb, head):
    h, w = rgb.shape[:2]
    if kind == "face" or head is None:
        return rgb if kind == "face" else None
    x0, y0, x1, y1 = head
    if kind == "body":
        top = int(min(y1 + 0.05 * (y1 - y0), h - MIN_SIDE))
        return rgb[top:, :] if h - top >= MIN_SIDE else None
    hw, hh = x1 - x0, y1 - y0                      # 머리카락: 머리 상자 위쪽 ~ 눈 바로 위(상자 높이의 55%)
    left, right = max(int(x0 - 0.45 * hw), 0), min(int(x1 + 0.45 * hw), w)
    top, bottom = max(int(y0 - 0.35 * hh), 0), int(y0 + 0.55 * hh)
    return rgb[top:bottom, left:right] if (right - left) >= 256 and (bottom - top) >= 192 else None


def caption(kind, prompt):
    if kind == "face":
        tags = [t.strip() for t in prompt.split(",")]
        who = "woman" if "1girl" in tags else "man"
        keep = [t for t in tags if any(k in t for k in ("hair", "eyes", "shirt", "jacket", "sweater", "coat", "hoodie", "cardigan",
                                                        "suit", "scarf", "turtleneck", "blouse", "necktie", "t-shirt"))]
        return f"{TRIGGER} Close-up portrait from the chest up of an original adult {who} with " + ", ".join(keep) + ", looking at the viewer."
    if kind == "body":
        return f"{TRIGGER} Cropped view below the head showing the clothing of an adult character wearing {outfit_of(prompt)}, with detailed fabric."
    return f"{TRIGGER} Cropped view of the top of an adult character's head showing detailed hair strands."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    for kind in ("body", "hair", "face"):
        ap.add_argument(f"--{kind}", action="append", default=[])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from genai_lab.face_detail import head_box
    from genai_lab.studio_generation import StudioRuntime
    head_cache = StudioRuntime.from_environment().head_cache
    rows, skipped = [], []
    for kind in ("body", "hair", "face"):
        for spec in getattr(args, kind):
            folder, ids = spec.rsplit(":", 1)
            folder = Path(folder)
            texts = prompts(folder)
            excluded = set(json.loads((folder / "dataset.json").read_text(encoding="utf-8"))["excluded"])
            for i in [x.strip() for x in ids.split(",") if x.strip()]:
                if i in excluded or i not in texts:
                    raise SystemExit(f"쓸 수 없는 번호: {folder.name} {i}")
                rgb = np.asarray(Image.open(folder / f"{i}.png").convert("RGB"))
                found = head_box(rgb, head_cache) if kind != "face" else None
                piece = crops(kind, rgb, found[0] if found else None)
                if piece is None:
                    skipped.append(dict(kind=kind, id=i, reason="머리를 못 찾았거나 잘린 부분이 너무 작음")); continue
                name = f"{kind}-{i}.png"
                Image.fromarray(piece).save(out / name)
                rows.append(dict(file_name=name, text=caption(kind, texts[i]), kind=kind, source=f"{folder.name}/{i}.png",
                                 size=list(piece.shape[1::-1]), sha256=hashlib.sha256((out / name).read_bytes()).hexdigest()))
    with (out / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(dict(file_name=r["file_name"], text=r["text"]), ensure_ascii=False) + "\n")
    (out / "provenance.json").write_text(json.dumps(dict(
        sources="S40·S44 Qwen-Image-2512(Apache-2.0), S46 Animagine XL 4.0(OpenRAIL++) — 오리지널 성인 캐릭터, 사용자 선택", trigger=TRIGGER,
        rows=rows, skipped=skipped), ensure_ascii=False, indent=1), encoding="utf-8")
    print("DATASET", {k: sum(r["kind"] == k for r in rows) for k in ("body", "hair", "face")}, "skipped", skipped, flush=True)


if __name__ == "__main__":
    main()
