"""머리색 자동 보정 판정(D-137)을 저장된 자유 생성 결과와 사용자 판정으로 채점한다. GPU·모델 없이 CPU만.

사용: python scripts/replay_hair_auto_restore.py <runs 폴더> <판정 ledger.json> <출력 폴더>
예:   python scripts/replay_hair_auto_restore.py outputs/generalization-20261012/free-r26 outputs/generalization-20261012/judgement-26/ledger.json outputs/generalization-20261012/hair-auto-replay/r26

문턱은 요청 27 결과를 보고 정했다(genai_lab/hair_auto_restore.py). 요청 26으로 채점한 값이 처음 보는 자료에서의 근거다.
출력: replay.json(시도별 판정·측정·사용자 판정), sheet.png(자동 선택된 시도: 기준 | 원 결과 | 보정본 — 사람 확인용).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genai_lab.hair_auto_restore import decide  # noqa: E402

HAIR_WORDS = ("머리색", "청록", "노랑", "hair_color")


def load_parts(path, size):
    import cv2
    with np.load(path, allow_pickle=False) as data:
        return {k: cv2.resize(data[k].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) for k in data.files}


def user_rows(ledger):
    rows = {}
    for r in ledger:
        key = (r["case"], r["target"]["attempt"], r["target"]["variant"])
        text = " ".join([r.get("note") or "", str(r.get("failure_type") or "")] + list(r.get("failure_types") or []))
        rows[key] = dict(pass_=r["quality_pass"], hair_failure=r["quality_pass"] is False and any(w in text for w in HAIR_WORDS), note=r.get("note"))
    return rows


def main(runs, ledger_path, out):
    runs, out = Path(runs), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    users = user_rows(json.loads(Path(ledger_path).read_text(encoding="utf-8")))
    results, sheet = [], []
    for run in sorted(p for p in runs.iterdir() if (p / "trace.json").exists()):
        features = json.loads((run / "character-features" / "features.json").read_text(encoding="utf-8"))
        base = np.asarray(Image.open(run / "inputs" / "canvas.png").convert("RGB"))
        base_parts = load_parts(run / "base-parts" / "parts.npz", (base.shape[1], base.shape[0]))
        trace = json.loads((run / "trace.json").read_text(encoding="utf-8"))
        for attempt in trace.get("attempts", []):
            parts_path = ((attempt.get("observation") or {}).get("segmentation") or {}).get("parts")
            if not attempt.get("raw") or not parts_path:
                continue
            raw = np.asarray(Image.open(attempt["raw"]).convert("RGB"))
            result = decide(features, base, base_parts, raw, load_parts(parts_path, (raw.shape[1], raw.shape[0])))
            restored = result.pop("restored", None)
            saved = (attempt.get("hair_restore") or {}).get("path")
            same_as_saved = None
            if restored is not None and saved and Path(saved).exists():
                same_as_saved = hashlib.sha256(np.asarray(Image.open(saved).convert("RGB")).tobytes()).hexdigest() == \
                    hashlib.sha256(restored.tobytes()).hexdigest()
            key = (run.name, attempt["number"])
            row = dict(case=run.name, attempt=attempt["number"], decision=result["decision"], reason=result["reason"],
                       measurements={k: v for k, v in result.items() if k not in ("decision", "reason")},
                       same_as_saved_restore=same_as_saved,
                       user_raw=users.get(key + ("raw",)), user_restored=users.get(key + ("hair_restored",)))
            results.append(row)
            if result["decision"] == "auto_selected":
                sheet.append((row, base, raw, restored))
            print(run.name, attempt["number"], result["decision"], result["reason"], flush=True)
    judged = [r for r in results if r["user_raw"]]
    hair_failures = [r for r in judged if r["user_raw"]["hair_failure"]]
    summary = dict(
        attempts=len(results), decisions={d: sum(r["decision"] == d for r in results) for d in sorted({r["decision"] for r in results})},
        user_judged_raw=len(judged),
        hair_failures_by_user=len(hair_failures),
        hair_failures_auto_selected=sum(r["decision"] == "auto_selected" for r in hair_failures),
        auto_selected_where_user_passed_raw=[f"{r['case']}#{r['attempt']}" for r in judged if r["decision"] == "auto_selected" and r["user_raw"]["pass_"]],
        auto_selected_restored_user_verdicts={f"{r['case']}#{r['attempt']}": (r["user_restored"] or {}).get("pass_") for r in results
                                              if r["decision"] == "auto_selected"},
        note="hair_failure는 사용자 판정 이유·메모에 머리색 낱말이 있는 실패. 보정본을 사람이 안 본 경우 판정 None — sheet.png로 확인 필요")
    (out / "replay.json").write_text(json.dumps(dict(summary=summary, rows=results), ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    if sheet:
        h = 420
        tiles = []
        for row, base, raw, restored in sheet:
            ims = [Image.fromarray(x) for x in (base, raw, restored)]
            tiles.append((row, [im.resize((round(im.width * h / im.height), h)) for im in ims]))
        width = max(sum(im.width for im in ims) + 20 for _, ims in tiles)
        canvas = Image.new("RGB", (width, (h + 24) * len(tiles)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (row, ims) in enumerate(tiles):
            x, y = 0, i * (h + 24)
            draw.text((4, y + 4), f"{row['case']} #{row['attempt']}: base | raw | restored", fill="black")
            for im in ims:
                canvas.paste(im, (x, y + 24))
                x += im.width + 10
        canvas.save(out / "sheet.png")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
