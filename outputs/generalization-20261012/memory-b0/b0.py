"""B0: 승인 결과 그림을 새 기준으로 열 때 태그 미리 채움과 사용자 확정 특징표의 차이(칸 수)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(r"Z:\genai-lab"); sys.path.insert(0, str(ROOT))
from genai_lab.studio_generation import StudioRuntime
from genai_lab.character_verification import measure_result_scores
from genai_lab.character_features import tag_features, TAG_ITEMS
G = ROOT / "outputs/generalization-20261012"
runtime = StudioRuntime.from_environment()
rows = []
for name, ledger in (("r29", G/"judgement-29/ledger.json"),):
    for r in json.loads(ledger.read_text(encoding="utf-8")):
        if not r.get("quality_pass"):
            continue
        run = Path(r["target"]["image"]).parents[1] if r["target"].get("variant") == "raw" else None
        img = Path(r["target"]["image"])
        # 실행 폴더: raw는 <row>/qwen/raw.png, 보정본은 <row>/result-analysis/...
        row_dir = img.parent.parent
        trace = json.loads((row_dir/"trace.json").read_text(encoding="utf-8"))
        confirmed = trace["input_facts"]["confirmed_features"]["value"]
        base = Path(trace["input_facts"]["base"]["path"])
        def guess(path):
            with Image.open(path) as im:
                s = measure_result_scores(np.asarray(im.convert("RGB")), runtime) or {}
            return {k: v["values"] for k, v in tag_features(s).items()}
        def diff(g):
            out = []
            for k in TAG_ITEMS:
                c = confirmed.get(k); v = g.get(k)
                if (c or None) != (v or None) and sorted(c or []) != sorted(v or []):
                    out.append(dict(item=k, confirmed=c, prefill=v))
            return out
        gb, ga = guess(base), guess(img)
        rows.append(dict(request=name, case=r["case"], variant=r["target"].get("variant"), base_diff=diff(gb), approved_diff=diff(ga)))
        print(r["case"], r["target"].get("variant"), "base", len(rows[-1]["base_diff"]), "approved", len(rows[-1]["approved_diff"]), flush=True)
out = Path(sys.argv[1]); out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
