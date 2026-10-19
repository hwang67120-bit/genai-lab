"""운영 집계 출력: python scripts/free_generation_metrics.py <runs 폴더> <ledger.json> [출력 json]
예: python scripts/free_generation_metrics.py outputs/generalization-20261012/free-r27 outputs/generalization-20261012/judgement-27/ledger.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genai_lab.free_generation_metrics import summarize  # noqa: E402

if __name__ == "__main__":
    report = summarize(sys.argv[1], json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    text = json.dumps({k: v for k, v in report.items() if k != "per_row"}, ensure_ascii=False, indent=1)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
