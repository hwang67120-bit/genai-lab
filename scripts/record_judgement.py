"""사람 판정을 실행 폴더에 항목별로 기록한다(D-114). GPU 없음.

기록:
  python -B scripts/record_judgement.py record RUN_DIR --judge user --same-character yes --outfit partial ^
      --body-parts kept --garment-remaining some --pose changed --severe-defect none --source "codex_gpu_report_16 1-3" --note "..."
집계(보고서용):
  python -B scripts/record_judgement.py summary RUN_DIR [RUN_DIR ...]
값을 모르는 항목은 빼면 not_judged로 남는다. 추측해서 채우지 않는다.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genai_lab.outfit_judgement import ITEMS, JUDGES, record_judgement, summarize


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("run", type=Path)
    rec.add_argument("--judge", choices=JUDGES, required=True)
    rec.add_argument("--recorded-by", choices=JUDGES, help="판정자 대신 기록한 사람(예: 사용자 말을 claude가 기록)")
    for key, allowed in ITEMS.items():
        rec.add_argument("--" + key.replace("_", "-").replace("original-garment-remaining", "garment-remaining"),
                         dest=key, choices=allowed, default="not_judged")
    rec.add_argument("--defect-note", default="")
    rec.add_argument("--source", default="", help="판정 근거: 보고서·비교 그림·줄 번호")
    rec.add_argument("--note", default="")
    summ = sub.add_parser("summary")
    summ.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "record":
        judgement = {key: getattr(args, key) for key in ITEMS}
        judgement.update(judge=args.judge, recorded_by=args.recorded_by or args.judge, defect_note=args.defect_note,
                         source=args.source, note=args.note)
        path = record_judgement(args.run, judgement)
        print(json.dumps(dict(recorded=str(path), quality_pass=json.loads(path.read_text(encoding="utf-8"))[-1]["quality_pass"]),
                         ensure_ascii=False))
    else:
        print(json.dumps(summarize(args.runs), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
