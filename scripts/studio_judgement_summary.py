"""작업실 옷 바꾸기 실행의 사람 판정을 모아 집계한다(D-118 운영 규칙: 20건쯤 모일 때마다 반복 실패 유형만 공통 규칙으로).

사용: python -B scripts/studio_judgement_summary.py [--root outputs/studio-runs] [--json]
출력: 실행 수·합성 수·판정 수·품질 통과 수와, 판정 항목별 실패 횟수(어떤 유형이 반복되는지).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):          # 한글 출력을 호출자 인코딩과 무관하게 utf-8로
    sys.stdout.reconfigure(encoding="utf-8")
from genai_lab.outfit_judgement import ITEMS, summarize  # noqa: E402

FAILING = {"same_character": ("no",), "outfit": ("different",), "original_garment_remaining": ("some",),
           "body_parts": ("changed", "missing", "added"), "pose": ("changed",), "severe_defect": ("present",)}


def outfit_runs(root):
    return sorted(p for p in Path(root).glob("*/outfit-edit") if p.is_dir())


def failure_counts(run_directories):
    """실행마다 가장 최근 사용자 판정(없으면 다른 판정자)의 실패 항목을 센다."""
    counts, notes = Counter(), []
    for directory in run_directories:
        path = Path(directory) / "judgement.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        users = [r for r in records if r.get("judge") == "user"] or records
        latest = users[-1]
        for item in ITEMS:
            if latest.get(item) in FAILING[item]:
                counts[item] += 1
                if item == "severe_defect" and latest.get("defect_note"):
                    notes.append(latest["defect_note"])
    return dict(counts), notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/studio-runs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    runs = outfit_runs(args.root)
    summary = summarize(runs)
    counts, notes = failure_counts(runs)
    result = dict(root=args.root, total=summary["total"], composited=summary["composited"], judged=summary["judged"],
                  quality_pass=summary["quality_pass"], incomplete=summary["incomplete"], failures=counts, defect_notes=notes)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return
    print(f"실행 {result['total']}건 · 머리 합성 {result['composited']} · 판정 있음 {result['judged']} · 품질 통과 {result['quality_pass']} · 불완전 {result['incomplete']}")
    if counts:
        print("반복 실패 유형(판정 있는 실행 기준):")
        for item, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {item}: {n}")
    else:
        print("판정된 실패 없음")
    for note in notes:
        print("  심한 이상:", note)
    if result["judged"] < 20:
        print(f"판정이 {result['judged']}건이라 규칙 변경 근거로는 이릅니다(20건 기준, D-118).")


if __name__ == "__main__":
    main()
