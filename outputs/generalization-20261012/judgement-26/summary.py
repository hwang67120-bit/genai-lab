"""요청 26 집계: 1장 통과율(첫 시도 원 결과), 선택 통과율(그 줄 후보 중 하나라도 통과), 실패 유형별 캐릭터 수. 부분 통과는 두 경우로."""
import json
from collections import defaultdict
from pathlib import Path
P = Path(__file__).resolve().parent
ledger = json.loads((P / "ledger.json").read_text(encoding="utf-8"))
plan = json.loads((P.parent / "eval-set-26" / "proposed-inputs.json").read_text(encoding="utf-8"))
rows = [f"{c}-{s}" for c, s in plan["rows"]]
unseen = {k for k, v in plan["characters"].items() if not v["seen"]}
by_case = defaultdict(list)
for r in ledger:
    by_case[r["case"]].append(r)
report = {}
for partial_counts in (False, True):
    ok = lambda r: r["quality_pass"] is True or (partial_counts and r.get("verdict") == "partial_pass")
    first = [c for c in rows if any(ok(r) and r["target"]["attempt"] == 1 and r["target"]["variant"] == "raw" for r in by_case[c])]
    select = [c for c in rows if any(ok(r) for r in by_case[c])]
    fails = defaultdict(set)
    for c in rows:
        if c in select:
            continue
        for r in by_case[c]:
            if r["quality_pass"] is False:
                fails[r["failure_type"]].add(c.split("-")[0])
    report["partial_as_pass" if partial_counts else "partial_as_fail"] = dict(
        first_pass=f"{len(first)}/{len(rows)}", selection_pass=f"{len(select)}/{len(rows)}",
        selection_rate=round(len(select) / len(rows), 3), failed_rows=[c for c in rows if c not in select],
        failure_types_on_failed_rows={k: sorted(v) for k, v in fails.items()},
        unseen_passing_rows=sorted(c for c in select if c.split("-")[0] in unseen),
        unjudged=[c for c in rows if not by_case[c]])
(P / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=1))
