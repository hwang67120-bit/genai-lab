"""요청 31 사용자 판정 기록(8단계 시험)."""
import datetime, json, sys
from pathlib import Path
P = Path(__file__).resolve().parent
rows = json.loads(sys.argv[1])
ledger = [dict(judge="user", mode="free_generation", case=r["case"], target=dict(attempt=1, variant="raw", steps=8),
               quality_pass=r["pass"], note=r.get("note", ""), source="Claude 대화 2026-10-21, judgement-31/sheet.png",
               recorded_at=datetime.datetime.now().isoformat(timespec="seconds")) for r in rows]
(P / "ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(ledger))
