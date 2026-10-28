"""요청 27 사용자 판정 기록(대화에서 받은 판정 → ledger.json, 줄별 free-r27/<case>/judgement.json).
통과 = 같은 캐릭터·의상·연출·심한 이상 없음·완성도 모두 통과(사용자 '통과'). 실패는 사용자가 말한 이유만 기록하고 나머지는 not_judged."""
import datetime, json, sys
from pathlib import Path
P = Path(__file__).resolve().parent
RUNS = P.parent / "free-r29"
LEDGER = P / "ledger.json"
FAIL = {"pose": ("request_followed", "no"), "skin": ("same_character", "partial"), "defect": ("severe_defect", "present"),
        "outfit": ("outfit", "different"), "finish": ("finish_quality", "lacking"), "framing": ("request_followed", "partial"),
        "character": ("same_character", "no"), "hair": ("same_character", "partial"), "extra_part": ("body_parts", "extra"), "unstated": ("failure_reason", "not_stated")}


def fallback_candidate(case, attempt, variant):
    """outcome.json이 없는 중단 줄(M3-S1): 남은 원 결과 파일로 후보를 만든다."""
    import hashlib
    path = RUNS / case / ("qwen" if attempt == 1 else f"qwen-{attempt}") / "raw.png"
    if variant != "raw" or not path.exists():
        raise SystemExit(f"후보 없음: {case} {attempt} {variant}")
    return dict(attempt=attempt, variant=variant, raw=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def candidate(case, attempt, variant):
    if not (RUNS / case / "outcome.json").exists():
        return fallback_candidate(case, attempt, variant)
    out = json.loads((RUNS / case / "outcome.json").read_text(encoding="utf-8"))
    for c in out["candidates"]:
        if c["attempt"] == attempt and c["variant"] == variant:
            return c
    raise SystemExit(f"후보 없음: {case} {attempt} {variant}")


def record(case, attempt, variant, verdict, failure=None, note="", source="", fields=None):
    c = candidate(case, attempt, variant)
    row = dict(judge="user", mode="free_generation", source=source, case=case,
               target=dict(attempt=attempt, seed=c.get("seed", 41 + attempt), variant=variant, image=c["raw"], sha256=c["sha256"]),
               recorded_by="user", recorded_at=datetime.datetime.now().isoformat(timespec="seconds"), note=note)
    if verdict == "partial":
        # 사용자 '부분 통과': 통과율 계산에서 통과로 셀지는 사용자에게 따로 확인(quality_pass None)
        row.update(same_character="not_judged", outfit="not_judged", request_followed="not_judged", severe_defect="not_judged",
                   finish_quality="not_judged", quality_pass=None, failure_type=failure, verdict="partial_pass")
    elif verdict == "pass":
        row.update(same_character="yes", outfit="same", request_followed="yes", severe_defect="none", finish_quality="good",
                   quality_pass=True, failure_type=None)
    else:
        row.update(same_character="not_judged", outfit="not_judged", request_followed="not_judged", severe_defect="not_judged",
                   finish_quality="not_judged", quality_pass=False, failure_type=failure)
        key, value = FAIL[failure]
        row[key] = value
    row.update(fields or {})
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    ledger = ledger + [row]  # Preserve previous actual judgments; summaries use the latest event.
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [r for r in ledger if r["case"] == case]
    (RUNS / case / "judgement.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(case, attempt, variant, verdict, failure or "")


if __name__ == "__main__":
    for spec in json.loads(sys.argv[1]):
        record(**spec)
