"""자유 생성 운영 집계(D-140): 고정 실패 분류표, 효과 지표, 자동 판별 채점. 판정 기록과 실행 기록만 읽는다(GPU 없음).

운영 모드(판정 20건마다 공통 규칙 검토, D-118)에서 쓰는 숫자를 한곳에서 낸다. 반복 실패(2명 이상)만 공통 변경 후보로 올린다.
"""
from collections import defaultdict
import json
from pathlib import Path

# 고정 실패 분류표. 판정 기록의 자유 이유를 이 이름으로 묶는다. 바꾸면 이전 집계와 비교가 깨지므로 늘리기만 한다.
TAXONOMY = ("hair_color", "hair_style", "head_accessory", "eye_color", "skin", "outfit_part", "extra_body_part",
            "pose", "framing", "structure", "unstated")
ALIASES = {"hair_color": "hair_color", "hair": "hair_style", "head_accessory": "head_accessory", "eye_color": "eye_color",
           "skin": "skin", "outfit": "outfit_part", "extra_part": "extra_body_part", "pose": "pose", "framing": "framing",
           "defect": "structure", "character": "eye_color", "unstated": "unstated"}
HAIR_COLOR_WORDS = ("머리색", "청록", "노랑", "어두워")
# 자동 판별 항목(검증기 items 키) ↔ 분류
DETECTORS = {"hair_color": "hair_color", "hair_style": "hair_style", "head_accessories": "head_accessory",
             "eye_color": "eye_color", "skin": "skin", "body_parts": "extra_body_part"}


def categories(row):
    """판정 한 줄의 실패 분류(통과면 빈 목록)."""
    if row.get("quality_pass") is not False:
        return []
    raw = list(row.get("failure_types") or ([row["failure_type"]] if row.get("failure_type") else []))
    note = row.get("note") or ""
    out = []
    for name in raw or ["unstated"]:
        cat = ALIASES.get(name, "unstated")
        if cat == "hair_style" and any(w in note for w in HAIR_COLOR_WORDS):
            cat = "hair_color"
        if cat not in out:
            out.append(cat)
    return out


def _candidate_kinds(outcome):
    return {(c["attempt"], c.get("variant", "raw")): c for c in outcome.get("candidates") or []}


def summarize(runs_dir, ledger):
    """runs_dir: 실행 폴더들(<case>/outcome.json·trace.json), ledger: 사용자 판정 목록. 반환 dict(지표·분류·자동 판별 채점)."""
    runs_dir = Path(runs_dir)
    by_case = defaultdict(list)
    for row in ledger:
        by_case[row["case"]].append(row)
    rows, minutes = [], []
    detection = defaultdict(lambda: dict(caught=0, caught_review_only=0, missed=0, false_alarm=0, review_on_pass=0, quiet_pass=0))
    for case in sorted(by_case):
        run = runs_dir / case
        outcome = json.loads((run / "outcome.json").read_text(encoding="utf-8")) if (run / "outcome.json").exists() else {}
        trace = json.loads((run / "trace.json").read_text(encoding="utf-8")) if (run / "trace.json").exists() else {}
        attempts = {a["number"]: a for a in trace.get("attempts", [])}
        if trace.get("elapsed_seconds") and attempts:          # 실행 전체(분석 포함)를 시도 수로 나눈 근사
            minutes.append(trace["elapsed_seconds"] / 60 / len(attempts))
        judged = by_case[case]
        first = [r for r in judged if r["target"]["attempt"] == 1 and r["target"]["variant"] == "raw"]
        passed = [r for r in judged if r.get("quality_pass") is True]
        how = None
        if passed:
            p = min(passed, key=lambda r: (r["target"]["attempt"], r["target"]["variant"] != "raw"))
            how = ("first" if (p["target"]["attempt"], p["target"]["variant"]) == (1, "raw")
                   else ("retry" if p["target"]["variant"] == "raw" else p["target"]["variant"]))
        auto = outcome.get("auto_selection")
        auto_row = next((r for r in judged if auto and (r["target"]["attempt"], r["target"]["variant"]) == (auto["attempt"], auto["variant"])), None)
        raw_row = next((r for r in judged if auto and (r["target"]["attempt"], r["target"]["variant"]) == (auto["attempt"], "raw")), None)
        rows.append(dict(case=case, judgements=len(judged), first_pass=bool(first and first[0].get("quality_pass") is True),
                         selection_pass=bool(passed), passed_by=how, candidates=len(_candidate_kinds(outcome)) or None,
                         auto_selected=bool(auto), auto_accepted=None if not auto_row else auto_row.get("quality_pass"),
                         auto_worsened=bool(auto_row and raw_row and raw_row.get("quality_pass") is True and auto_row.get("quality_pass") is False),
                         failures=sorted({c for r in judged for c in categories(r)}) if not passed else []))
        for r in judged:
            a = attempts.get(r["target"]["attempt"])
            if not a or r["target"]["variant"] != "raw":
                continue
            items = ((a.get("observation") or {}).get("verification") or {}).get("items") or {}
            human = set(categories(r))
            for key, cat in DETECTORS.items():
                strong = (items.get(key) or {}).get("status") in ("changed", "added", "missing")
                review = bool((items.get(key) or {}).get("review"))
                d = detection[cat]
                if cat in human:
                    d["caught" if strong else ("caught_review_only" if review else "missed")] += 1
                elif r.get("quality_pass") is True:
                    d["false_alarm" if strong else ("review_on_pass" if review else "quiet_pass")] += 1
    n = len(rows)
    by_category = defaultdict(set)
    for r in rows:
        for c in r["failures"]:
            by_category[c].add(r["case"].split("-")[0])
    return dict(
        schema="free_generation_metrics/1", rows=n,
        human_judgements=sum(r["judgements"] for r in rows), judgements_per_row=round(sum(r["judgements"] for r in rows) / n, 2) if n else None,
        first_pass=sum(r["first_pass"] for r in rows), selection_pass=sum(r["selection_pass"] for r in rows),
        passed_by={k: sum(r["passed_by"] == k for r in rows) for k in ("first", "retry", "hair_restored", "hair_auto")},
        auto_selected=sum(r["auto_selected"] for r in rows), auto_accepted=sum(r["auto_accepted"] is True for r in rows),
        auto_worsened=sum(r["auto_worsened"] for r in rows),
        failure_categories={c: dict(rows=sum(c in r["failures"] for r in rows), characters=sorted(by_category[c]),
                                    repeated=len(by_category[c]) >= 2) for c in TAXONOMY if by_category[c]},
        detection={c: dict(v) for c, v in detection.items()},
        attempt_minutes=round(sum(minutes) / len(minutes), 1) if minutes else None,
        per_row=rows,
        note="detection: 사람이 그 분류로 실패시킨 원 결과에 자동 검사가 위반(changed·added·missing)을 냈으면 caught, '확인 필요'만이면 "
             "caught_review_only, 아무것도 없으면 missed. 사람 통과 원 결과에 위반이면 false_alarm, 확인 필요만이면 review_on_pass. "
             "attempt_minutes는 실행 전체 시간/시도 수(분석 포함 근사). repeated = 2명 이상 → 공통 변경 후보(D-140 조건 1·2).")
