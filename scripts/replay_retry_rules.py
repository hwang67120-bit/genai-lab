"""재시도 기준 오프라인 채점(판정 쪽 변경, GPU 없음). 저장된 시도 기록과 사용자 판정으로 '이 기준이었으면' 재시도 수·통과 줄을 다시 센다.

사용: python scripts/replay_retry_rules.py <runs 폴더> <ledger.json> [<runs 폴더> <ledger.json> ...]
규칙은 실제 재시도 이유(retry_items)에서 항목을 빼는 방식만 본다(새 재시도는 만들 수 없음 — 그 그림이 없다).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

SCENE_SKIN_LIGHTER = 15     # D-136 ③ 보류안: 장면 요청에서 피부가 기준보다 L*로 이만큼 넘게 밝아질 때만 위반


def lightness(rgb):
    import numpy as np
    c = np.array(rgb, dtype=float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    y = float(c @ np.array([0.2126, 0.7152, 0.0722]))
    return 116 * (y ** (1 / 3) if y > 0.008856 else 7.787 * y + 16 / 116) - 16


def skin_lighter(attempt):
    pixels = ((((attempt.get("observation") or {}).get("verification") or {}).get("items") or {}).get("skin") or {}).get("pixels") or {}
    if not pixels.get("reference_rgb") or not pixels.get("result_rgb"):
        return None
    return lightness(pixels["result_rgb"]) - lightness(pixels["reference_rgb"])


RULES = {
    "R0 지금": lambda items, a: items,
    "R1 장면 피부는 크게 밝아질 때만": lambda items, a: [i for i in items if i != "skin" or (skin_lighter(a) or 0) > SCENE_SKIN_LIGHTER],
    "R2 피부 빼기": lambda items, a: [i for i in items if i != "skin"],
    "R3 헤어스타일·머리 장식 태그 빼기": lambda items, a: [i for i in items if i not in ("hair_style", "head_accessories")],
    "R4 = R1 + R3": lambda items, a: [i for i in items if i not in ("hair_style", "head_accessories")
                                      and (i != "skin" or (skin_lighter(a) or 0) > SCENE_SKIN_LIGHTER)],
}


def score(runs, ledger):
    verdict = {(r["case"], r["target"]["attempt"], r["target"]["variant"]): r["quality_pass"] for r in ledger}
    rows = []
    for run in sorted(p for p in Path(runs).iterdir() if (p / "trace.json").exists()):
        attempts = json.loads((run / "trace.json").read_text(encoding="utf-8")).get("attempts") or []
        if not attempts:
            continue
        def passed(number):
            return any(v is True for (c, n, _), v in verdict.items() if c == run.name and n == number)
        rows.append(dict(case=run.name, a1=attempts[0], items=list(attempts[0].get("retry_items") or []), retried=len(attempts) > 1,
                         a1_pass=passed(1), a2_pass=passed(2) if len(attempts) > 1 else False,
                         a1_judged=any(c == run.name and n == 1 for c, n, _ in verdict)))
    out = {}
    for name, rule in RULES.items():
        retries = kept = lost = wasted_avoided = 0
        for r in rows:
            retry = r["retried"] and bool(rule(r["items"], r["a1"]))
            retries += retry
            passes = r["a1_pass"] or (retry and r["a2_pass"])
            kept += passes
            lost += (r["retried"] and r["a2_pass"] and not r["a1_pass"] and not retry)
            wasted_avoided += (r["retried"] and not retry and r["a1_pass"])
        out[name] = dict(retries=retries, selection_pass=kept, rows_lost_vs_R0=lost, needless_retries_avoided=wasted_avoided)
    detail = [dict(case=r["case"], items=r["items"], skin_lighter=None if skin_lighter(r["a1"]) is None else round(skin_lighter(r["a1"]), 1),
                   a1_pass=r["a1_pass"], a2_pass=r["a2_pass"]) for r in rows if r["retried"]]
    return dict(rows=len(rows), rules=out, retried_rows=detail)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    total = defaultdict(lambda: defaultdict(int))
    for runs, ledger in zip(args[::2], args[1::2]):
        result = score(runs, json.loads(Path(ledger).read_text(encoding="utf-8")))
        print(f"== {runs} ({result['rows']}줄)")
        for name, v in result["rules"].items():
            print(f"  {name}: {v}")
            for k, x in v.items():
                total[name][k] += x
        for d in result["retried_rows"]:
            print("   ", d)
    print("== 합계")
    for name, v in total.items():
        print(f"  {name}: {dict(v)}")
