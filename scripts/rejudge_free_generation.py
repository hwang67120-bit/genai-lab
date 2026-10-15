"""저장된 자유 생성 실행을 현재 자동 관찰 규칙으로 다시 판정한다(CPU, D-126 확인용). 생성은 하지 않는다.

사용: python -B scripts/rejudge_free_generation.py RUN_DIR [RUN_DIR ...] --out OUT_DIR [--json]
RUN_DIR: run_free_generation 실행 폴더(trace.json, inputs/canvas.png, base-parts/parts.npz, result-analysis*/canvas.png·parts).
기준 특징표는 기록된 사용자 확정값·기준 태그로 지금 규칙의 extract_features를 다시 돌려 만든다(두 눈 색·머리 군집 5개 포함).
결과 태그 점수는 OUT_DIR/result-scores.json에 캐시한다. 원래 실행 폴더는 바꾸지 않는다.
OUT_DIR/<실행>-reference.png: 지금 규칙의 머리 참조 크롭(사람이 먼저 보는 용도, D-126 ③).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from genai_lab.character_features import DISTINCTIVE, extract_features  # noqa: E402
from genai_lab.character_verification import retry_reasons, verify_features  # noqa: E402
from genai_lab.free_generation import garment_tag_changes, request_conflicts, request_observations, staging_text  # noqa: E402
from genai_lab.free_generation_pipeline import head_box, WIDE_REFERENCE  # noqa: E402

WATCH = ("eye_color", "hair_color", "head_accessories", "skin", "body_parts")


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_parts(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].astype(bool) for k in data.files}


def reference_sheet(run, trace):
    facts = trace["input_facts"]
    confirmed = dict(facts["confirmed_features"]["value"])
    confirmed.setdefault(DISTINCTIVE, None)
    features = extract_features(load_rgb(run / "inputs" / "canvas.png"), load_parts(run / "base-parts" / "parts.npz"),
                                facts["automatic_tags"]["scores"], gender=facts["gender"]["value"],
                                body_parts=facts["body_parts"]["value"], confirmed=confirmed)
    for key, values in confirmed.items():           # 파이프라인과 같이: 모름은 unknown
        if values is None and key in features["items"]:
            features["items"][key].update(state="unknown", reason="사용자가 '모름'으로 둔 항목")
    return features


def short(item):
    extra = item.get("reason") or ((item.get("pixels") or {}).get("reason") if item["status"] == "unknown" else "") or ""
    return (item["status"] + (f"({extra})" if extra else "") + ("!" if item.get("confident") else "")
            + (f"[확인 필요: {'; '.join(item['review'])}]" if item.get("review") else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "result-scores.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    tagger = None
    rows = []
    try:
        for run in args.runs:
            trace = json.loads((run / "trace.json").read_text(encoding="utf-8"))
            reference = reference_sheet(run, trace)
            base = Image.open(run / "inputs" / "canvas.png").convert("RGB")
            box = head_box(load_parts(run / "base-parts" / "parts.npz"), base.size)
            if box:
                base.crop(box).save(args.out / f"{run.name}-reference.png")
                ratio = (box[2] - box[0]) * (box[3] - box[1]) / (base.width * base.height)
            request = trace["input_facts"]["request"]["value"]
            seed = next((c["basis"]["seed"] for c in trace["conditions"] if c["key"] == "request" and isinstance(c.get("basis"), dict)), 42)
            request = staging_text(request, seed)          # 빈 요청이면 실제로 고른 고정 자세(D-128)
            skip = request_conflicts(request)
            base_scores = trace["input_facts"]["automatic_tags"]["scores"]
            for attempt in trace["attempts"]:
                suffix = "" if attempt["number"] == 1 else f"-{attempt['number']}"
                analysis = run / f"result-analysis{suffix}"
                key = f"{run.name}/{analysis.name}"
                if key not in cache:
                    if tagger is None:
                        from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
                        from genai_lab.studio_generation import StudioRuntime
                        runtime = StudioRuntime.from_environment()
                        tagger = WdTagSession(ClothingDesignAnalysisSettings(cache_dir=str(runtime.model_cache), local_files_only=True))
                        tagger.__enter__()
                    with Image.open(analysis / "canvas.png") as image:
                        cache[key] = dict(tagger.analyze(image.convert("RGB")).raw_general_scores)
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")
                report = verify_features(reference, cache[key], load_rgb(analysis / "canvas.png"),
                                         load_parts(analysis / "parts" / "parts.npz"), skip)
                old = attempt["observation"]["verification"]
                rows.append(dict(run=run.name, attempt=attempt["number"], seed=attempt["seed"], request=request,
                                 old_retry_reason=attempt.get("retry_reason"), old_summary=old.get("summary"),
                                 new_retry=retry_reasons(report), new_items={k: short(report["items"][k]) for k in WATCH if k in report["items"]},
                                 new_hair_colors=report["items"]["hair_color"].get("new_colors"),
                                 reduced_hair_colors=report["items"]["hair_color"].get("reduced_colors"),
                                 request_tags=request_observations(request, cache[key]),
                                 garment_tags=garment_tag_changes(base_scores, cache[key]),
                                 reference_box=list(box) if box else None, reference_scope=("wide" if box and ratio > WIDE_REFERENCE else "head") if box else None,
                                 old_reference_box=trace["input_facts"]["derived_reference"]["box"]))
    finally:
        if tagger is not None:
            tagger.__exit__(None, None, None)
    (args.out / "rejudge.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for r in rows:
        changed = r["garment_tags"]
        print(f"{r['run']} #{r['attempt']} seed{r['seed']}  예전 재시도: {r['old_retry_reason'] or '-'}  →  지금 재시도: {', '.join(r['new_retry']) or '-'}")
        print("    " + "  ".join(f"{k}={v}" for k, v in r["new_items"].items()))
        if r["new_hair_colors"]:
            print("    새 머리색:", [(c["rgb"], c["fraction"]) for c in r["new_hair_colors"]])
        if r["reduced_hair_colors"]:
            print("    줄어든 머리색:", [(c["rgb"], c["reference_share"], c["result_share"]) for c in r["reduced_hair_colors"]])
        for t in r["request_tags"]:
            print(f"    요청 관찰 {t['key']}: {t['status']} ({t.get('score')})")
        if changed.get("lost") or changed.get("added"):
            print("    의상 태그: 없어짐", changed["lost"], "생김", changed["added"])
    print(f"참조 크롭: {sorted({(r['run'], tuple(r['old_reference_box']), tuple(r['reference_box'] or ())) for r in rows})}")


if __name__ == "__main__":
    main()
