"""저장된 실행에서 결과 검증기(D-121)를 돌려 기준 특징표가 결과에서 유지됐는지 적고, 사람 판정과 대조한다(CPU).

사용: python -B scripts/verify_character_features.py RUN_DIR [RUN_DIR ...] [--json]
RUN_DIR: inputs/canvas.png, parts/parts.npz, qwen/raw.png(또는 compose/aligned.png)가 있는 실행 폴더.
기준 특징표는 RUN_DIR/character-features/features.json(없으면 그 자리에서 추출). 결과는 compose/aligned.png(원본 좌표),
없으면 qwen/raw.png를 캔버스 크기로 줄인 것. 결과를 RUN_DIR/character-features/verification.json에 쓴다.
judgement.json이 있으면 body_parts·same_character 사람 판정을 나란히 보여 준다.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from genai_lab.character_features import extract_features, write_features  # noqa: E402
from genai_lab.character_verification import verify_features  # noqa: E402
from extract_character_features import confirmed_facts  # noqa: E402


def result_image(run, size):
    aligned = run / "compose" / "aligned.png"
    raw = run / "qwen" / "raw.png"
    path = aligned if aligned.exists() else raw
    if not path.exists():
        return None, None
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.LANCZOS)
        return np.asarray(image), path.name


def human_judgement(run):
    path = run / "judgement.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    users = [r for r in records if r.get("judge") == "user"] or records
    latest = users[-1]
    return dict(judge=latest["judge"], body_parts=latest.get("body_parts"), same_character=latest.get("same_character"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.studio_generation import StudioRuntime
    runtime = StudioRuntime.from_environment()
    settings = ClothingDesignAnalysisSettings(cache_dir=str(runtime.model_cache), local_files_only=True)
    rows = []
    with WdTagSession(settings) as tagger:
        for run in args.runs:
            canvas, parts_path = run / "inputs" / "canvas.png", run / "parts" / "parts.npz"
            if not canvas.exists() or not parts_path.exists():
                print(f"{run}: canvas 또는 parts 없음 — 건너뜀")
                continue
            with Image.open(canvas) as source:
                image = np.asarray(source.convert("RGB"))
                size = source.size
            parts = {k: v.astype(bool) for k, v in np.load(parts_path).items()}
            features_path = run / "character-features" / "features.json"
            if features_path.exists():
                features = json.loads(features_path.read_text(encoding="utf-8"))
            else:
                scores = dict(tagger.analyze(Image.fromarray(image)).raw_general_scores)
                gender, body_parts, provenance = confirmed_facts(run)
                features = extract_features(image, parts, scores, gender=gender, body_parts=body_parts, provenance=provenance)
                write_features(run / "character-features", features)
            result, used = result_image(run, size)
            if result is None:
                print(f"{run}: 결과 그림 없음 — 건너뜀")
                continue
            result_scores = dict(tagger.analyze(Image.fromarray(result)).raw_general_scores)
            report = dict(verify_features(features, result_scores, result, parts), result_image=used, status="measured")
            (run / "character-features" / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            human = human_judgement(run)
            row = dict(run=f"{run.parent.name}/{run.name}", result=used, summary=report["summary"],
                       body_parts=report["items"]["body_parts"], gender=report["items"]["gender"]["status"], human=human)
            rows.append(row)
            if not args.json:
                s = report["summary"]
                print(f"{row['run']} [{used}] kept={s['kept']} changed={s['changed']} missing={s['missing']} added={s['added']} unknown={s['unknown']}"
                      f" | 부위 자동 {report['items']['body_parts']['status']} vs 사람 {human and human['body_parts']}"
                      f" | 같은 캐릭터(사람) {human and human['same_character']}")
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
