"""저장된 옷 바꾸기 실행 폴더에서 기준 캐릭터 특징표를 뽑는다(CPU, 2단계 1번 / D-118).

사용: python -B scripts/extract_character_features.py RUN_DIR [RUN_DIR ...] [--sheet PATH]
RUN_DIR에는 inputs/canvas.png 와 parts/parts.npz 가 있어야 한다(요청 14~19, 작업실 outfit-edit 폴더).
태그는 태그 분석기를 CPU로 돌리고, 성별·부위는 qwen-inputs/conditions.json의 input_facts(확정값)에서 읽는다.
결과: RUN_DIR/character-features/features.json 과 한 줄 요약. --sheet 를 주면 머리·눈·피부 색 견본과 요약을 한 장에 모은다.
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
from genai_lab.character_features import extract_features, summary_line, write_features  # noqa: E402


def confirmed_facts(run):
    """conditions.json의 input_facts에서 확정된 성별·부위를 읽는다. 없으면 unknown."""
    path = run / "qwen-inputs" / "conditions.json"
    if not path.exists():
        return "unknown", None, {}
    facts = json.loads(path.read_text(encoding="utf-8")).get("input_facts", {})
    gender = facts.get("gender", {}).get("value", "unknown")
    parts_fact = facts.get("body_parts") or {}
    states = {k: v for k, v in parts_fact.items() if isinstance(v, dict) and "state" in v}
    if states and all(v["state"] != "unknown" for v in states.values()):
        body_parts = tuple(k for k, v in states.items() if v["state"] == "present")
    else:
        body_parts = None
    provenance = dict(gender=facts.get("gender", {}).get("source", "user"),
                      body_parts=next((v.get("source", "user") for v in states.values()), "user"))
    return gender, body_parts, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--sheet", type=Path)
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
            parts = {k: v.astype(bool) for k, v in np.load(parts_path).items()}
            scores = dict(tagger.analyze(Image.fromarray(image)).raw_general_scores)
            gender, body_parts, provenance = confirmed_facts(run)
            features = extract_features(image, parts, scores, gender=gender, body_parts=body_parts, provenance=provenance)
            path = write_features(run / "character-features", features)
            print(f"{run.name}: {summary_line(features)}  → {path}")
            rows.append((run.name, image, features))
    if args.sheet and rows:
        import cv2
        tiles = []
        for name, image, features in rows:
            it = features["items"]
            thumb = cv2.resize(image, (150, 200))[:, :, ::-1]
            swatches = np.full((200, 60, 3), 255, np.uint8)
            y = 0
            for color in it["hair_color"]["colors"][:3]:
                swatches[y:y + 40, :] = color["rgb"][::-1]; y += 40
            if it["eye_color"]["color"]:
                swatches[140:170, :] = it["eye_color"]["color"]["rgb"][::-1]
            if it["skin"]["color"]:
                swatches[170:200, :] = it["skin"]["color"]["rgb"][::-1]
            text = np.full((200, 900, 3), 255, np.uint8)
            cv2.putText(text, name, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            lines = [f"hair: {it['hair_length']['value']} {' '.join(it['hair_style']['values'][:5])}",
                     f"hair color tags: {' '.join(it['hair_color']['tags'][:4])}",
                     f"eyes: {' '.join(it['eye_color']['tags'][:3])}   accessories: {' '.join(it['head_accessories']['values'][:5])}",
                     f"skin tags: {' '.join(it['skin']['tags'])}   body parts: {it['body_parts']['values']}   gender: {it['gender']['value']}",
                     f"unknown: {features['unknown']}"]
            for i, line in enumerate(lines):
                cv2.putText(text, line, (6, 54 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            tiles.append(np.concatenate([thumb, swatches, text], 1))
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.sheet), np.concatenate(tiles, 0))
        print("sheet:", args.sheet)


if __name__ == "__main__":
    main()
