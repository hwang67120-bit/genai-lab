"""S14 고정 특징표 맞추기(D-228, CPU): 사용자가 % 판정한 결과들에 자동 고정 특징 비교를 적용해 판정과 맞는지 본다.
사용: python anchor_calibration_s14.py <출력 json>
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
from PIL import Image
from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
from genai_lab.identity_anchors import anchors, compare, summary_ko

B = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025\character-style-binding-20261027")
# (이름, 원본, 결과 그림들, 사용자 판정 %)
CASES = [
    ("U002", B / "s6b-scene-unseen-20261031/U002-base/character.png", [B / f"s6b-scene-unseen-20261031/U002-{s}.png" for s in ("walk", "bench", "library")], 90),
    ("U003(꼬리 전)", B / "s6b-scene-unseen-20261031/U003-base/character.png", [B / f"s6b-scene-unseen-20261031/U003-{s}.png" for s in ("walk", "bench", "library")], 85),
    ("U003(꼬리 후)", B / "s6c-scene-tail-20261031/U003-base/character.png", [B / f"s6c-scene-tail-20261031/U003-{s}.png" for s in ("walk", "bench", "library")], 90),
    ("U004", B / "s6b-scene-unseen-20261031/U004-base/character.png", [B / f"s6b-scene-unseen-20261031/U004-{s}.png" for s in ("walk", "bench", "library")], 90),
    ("U005", B / "s6b-scene-unseen-20261031/U005-base/character.png", [B / f"s6b-scene-unseen-20261031/U005-{s}.png" for s in ("walk", "bench", "library")], 85),
    ("U006", B / "s6b-scene-unseen-20261031/U006-base/character.png", [B / f"s6b-scene-unseen-20261031/U006-{s}.png" for s in ("walk", "bench", "library")], 90),
    ("U007", B / "s6c-scene-tail-20261031/U007-base/character.png", [B / f"s6c-scene-tail-20261031/U007-{s}.png" for s in ("walk", "bench", "library")], 85),
    ("U008", B / "s6c-scene-tail-20261031/U008-base/character.png", [B / f"s6c-scene-tail-20261031/U008-{s}.png" for s in ("walk", "bench", "library")], 80),
    ("U009", B / "s13-setB-20261031/U009-base/character.png", [B / f"s13-setB-20261031/U009-{s}.png" for s in ("walk", "bench", "library")], 85),
]


def main():
    rows = []
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(r"D:\genai-cache\huggingface"), local_files_only=True)) as tagger:
        tag = lambda p: dict(tagger.analyze(Image.open(p).convert("RGB")).raw_general_scores)
        for name, original, results, user in CASES:
            found = anchors(tag(original))
            reports = [compare(found, tag(r)) for r in results]
            scores = [r["score"] for r in reports if r["score"] is not None]
            row = dict(name=name, user=user, anchors=[f"{a['label']}:{a['tag']}" for a in found],
                       auto_mean=round(sum(scores) / len(scores), 3) if scores else None, per_scene=[summary_ko(r) for r in reports])
            rows.append(row)
            print(f"{name} 사용자 {user}% · 자동 {row['auto_mean']} · 특징 {row['anchors']}")
            for s in row["per_scene"]:
                print("    ", s)
    Path(sys.argv[1]).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
