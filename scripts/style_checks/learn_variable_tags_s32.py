"""S32 '장면 때문에 바뀌어도 되는 태그'를 데이터에서 배운다(사용자 11/1 "수동 조정이 아닌 자동 조정").
지금까지 FLUX로 다시 그린 원본·결과 쌍(작업 폴더의 character.png ↔ result-raw-seed*.png, 장면 시험의 <이름>-base ↔ <이름>-<장면>.png)마다
WD 태그를 비교해, 원본에 확실히(0.5 이상) 있던 태그가 결과에서 사라진(0.35 미만) 비율을 센다. 서로 다른 캐릭터 3명 이상에서
절반 이상 사라진 태그 = 자세·배경·표정·구도처럼 장면을 따라 바뀌는 태그. 나머지(한 번도 못 본 태그 포함)는 기본으로 '지켜야 할 태그'.
사람이 항목을 고르지 않는다 — 털 수인(F1)의 furry·animal_nose처럼 처음 보는 특징도 자동으로 지켜야 할 쪽에 든다.
사용: python learn_variable_tags_s32.py <출력 json>   (CPU, WD 태거 로컬 캐시)
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
from PIL import Image

ROOT = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025")
PRESENT, GONE = 0.5, 0.35
MIN_CHARACTERS, MIN_LOSS_RATE = 3, 0.5


def pairs():
    found = []
    for character in ROOT.rglob("character.png"):
        folder = character.parent
        results = sorted(folder.glob("result-raw-seed*.png"))
        if not results and folder.name.endswith("-base"):
            name = folder.name[:-5]
            results = [p for p in folder.parent.glob(f"{name}-*.png") if not p.stem.endswith("-raw") and p.stem.split("-")[-1] != "base"]
        found += [(character, r) for r in results]
    return found


def main():
    out = Path(sys.argv[1])
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    found = pairs()
    scores, sha = {}, {}
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(r"D:\genai-cache\huggingface"), local_files_only=True)) as tagger:
        for path in dict.fromkeys(p for pair in found for p in pair):
            scores[path] = dict(tagger.analyze(Image.open(path).convert("RGB")).raw_general_scores)
            sha[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    present, lost = defaultdict(set), defaultdict(int)
    seen = defaultdict(int)
    for original, result in found:
        who = sha[original]
        for tag, value in scores[original].items():
            if value >= PRESENT:
                seen[tag] += 1
                present[tag].add(who)
                if scores[result].get(tag, 0.0) < GONE:
                    lost[tag] += 1
    variable = {t: dict(lost=lost[t], present=seen[t], characters=len(present[t]), rate=round(lost[t] / seen[t], 2))
                for t in seen if len(present[t]) >= MIN_CHARACTERS and lost[t] / seen[t] >= MIN_LOSS_RATE}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        schema=1, made=time.strftime("%Y-%m-%d %H:%M:%S"), rule=dict(present=PRESENT, gone=GONE, min_characters=MIN_CHARACTERS,
                                                                    min_loss_rate=MIN_LOSS_RATE),
        source=dict(pairs=len(found), characters=len({sha[o] for o, _ in found}), root=str(ROOT)),
        note="장면을 따라 바뀌는 태그(데이터에서 배움). 여기에 없는 원본 태그는 모두 지켜야 할 태그.",
        variable=dict(sorted(variable.items()))), ensure_ascii=False, indent=1), encoding="utf-8")
    print("pairs", len(found), "characters", len({sha[o] for o, _ in found}), "variable", len(variable))
    print(sorted(variable))


if __name__ == "__main__":
    main()
