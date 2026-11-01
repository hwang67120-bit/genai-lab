"""S32 자동 동일성 점수 맞춰 보기(사용자 11/1 "수동 조정이 아닌 AI가 스스로 고칠 수 있는 계산법"): CCIP(deepghs/ccip_onnx
eb2acdd2, OpenRAIL, 같은 캐릭터인지 학습한 판별기)의 차이값이 사용자 동일성 판정과 맞는가. 모델 기준: 차이 0.178 미만 = 같은 캐릭터.
묶음: 사용자가 % 판정한 장면 결과(D-219·D-221: 80~90%), 오늘 LoRA 결과(귀가 여우 귀로 바뀐 것·되돌린 것), F1 털 수인(털·동물 얼굴을
잃음), 서로 다른 캐릭터 원본끼리(같으면 안 됨). CPU, 이 PC.
사용: python ccip_calibration_s32.py <출력 폴더>
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import itertools, json, sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
from PIL import Image

CACHE = Path(r"D:\genai-cache\huggingface")
OUT = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025")
B = OUT / "character-style-binding-20261027"
JUDGED = {("s6b-scene-unseen-20261031", "U002"): 90, ("s6b-scene-unseen-20261031", "U003"): 85, ("s6b-scene-unseen-20261031", "U004"): 90,
          ("s6b-scene-unseen-20261031", "U005"): 85, ("s6b-scene-unseen-20261031", "U006"): 90, ("s6c-scene-tail-20261031", "U003"): 90,
          ("s6c-scene-tail-20261031", "U007"): 85, ("s6c-scene-tail-20261031", "U008"): 80}


def ccip():
    import huggingface_hub.constants as hub
    hub.HF_HUB_CACHE = str(CACHE)
    from imgutils.metrics import ccip_difference, ccip_extract_feature
    return ccip_extract_feature, ccip_difference


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    extract, difference = ccip()
    feat = {}
    f = lambda p: feat.setdefault(str(p), extract(Image.open(p).convert("RGB")))
    rows = []
    for (folder, name), percent in JUDGED.items():
        base = B / folder / f"{name}-base" / "character.png"
        for scene in ("bench", "library", "walk"):
            result = B / folder / f"{name}-{scene}.png"
            if result.exists():
                rows.append(dict(group="judged_scene", name=name, case=f"{folder[:3]}-{scene}", user_percent=percent,
                                 diff=round(float(difference(f(base), f(result))), 3)))
    s30 = OUT / "flux-style-lora-product-s30-20261101"
    for run in sorted(s30.glob("**/runs/*/flux")):
        base = run / "character.png"
        for seed in (42, 43, 44):
            rows.append(dict(group="lora_product", name=run.parent.parent.parent.name + "/" + run.parent.name[:6], case=f"seed{seed}",
                             diff=round(float(difference(f(base), f(run / f"result-raw-seed{seed}.png"))), 3)))
    f1 = next((OUT / "furry-autolearn-s31-20261101" / "baseline" / "runs").glob("*/flux"))
    for seed in (42, 43, 44):
        rows.append(dict(group="furry_lost", name="F1", case=f"seed{seed}",
                         diff=round(float(difference(f(f1 / "character.png"), f(f1 / f"result-raw-seed{seed}.png"))), 3)))
    originals = {name: B / folder / f"{name}-base" / "character.png" for folder, name in JUDGED}
    originals["F1"] = f1 / "character.png"
    for a, b in itertools.combinations(sorted(originals), 2):
        rows.append(dict(group="different_characters", name=f"{a}-{b}", case="originals",
                         diff=round(float(difference(f(originals[a]), f(originals[b]))), 3)))
    summary = {}
    for g in dict.fromkeys(r["group"] for r in rows):
        d = sorted(r["diff"] for r in rows if r["group"] == g)
        summary[g] = dict(n=len(d), min=d[0], median=d[len(d) // 2], max=d[-1], under_0178=sum(x < 0.178 for x in d))
    (out / "ccip-calibration.json").write_text(json.dumps(dict(model="deepghs/ccip_onnx ccip-caformer-24-randaug-pruned eb2acdd2 (OpenRAIL)",
                                                               threshold=0.178, summary=summary, rows=rows), ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    for g, s in summary.items():
        print(g, s)
    for r in rows:
        if r["group"] != "different_characters":
            print(r["group"], r["name"], r["case"], r.get("user_percent", ""), r["diff"])


if __name__ == "__main__":
    main()
