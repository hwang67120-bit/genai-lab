"""다른 GPU 일관성 비교(D-231, CPU): A100에서 만든 원래 그림 ↔ 이 PC(RTX 4060) run-a 원래 그림. 같은 입력·설정·패키지 버전.
비교: 원래 그림 SHA·픽셀 차이, 그리고 A100 원래 그림에 이 PC와 같은 후처리(색 맞춤·손 개수·부위 태그·고정 특징)를 적용한 값.
사용: python xgpu_compare_s21.py
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
import numpy as np
from PIL import Image, ImageDraw

B = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025\character-style-binding-20261027")
LOCAL, REMOTE = B / "s20-consistency-highres-20261031", B / "s21-cross-gpu-a100-20261031"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from style_redraw_worker import count_hands, person_and_parts
    from genai_lab.style_redraw_flux import IDENTITY_PARTS, IDENTITY_SHIFT_CAP, PARTS, body_part_tags, colour_match, missing_body_parts
    from genai_lab.identity_anchors import compare, summary_ko
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    report = {}
    sheets = []
    for case in ("K1-V002-garment-scene", "K2-U003-scene"):
        local = LOCAL / case / "run-a"
        req = json.loads((local / "request.json").read_text(encoding="utf-8"))
        prepared = json.loads((local / "prepare.json").read_text(encoding="utf-8"))
        gen = {c["seed"]: c for c in json.loads((local / "generation.json").read_text(encoding="utf-8"))["candidates"]}
        original = np.asarray(Image.open(local / "character.png").convert("RGB"))
        om = dict(np.load(local / "original-masks.npz"))
        rows = []
        raws = {s: np.asarray(Image.open(REMOTE / case / f"result-raw-seed{s}.png").convert("RGB")) for s in (42, 43, 44)}
        hands = dict(zip((42, 43, 44), count_hands([raws[s] for s in (42, 43, 44)], req)))
        with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(req["model_cache"]), local_files_only=True)) as tagger:
            for seed in (42, 43, 44):
                a = np.asarray(Image.open(local / f"result-raw-seed{seed}.png").convert("RGB"), np.int16)
                b = raws[seed].astype(np.int16)
                d = np.abs(a - b)
                _, masks = person_and_parts(raws[seed], req)
                scene = bool(prepared.get("scene"))
                fixed, shifts = (colour_match(raws[seed], original, masks, om, parts=IDENTITY_PARTS, cap=IDENTITY_SHIFT_CAP, scene_lighting=scene)
                                 if prepared.get("with_garment") else colour_match(raws[seed], original, masks, om, parts=PARTS, scene_lighting=scene))
                Image.fromarray(fixed).save(REMOTE / case / f"result-seed{seed}.png")
                scores = dict(tagger.analyze(Image.fromarray(fixed)).raw_general_scores)
                anchors = compare(prepared.get("identity_anchors") or [], scores)
                here = gen[seed]
                row = dict(seed=seed, raw_same_sha=sha(local / f"result-raw-seed{seed}.png") == sha(REMOTE / case / f"result-raw-seed{seed}.png"),
                           pixel_max=int(d.max()), pixel_mean=round(float(d.mean()), 3), changed_fraction=round(float((d.max(-1) > 8).mean()), 4),
                           hands=(here.get("hands"), hands[seed]), missing=(here.get("missing_body_parts"), missing_body_parts(prepared.get("body_parts"), scores)),
                           anchors=(summary_ko(here.get("identity_anchors")), summary_ko(anchors)),
                           part_tags_same=here.get("body_part_tags") == body_part_tags(scores),
                           colour_shift=(here.get("colour_shift"), shifts))
                rows.append(row)
                print(case, json.dumps({k: v for k, v in row.items() if k != "colour_shift"}, ensure_ascii=False), flush=True)
                sheets.append((case, seed, local / f"result-seed{seed}.png", REMOTE / case / f"result-seed{seed}.png"))
        report[case] = rows
    (REMOTE / "cross-gpu-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    W, H = 240, 319
    sheet = Image.new("RGB", (W * 6, (H + 22) * 2), "white"); dr = ImageDraw.Draw(sheet)
    for i, (case, seed, a, b) in enumerate(sheets):
        r, c = divmod(i, 3)
        for j, (p, t) in enumerate(((a, "RTX4060"), (b, "A100"))):
            sheet.paste(Image.open(p).convert("RGB").resize((W, H)), ((c * 2 + j) * W, r * (H + 22) + 22))
            dr.text(((c * 2 + j) * W + 4, r * (H + 22) + 5), f"{case[:6]} s{seed} {t}", fill="black")
    sheet.save(REMOTE / "cross-gpu-pairs.png")


if __name__ == "__main__":
    main()
