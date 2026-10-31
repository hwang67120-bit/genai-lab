"""S9 일관성 시험(사용자 10/31 "제품화할 수 있는 일관적인 값"): 같은 입력을 정식 진입점(run_style_redraw)에 두 번 넣고 비교한다.
같은 PC, 같은 코드. 장면 문장은 고정(화면은 누를 때마다 새 장면을 고르는 것이 설계이므로 그 무작위는 뺀다).
비교: 준비 단계(특징 문장·부위·최종 지시문), 후보별 생성 그림·최종 그림(SHA·픽셀 차이), 색 이동량·손 개수·부위 태그.
사용: python consistency_check_s9.py <출력 폴더>
"""
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
import numpy as np
from PIL import Image
from genai_lab.scene_composer import compose_scene
from genai_lab.style_redraw_pipeline import run_style_redraw

PREP = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025")
DL = Path(r"C:\Users\user\Downloads")
CASES = [
    ("K1-V002-garment-scene", PREP / "style-review-20261026/images/V002.png", DL / "참조 의상/c8dddb317bb3355b7727407d4dce9c30.jpg", compose_scene(11)["text"]),
    ("K2-U003-scene", DL / "수인캐릭터/51eed7ade89a8e6c31f1505081611335.jpg", None, compose_scene(23)["text"]),
]
PREPARE_KEYS = ("traits", "body_parts", "prompt", "original_hands", "warnings", "with_garment")
CANDIDATE_KEYS = ("colour_shift", "hands", "extra_hands", "missing_body_parts", "body_part_tags")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_diff(a, b):
    x = np.asarray(Image.open(a).convert("RGB"), np.int16)
    y = np.asarray(Image.open(b).convert("RGB"), np.int16)
    d = np.abs(x - y)
    return dict(max=int(d.max()), mean=round(float(d.mean()), 4), changed_fraction=round(float((d.max(-1) > 0).mean()), 6))


def compare(a, b):
    pa, pb = (json.loads((d / "prepare.json").read_text(encoding="utf-8")) for d in (a, b))
    ga, gb = (json.loads((d / "generation.json").read_text(encoding="utf-8")) for d in (a, b))
    report = dict(prepare={k: pa.get(k) == pb.get(k) for k in PREPARE_KEYS}, candidates=[])
    for ca, cb in zip(ga["candidates"], gb["candidates"]):
        row = dict(seed=ca["seed"])
        for kind, key in (("raw", "raw_file"), ("final", "file")):
            same = sha(a / ca[key]) == sha(b / cb[key])
            row[kind] = dict(same_sha=same, **({} if same else pixel_diff(a / ca[key], b / cb[key])))
        row["values_same"] = {k: ca.get(k) == cb.get(k) for k in CANDIDATE_KEYS}
        row["values_a"] = {k: ca.get(k) for k in ("colour_shift", "hands", "missing_body_parts")}
        row["values_b"] = {k: cb.get(k) for k in ("colour_shift", "hands", "missing_body_parts")}
        report["candidates"].append(row)
    return report


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, character, garment, scene in CASES:
        dirs = []
        for run in ("a", "b"):
            d = out / name / f"run-{run}"
            t0 = time.time()
            o = run_style_redraw(character, d, garment_path=garment, scene=scene)
            print(name, run, o["state"], round(time.time() - t0), "s", flush=True)
            if o["state"] != "review_pending":
                raise SystemExit(f"{name} {run}: {o}")
            dirs.append(d)
        reports[name] = compare(*dirs)
        print(json.dumps({name: reports[name]}, ensure_ascii=False), flush=True)
    (out / "consistency-report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
