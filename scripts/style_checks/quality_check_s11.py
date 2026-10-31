"""S11 품질 문제 확인(사용자 10/31 "(나) 품질 문제"): 같은 입력·같은 장면으로 고치기 전/후를 비교한다(같은 입력은 비트 단위 재현, D-223).
묶음 A(문제가 보인 캐릭터): Q1 U003 + 황금빛 역광 장면(머리색 회색), Q2 U005 + 벤치(열린 셔츠 잠김), Q3 U002 + 비 오는 거리 걷기(걷기·간판 글자).
사용: python quality_check_s11.py <출력 폴더> <before|after>
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
from genai_lab.scene_composer import compose_scene
from genai_lab.style_redraw_pipeline import run_style_redraw

DL = Path(r"C:\Users\user\Downloads")
CASES = [
    ("Q1-U003-backlight", DL / "수인캐릭터/51eed7ade89a8e6c31f1505081611335.jpg", compose_scene(23)["text"]),
    ("Q2-U005-bench", DL / "참조 이미지/f87a8a7331065fff23e7ffb25641dc58.jpg",
     "sitting on a wooden park bench on an autumn afternoon, relaxed, hands resting on the knees"),
    ("Q3-U002-walk", DL / "참조 이미지/734da0ec4fe594cacfddd664a143af2a.jpg",
     "walking along a rainy city street at night with glowing shop signs, mid-stride, one hand in a pocket"),
]


def main():
    out, label = Path(sys.argv[1]), sys.argv[2]
    rows = []
    for name, character, scene in CASES:
        t0 = time.time()
        o = run_style_redraw(character, out / label / name, scene=scene, adult_confirmed=True)
        row = dict(name=name, state=o["state"], seconds=round(time.time() - t0), scene=scene)
        if o["state"] == "review_pending":
            prepared = json.loads((out / label / name / "prepare.json").read_text(encoding="utf-8"))
            row.update(traits=[(t["name"], t["text"]) for t in prepared["traits"]], prompt=prepared["prompt"])
        rows.append(row)
        print(name, label, o["state"], row["seconds"], "s", flush=True)
    (out / f"{label}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
