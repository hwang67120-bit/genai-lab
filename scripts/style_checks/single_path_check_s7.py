"""S7 FLUX 한 경로 확인(D-222, 사용자 10/31 Qwen 제외 확정): 정식 진입점 run_style_redraw에 옷 그림·장면을 함께 넣는다. 이 PC, GPU.
경우: ① 옷 그림 + 자동 장면, ② 장면만(꼬리 캐릭터), ③ 한국어 사용자 장면. 장면 seed는 고정(같은 입력 비교용).
사용: python single_path_check_s7.py <출력 폴더>
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
from genai_lab.scene_composer import complete_user_scene, compose_scene
from genai_lab.style_redraw_pipeline import run_style_redraw

PREP = Path(r"D:\Code\genai-lab\outputs\style-training-prep-20261025")
DL = Path(r"C:\Users\user\Downloads")
CASES = [
    ("C1-V002-garment-scene", PREP / "style-review-20261026/images/V002.png", DL / "참조 의상/c8dddb317bb3355b7727407d4dce9c30.jpg",
     compose_scene(11)["text"]),
    ("C2-U003-scene", DL / "수인캐릭터/51eed7ade89a8e6c31f1505081611335.jpg", None, compose_scene(23)["text"]),
    ("C3-U002-korean", DL / "참조 이미지/734da0ec4fe594cacfddd664a143af2a.jpg", None,
     complete_user_scene("비 오는 밤 네온 거리에서 우산을 들고 걷는", 5)["text"]),
]


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, character, garment, scene in CASES:
        t0 = time.time()
        o = run_style_redraw(character, out / name, garment_path=garment, scene=scene)
        row = dict(name=name, state=o["state"], message=o.get("message", ""), seconds=round(time.time() - t0), scene=scene,
                   peak=o.get("peak_reserved_gib"))
        if o["state"] == "review_pending":
            gen = json.loads((out / name / "generation.json").read_text(encoding="utf-8"))
            row["candidates"] = [dict(seed=c["seed"], hands=c["hands"], missing=c["missing_body_parts"], tags=c["body_part_tags"])
                                 for c in gen["candidates"]]
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
