"""그림체 LoRA 맛보기 GPU 시험 실행기(D-159, docs/STYLE_STAGE_DESIGN.md 3-3). 시험 전용 — 제품 화면·기준선과 섞지 않는다.

줄마다 같은 캐릭터·같은 장면·같은 seed로 '그림체 LoRA 없음 / 강도 0.6 / 강도 1.0'을 8단계 가속(재시도 없음)으로 만든다.
사용: python scripts/style_trial_run.py <줄 목록 json> <출력 폴더>
  줄 목록: [{"case": "N1", "base": "...", "features": {...}, "gender": "...", "body_parts": [...], "scene_seed": 101}, ...]
출력: <출력 폴더>/runs/<case>-<강도>/, batch-status.json, summary.json, sheet.png([원본 | 없음 | 0.6 | 1.0])
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STRENGTHS = (0.0, 0.6, 1.0)
STYLE = "anime_borrowed"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sheet(out, rows):
    from PIL import Image, ImageDraw
    height, tiles = 360, []
    for row in rows:
        cells = [row["base"]] + [row.get(f"result_{s}") for s in STRENGTHS]
        images = []
        for cell in cells:
            if cell and Path(cell).is_file():
                with Image.open(cell) as im:
                    im = im.convert("RGB")
                    images.append(im.resize((int(im.width * height / im.height), height)))
            else:
                images.append(Image.new("RGB", (270, height), "white"))
        tile = Image.new("RGB", (sum(i.width for i in images) + 10 * len(images), height + 24), "white")
        x = 0
        for label, image in zip(["원본"] + [("없음" if s == 0 else f"강도 {s}") for s in STRENGTHS], images):
            tile.paste(image, (x, 24))
            ImageDraw.Draw(tile).text((x + 4, 4), f"{row['case']} {label}", fill="black")
            x += image.width + 10
        tiles.append(tile)
    canvas = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)), "white")
    y = 0
    for tile in tiles:
        canvas.paste(tile, (0, y))
        y += tile.height
    canvas.save(out / "sheet.png")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    from genai_lab.free_generation_pipeline import run_free_generation
    from genai_lab.qwen_pose_settings import QwenPoseSettings
    from genai_lab.scene_composer import compose_scene
    from genai_lab.studio_generation import StudioRuntime
    cases, out = json.loads(Path(argv[0]).read_text(encoding="utf-8")), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    runtime = StudioRuntime.from_environment()
    settings = QwenPoseSettings.load(runtime.qwen_settings)
    status = dict(state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), trial="style_lora_borrowed",
                  note="그림체 맛보기 — 품질 기준선·운영 집계에 넣지 않는다", rows=[])
    for case in cases:
        scene = compose_scene(case["scene_seed"])
        base = Path(case["base"]) if Path(case["base"]).is_absolute() else ROOT / case["base"]
        row = dict(case=case["case"], base=str(base), scene=scene["text"], scene_seed=case["scene_seed"])
        status["rows"].append(row)
        for strength in STRENGTHS:
            key = f"{case['case']}-{'none' if strength == 0 else strength}"
            start = time.time()
            row[f"state_{strength}"] = "running"
            save(out / "batch-status.json", status)
            try:
                outcome = run_free_generation(
                    str(base), case["features"], scene["text"], base_kind="original", gender=case["gender"],
                    body_parts=list(case["body_parts"]), confirmed=True, adult_confirmed=True, settings=settings, runtime=runtime,
                    directory=out / "runs" / key, selection_seed=42, experiment=dict(lightning8=True, single_attempt=True),
                    style_trial=None if strength == 0 else dict(name=STYLE, strength=strength))
                row[f"result_{strength}"] = outcome["result"]
                row[f"state_{strength}"] = outcome["status"]
            except Exception as error:  # noqa: BLE001 — 줄 실패는 기록하고 다음으로
                row[f"state_{strength}"] = "failed"
                row[f"error_{strength}"] = traceback.format_exc()[-2000:] if not str(error) else str(error)[:500]
            row[f"seconds_{strength}"] = round(time.time() - start)
            save(out / "batch-status.json", status)
    status.update(state="finished", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(out / "batch-status.json", status)
    save(out / "summary.json", status)
    sheet(out, status["rows"])
    print(json.dumps({r["case"]: [r.get(f"state_{s}") for s in STRENGTHS] for r in status["rows"]}, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main()
