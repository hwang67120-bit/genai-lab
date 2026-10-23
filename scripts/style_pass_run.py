"""그림체 후처리 한 번(D-161 시험 A) GPU 실행기. 시험 전용 — 제품 화면·기준선과 섞지 않는다.

사용자가 판정한 결과 그림을 같은 장면 그대로 트리거 문장 하나로 변환한다(새 자세·배경 지시 없음, 강도·seed 고정).
사용: python scripts/style_pass_run.py <계획 json> <출력 폴더>
  계획: {"style_lora": "anime_borrowed", "strength": 1.0, "variants": ["style"] 또는 ["style", "none"],
         "cases": [{"case": "A1", "image": "...", "sha256": "...", "original_seconds": 4324}, ...]}
출력: <출력 폴더>/runs/<case>-<variant>/, batch-status.json, summary.json, sheet.png([입력 | 변환 | (LoRA 없는 대조)])
"""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = ("style", "none")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sheet(out, rows, variants):
    from PIL import Image, ImageDraw
    height, tiles = 480, []
    labels = ["입력(판정된 결과)"] + [("변환 LoRA" if v == "style" else "변환 LoRA 없음") for v in variants]
    for row in rows:
        images = []
        for cell in [row["image"]] + [row.get(f"result_{v}") for v in variants]:
            if cell and Path(cell).is_file():
                with Image.open(cell) as im:
                    im = im.convert("RGB")
                    images.append(im.resize((int(im.width * height / im.height), height)))
            else:
                images.append(Image.new("RGB", (363, height), "white"))
        tile = Image.new("RGB", (sum(i.width for i in images) + 10 * len(images), height + 24), "white")
        x = 0
        for label, image in zip(labels, images):
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
    from genai_lab.qwen_pose_settings import QwenPoseSettings
    from genai_lab.studio_generation import StudioRuntime
    from genai_lab.style_pass import StylePassWorkflow, make_style_request, prepare_style_spec, run_style_pass_edit
    plan, out = json.loads(Path(argv[0]).read_text(encoding="utf-8")), Path(argv[1])
    variants = list(plan["variants"])
    if not variants or not set(variants) <= set(VARIANTS) or variants[0] != "style":
        raise SystemExit("variants는 ['style'] 또는 ['style', 'none']")
    out.mkdir(parents=True, exist_ok=True)
    settings = QwenPoseSettings.load(StudioRuntime.from_environment().qwen_settings)
    status = dict(state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), trial="style_pass_borrowed",
                  note="그림체 후처리 시험 A — 품질 기준선·운영 집계에 넣지 않는다", style_lora=plan["style_lora"],
                  strength=plan["strength"], variants=variants, rows=[])
    for case in plan["cases"]:
        image = Path(case["image"]) if Path(case["image"]).is_absolute() else ROOT / case["image"]
        row = dict(case=case["case"], image=str(image), original_seconds=case["original_seconds"])
        status["rows"].append(row)
        for variant in variants:
            directory = out / "runs" / f"{case['case']}-{variant}"
            start = time.time()
            row[f"state_{variant}"] = "running"
            save(out / "batch-status.json", status)
            try:
                if hashlib.sha256(image.read_bytes()).hexdigest() != case["sha256"]:
                    raise ValueError("입력 그림 SHA가 판정 기록과 다릅니다.")
                spec = prepare_style_spec(image, directory=directory / "inputs", style_lora=plan["style_lora"],
                                          apply_lora=variant == "style", strength=plan["strength"])
                request = make_style_request(spec, settings=settings)
                run_style_pass_edit(request, directory / "qwen", StylePassWorkflow(spec))
                final = json.loads((directory / "qwen" / "run.final.json").read_text(encoding="utf-8"))
                row[f"result_{variant}"] = str(directory / "qwen" / "raw.png")
                row[f"state_{variant}"] = "completed"
                row[f"worker_seconds_{variant}"] = round(final.get("seconds") or 0, 1)
                row[f"reserved_gib_{variant}"] = round(((final.get("generation_memory") or {}).get("reserved_bytes") or 0) / 2**30, 3)
                row[f"style_lora_{variant}"] = final.get("style_lora")
                row[f"call_{variant}"] = {k: (final.get("call_arguments") or {}).get(k)
                                          for k in ("num_inference_steps", "true_cfg_scale", "seed", "width", "height", "positive", "negative", "images")}
            except Exception as error:  # noqa: BLE001 — 칸 실패는 기록하고 다음으로
                row[f"state_{variant}"] = "failed"
                row[f"error_{variant}"] = (str(error) or traceback.format_exc())[-2000:]
            row[f"seconds_{variant}"] = round(time.time() - start)
            row[f"total_seconds_{variant}"] = row["original_seconds"] + row[f"seconds_{variant}"]
            save(out / "batch-status.json", status)
    status.update(state="finished", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(out / "batch-status.json", status)
    save(out / "summary.json", status)
    sheet(out, status["rows"], variants)
    print(json.dumps({r["case"]: [r.get(f"state_{v}") for v in variants] for r in status["rows"]}, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main()
