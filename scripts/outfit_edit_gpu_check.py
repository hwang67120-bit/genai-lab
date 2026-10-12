"""원본 편집형 의상 교체 제품 경로 GPU 확인(메인 환경에서 실행, D-094 2~3단계).
제품과 같은 진입점(genai_lab.outfit_edit_pipeline.run_outfit_pipeline)을 캐릭터마다 한 번씩 부른다:
캔버스 준비 → See-through 분할(별도 프로세스) → Qwen 편집(별도 프로세스, 6.5GiB 상한·VAE 타일·lm_head CPU) → 머리 합성.
자동 재시도는 없다. 한 명이 실패하면 기록을 남기고 다음 명으로 간다.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from genai_lab.outfit_edit import write_json
from genai_lab.outfit_edit_pipeline import run_outfit_pipeline
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.studio_generation import StudioRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", metavar=("NAME", "IMAGE"), required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--settings", required=True, help="Qwen 실행 설정 JSON(꼬리 편집과 같은 파일)")
    parser.add_argument("--head-accessories", choices=("keep", "remove"), default="keep")
    parser.add_argument("--garment-terms", default="", help="쉼표로 나눈 옷 이름(제품의 의상 자동 분석 결과). 비우면 목록 없이")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = QwenPoseSettings.load(args.settings)
    runtime = StudioRuntime.from_environment()
    summary, rows = [], []
    for name, image in args.case:
        start = time.perf_counter()
        try:
            outcome = run_outfit_pipeline(image, args.garment, settings=settings, head_accessories=args.head_accessories,
                                          runtime=runtime, directory=args.output / name,
                                          garment_terms=[t for t in args.garment_terms.split(",") if t.strip()],
                                          progress=lambda text: print(f"[{name}] {text}", flush=True))
            memory = json.loads((args.output / name / "qwen" / "memory.json").read_text(encoding="utf-8"))
            record = dict(case=name, status="completed", composited=outcome["composited"], reason=outcome.get("reason"),
                          decision=outcome["alignment"]["decision"], hair_shape=outcome.get("hair_shape"),
                          canvas_pad=outcome["alignment"].get("canvas_pad"), memory_stopped=memory.get("stopped"),
                          max_stage_peak_reserved_gib=max(s.get("stage_peak_reserved_gib", 0) for s in memory["stages"]))
            rows.append(np.asarray(Image.open(args.output / name / "compose" / "review.png").convert("RGB").resize((1152, 512))))
        except Exception as error:
            record = dict(case=name, status="failed", error=str(error), traceback=traceback.format_exc())
        record["seconds"] = round(time.perf_counter() - start, 1)
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False)[:600], flush=True)
        write_json(args.output / "summary.json", dict(head_accessories=args.head_accessories, garment=args.garment,
                   garment_terms=args.garment_terms, results=summary))
    if rows:
        Image.fromarray(np.concatenate(rows, 0)).save(args.output / "review-sheet.png")


if __name__ == "__main__":
    main()
