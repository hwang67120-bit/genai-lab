"""제품 함수로 R6를 재현한다. 기본은 검증만 하며 --run을 명시해야 GPU를 사용한다."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from genai_lab.proportion_inputs import (
    HeadOutline, ProportionOptions, FOREGROUND_SHA, sha, require, validate_proportion_request,
    make_sketch, white_background, json_value)
from genai_lab.proportion_generation import generate_proportion_batch
from genai_lab.proportion_foreground import AnimeForeground
from genai_lab.qwen_record_io import write_json
from scripts.head_contour_trial_inputs import (
    DESIGN, BASE_DIR, POSE_DIR, restore_request, verify_download, verify_locks, read, CONTROL_SHA, FACE_SHA, CONTOUR_SHA)

SEEDS = (209212003, 209212001, 209212002, 209212004, 209212005, 209212006)


def replay_inputs(foreground_model):
    verify_locks(DESIGN)
    require(sha(BASE_DIR / "plan.json") == (BASE_DIR / "plan.sha256").read_text().strip(), "기존 BASE 계획 SHA 불일치")
    inputs, settings = restore_request(read(BASE_DIR / "plan.json"))
    require(inputs.control_sha256 == CONTROL_SHA and inputs.face_sha256 == FACE_SHA, "기존 골격·얼굴 SHA 불일치")
    confirmed = read(DESIGN / "head_draft/head_confirmed.json")
    require(confirmed["status"] == "user_confirmed" and confirmed["thickness_locked"] == 4, "머리 확인 기록 불일치")
    points = {point["joint_name"]: point for point in read(POSE_DIR / "observations.json")["observations"][1]["joints"]}
    for key in ("nose", "neck"):
        require(points[key]["detected"] and points[key]["confidence_score"] >= .3, "코·목 관절 확인 불가")
    head = HeadOutline(POSE_DIR / "normalized.png", confirmed["normalized_sha256"], confirmed["control_sha256"],
                       DESIGN / "head_draft/head_mask_draft.png", sha(DESIGN / "head_draft/head_mask_draft.png"),
                       DESIGN / "head_draft/contour_4px.png", CONTOUR_SHA,
                       tuple(points["nose"][axis] for axis in ("x", "y")),
                       tuple(points["neck"][axis] for axis in ("x", "y")), confirmed=True)
    sketch_root, _ = verify_download(DESIGN / "download.json")
    options = ProportionOptions(True, head, sketch_root, Path(foreground_model), FOREGROUND_SHA)
    return inputs, settings, options


def replay_expectations():
    maps = read(DESIGN / "twopass/maps.json")
    bases, sketches, raws = {}, {}, {}
    for seed in SEEDS:
        record = maps[str(seed)]
        source = DESIGN / ("gpu-02" if seed in SEEDS[:2] else "gpu-04") / f"BASE_{seed}/raw.png"
        require(sha(source) == record["base_raw_sha256"], "기존 BASE raw SHA 불일치")
        require(sha(DESIGN / f"twopass/maps/sketch_{seed}.png") == record["sketch_sha256"], "잠금 스케치 SHA 불일치")
        raw = DESIGN / f"gpu-05/CONTOUR_{seed}/raw.png"
        previous = read(raw.parent / "run.json")
        require(previous["valid"] and sha(raw) == previous["raw_sha256"], "기존 gpu-05 raw SHA 불일치")
        bases[seed], sketches[seed], raws[seed] = record["base_raw_sha256"], record["sketch_sha256"], previous["raw_sha256"]
    return bases, sketches, raws


def check_cpu_pixels(directory, settings, options, head_mask, head_contour, expected):
    """기존 결과만 CPU로 추론한다. 생성은 없으며 모든 잠금 자료를 보존한다."""
    import numpy as np
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=False)
    foreground, record = None, {"status": "started", "cases": [], "gpu_generation": False}
    started = time.monotonic()
    try:
        foreground = AnimeForeground(options)
        for seed in SEEDS:
            base = DESIGN / ("gpu-02" if seed in SEEDS[:2] else "gpu-04") / f"BASE_{seed}/raw.png"
            with Image.open(base) as image:
                sketch = make_sketch(foreground.alpha(image), head_mask, head_contour)
            path = directory / f"sketch_{seed}.png"
            Image.fromarray(sketch).save(path)
            require(sha(path) == expected[seed], f"스케치 불일치: {seed}")
            with Image.open(DESIGN / f"gpu-05/CONTOUR_{seed}/raw.png") as image:
                white = white_background(np.asarray(image), foreground.alpha(image))
            product = directory / f"white_{seed}.png"
            Image.fromarray(white).save(product)
            previous = DESIGN / f"gen4/TP_{seed}/white.png"
            require(sha(product) == sha(previous), f"흰 배경 결과 불일치: {seed}")
            record["cases"].append({"seed": seed, "sketch_sha256": sha(path), "white_sha256": sha(product), "match": True})
            write_json(directory / "result.json", record)
            print(f"{seed}: sketch + white SHA match", flush=True)
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        if foreground is not None:
            foreground.close()
        record["seconds"] = time.monotonic() - started
        write_json(directory / "result.json", record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foreground-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicit GPU: 6 BASE + 6 CONTOUR, no retries")
    mode.add_argument("--check-cpu-pixels", action="store_true", help="Existing images only, CPU isnet inference")
    args = parser.parse_args(argv)
    inputs, settings, options = replay_inputs(args.foreground_model)
    head_mask, head_contour, models = validate_proportion_request(inputs, settings, options)
    bases, sketches, raws = replay_expectations()
    if args.run:
        generate_proportion_batch(inputs, SEEDS, args.output, settings=settings, options=options,
                                  expected_base=bases, expected_sketch=sketches, expected_raw=raws)
    elif args.check_cpu_pixels:
        check_cpu_pixels(args.output, settings, options, head_mask, head_contour, sketches)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "preflight.json", json_value({"status": "validated_no_generation",
                   "seeds": SEEDS, "models": models, "options": asdict(options), "expected_base": bases,
                   "expected_sketch": sketches, "expected_raw": raws}))
        print("Preflight passed; GPU generation not executed")


if __name__ == "__main__":
    main()
