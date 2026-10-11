"""골격 얼굴 점 빼기 시험(D-086): 완료된 원본 비율 참고 실행의 골격에서 얼굴 점을 빼고, 같은 seed로 2단계 생성을 다시 한다.
CPU 준비(기본): 조건별 골격·머리 기록·어깨 보정 기록을 만들고 입력 계약을 검사해 잠근다. 생성은 하지 않는다.
--run: 잠금 SHA를 확인한 뒤 조건마다 BASE+CONTOUR를 생성한다. 원래 실행 폴더는 읽기만 한다."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from genai_lab.pose_face_strip import FACE_MODES, strip_face_control, stripped_joints
from genai_lab.proportion_inputs import sha, require, json_value, checked_image, validate_proportion_request
from genai_lab.qwen_record_io import write_json
from scripts.body_outline_source_trial import load_studio, restore_options, verify_manifest
from scripts.head_contour_trial_inputs import read, restore_request

CODE_FILES = (Path(__file__), ROOT / "genai_lab/pose_face_strip.py", ROOT / "genai_lab/shoulder_control.py",
              ROOT / "genai_lab/proportion_generation.py", ROOT / "genai_lab/proportion_inputs.py",
              ROOT / "genai_lab/proportion_backend.py", ROOT / "scripts/body_outline_source_trial.py")


def prepare(run, output, modes, seed_count):
    from genai_lab.shoulder_control import prepare_shoulder_control, resolve_shoulder_inputs
    inputs, settings, options, seeds, sources = load_studio(run, outline_source="base")
    require(options.shoulder_pull == "0.85" and options.shoulder_record_file is not None, "어깨 보정 실행만 지원합니다.")
    shoulder = read(options.shoulder_record_file)
    control = checked_image(inputs.control_file, inputs.control_sha256, (736, 1232), "RGB")
    mask = checked_image(options.head.mask_file, options.head.mask_sha256, (736, 1232), "L")
    seeds = list(seeds)[:seed_count]
    output.mkdir(parents=True, exist_ok=False)
    conditions = {}
    for mode in modes:
        folder = output / mode / "inputs"
        folder.mkdir(parents=True)
        pixels, details = strip_face_control(control, shoulder["joints"], mask, mode)
        Image.fromarray(pixels).save(folder / "control.png")
        control_sha = sha(folder / "control.png")
        joints = stripped_joints(shoulder["joints"], mode)
        head = replace(options.head, control_sha256=control_sha)
        record = prepare_shoulder_control(folder / "control.png", control_sha, joints, head, folder / "shoulder")
        require(record["status"] == "applied", "얼굴 점을 뺀 골격에 어깨 보정을 적용하지 못했습니다.")
        variant = replace(options, head=head, shoulder_record_file=folder / "shoulder" / "shoulder.json",
                          shoulder_record_sha256=sha(folder / "shoulder" / "shoulder.json"))
        stage_inputs = replace(inputs, control_file=folder / "control.png", control_sha256=control_sha)
        _, _, models = validate_proportion_request(stage_inputs, settings, variant)
        resolved, _ = resolve_shoulder_inputs(stage_inputs, variant)
        conditions[mode] = dict(details=details, inputs=asdict(stage_inputs), options=asdict(variant),
                                shoulder_control_sha256=resolved.control_sha256, models=models)
    files = {str(p): sha(p) for p in output.rglob("*") if p.is_file()}
    lock = json_value(dict(run=str(run), seeds=seeds, settings=asdict(settings), conditions=conditions,
                           source_files=sources, prepared_files=files,
                           code_sha256={str(p): sha(p) for p in CODE_FILES}, new_generation_count=len(seeds) * len(modes)))
    write_json(output / "preflight.json", lock)
    digest = sha(output / "preflight.json")
    (output / "preflight.sha256").write_text(digest + "\n")
    preview(control, conditions, mask, output / "preview.png")
    return digest


def preview(control, conditions, mask, path):
    ys, xs = np.nonzero(mask > 127)
    y0, y1 = max(0, ys.min() - 40), min(1232, ys.max() + 220)
    x0, x1 = max(0, xs.min() - 140), min(736, xs.max() + 140)
    tiles = [control] + [checked_image(c["inputs"]["control_file"], c["inputs"]["control_sha256"], (736, 1232), "RGB")
                         for c in conditions.values()]
    Image.fromarray(np.concatenate([np.asarray(Image.fromarray(t[y0:y1, x0:x1]).resize((360, 360))) for t in tiles], 1)).save(path)


def run_prepared(output, review_sha):
    from genai_lab.proportion_generation import generate_proportion_batch
    require(review_sha and sha(output / "preflight.json") == review_sha
            and (output / "preflight.sha256").read_text().strip() == review_sha, "준비 잠금 SHA가 다릅니다.")
    lock = read(output / "preflight.json")
    verify_manifest({**lock["source_files"], **lock["prepared_files"], **lock["code_sha256"]})
    state = dict(status="started", conditions={})
    try:
        for mode, condition in lock["conditions"].items():
            inputs, settings = restore_request({"inputs": {"BASE": condition["inputs"]}, "settings": lock["settings"]})
            options = restore_options(condition["options"])
            require(validate_proportion_request(inputs, settings, options)[2] == condition["models"], "준비 이후 모델 변경")
            verify_manifest({**lock["source_files"], **lock["prepared_files"], **lock["code_sha256"]})
            state["conditions"][mode] = "generating"
            write_json(output / "run-state.json", state)
            generate_proportion_batch(inputs, tuple(lock["seeds"]), output / mode / "generation",
                                      settings=settings, options=options)
            state["conditions"][mode] = "review_pending"
        state["status"] = "review_pending"
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "run-state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-run", type=Path)
    parser.add_argument("--modes", nargs="+", choices=FACE_MODES[1:], default=["no_face", "no_face_nose"])
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha")
    args = parser.parse_args()
    if args.run:
        run_prepared(args.output, args.review_sha)
        return
    require(args.studio_run is not None, "CPU 준비에는 --studio-run이 필요합니다.")
    digest = prepare(args.studio_run, args.output, args.modes, args.seed_count)
    print(json.dumps(dict(status="prepared", gpu_generation=0, review_sha256=digest, output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
