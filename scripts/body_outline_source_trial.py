"""원본 윤곽 시험이다. 기본은 오프라인 CPU 사전 검사이며 --run을 명시해야 GPU를 사용한다."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from genai_lab.proportion_inputs import (ProportionOptions, sha, require, json_value, validate_proportion_request)
from genai_lab.proportion_generation import generate_proportion_batch, build_original_maps
from genai_lab.proportion_foreground import AnimeForeground
from genai_lab.studio_proportion import restore_head, verify_pose
from genai_lab.qwen_record_io import write_json
from scripts.head_contour_trial_inputs import restore_request, read, canonical
from scripts.verify_proportion_product import replay_inputs

TRIAL = ROOT / "outputs/body-outline-test-20261008"
R6_SEEDS = (209212001, 209212002, 209212003)
R6_SKETCH_SHA = "a253333d1e3e7f89d60987d927958c48cd1aabe5ea3286a4e9f6196dd9f32f56"


def checked_record(path, manifest):
    path = Path(path)
    digest = sha(path)
    value = read(path)
    require(sha(path) == digest, "기록을 읽는 중 파일이 변경됐습니다.")
    manifest[str(path)] = digest
    return value


def check_trial_locks():
    manifest = {str(TRIAL / "sha256.txt"): sha(TRIAL / "sha256.txt")}
    for line in (TRIAL / "sha256.txt").read_text(encoding="utf-8-sig").splitlines():
        digest, name = line.split(maxsplit=1)
        path = (TRIAL / name.lstrip("*")).resolve()
        require(path.is_relative_to(TRIAL.resolve()), "시험 잠금 경로가 폴더 밖입니다.")
        require(sha(path) == digest, "시험 기준·자료 SHA 불일치: " + str(path))
        manifest[str(path)] = digest
    require(str((TRIAL / "criteria.md").resolve()) in manifest, "판정 기준 잠금이 없습니다.")
    return manifest


def restore_options(saved):
    values = dict(saved)
    values["head"] = restore_head({"head": values["head"]})
    for key in ("sketch_root", "foreground_model"):
        values[key] = Path(values[key])
    options = ProportionOptions(**values)
    require(options.enabled and options.outline_source == "base", "현재 방식으로 완료한 비율 실행만 비교할 수 있습니다.")
    return options


def check_studio_approval(directory, inputs, options, manifest):
    approval = checked_record(directory / "inputs/approval.json", manifest)
    require(approval["approved"] is True and approval["proportion_mode"] == "two_pass", "GUI 비율 입력 승인이 없습니다.")
    require(approval["prompt"] == inputs.prompt.positive and approval["negative"] == inputs.prompt.negative
            and approval["face_sha256"] == inputs.face_sha256, "승인 프롬프트·얼굴이 실행 기록과 다릅니다.")
    for name, item in approval["references"].items():
        source, snapshot = Path(item["source"]), directory / "inputs" / (name + ".png")
        for path, digest in ((source, item["sha256"]), (snapshot, item["preprocessing"]["analysis_sha256"])):
            require(sha(path) == digest, "GUI 원본·분석 사본이 변경됐습니다.")
            manifest[str(path)] = digest
    confirmation = checked_record(options.head.contour_file.parent / "user-confirmation.json", manifest)
    require(confirmation["reviewer"] == "user" and confirmation["automatic_approval"] is False,
            "사용자의 머리 확인 기록이 없습니다.")
    require(canonical(confirmation["head"]) == canonical(asdict(options.head)), "확인한 머리와 생성에 쓴 머리가 다릅니다.")
    pose = confirmation["pose"]
    verify_pose(pose)
    require(pose["normalized_sha256"] == options.head.normalized_sha256
            and pose["control_sha256"] == inputs.control_sha256, "원본·머리·골격 연결이 다릅니다.")
    require(Path(pose["source_file"]).resolve() == (directory / "inputs/character.png").resolve(),
            "다른 캐릭터의 원본 자세입니다.")


def check_studio_outputs(generation, seeds, inputs, manifest):
    """대조군 A는 네 쌍이 모두 완성돼야 한다. 부분 묶음과 비교하지 않는다."""
    for seed in seeds:
        for stage in ("BASE", "CONTOUR"):
            folder = generation / f"{stage}_{seed}"
            record = checked_record(folder / "run.json", manifest)
            require(record["valid"] and record["completed"] and record["seed"] == seed, "비교할 A 생성이 미완료입니다.")
            require(canonical(record["prompt"]) == canonical(asdict(inputs.prompt)), "A 프롬프트가 잠금 입력과 다릅니다.")
            require(sha(folder / "raw.png") == record["raw_sha256"], "A 원시 이미지가 변경됐습니다.")
            manifest[str(folder / "raw.png")] = record["raw_sha256"]
        folder = generation / f"seed-{seed}"
        product = checked_record(folder / "run.json", manifest)
        require(product["valid"] and product["completed"] and sha(folder / "product.png") == product["product_sha256"],
                "비교할 A 최종 이미지가 변경됐습니다.")
        manifest[str(folder / "product.png")] = product["product_sha256"]
    maps = checked_record(generation / "maps.json", manifest)
    require(set(maps) == {str(seed) for seed in seeds}, "A 지도 seed가 다릅니다.")
    for entry in maps.values():
        require(sha(entry["path"]) == entry["sha256"], "A 스케치가 변경됐습니다.")
        manifest[entry["path"]] = entry["sha256"]


def load_studio(directory, outline_source="original"):
    directory = Path(directory).resolve()
    generation = directory / "generation"
    manifest = {}
    lock_path = generation / "preflight.json"
    lock_sha = (generation / "preflight.sha256").read_text().strip()
    require(sha(lock_path) == lock_sha, "GUI preflight 잠금 SHA 불일치")
    lock = checked_record(lock_path, manifest)
    manifest[str(generation / "preflight.sha256")] = sha(generation / "preflight.sha256")
    run = checked_record(generation / "run.json", manifest)
    for key, value in lock.items():
        require(canonical(run.get(key)) == canonical(value), "GUI 실행과 preflight 불일치: " + key)
    seeds = tuple(lock["seeds"])
    require(len(seeds) == 4 and len(set(seeds)) == 4, "GUI 비교는 4개 seed가 필요합니다.")
    require(run["status"] == "review_pending" and run["phase"] == "done"
            and all(run[key] == list(seeds) for key in ("BASE", "CONTOUR", "products")), "GUI A 실행이 완료되지 않았습니다.")
    # 어깨 보정 실행은 보정 전 입력(parent_inputs)을 잠금에 따로 둔다. 제품처럼 보정 전 입력으로 검사하고,
    # 생성 직전에 resolve_shoulder_inputs로 보정 골격을 다시 만들어 검증한다.
    source = lock.get("parent_inputs", lock["inputs"])
    inputs, settings = restore_request({"inputs": {"BASE": source}, "settings": lock["settings"]})
    options = restore_options(lock["options"])
    check_studio_approval(directory, inputs, options, manifest)
    check_studio_outputs(generation, seeds, inputs, manifest)
    _, _, models = validate_proportion_request(inputs, settings, options)
    require(models == lock["models"], "A 실행 때와 모델 파일이 다릅니다.")
    # 원본 윤곽 비교는 "original", 머리 안쪽 선 시험(어깨 보정 실행 포함)은 기존과 같은 "base"를 쓴다.
    return inputs, settings, replace(options, outline_source=outline_source), seeds, manifest


def verify_manifest(manifest):
    for path, digest in manifest.items():
        require(sha(path) == digest, "사전 검사 뒤 자료가 변경됐습니다: " + path)


def prepare_trial(inputs, settings, options, seeds, directory, sources, expected,
                  foreground_factory=AnimeForeground):
    """GPU 분기 전에 CPU 지도를 만들고 JSON으로 저장 가능한 잠금을 만든다."""
    mask, contour, models = validate_proportion_request(inputs, settings, options)
    directory.mkdir(parents=True, exist_ok=False)
    state = {"status": "preflight_started", "gpu_generation": False}
    try:
        map_dir = directory / "cpu-maps"
        map_dir.mkdir()
        maps = build_original_maps(seeds, mask, contour, map_dir, foreground_factory,
                                   options, lambda: False, expected)
        lock = json_value({"inputs": asdict(inputs), "settings": asdict(settings), "options": asdict(options),
            "seeds": seeds, "outline_source": "original", "base_stage": "skipped_unused_outline",
            "original_sha256": options.head.normalized_sha256, "maps": maps, "models": models,
            "source_files": sources, "code_sha256": {str(path): sha(path) for path in (
                Path(__file__), ROOT / "genai_lab/proportion_generation.py", ROOT / "genai_lab/proportion_inputs.py",
                ROOT / "genai_lab/proportion_backend.py", ROOT / "genai_lab/proportion_foreground.py")}})
        write_json(directory / "trial-preflight.json", lock)
        (directory / "trial-preflight.sha256").write_text(sha(directory / "trial-preflight.json") + "\n")
        state["status"] = "validated_no_generation"
        return {seed: entry["sha256"] for seed, entry in maps.items()}, lock
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(directory / "trial-status.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--r6", action="store_true")
    source.add_argument("--studio-run", type=Path)
    parser.add_argument("--foreground-model", type=Path, help="Required only for locked R6 inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Generate CONTOUR only after CPU validation")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "새 출력 폴더만 사용할 수 있습니다.")
    sources = check_trial_locks()
    if args.r6:
        require(args.foreground_model is not None, "R6에는 로컬 foreground-model 경로가 필요합니다.")
        inputs, settings, options = replay_inputs(args.foreground_model)
        options = replace(options, outline_source="original")
        seeds, expected = R6_SEEDS, {seed: R6_SKETCH_SHA for seed in R6_SEEDS}
    else:
        require(args.foreground_model is None, "GUI 비교는 기존 모델 경로를 그대로 사용합니다.")
        inputs, settings, options, seeds, previous = load_studio(args.studio_run)
        sources.update(previous)
        expected = {}
    hashes, lock = prepare_trial(inputs, settings, options, seeds, args.output, sources, expected)
    if not args.run:
        print("CPU preflight passed; GPU generation: 0", flush=True)
        return
    state = {"status": "started", "gpu_generation_requested": True, "outline_source": "original", "seeds": list(seeds)}
    try:
        verify_manifest({**sources, **lock["code_sha256"]})
        generate_proportion_batch(inputs, seeds, args.output / "generation", settings=settings, options=options,
                                  expected_sketch=hashes)
        state["status"] = "review_pending"
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "trial-status.json", state)


if __name__ == "__main__":
    main()
