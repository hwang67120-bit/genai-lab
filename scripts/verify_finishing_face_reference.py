"""Copy the 16 existing product raws and replay only finishing with face reference."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genai_lab.onepass_generation import OnePassBatch, OnePassCandidate, validate_prompt
from genai_lab.onepass_prompt import OnePassPrompt, EncoderChunkPlan, PromptChunk
from genai_lab.proportion_inputs import sha, require, json_value
from genai_lab.qwen_record_io import write_json
from genai_lab.studio_generation import StudioRuntime
from genai_lab.studio_finishing import finish_batch, validate_candidate, read_finishing
from genai_lab.finishing_reference import checked_reference
from genai_lab.studio_background import prepare_backgrounds, read_background

SOURCE = ROOT/"outputs/finishing-test-20261008/product_path"


def restore_prompt(record):
    """Use the recorded token IDs, without changing tags or re-tokenizing."""
    encoders = []
    for encoder in record["encoders"]:
        chunks = {pol:tuple(PromptChunk(c["kind"],c["text"],tuple(c["token_ids"]))
                            for c in encoder[pol]) for pol in ("positive","negative")}
        encoders.append(EncoderChunkPlan(encoder["positive_tokens"],encoder["negative_tokens"],**chunks))
    prompt = OnePassPrompt(record["positive"],record["negative"],tuple(encoders),record["rules"])
    validate_prompt(prompt)
    require(json_value(asdict(prompt)) == record, "기록된 프롬프트와 복원 결과가 다릅니다.")
    return prompt


def read_batch(directory, seeds):
    candidates = []
    for seed in seeds:
        folder = directory/"generation"/f"seed-{seed}"
        record = json.loads((folder/"run.json").read_text(encoding="utf-8"))
        require(record["seed"] == seed, "폴더와 기록의 seed가 다릅니다.")
        candidates.append(OnePassCandidate(seed,folder/"raw.png",folder/"run.json",record))
    return OnePassBatch(directory/"generation",tuple(candidates))


def preflight(source, runtime):
    summary = json.loads((source/"summary.json").read_text(encoding="utf-8"))
    require(len(summary) == 4 and len({v["case"] for v in summary}) == 4, "제품 비교 4건이 필요합니다.")
    cases = []
    for item in summary:
        name = f"run-{item['case']}"
        require(Path(name).name == name and len(item["case"]) == 8
                and all(c in "0123456789abcdef" for c in item["case"]), "사례 경로 오류")
        directory = source/name
        seeds = item["seeds"]
        require(len(seeds) == 4 and len(set(seeds)) == 4
                and all(type(s) is int and s >= 0 for s in seeds), "사례당 기존 seed 4개가 필요합니다.")
        request = json.loads((directory/"generation/request.json").read_text(encoding="utf-8"))
        require(request["seeds"] == seeds and request["completed_seeds"] == seeds
                and request["status"] == "raw_completed", "원본 생성 요청이 완료되지 않았습니다.")
        batch = read_batch(directory,seeds)
        prompt_record = batch.candidates[0].record["prompt"]
        prompt = restore_prompt(prompt_record)
        for candidate in batch.candidates:
            validate_candidate(candidate,prompt)
            require(candidate.record["prompt"] == prompt_record, "한 배치의 프롬프트가 서로 다릅니다.")
            require(Path(candidate.record["settings"]["model_root"]).resolve() == runtime.generation.model_root.resolve(), "기존 생성과 base 모델 경로가 다릅니다.")
        reference = checked_reference(batch,runtime.generation,face_file=directory/"inputs/face.png")
        files = [Path("inputs/face.png"),Path("inputs/approval.json"),Path("generation/request.json")]
        files += [Path("generation")/f"seed-{seed}"/name for seed in seeds for name in ("raw.png","run.json")]
        cases.append({"name":name,"seeds":seeds,"face_sha256":reference.sha256,
                      "files":{p.as_posix():sha(directory/p) for p in files}})
    return {"status":"passed","source":str(source.resolve()),"summary_sha256":sha(source/"summary.json"),
            "face_reference":"on","ip_scale":0.9,"new_generation":False,"cases":cases}


def copy_case(source, destination, case):
    destination.mkdir(parents=True,exist_ok=False)
    for name,expected in case["files"].items():
        original, copied = source/name, destination/name
        require(sha(original) == expected, "사전 검사 이후 입력이 바뀌었습니다.")
        copied.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,copied)
        require(sha(copied) == expected, "입력 복사 SHA 불일치")
    write_json(destination/"replay-source.json",{**case,"source":str(source),"new_generation":False})
    batch = read_batch(destination,case["seeds"])
    return batch, restore_prompt(batch.candidates[0].record["prompt"])


def run_replay(report, output, runtime):
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"preflight.json",report)
    state = {"status":"started","new_generation":False,"face_reference":"on","cases":[]}
    try:
        source = Path(report["source"])
        require(sha(source/"summary.json") == report["summary_sha256"], "사전 검사 이후 목록이 바뀌었습니다.")
        for case in report["cases"]:
            directory = output/case["name"]
            batch,prompt = copy_case(source/case["name"],directory,case)
            case_runtime = replace(runtime,quality_tags=prompt.rules.get("quality_format") == "diagnostic_2b")
            finish_batch(batch,prompt,case_runtime,face_file=directory/"inputs/face.png",progress=print)
            prepare_backgrounds(batch,runtime.model_cache,finishing=True,progress=print)
            results = []
            for candidate in batch.candidates:
                results.append({"seed":candidate.seed,"raw_sha256":sha(candidate.path),
                                "finishing":read_finishing(candidate),"background":read_background(candidate)})
            state["cases"].append({"name":case["name"],"results":results})
            write_json(output/"replay.json",state)
            require(all(v["background"]["status"] == "completed" for v in results), "흰 배경 정리 실패")
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write_json(output/"replay.json",state)
    return state


def parser():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--source",type=Path,default=SOURCE)
    args.add_argument("--output",type=Path)
    args.add_argument("--run",action="store_true",help="Explicitly enable GPU finishing only")
    return args


def main(argv=None):
    args = parser().parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1")
    runtime = replace(StudioRuntime.from_environment(),finishing=True,finishing_face_reference=True)
    report = preflight(args.source,runtime)
    if not args.run:
        print(json.dumps({"status":"passed","cases":len(report["cases"]),"raws":sum(len(c["seeds"]) for c in report["cases"]),
                          "face_reference":"on","gpu_executed":False,"new_generation":False}))
        return 0
    require(args.output is not None, "GPU 실행은 새 --output 폴더를 지정해야 합니다.")
    run_replay(report,args.output,runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
