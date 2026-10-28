"""고정된 F0 이미지를 제품 마무리로 재현한다. --run이 없으면 CPU 사전 검사만 한다."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genai_lab.finishing_prompt import quality_strings
from genai_lab.onepass_generation import OnePassBatch, OnePassCandidate
from genai_lab.onepass_generation_settings import scheduler_config
from genai_lab.onepass_prompt import OnePassPrompt, plan_prompt_chunks
from genai_lab.onepass_prompt_tokenizers import load_onepass_tokenizers
from genai_lab.proportion_inputs import json_value, require, sha
from genai_lab.qwen_record_io import write_json
from genai_lab.studio_generation import StudioRuntime
from genai_lab.studio_finishing import finish_batch, read_finishing
from genai_lab.studio_background import prepare_backgrounds, read_background

PLAN = "outputs/model-ceiling-diag-20261008/plan.json"
LOG = "outputs/finishing-test-20261008/gen_log.json"
PLAN_SHA = "f5abe5668b310df659bcd7d7b1f542addad747d9d077b31b3bfe0916731d73f2"
LOG_SHA = "467e92f3927fb23bbfc8bfef1339d3a1dda6f7b170aa38bcaad7de2ee07cda70"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cases", type=int, nargs="+", choices=range(1,17), default=[1])
    result.add_argument("--output", type=Path)
    result.add_argument("--run", action="store_true", help="Explicitly allow GPU finishing")
    return result


def preflight(root, cases, runtime):
    """추론 파이프라인을 로드하기 전에 고정 문구와 이미지 바이트를 검증한다."""
    require(len(set(cases)) == len(cases), "중복 사례 번호입니다.")
    require(sha(root/PLAN) == PLAN_SHA and sha(root/LOG) == LOG_SHA, "잠금된 시험 기록 SHA 불일치")
    items = json.loads((root/PLAN).read_text(encoding="utf-8"))
    logs = {(item["run"], item["seed"]):item for item in json.loads((root/LOG).read_text(encoding="utf-8"))}
    selected = []
    for case in cases:
        item = items[case-1]
        expected = logs[(item["run"],item["seed"])]
        require(Path(item["model_root"]).resolve() == runtime.generation.model_root.resolve(), "시험과 모델 경로가 다릅니다.")
        require(item["scheduler"] == scheduler_config(), "시험과 스케줄러가 다릅니다.")
        require(quality_strings(item["a"]["positive"],item["a"]["negative"]) == (item["b"]["positive"],item["b"]["negative"]), "②b 프롬프트 불일치")
        require(expected["scale"] == 1.5 and expected["size"] == [1104,1848], "다른 배율의 시험입니다.")
        files = {"f0": root/"outputs/model-ceiling-diag-20261008/gen"/f"{item['run']}_{item['seed']}_b.png"}
        for stage in ("f1","f2"):
            files[stage] = root/"outputs/finishing-test-20261008/gen"/f"{item['run']}_{item['seed']}_{stage}.png"
        for stage, path in files.items():
            require(sha(path) == expected[f"{stage}_sha256"], f"기존 {stage} SHA 불일치")
        selected.append({"case":case,"item":item,"expected":expected,"files":files})
    return {"status":"passed","plan_sha256":PLAN_SHA,"log_sha256":LOG_SHA,"cases":selected}


def copy_source_case(entry, directory, runtime, tokenizers):
    """기존 F0를 복사하고 출처를 기록한다. 새로 생성했다고 표시하지 않는다."""
    directory.mkdir(parents=True, exist_ok=False)
    item = entry["item"]
    text = item["b"]
    prompt = OnePassPrompt(text["positive"],text["negative"],
                          plan_prompt_chunks(text["positive"],text["negative"],tokenizers),
                          {"quality_format":"diagnostic_2b","replay_source":True})
    raw = directory/"raw.png"
    shutil.copyfile(entry["files"]["f0"],raw)
    require(sha(raw) == entry["expected"]["f0_sha256"], "F0 복사 SHA 불일치")
    record = {"valid":True,"completed":True,"origin":"copied_locked_trial_f0", "generated_here":False,
              "source_file":str(entry["files"]["f0"]),"seed":item["seed"],
              "raw_sha256":sha(raw),"prompt":asdict(prompt)}
    write_json(directory/"run.json",record)
    candidate = OnePassCandidate(item["seed"],raw,directory/"run.json",record)
    return OnePassBatch(directory,(candidate,)), prompt


def run_replay(report, destination, runtime):
    """첫 불일치에서 멈추고 Claude 비교용으로 모든 파일을 보존한다."""
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/"preflight.json",json_value(report))
    state = {"status":"started","comparisons":[],"quality_default_decision":"pending"}
    try:
        tokenizers = load_onepass_tokenizers(runtime.prompt)
        for entry in report["cases"]:
            folder = destination/f"case-{entry['case']:02d}"
            batch, prompt = copy_source_case(entry,folder,runtime,tokenizers)
            finish_batch(batch,prompt,runtime,progress=lambda text:print(text,flush=True))
            candidate = batch.candidates[0]
            actual = read_finishing(candidate)
            expected = entry["expected"]
            comparison = {"case":entry["case"],"seed":candidate.seed,
                "f1_match":actual["stages"]["hires"]["sha256"] == expected["f1_sha256"],
                "f2_match":actual["finished_sha256"] == expected["f2_sha256"],
                "expected":expected,"actual":actual}
            state["comparisons"].append(comparison)
            write_json(destination/"replay.json",json_value(state))
            require(comparison["f1_match"] and comparison["f2_match"],
                    "시험 SHA 불일치: 생성물은 보존했습니다. 원인과 육안 비교를 확인한 뒤 계속 여부를 결정하세요.")
            prepare_backgrounds(batch,runtime.model_cache,finishing=True)
            comparison["background"] = read_background(candidate)
            require(comparison["background"]["status"] == "completed", "흰 배경 정리 실패: 마무리 이미지는 보존했습니다.")
            print(f"case {entry['case']}: F1/F2 SHA match",flush=True)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write_json(destination/"replay.json",json_value(state))
    return state


def main(argv=None):
    args = parser().parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1")
    runtime = replace(StudioRuntime.from_environment(),quality_tags=True,finishing=True,finishing_face_reference=False)
    report = preflight(ROOT,args.cases,runtime)
    if not args.run:
        print(json.dumps({"status":report["status"],"cases":args.cases,"gpu_executed":False},ensure_ascii=False))
        return 0
    require(args.output is not None, "GPU 실행에는 새 --output 폴더가 필요합니다.")
    run_replay(report,args.output,runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
