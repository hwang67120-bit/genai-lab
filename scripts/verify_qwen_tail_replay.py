"""Prepare R1-R5 with production tail functions; GPU only with --execute --case Rn.

Existing experiment outputs are read-only. Every invocation uses a new output
folder. This script never retries, installs packages, or downloads models.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from genai_lab.qwen_tail_edit import (prepare_tail_spec, assemble_tail_prompt, make_tail_request,
    run_tail_edit, TailEditWorkflow, SPIRAL_EXAMPLE)
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.qwen_preservation import file_sha
from genai_lab.qwen_record_io import write_json

CASES = {
    "R1": ("cases/B1/E2/run.json", "RAC", (560,1100,936,1700), True, ""),
    "R2": ("stripe2/cases/B3/E2/run.json", "T04", (700,340,860,640), False, ""),
    "R3": ("shape/cases/B4/E2P/run.json", "T10", (560,485,900,800), True, SPIRAL_EXAMPLE),
    "R4": ("integrated/cases/B6/RT10/run.json", "T10", (560,485,900,800), True, SPIRAL_EXAMPLE),
    "R5": ("integrated/cases/B7/RRAC/run.json", "RAC", (560,1100,936,1700), True, ""),
}


def recorded_path(value):
    """Recorded checkout paths map to this checkout; no arbitrary external inputs."""
    text = str(value).replace("\\", "/")
    marker = "/genai-lab/"
    if marker not in text:
        raise ValueError("시험 파일 경로가 저장소 밖입니다: " + text)
    path = (ROOT / text.split(marker,1)[1]).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("저장소 밖 경로")
    return path


def prepare_case(case, output, settings=None):
    experiment = ROOT / "outputs/qwen-tail-edit-20261004"
    relative, character, box, pattern, tip = CASES[case]
    run = json.loads((experiment / relative).read_text(encoding="utf-8"))
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    reference = manifest["refs"][character]["E1"]
    source = recorded_path(reference["path"])
    basis = recorded_path(run["start"])
    if file_sha(source) != reference["sha256"] or file_sha(basis) != run["start_sha256"]:
        raise ValueError("기존 시험 입력 SHA 불일치")
    old_raw = experiment / Path(relative).parent / "raw.png"
    if file_sha(old_raw) != run["raw_sha256"]:
        raise ValueError("기존 시험 원시 결과 SHA 불일치")
    spec = prepare_tail_spec(basis, source, box, pattern=pattern, tip=tip, confirmed=True,
        directory=output / case / "inputs", source_sha256=reference["sha256"], image_sha256=run["start_sha256"])
    prompt = assemble_tail_prompt(pattern,tip)
    matches = {"crop_sha": spec.crop_sha256 == run["reference_sha256"],
               "positive": prompt["positive"] == run["prompt"], "negative": prompt["negative"] == run["negative"],
               "seed": run["seed"] == 209212001}
    request = None
    if settings is not None:
        matches.update({
            "steps": settings.steps == run["steps"],
            "true_cfg_scale": settings.true_cfg_scale == run["true_cfg_scale"],
            "guidance_scale": settings.guidance_scale == run["guidance_scale"],
            "offload": settings.offload == run["offload_mode"],
            "model_revision": settings.model_revision == run["model_rev"],
            "gguf_revision": settings.gguf_revision == run["gguf_rev"],
            "gguf_sha": settings.gguf_sha256 == run["gguf_sha256_locked"],
        })
        request = make_tail_request(spec,settings=settings)
    report = {"case":case,"spec":spec.record(),"prompt":prompt,"matches":matches,
              "expected_raw_sha256":run["raw_sha256"],"gpu_executed":False,
              "source_record":str(experiment/relative)}
    write_json(output / case / "preflight.json",report)
    if not all(matches.values()):
        raise ValueError(f"{case} CPU 대조 불일치: {matches}")
    if request:
        write_json(output / case / "prepared-request.json",request)
    return spec,request,report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--settings",type=Path)
    parser.add_argument("--case",choices=CASES)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    if args.execute and (not args.case or not args.settings):
        parser.error("GPU 실행은 --case Rn 및 --settings가 필요합니다.")
    settings=QwenPoseSettings.load(args.settings) if args.settings else None
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for case in ([args.case] if args.case else CASES):
        spec,request,report=prepare_case(case,args.output,settings)
        if args.execute:
            directory=args.output / case / "edit"
            run_tail_edit(request,directory,TailEditWorkflow(spec),
                on_progress=lambda r:print("steps",len(r.get("steps",[])),flush=True))
            report.update(gpu_executed=True,raw_sha256=file_sha(directory/"raw.png"))
            report["raw_sha_matches"]=report["raw_sha256"]==report["expected_raw_sha256"]
        reports.append(report)
        write_json(args.output/"summary.json",reports)
        print(case,report["matches"],"GPU",report["gpu_executed"],flush=True)
        if report.get("raw_sha_matches") is False:
            raise SystemExit("원시 SHA 불일치. 재시도 없이 비교 기록을 검토하세요.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
