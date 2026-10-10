"""긴 머리 경로(L1) 확인: 완료된 2단계 실행을 출력 폴더에 복사하고 실제 자동 머리 붙이기 경로로 4장을 처리한다.
원래 실행 폴더는 읽기만 한다. --gpu가 없으면 입력 계약(후보·머리 영역 확인·경로 선택)만 검사한다."""
import argparse
import shutil
import time
import traceback
import uuid
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from genai_lab.onepass_generation import OnePassCandidate
from genai_lab.proportion_generation import ProportionBatch, ProportionCandidate
from genai_lab.proportion_inputs import require, sha
from genai_lab.qwen_record_io import write_json
from genai_lab.studio_generation import StudioRuntime, StudioResults
from genai_lab.studio_head_paste import (read, selected_request, verify_preview, apply_head_batch,
    automatic_skip_reason, preview_skip_reason)
from genai_lab.head_paste_rules import paste_profile, rule_name


def copy_run(run, output):
    """생성 원본과 입력 기록만 복사한다. 이전 머리 붙이기 결과는 가져오지 않는다."""
    target = output / "run-copy"
    shutil.copytree(run / "inputs", target / "inputs")
    shutil.copytree(run / "generation", target / "generation",
                    ignore=shutil.ignore_patterns("head-pastes", "auto-head-pastes", "user-review.json",
                                                  "auto-head-result.json"))
    # 복사본 기록만 복사한 파일을 가리키게 고친다. 이전 수동 머리 붙이기 연결은 지운다.
    for record_file in sorted((target / "generation").glob("seed-*/run.json")):
        record = read(record_file)
        seed = record_file.parent.name.removeprefix("seed-")
        record.update(raw_file=str(target / "generation" / f"CONTOUR_{seed}" / "raw.png"),
                      product_file=str(record_file.parent / "product.png"))
        record.pop("head_paste", None)
        write_json(record_file, record)
    return target


def load_batch(directory):
    lock = read(directory / "preflight.json")
    candidates = []
    for seed in lock["seeds"]:
        raw_folder = directory / f"CONTOUR_{seed}"
        raw_record = read(raw_folder / "run.json")
        raw = OnePassCandidate(seed, raw_folder / "raw.png", raw_folder / "run.json", raw_record)
        folder = directory / f"seed-{seed}"
        record = read(folder / "run.json")
        require(record["product_sha256"] == sha(folder / "product.png"), "복사한 후보가 다릅니다.")
        candidates.append(ProportionCandidate(seed, raw, folder / "product.png", folder / "run.json", record))
    return ProportionBatch(directory, tuple(candidates))


def confirmed_preview(folder):
    info = read(folder / "result.json")
    choice = read(folder / "confirmation.json")
    require(choice["confirmed"] is True and choice["reviewer"] == "user", "사용자가 확인한 머리 영역이 아닙니다.")
    require(choice["preview_sha256"] == sha(folder / "result.json"), "머리 영역 확인 기록이 변경됐습니다.")
    return info


def head_sheet(results, report, box, path):
    """장마다 [붙이기 전 | 자동 붙이기] 머리 확대와 전신을 한 줄로 놓는다(확인용, 블라인드 아님)."""
    rows = []
    for index, item in enumerate(report["items"]):
        before = Image.open(results.batch.candidates[index].path).convert("RGB")
        after = Image.open(item["product_file"]).convert("RGB") if item["status"] == "completed" else before
        crops = [image.crop(box).resize((384, 384)) for image in (before, after)]
        bodies = [image.resize((230, 384)) for image in (before, after)]
        rows.append(np.concatenate([np.asarray(image) for image in (*crops, *bodies)], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def probe_write(directory, attempts=3, wait=2.0):
    """작업 프로세스와 같은 방식(새 폴더 + 파일)으로 쓰기를 먼저 확인한다. 모델은 아직 올리지 않는다."""
    errors = []
    for attempt in range(1, attempts + 1):
        probe = directory / "auto-head-pastes" / ("write-probe-" + uuid.uuid4().hex)
        try:
            probe.mkdir(parents=True, exist_ok=False)
            (probe / "probe.txt").write_text("ok", encoding="utf-8")
            (probe / "probe.txt").unlink()
            probe.rmdir()
            return dict(status="ok", attempts=attempt, errors=errors, path=str(directory / "auto-head-pastes"))
        except OSError as error:
            errors.append(dict(attempt=attempt, type=type(error).__name__, winerror=getattr(error, "winerror", None),
                               message=str(error), path=str(probe)))
            time.sleep(wait)
    return dict(status="failed", attempts=attempts, errors=errors, path=str(directory / "auto-head-pastes"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="완료된 원본 비율 참고 실행 폴더(studio-runs/…)")
    parser.add_argument("--preview", type=Path, required=True, help="사용자가 확인한 머리카락 영역 폴더(hair-confirmations/…/…)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="명시적 GPU 실행 승인. 없으면 입력 계약만 검사")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(status="started", run=str(args.run), preview=str(args.preview), gpu_executed=False)
    try:
        run(args, summary)
    except BaseException as error:
        # 작업 프로세스 밖 실패도 상태·호출 경로를 남긴다.
        summary.update(status="failed", error_type=type(error).__name__, error=str(error),
                       winerror=getattr(error, "winerror", None), traceback=traceback.format_exc())
        write_json(args.output / "summary.json", summary)
        raise


def run(args, summary):
    copy = copy_run(args.run, args.output)
    batch = load_batch(copy / "generation")
    analysis = read(copy / "inputs/analysis.json")
    appearance = analysis["groups"]["appearance"]
    preview = confirmed_preview(args.preview)
    profile = paste_profile(appearance)
    reason = automatic_skip_reason(appearance, analysis.get("garment_tags", []))
    skip, background = preview_skip_reason(preview, appearance)
    results = StudioResults(batch)
    for index in range(len(batch.candidates)):
        results.select(index)
        verify_preview(preview, selected_request(results, StudioRuntime())["identity"])
    summary.update(rule=rule_name(profile), appearance=appearance, tag_skip_reason=reason, preview_skip_reason=skip,
                   background=background, seeds=[c.seed for c in batch.candidates],
                   write_probe=probe_write(batch.directory))
    write_json(args.output / "summary.json", summary)
    print("입력 계약 통과:", summary["rule"], "건너뛰기:", reason or skip or "없음",
          "쓰기 확인:", summary["write_probe"]["status"], flush=True)
    require(summary["write_probe"]["status"] == "ok", "작업 폴더를 만들 수 없어 모델을 시작하지 않았습니다: "
            + summary["write_probe"]["path"])
    if not args.gpu or reason or skip:
        summary["status"] = "contract_checked"
        write_json(args.output / "summary.json", summary)
        return
    report = apply_head_batch(batch, preview, StudioRuntime(), progress=print)
    summary.update(status="completed", gpu_executed=True, items=[dict(seed=item["seed"], status=item["status"], reason=item.get("reason", ""),
                   long_hair=read(Path(item["directory"]) / "result.json").get("long_hair")
                   if item["status"] == "completed" else None) for item in report["items"]],
                   worker=str(Path(report["items"][0]["directory"]).parent))
    write_json(args.output / "summary.json", summary)
    head_sheet(results, report, tuple(preview["box"]), args.output / "before_after.png")
    print("완료:", [item["status"] for item in report["items"]], flush=True)


if __name__ == "__main__":
    main()
