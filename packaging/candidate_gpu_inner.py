"""배포 후보 GPU 검증 P4·P5의 안쪽(D-175) — 배포 후보 runtime-main\\python.exe로 실행한다(바깥 `run_candidate_gpu.py`가 띄움).

  p4 <출력> <모델 폴더> <저장소 루트> [--dry]   머리 분할 작업 프로세스를 개발 결과(요청 34)와 같은 캔버스로 돌리고 parts.npz를 대조
  p5 <출력> <줄 목록 json> [--dry]              사용자 화면 흐름(ProductFlow) — 장면 seed만 요청 34 기록값으로 고정, 그 밖은 화면 그대로
--dry: GPU 단계(분할·생성)를 돌리지 않고 입력 대조·경로·장면 문장·빠른 검사만 확인한다(CPU).
제품 파일은 고치지 않는다. 장면 seed 고정은 이 실행 안에서만 ProductFlow.choose_scene을 바꿔 끼운다.
"""
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 요청 34(개발 환경, 2026-10-22, 코드 1eb011f9)의 분할 결과 — 같은 캔버스(SHA)로 다시 돌려 비교한다
P4_DEV = (("B2-outfit-parts", "b5cb43fef5bf4277a15f16db4fd23644/outfit-edit/parts"),
          ("B2-base-parts", "b5cb43fef5bf4277a15f16db4fd23644/free-generation/base-parts"),
          ("B2-result-parts", "b5cb43fef5bf4277a15f16db4fd23644/free-generation/result-analysis/parts"),
          ("A3-base-parts", "eaf9b7c7102c4a67ad3859c611e01762/free-generation/base-parts"),
          ("A3-result-parts", "eaf9b7c7102c4a67ad3859c611e01762/free-generation/result-analysis/parts"))
DEV_RUNS = "outputs/product-run-20261022/runs"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def compare_parts(new_npz, dev_npz):
    import numpy as np
    new, dev = np.load(new_npz), np.load(dev_npz)
    keys = sorted(set(new.files) | set(dev.files))
    out = dict(same_file_sha=sha(new_npz) == sha(dev_npz), keys_only_new=sorted(set(new.files) - set(dev.files)),
               keys_only_dev=sorted(set(dev.files) - set(new.files)), arrays={})
    for k in keys:
        if k in new.files and k in dev.files:
            a, b = new[k], dev[k]
            if a.shape != b.shape:
                out["arrays"][k] = dict(equal=False, shape_new=list(a.shape), shape_dev=list(b.shape))
            else:
                out["arrays"][k] = dict(equal=bool(np.array_equal(a, b)), differing=int((a != b).sum()), size=int(a.size))
    out["all_equal"] = not out["keys_only_new"] and not out["keys_only_dev"] and all(v["equal"] for v in out["arrays"].values())
    return out


def p4(out, home, repo, dry):
    os.environ["GENAI_MODEL_HOME"] = str(home)
    from genai_lab.outfit_edit_pipeline import segment_canvas
    from genai_lab.studio_generation import StudioRuntime
    runtime = StudioRuntime.from_environment()
    candidate = Path(os.environ["GENAI_APP_HOME"])
    report = dict(stage="p4", dry=dry, executable=sys.executable, analysis_python=str(runtime.analysis_python),
                  analysis_python_in_candidate=inside(runtime.analysis_python, candidate),
                  semantic_repo=str(runtime.head_semantic_repo), semantic_checkpoint=str(runtime.head_semantic_checkpoint),
                  semantic_in_model_home=inside(runtime.head_semantic_checkpoint, home), rows={})
    for name, rel in P4_DEV:
        row = report["rows"][name] = {}
        dev = repo / DEV_RUNS / rel
        try:
            request = json.loads((dev / "request.json").read_text(encoding="utf-8"))
            canvas = repo / request["canvas"]
            row.update(dev_dir=str(dev), canvas_sha_match=sha(canvas) == request["canvas_sha256"])
            if (dev / "result.json").is_file():
                dev_result = json.loads((dev / "result.json").read_text(encoding="utf-8"))
                row["dev"] = dict(protect_px=dev_result["protect_px"], face_px=dev_result["face_px"],
                                  accessory_px=dev_result["accessory_px"], segmenter=dev_result["segmenter"])
            else:       # 개발 실행에서도 분할이 거절한 건(A3 결과 분석: 얼굴·머리 못 찾음) → 같은 거절이 나오는지 본다
                row["dev_error"] = json.loads((dev / "worker-error.json").read_text(encoding="utf-8"))["error"]
            if not row["canvas_sha_match"]:
                row["state"] = "skipped: canvas SHA 불일치"
                continue
            work = out / name
            work.mkdir(parents=True)
            local = work / "canvas.png"                        # 시험 입력을 출력 폴더로 복사(배포 후보 프로세스가 저장소를 읽지 않게)
            shutil.copy2(canvas, local)
            prepared = dict(canvas=str(local), canvas_sha256=request["canvas_sha256"])
            if dry:
                row["state"] = "dry"
                continue
            start = time.time()
            if "dev_error" in row:
                try:
                    segment_canvas(prepared, work / "parts", runtime)
                    row.update(state="different: 개발은 거절, 후보는 분할 완료", seconds=round(time.time() - start, 1))
                except ValueError as error:
                    error_file = work / "parts" / "worker-error.json"
                    new_error = json.loads(error_file.read_text(encoding="utf-8"))["error"] if error_file.is_file() else str(error)
                    row.update(state="ok" if new_error == row["dev_error"] else "different: 거절 문구가 다름",
                               seconds=round(time.time() - start, 1), new_error=new_error)
                continue
            result = segment_canvas(prepared, work / "parts", runtime)
            row.update(state="ok", seconds=round(time.time() - start, 1),
                       new=dict(protect_px=result["protect_px"], face_px=result["face_px"], accessory_px=result["accessory_px"],
                                segmenter=result["segmenter"]),
                       compare=compare_parts(result["parts"], dev / "parts.npz"))
        except Exception as error:  # noqa: BLE001
            row.update(state="failed", error=f"{type(error).__name__}: {error}"[:1500], tb=traceback.format_exc()[-1500:])
    report.update(p4_verdict(report["rows"], dry))
    return report


def classify(row):
    """P4 한 건의 결론(2026-10-23 사용자: 예정된 거절 재현과 예상 밖 실행 실패를 구분).
    expected_refusal_reproduced  개발에서도 거절한 건이 같은 문구로 거절됨 → 통과
    match                        분할 완료 + 배열이 개발 결과와 같음 → 통과
    array_difference             분할 완료, 배열이 다름 → 실패는 아님(원인 미확정으로 보고)
    unexpected                   그 밖(실행 오류·SHA 불일치·거절이 재현되지 않거나 문구가 다름) → P5를 시작하지 않음"""
    state = row.get("state", "")
    if state == "dry":
        return "dry"
    if state == "ok" and "dev_error" in row:
        return "expected_refusal_reproduced"
    if state == "ok":
        return "match" if row.get("compare", {}).get("all_equal") else "array_difference"
    return "unexpected"


def p4_verdict(rows, dry):
    for row in rows.values():               # 모든 건을 끝에서 한 번에 판정(건너뛴 건도 빠지지 않게)
        row.setdefault("outcome", classify(row))
    outcomes = {name: row["outcome"] for name, row in rows.items()}
    unexpected = sorted(name for name, outcome in outcomes.items() if outcome == "unexpected")
    return dict(outcomes=outcomes, unexpected=unexpected,
                p5_allowed=not dry and len(rows) == len(P4_DEV) and not unexpected)


def p5(out, cases_path, dry):
    from genai_lab.child_env import isolate_hf_credentials
    isolate_hf_credentials()
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from genai_lab import model_setup
    from genai_lab.app_home import app_home
    from genai_lab.scene_composer import compose_scene
    decision, value = model_setup.startup_decision(quick_check=None, packaged=app_home() is not None)
    report = dict(stage="p5", dry=dry, executable=sys.executable, decision=decision, decision_value=str(value), cases={})
    if decision != "studio":
        report["error"] = "검증 기록으로 작업실을 열 수 없음(준비 단계가 먼저 통과해야 함)"
        return report
    sys.path.insert(0, str(Path(os.environ["GENAI_APP_HOME"]) / "app" / "scripts"))
    import gui_main
    import product_flow_run as flow                  # 배포 후보 안의 사용자 화면 실행기(요청 34와 같은 것)
    window = gui_main.open_studio(value)
    app.processEvents()
    from genai_lab import model_check
    from genai_lab.qwen_pose_settings import QwenPoseSettings
    runtime = window.studio.runtime if hasattr(window.studio, "runtime") else None
    from genai_lab.studio_generation import StudioRuntime
    runtime = runtime or StudioRuntime.from_environment()
    settings = QwenPoseSettings.load(runtime.qwen_settings)
    report["runtime"] = dict(analysis_python=str(runtime.analysis_python), pose_python=str(runtime.pose_python),
                             qwen_settings=str(runtime.qwen_settings), qwen_python=str(settings.python_executable),
                             model_home=os.environ.get("GENAI_MODEL_HOME"))
    for garment in (False, True):
        report[f"quick_check_{'outfit' if garment else 'staging'}"] = model_check.require_product_models(runtime, settings, garment=garment)["ok"]
    flow.OUT = out
    out.mkdir(parents=True, exist_ok=True)
    window.studio.output_root = out / "runs"
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    for case in cases:
        row = report["cases"][case["case"]] = {}
        row["inputs_sha_match"] = (sha(case["character"]) == case["character_sha256"]
                                   and (case.get("garment") is None or sha(case["garment"]) == case["garment_sha256"]))
        scene = compose_scene(case["scene_seed"])
        row["scene_text_match"] = scene["text"] == case["scene_text"]
        if not (row["inputs_sha_match"] and row["scene_text_match"]):
            row["state"] = "skipped: 입력 SHA 또는 장면 문장이 요청 34 기록과 다름"
            continue
        if dry:
            row["state"] = "dry"
            continue
        window.studio.product.choose_scene = lambda seed=case["scene_seed"]: compose_scene(seed)   # 장면 seed 고정(이 실행 안에서만)
        try:
            flow.run_case(window, case)
        except Exception:  # noqa: BLE001
            flow.status["cases"][case["case"]] = dict(state="crashed", traceback=traceback.format_exc()[-3000:])
        row.update(flow=flow.status["cases"].get(case["case"]))
    flow.status.update(state="finished", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not dry:
        flow.save_status()
        (out / "summary.json").write_text(json.dumps(flow.status, ensure_ascii=False, indent=2), encoding="utf-8")
    window.close()
    return report


def main(argv):
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    stage, out = argv[0], Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    try:
        if stage == "p4":
            report = p4(out, Path(argv[2]), Path(argv[3]), dry)
        elif stage == "p5":
            report = p5(out, Path(argv[2]), dry)
        else:
            raise SystemExit("usage: p4|p5 ...")
    except Exception as error:  # noqa: BLE001
        report = dict(stage=stage, error=f"{type(error).__name__}: {error}", tb=traceback.format_exc()[-2000:])
    report.update(base_prefix=sys.base_prefix, sys_path=list(sys.path))
    (out / f"inner-{stage}{'-dry' if dry else ''}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                                                     encoding="utf-8")
    print(stage, "dry" if dry else "", report.get("error", "ok"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
