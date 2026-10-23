"""배포 후보 GPU 검증 P4·P5의 안쪽(D-175) — 배포 후보 runtime-main\\python.exe로 실행한다(바깥 `run_candidate_gpu.py`가 띄움).

  p4 <출력> <모델 폴더> <저장소 루트> [--dry]   머리 분할 작업 프로세스를 개발 결과(요청 34)와 같은 캔버스로 돌리고 parts.npz를 대조
  p5 <출력> <줄 목록 json> [--dry]              사용자 화면 흐름(ProductFlow) — 장면 seed만 요청 34 기록값으로 고정, 그 밖은 화면 그대로
--dry: GPU 단계(분할·생성)를 돌리지 않고 입력 대조·경로·장면 문장·빠른 검사만 확인한다(CPU).
제품 파일은 고치지 않는다. 장면 seed 고정은 이 실행 안에서만 ProductFlow.choose_scene을 바꿔 끼운다.
"""
import hashlib
import json
import os
import re
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
# 분할 작업 프로세스가 '얼굴·머리를 찾지 못함'으로 거절하는 검사(outfit_edit_worker.segment: 보호 영역 또는 얼굴 영역이 비어 있음).
# 요청 34 이후 이 파일은 바뀌지 않았고(1eb011f9..D-176) 후보 안의 파일도 같다 → 같은 파일·함수·줄·오류 종류면 같은 검사에서 거절한 것이다.
FRAME = re.compile(r'File "[^"]*[\\/](?P<file>[^"\\/]+)", line (?P<line>\d+), in (?P<func>\w+)')
EXCEPTION = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(Error|Exception)):", re.M)
NEAR_PX = 3            # 경계 흔들림으로 볼 거리(픽셀) — 이보다 멀리 생기거나 사라진 영역을 '실질적 차이 후보'로 센다


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


def dilate(mask, radius):
    import numpy as np
    padded = np.pad(mask, radius)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def region_difference(new, dev):
    """한 영역(불 배열)의 차이를 '머리 영역 누락'·'배경 포함' 판단에 쓰는 수치로. 판정은 사람이 그림과 함께 한다."""
    import numpy as np
    new, dev = new.astype(bool), dev.astype(bool)
    union = int((new | dev).sum())
    return dict(dev_px=int(dev.sum()), new_px=int(new.sum()), iou=round(float((new & dev).sum()) / union, 4) if union else 1.0,
                missing_px=int((dev & ~new).sum()),                               # 개발에는 있고 후보에는 없음
                added_px=int((new & ~dev).sum()),                                 # 후보에만 있음
                missing_far_px=int((dev & ~dilate(new, NEAR_PX)).sum()),          # 경계 흔들림보다 먼 누락 → 머리 영역 누락 후보
                added_far_px=int((new & ~dilate(dev, NEAR_PX)).sum()))            # 경계 흔들림보다 먼 추가 → 배경 포함 후보


def substantive_evidence(new_npz, dev_npz, canvas, work):
    """배열이 다를 때: 보호 영역(머리·얼굴·장식)과 부위별 차이 수치 + 겹쳐 본 그림(초록 둘 다, 빨강 개발에만, 파랑 후보에만)."""
    import numpy as np
    from PIL import Image
    from genai_lab.outfit_edit import HEAD_ACCESSORIES, HEAD_KEEP
    from genai_lab.outfit_edit_worker import FACE_PARTS
    new, dev = np.load(new_npz), np.load(dev_npz)
    shared = sorted(set(new.files) & set(dev.files))

    def union(data, names):
        names = [n for n in names if n in data.files]
        return np.logical_or.reduce([data[n].astype(bool) for n in names]) if names else None

    regions = {}
    for label, names in (("protect", HEAD_KEEP + HEAD_ACCESSORIES), ("face", FACE_PARTS)):
        a, b = union(new, names), union(dev, names)
        if a is not None and b is not None and a.shape == b.shape:
            regions[label] = region_difference(a, b)
    parts = {k: region_difference(new[k], dev[k]) for k in shared if new[k].shape == dev[k].shape and not np.array_equal(new[k], dev[k])}
    pictures = []
    for label, names in (("protect", HEAD_KEEP + HEAD_ACCESSORIES),):
        a, b = union(new, names), union(dev, names)
        if a is None or b is None or a.shape != b.shape:
            continue
        with Image.open(canvas) as image:
            base = np.asarray(image.convert("RGB")).astype(np.float32) * 0.45
        colour = np.zeros_like(base)
        colour[a & b] = (0, 200, 0)
        colour[b & ~a] = (255, 0, 0)
        colour[a & ~b] = (0, 90, 255)
        mixed = np.where((a | b)[..., None], base * 0.4 + colour * 0.6, base).clip(0, 255).astype(np.uint8)
        path = work / f"diff-{label}.png"
        Image.fromarray(mixed).save(path)
        pictures.append(str(path))
    return dict(regions=regions, parts=parts, pictures=pictures,
                note="빨강 = 개발에만(후보에서 빠짐), 파랑 = 후보에만(새로 들어감). '먼' 수치는 경계에서 3픽셀보다 먼 차이")


def refusal_site(log_path):
    """작업 프로세스 기록(worker.log)의 오류 추적에서 거절한 자리: 마지막 프레임(파일·줄·함수) + 오류 종류. 메시지 인코딩과 무관."""
    try:
        text = Path(log_path).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None
    frames = list(FRAME.finditer(text))
    exceptions = list(EXCEPTION.finditer(text))
    if not frames or not exceptions:
        return None
    last = frames[-1]
    return dict(file=last["file"], line=int(last["line"]), function=last["func"], exception=exceptions[-1]["type"])


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
            else:       # 개발 실행에서도 분할이 거절한 건(A3 결과 분석: 얼굴·머리 못 찾음) → 같은 이유로 거절되는지 본다
                dev_error = json.loads((dev / "worker-error.json").read_text(encoding="utf-8"))
                row.update(dev_error=dev_error["error"], dev_stage=dev_error.get("stage"), dev_site=refusal_site(dev / "worker.log"))
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
                    row.update(state="refusal_not_reproduced", seconds=round(time.time() - start, 1))
                except ValueError as error:
                    error_file = work / "parts" / "worker-error.json"
                    new_error = json.loads(error_file.read_text(encoding="utf-8")) if error_file.is_file() else dict(error=str(error))
                    row.update(state="refused", seconds=round(time.time() - start, 1), new_error=new_error.get("error"),
                               new_stage=new_error.get("stage"), new_site=refusal_site(work / "parts" / "worker.log"))
                continue
            result = segment_canvas(prepared, work / "parts", runtime)
            row.update(state="ok", seconds=round(time.time() - start, 1),
                       new=dict(protect_px=result["protect_px"], face_px=result["face_px"], accessory_px=result["accessory_px"],
                                segmenter=result["segmenter"]),
                       compare=compare_parts(result["parts"], dev / "parts.npz"))
            if not row["compare"]["all_equal"]:
                row["evidence"] = substantive_evidence(result["parts"], dev / "parts.npz", local, work)
        except Exception as error:  # noqa: BLE001
            row.update(state="failed", error=f"{type(error).__name__}: {error}"[:1500], tb=traceback.format_exc()[-1500:])
    report.update(p4_verdict(report["rows"], dry))
    return report


def classify(row):
    """P4 한 건의 결론(2026-10-24 사용자: 통과 / 검토 필요 / 예상 밖 실패).
    pass          분할 완료 + 배열이 개발 결과와 같음, 또는 개발에서도 거절한 건이 같은 문구·같은 단계·같은 검사(파일·줄·함수·오류 종류)로 거절됨
    review_needed 실행은 정상인데 결과가 다름 — 배열 차이(수치·겹친 그림을 보고 실질적 차이인지 사람이 확인), 거절이 재현되지 않음,
                  같은 문구지만 거절 자리를 확인할 수 없음. 해결 전에는 P5를 시작하지 않음
    unexpected    실행 실패: 오류, 캔버스 SHA 불일치, 다른 문구·다른 검사로 거절(예: CUDA 오류)
    반환: (결론, 설명)"""
    state = row.get("state", "")
    if state == "dry":
        return "dry", ""
    if state == "ok":
        if row.get("compare", {}).get("all_equal"):
            return "pass", "배열이 개발 결과와 같음"
        return "review_needed", "배열이 다름 — 수치·겹친 그림으로 머리 영역 누락·배경 포함 같은 실질적 차이인지 확인"
    if state == "refusal_not_reproduced":
        return "review_needed", "개발은 거절했는데 후보는 분할을 끝냄"
    if state == "refused":
        if row.get("new_error") != row.get("dev_error") or row.get("new_stage") != row.get("dev_stage"):
            return "unexpected", "다른 문구 또는 다른 단계에서 실패"
        if row.get("dev_site") and row.get("new_site") and row["dev_site"] == row["new_site"]:
            return "pass", "같은 문구·단계·검사({file} {line}행 {function}, {exception})에서 거절 — 예정된 거절 재현".format(**row["new_site"])
        if row.get("dev_site") and row.get("new_site"):
            return "unexpected", "같은 문구지만 다른 자리에서 거절"
        return "review_needed", "같은 문구지만 거절 자리를 기록에서 확인하지 못함"
    return "unexpected", "실행 실패: " + state


def p4_verdict(rows, dry):
    for row in rows.values():               # 모든 건을 끝에서 한 번에 판정(건너뛴 건도 빠지지 않게)
        if "outcome" not in row:
            row["outcome"], row["outcome_reason"] = classify(row)
    outcomes = {name: row["outcome"] for name, row in rows.items()}
    unexpected = sorted(name for name, outcome in outcomes.items() if outcome == "unexpected")
    review = sorted(name for name, outcome in outcomes.items() if outcome == "review_needed")
    complete = not dry and len(rows) == len(P4_DEV)
    return dict(outcomes=outcomes, unexpected=unexpected, review_needed=review,
                p5_allowed=complete and not unexpected and not review,
                p5_allowed_after_review=complete and not unexpected)


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
