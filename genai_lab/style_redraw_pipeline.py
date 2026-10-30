"""모드 B 공통 진입점(STYLE_MODE_APP_DESIGN.md S4): 캐릭터 그림 1장 → 목표 그림체로 다시 그린 전신 1장 → 검토 대기.

작업실 화면과 외부 API가 같이 쓴다. 확인 창을 띄우지 않고 결과(outcome.json)를 돌려준다.
단계: 캔버스(768x1024) → 관절(제품 관절 분석, 별도 Python) → 준비(CPU 작업 프로세스) → 생성(GPU 작업 프로세스).
"""
import json
import subprocess
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_BINDING = ROOT / "outputs/style-training-prep-20261025/character-style-binding-20261027"
# 모델 폴더(GENAI_MODEL_HOME)가 없을 때의 개발 PC 경로(D-195 시험과 같은 파일).
LEGACY = {"style_models": _BINDING / "cpu-v2-staging",
          "style_reference": ROOT / "outputs/style-training-prep-20261025/style-review-20261026/references/target-02.png",
          "style_reference_pose": _BINDING / "pose-target-02.json",
          "style_lora": ROOT / "outputs/style-training-prep-20261025/style-lora-run-v1-20261030/step-00500",
          "flux_model": Path("D:/genai-cache/models/FLUX.2-klein-4B-e7b7dc27"),
          "hand_cache": Path("D:/genai-cache/huggingface")}
TIMEOUTS = {"pose": 600.0, "prepare": 900.0, "generate": 1800.0, "refine": 900.0, "face": 900.0}


class StyleRedrawFailed(RuntimeError):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def request_paths(runtime, overrides=None):
    """작업 프로세스에 넘길 모델·참조 경로. 모델 폴더가 정해졌으면 그 배치, 아니면 개발 PC 경로."""
    from genai_lab.model_home import model_path
    paths = {key: str(model_path(key, legacy)) for key, legacy in LEGACY.items()}
    paths.update(model_cache=str(runtime.model_cache), head_semantic_repo=str(runtime.head_semantic_repo),
                 head_semantic_checkpoint=str(runtime.head_semantic_checkpoint))
    paths.update({k: str(v) for k, v in (overrides or {}).items()})
    return paths


def default_run(runtime):
    """단계별 하위 프로세스. 필요한 환경 변수만 넘긴다(D-170)."""
    from genai_lab.child_env import child_environment
    worker = Path(__file__).resolve().with_name("style_redraw_worker.py")
    analysis = Path(__file__).resolve().with_name("studio_analysis.py")
    def run(stage, directory):
        if stage == "pose":
            command = [str(runtime.pose_python), str(analysis), "pose", str(directory), str(runtime.pose_models),
                       str(runtime.model_cache), str(runtime.head_cache)]
        else:
            command = [str(runtime.analysis_python), "-X", "utf8", str(worker), stage, str(directory)]
        with open(directory / f"{stage}.log", "w", encoding="utf-8") as log:
            done = subprocess.run(command, env=child_environment(), stdout=log, stderr=subprocess.STDOUT,
                                  timeout=TIMEOUTS[stage])
        return done.returncode
    return run


def _stage(run, stage, directory, expected):
    t0 = time.time()
    try:
        code = run(stage, directory)
    except subprocess.TimeoutExpired:
        raise StyleRedrawFailed(stage, f"{stage} 단계가 시간 안에 끝나지 않았습니다.")
    if code != 0 or not (directory / expected).is_file():
        error = directory / f"{stage}-error.json"
        detail = json.loads(error.read_text(encoding="utf-8")).get("message", "") if error.is_file() else ""
        raise StyleRedrawFailed(stage, f"{stage} 단계 실패(코드 {code}). {detail}".strip())
    return round(time.time() - t0, 1)


ENGINES = ("flux", "sdxl_lora")


def run_style_redraw(source_path, directory, runtime=None, *, confirmed_gender=None, paths=None, run=None, engine="flux"):
    """캐릭터 그림 1장 → 후보 3장. 반환: outcome(state = review_pending | refused | failed). 결과를 자동 승인하지 않는다.
    confirmed_gender: 사람이 확정한 성별(male·female, 캐릭터 기억 등). 없으면 성별 낱말 없이 그린다(D-198).
    engine: "flux"(기본, D-210 FLUX.2-klein 편집) 또는 "sdxl_lora"(이전 RealVisXL + 스타일 LoRA + 참조 경로)."""
    if confirmed_gender not in (None, "male", "female"):
        raise ValueError("확정 성별은 male·female 또는 None이어야 합니다.")
    if engine not in ENGINES:
        raise ValueError(f"모르는 생성 방식입니다: {engine}")
    from genai_lab.outfit_edit import canvas
    if runtime is None:
        from genai_lab.studio_generation import StudioRuntime
        runtime = StudioRuntime.from_environment()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    run = run or default_run(runtime)
    outcome = dict(mode="style_redraw", source=str(source_path), started=time.time(), seconds={})
    try:
        with Image.open(source_path) as image:
            canvas(image).save(directory / "character.png")
        _save(directory / "request.json", dict(request_paths(runtime, paths), confirmed_gender=confirmed_gender, engine=engine))
        outcome["seconds"]["pose"] = _stage(run, "pose", directory, "pose.json")
        outcome["seconds"]["prepare"] = _stage(run, "prepare", directory, "prepare.json")
        prepared = json.loads((directory / "prepare.json").read_text(encoding="utf-8"))
        outcome.update(warnings=prepared.get("warnings", []), traits=prepared.get("traits"), gender=prepared.get("gender"),
                       guessed_gender=prepared.get("guessed_gender"))
        if prepared["state"] == "refused":
            outcome.update(state="refused", code=prepared["code"], message=prepared["message"])
            return outcome
        outcome["seconds"]["generate"] = _stage(run, "generate", directory, "generation.json")
        generated = json.loads((directory / "generation.json").read_text(encoding="utf-8"))
        candidates = [str(directory / c["file"]) for c in generated.get("candidates", []) if (directory / c["file"]).is_file()]
        if not candidates:
            raise StyleRedrawFailed("generate", "생성 결과 그림이 없습니다.")
        outcome.update(state="review_pending", candidates=candidates, review="사용자가 후보 중 1장을 고른다(동일성·그림체 확인 전)",
                       peak_reserved_gib=generated.get("peak_reserved_gib"))
        return outcome
    except StyleRedrawFailed as error:
        outcome.update(state="failed", stage=error.stage, message=str(error))
        return outcome
    except (OSError, ValueError) as error:          # 그림을 열지 못함 등 — 원인과 함께 실패로 돌려준다
        outcome.update(state="failed", stage="input", message=f"{type(error).__name__}: {error}")
        return outcome
    finally:
        outcome["finished"] = time.time()
        _save(directory / "outcome.json", outcome)


def _run_on_candidate(directory, candidate, runtime, run, stage, mode, review):
    """고른 후보 1장에 단계 하나(refine·face)를 돌린다. 후보는 이 실행의 후보여야 하고 SHA로 확인, 원래 그림은 바꾸지 않는다."""
    import hashlib
    directory = Path(directory)
    outcome = json.loads((directory / "outcome.json").read_text(encoding="utf-8"))
    if outcome.get("state") != "review_pending":
        raise ValueError("후보가 준비된 실행만 다듬을 수 있습니다.")
    chosen = Path(candidate)
    if str(chosen) not in outcome.get("candidates", []) or not chosen.is_file():
        raise ValueError("다듬을 그림은 이 실행의 후보 중 하나여야 합니다.")
    if runtime is None and run is None:
        from genai_lab.studio_generation import StudioRuntime
        runtime = StudioRuntime.from_environment()
    run = run or default_run(runtime)
    seed = int(chosen.stem.replace("result-seed", ""))
    _save(directory / f"{stage}-request.json", dict(candidate=chosen.name, seed=seed,
                                                    candidate_sha256=hashlib.sha256(chosen.read_bytes()).hexdigest()))
    result = dict(mode=mode, candidate=str(chosen), started=time.time())
    try:
        result["seconds"] = _stage(run, stage, directory, f"{stage}-result.json")
        done = json.loads((directory / f"{stage}-result.json").read_text(encoding="utf-8"))
        result.update(state="review_pending", result=str(directory / done["file"]), raw=str(directory / done["raw_file"]),
                      peak_reserved_gib=done.get("peak_reserved_gib"), review=review)
    except StyleRedrawFailed as error:
        result.update(state="failed", stage=error.stage, message=str(error))
    result["finished"] = time.time()
    _save(directory / f"{stage}-outcome-seed{seed}.json", result)
    return result


def run_style_refine(directory, candidate, runtime=None, *, run=None):
    """고른 후보 1장을 그림체 참조로 다듬는다(D-203, 그림체 참조 방식 기록용 — LoRA 흐름에서는 쓰지 않음)."""
    result = _run_on_candidate(directory, candidate, runtime, run, "refine", "style_refine", "사람 확인 전(다듬기 결과)")
    if "result" in result:
        result["refined"] = result["result"]
    return result


def run_style_face_detail(directory, candidate, runtime=None, *, run=None):
    """고른 후보 1장의 얼굴을 원본 쪽으로 다듬고 눈 색을 되돌린다(D-206). 반환: state = review_pending | failed."""
    return _run_on_candidate(directory, candidate, runtime, run, "face", "style_face_detail", "사람 확인 전(얼굴 다듬기 결과)")
