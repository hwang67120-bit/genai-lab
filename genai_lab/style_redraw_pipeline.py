"""모드 B 공통 진입점(STYLE_MODE_APP_DESIGN.md S4): 캐릭터 그림 1장(+ 옷 그림) → 목표 그림체로 다시 그린 후보 3장 → 검토 대기.

작업실 화면과 외부 API가 같이 쓴다. 확인 창을 띄우지 않고 결과(outcome.json)를 돌려준다.
단계: 캔버스(768x1024) → 관절(제품 관절 분석, 별도 Python) → 준비(CPU 작업 프로세스) → 생성(GPU, FLUX.2-klein-4B).
후보를 받은 뒤: 부위별 고르기(run_style_part_combine, CPU).
"""
import json
import subprocess
import time
from pathlib import Path

from PIL import Image

# 모델 폴더(GENAI_MODEL_HOME)가 없을 때의 개발 PC 경로.
LEGACY = {"flux_model": Path("D:/genai-cache/models/FLUX.2-klein-4B-e7b7dc27"),
          "hand_cache": Path("D:/genai-cache/huggingface")}
TIMEOUTS = {"pose": 600.0, "prepare": 900.0, "generate": 1800.0, "combine": 600.0}


class StyleRedrawFailed(RuntimeError):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


class StyleRedrawCancelled(StyleRedrawFailed):
    """사용자 취소: 실행 중인 작업 프로세스를 끝냈다(GPU 메모리는 프로세스와 함께 돌아간다)."""


POLL_SECONDS = 0.5


def _stop(process):
    """프로세스와 그 자식까지 끝낸다. Windows 가상환경의 python.exe는 실제 Python을 자식으로 띄우는 실행기라,
    바깥만 끝내면 안쪽 작업(GPU 사용)이 계속 돈다(D-224 시험에서 발견) → 트리째(taskkill /T /F)."""
    import os
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_process(command, log_path, timeout, cancelled=None, stage="", env=None):
    """하위 프로세스를 띄우고 끝날 때까지 기다리되, POLL_SECONDS마다 취소를 확인한다(D-224).
    취소면 프로세스를 끝내고 StyleRedrawCancelled, 시간 초과면 끝내고 subprocess.TimeoutExpired. 반환: 종료 코드."""
    deadline = time.monotonic() + timeout
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                return process.wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                if cancelled is not None and cancelled():
                    _stop(process)
                    raise StyleRedrawCancelled(stage, "취소했습니다.")
                if time.monotonic() > deadline:
                    _stop(process)
                    raise subprocess.TimeoutExpired(command, timeout)


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


def default_run(runtime, cancelled=None):
    """단계별 하위 프로세스. 필요한 환경 변수만 넘긴다(D-170). cancelled: 취소 확인 함수(작업실 '취소', D-224)."""
    from genai_lab.child_env import child_environment
    worker = Path(__file__).resolve().with_name("style_redraw_worker.py")
    analysis = Path(__file__).resolve().with_name("studio_analysis.py")
    def run(stage, directory):
        if stage == "pose":
            command = [str(runtime.pose_python), str(analysis), "pose", str(directory), str(runtime.pose_models),
                       str(runtime.model_cache), str(runtime.head_cache)]
        else:
            command = [str(runtime.analysis_python), "-X", "utf8", str(worker), stage, str(directory)]
        return run_process(command, directory / f"{stage}.log", TIMEOUTS[stage], cancelled, stage, child_environment())
    return run


def _stage(run, stage, directory, expected, cancelled=None):
    if cancelled is not None and cancelled():          # 단계 사이에서도 취소를 본다
        raise StyleRedrawCancelled(stage, "취소했습니다.")
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


def run_style_redraw(source_path, directory, runtime=None, *, confirmed_gender=None, confirmed_body_parts=None, paths=None, run=None,
                     garment_path=None, scene=None, cancelled=None, adult_confirmed=False):
    """캐릭터 그림 1장 → 후보 3장. 반환: outcome(state = review_pending | refused | failed | cancelled). 결과를 자동 승인하지 않는다.
    cancelled: 취소 확인 함수(작업실 '취소'). 참이면 실행 중인 작업 프로세스를 끝내고 state=cancelled(D-224).
    adult_confirmed: 사용자가 "이 캐릭터는 성인"이라고 확인했는지(D-227). True가 아니면 아무것도 하지 않고 refused —
    태그 분석기는 노출 없는 미성년·아이 캐릭터를 걸러내지 못한다(D-226).
    confirmed_gender: 사람이 확정한 성별(male·female, 캐릭터 기억 등). 없으면 성별 낱말 없이 그린다(D-198).
    confirmed_body_parts: 사람이 확정한 동물 귀·꼬리·날개(ears·tail·wings 목록, 없으면 빈 목록). None이면 태그로 추정(D-221).
    garment_path: 옷 그림(선택, D-212). 있으면 그 옷으로 갈아입히면서 목표 그림체로 그린다(모드 A·B 통합).
    scene: 새 자세·장면 문장(선택, D-222 FLUX 한 경로). scene_composer 문장 또는 사용자 문장 — 외형·의상을 바꾸는 말은 거절한다.
    없으면 원본 자세 그대로 다시 그린다."""
    if scene is not None:
        from genai_lab.free_generation import check_request
        checked = check_request(scene)
        if not checked["ok"]:
            raise ValueError("장면에는 자세·배경·상황만 적어 주세요(외형·의상을 바꾸는 말은 쓸 수 없습니다): " + ", ".join(checked["matched"]))
        scene = scene.strip() or None
    if confirmed_body_parts is not None:
        confirmed_body_parts = list(confirmed_body_parts)
        if not set(confirmed_body_parts) <= {"ears", "tail", "wings"} or len(set(confirmed_body_parts)) != len(confirmed_body_parts):
            raise ValueError("확정 부위는 ears·tail·wings 중에서 겹치지 않게 골라야 합니다.")
    if confirmed_gender not in (None, "male", "female"):
        raise ValueError("확정 성별은 male·female 또는 None이어야 합니다.")
    from genai_lab.outfit_edit import canvas
    if runtime is None:
        from genai_lab.studio_generation import StudioRuntime
        runtime = StudioRuntime.from_environment()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    run = run or default_run(runtime, cancelled)
    outcome = dict(mode="style_redraw", source=str(source_path), started=time.time(), seconds={})
    if adult_confirmed is not True:
        outcome.update(state="refused", code="adult_unconfirmed", finished=time.time(),
                       message="성인 캐릭터만 만들 수 있습니다. '이 캐릭터는 성인입니다'를 확인해 주세요.")
        _save(directory / "outcome.json", outcome)
        return outcome
    try:
        with Image.open(source_path) as image:
            canvas(image).save(directory / "character.png")
        if garment_path is not None:
            with Image.open(garment_path) as garment:            # 옷 그림은 비율 그대로(흰 여백으로 캔버스에 맞춤)
                canvas(garment).save(directory / "garment.png")
            outcome["garment"] = str(garment_path)
        _save(directory / "request.json", dict(request_paths(runtime, paths), confirmed_gender=confirmed_gender, confirmed_body_parts=confirmed_body_parts, scene=scene, adult_confirmed=True, engine="flux",
                                              garment=str(garment_path) if garment_path is not None else None))
        outcome["seconds"]["pose"] = _stage(run, "pose", directory, "pose.json", cancelled)
        outcome["seconds"]["prepare"] = _stage(run, "prepare", directory, "prepare.json", cancelled)
        prepared = json.loads((directory / "prepare.json").read_text(encoding="utf-8"))
        outcome.update(warnings=prepared.get("warnings", []), traits=prepared.get("traits"), gender=prepared.get("gender"),
                       guessed_gender=prepared.get("guessed_gender"))
        if prepared["state"] == "refused":
            outcome.update(state="refused", code=prepared["code"], message=prepared["message"])
            return outcome
        outcome["seconds"]["generate"] = _stage(run, "generate", directory, "generation.json", cancelled)
        generated = json.loads((directory / "generation.json").read_text(encoding="utf-8"))
        candidates = [str(directory / c["file"]) for c in generated.get("candidates", []) if (directory / c["file"]).is_file()]
        if not candidates:
            raise StyleRedrawFailed("generate", "생성 결과 그림이 없습니다.")
        outcome.update(state="review_pending", candidates=candidates, review="사용자가 후보 중 1장을 고른다(동일성·그림체 확인 전)",
                       peak_reserved_gib=generated.get("peak_reserved_gib"))
        return outcome
    except StyleRedrawCancelled as error:
        outcome.update(state="cancelled", stage=error.stage, message=str(error))
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


def run_style_part_combine(directory, face_candidate, body_candidate, runtime=None, *, run=None):
    """부위별 고르기(D-214): 후보 하나의 얼굴·머리 + 다른 후보의 몸·옷 → 1장(CPU). 반환: state = review_pending | failed.
    두 후보는 이 실행의 서로 다른 후보여야 하고 SHA로 확인한다. 원래 후보 그림은 바꾸지 않는다."""
    import hashlib
    directory = Path(directory)
    outcome = json.loads((directory / "outcome.json").read_text(encoding="utf-8"))
    if outcome.get("state") != "review_pending":
        raise ValueError("후보가 준비된 실행만 부위를 고를 수 있습니다.")
    face, body = Path(face_candidate), Path(body_candidate)
    for chosen in (face, body):
        if str(chosen) not in outcome.get("candidates", []) or not chosen.is_file():
            raise ValueError("고를 그림은 이 실행의 후보 중 하나여야 합니다.")
    if face == body:
        raise ValueError("얼굴·머리와 몸·옷은 서로 다른 후보에서 골라야 합니다.")
    if runtime is None and run is None:
        from genai_lab.studio_generation import StudioRuntime
        runtime = StudioRuntime.from_environment()
    run = run or default_run(runtime)
    seeds = [int(p.stem.replace("result-seed", "")) for p in (face, body)]
    _save(directory / "combine-request.json", dict(
        face_candidate=face.name, body_candidate=body.name, face_seed=seeds[0], body_seed=seeds[1],
        face_sha256=hashlib.sha256(face.read_bytes()).hexdigest(), body_sha256=hashlib.sha256(body.read_bytes()).hexdigest()))
    result = dict(mode="style_part_combine", face_candidate=str(face), body_candidate=str(body), started=time.time())
    try:
        result["seconds"] = _stage(run, "combine", directory, "combine-result.json")
        done = json.loads((directory / "combine-result.json").read_text(encoding="utf-8"))
        result.update(state="review_pending", result=str(directory / done["file"]), shift=done.get("shift"),
                      review="사람 확인 전(부위별 고르기 결과)")
    except StyleRedrawFailed as error:
        result.update(state="failed", stage=error.stage, message=str(error))
    result["finished"] = time.time()
    _save(directory / f"combine-outcome-face{seeds[0]}-body{seeds[1]}.json", result)
    return result
