"""최초 모델 준비(D-173) — 기존 로컬 모델 폴더 선택 · 공통 깊은 검사 · 결과 저장 · 재검사.

- 검사는 공통 검사(model_check.check_models, deep=True)를 그대로 쓴다. '이미지 생성' 전 빠른 검사와 같은 목록·같은 함수.
- 화면을 막지 않도록 검사는 별도 프로세스(`python -m genai_lab.model_setup check …`)에서 돈다. 취소 = 그 프로세스를 끝냄.
- 결과에는 실행 번호와 검사한 폴더가 들어가며, 화면은 지금 실행·지금 폴더의 결과만 받는다.
- 저장: 깊은 검사를 끝까지 마치고 문제 0일 때만 '모델 파일 검증 완료'. 취소·중단은 저장하지 않는다.
  'GPU 생성 검증'은 별개다 — 이 기록은 항상 아직 안 됨(False)으로 둔다(첫 생성·최종 패키징 검증 P4·P5 몫).
- 이번 범위 밖: 자동 내려받기·모델 이동·삭제·복구, SHA 정책 변경.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from genai_lab.model_home import ENV, LAYOUT

ROOT = Path(__file__).resolve().parents[1]
HOME_SETTINGS = ROOT / "configs" / "qwen_edit_settings.model_home.json"
SCHEMA = "model_setup/1"
FILES_VERIFIED = "files_verified"
FILES_CHECK_FAILED = "files_check_failed"


def record_path():
    """설정 기록 위치: %APPDATA%\\GenAILab\\model-setup.json(GENAI_MODEL_SETUP_RECORD로 바꿀 수 있음 — 시험용)."""
    override = os.environ.get("GENAI_MODEL_SETUP_RECORD")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "GenAILab" / "model-setup.json"


def runtime_for_home(home):
    """선택한 모델 폴더의 배치(model_home.LAYOUT)로 공통 검사가 볼 위치. 환경 변수는 바꾸지 않는다."""
    home = Path(home)
    return SimpleNamespace(model_cache=home / LAYOUT["model_cache"], pose_models=home / LAYOUT["pose_models"],
                           head_cache=home / LAYOUT["head_cache"], head_semantic_repo=home / LAYOUT["head_semantic_repo"],
                           head_semantic_checkpoint=home / LAYOUT["head_semantic_checkpoint"])


def settings_for_home(home):
    """모델 폴더용 Qwen 설정을 이 폴더 기준으로 읽는다(설정 파일과 경로 3개만 다름, D-166)."""
    from genai_lab.qwen_pose_settings import QwenPoseSettings
    values = json.loads(HOME_SETTINGS.read_text(encoding="utf-8-sig"))
    values["artifact_manifest"] = str((HOME_SETTINGS.parent / values["artifact_manifest"]).resolve())
    for key in ("model_root", "gguf_file"):
        values[key] = str(Path(home) / values[key])
    if values.get("python_executable") == "@app":
        values["python_executable"] = sys.executable
    return QwenPoseSettings(**values)


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_fingerprint():
    """검사 목록이 바뀌면(앱 갱신) 이전 검증 기록은 쓰지 않는다."""
    settings = json.loads(HOME_SETTINGS.read_text(encoding="utf-8-sig"))
    return dict(model_home_files=_file_sha(ROOT / "configs" / "model_home_files.json"),
                qwen_model_files=settings["artifact_manifest_sha256"], gguf_sha256=settings["gguf_sha256"])


def run_check(home, run_id, *, progress=print):
    """깊은 검사 한 번 → 결과 dict. 문제가 있으면 ok=False와 안내 문구. 예외도 결과로 돌려준다(성공으로 바꾸지 않음)."""
    from genai_lab import model_check
    start = time.time()
    home = Path(home).resolve()
    result = dict(schema=SCHEMA, run_id=run_id, home=str(home), deep=True, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        if not home.is_dir():
            raise FileNotFoundError(f"모델 폴더가 없습니다: {home}")
        checked = model_check.check_models(runtime_for_home(home), settings_for_home(home), "outfit", deep=True, progress=progress)
        result.update(ok=checked["ok"], problems=checked["problems"][:200], problem_count=len(checked["problems"]),
                      message="" if checked["ok"] else model_check.message(checked), locations=checked["locations"])
    except Exception as error:  # noqa: BLE001 — 검사 실패도 결과(완료로 저장하지 않음)
        result.update(ok=False, problems=[], problem_count=None, message=f"검사를 마치지 못했습니다: {type(error).__name__}: {error}")
    result.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - start, 1), fingerprint=manifest_fingerprint())
    return result


def save_result(result, *, path=None):
    """검사 결과 저장. 끝까지 마친 결과만 받는다(취소·중단은 부르지 않음). 통과만 FILES_VERIFIED."""
    if "finished" not in result or result.get("schema") != SCHEMA:
        raise ValueError("끝나지 않은 검사 결과는 저장하지 않습니다.")
    record = dict(schema=SCHEMA, model_home=result["home"], status=FILES_VERIFIED if result["ok"] is True else FILES_CHECK_FAILED,
                  checked_at=result["finished"], run_id=result["run_id"], seconds=result["seconds"], fingerprint=result["fingerprint"],
                  problem_count=result.get("problem_count"), message=result.get("message", ""),
                  gpu_generation_verified=False,
                  note="모델 파일 검증(있음·크기·SHA)일 뿐 GPU 생성 검증이 아니다. 생성 확인은 첫 생성·최종 패키징 검증 P4·P5")
    path = Path(path) if path else record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return record


def load_record(path=None):
    path = Path(path) if path else record_path()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if record.get("schema") == SCHEMA else None


def verified_home(record=None):
    """작업실이 쓸 모델 폴더: 검증 완료 기록 + 폴더가 있음 + 검사 목록이 그대로일 때만. 아니면 None."""
    record = load_record() if record is None else record
    if not record or record.get("status") != FILES_VERIFIED or record.get("fingerprint") != manifest_fingerprint():
        return None
    home = Path(record["model_home"])
    return home if home.is_dir() else None


def startup_decision(environ=None, record=None, quick_check=None):
    """작업실 시작 때: ('studio', 모델 폴더 또는 None) 또는 ('setup', 이유).
    - 사용자가 GENAI_MODEL_HOME을 직접 정했으면 그대로(검사는 '이미지 생성' 전 빠른 검사).
    - 검증 완료 기록이 있으면 그 폴더로 작업실.
    - 기록이 없어도 지금 경로가 빠른 검사를 통과하면 작업실(개발 PC 기존 사용 방식 유지).
    - 그 밖에는 준비 화면."""
    environ = os.environ if environ is None else environ
    if environ.get(ENV):
        return "studio", Path(environ[ENV])
    home = verified_home(record)
    if home is not None:
        return "studio", home
    if quick_check is not None and quick_check():
        return "studio", None
    return "setup", "모델 파일 검증 기록이 없거나 맞지 않습니다."


def current_paths_pass():
    """지금 실행 설정(모델 폴더 없음 = 기존 경로)으로 두 흐름의 빠른 검사가 통과하는지."""
    from genai_lab import model_check
    from genai_lab.qwen_pose_settings import QwenPoseSettings
    from genai_lab.studio_generation import StudioRuntime
    try:
        runtime = StudioRuntime.from_environment()
        settings = QwenPoseSettings.load(runtime.qwen_settings)
        return all(model_check.check_models(runtime, settings, flow)["ok"] for flow in ("outfit", "staging"))
    except Exception:  # noqa: BLE001 — 판단 불가면 준비 화면으로
        return False


def main(argv):
    """검사 프로세스: python -m genai_lab.model_setup check <모델 폴더> <결과 json> <실행 번호>. 결과 파일은 끝까지 마쳤을 때만 쓴다."""
    if len(argv) != 4 or argv[0] != "check":
        raise SystemExit("usage: python -m genai_lab.model_setup check <home> <out.json> <run_id>")
    _, home, out, run_id = argv
    def progress(text):
        print("progress " + text, flush=True)
    result = run_check(home, run_id, progress=progress)
    out = Path(out)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, out)
    print("done " + ("ok" if result["ok"] else "problems"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
