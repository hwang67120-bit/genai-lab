"""단일 작성자가 JSON을 공개하고 Qwen 진행 자료는 변경 없이 유지한다. 모델은 가져오지 않는다. Windows에서 FILE_SHARE_DELETE 없는
읽기 작업은 파일 교체를 막을 수 있다. 작성 중인 최종 기록을 반복해서 열지 않는다. 같은 폴더의 임시 파일로 완전한 JSON만 공개한다. 재시도는 제한하며
영구 실패는 오류로 남기고 임시 파일을 보존한다.
"""
import json
import os
from pathlib import Path
import tempfile
import time

REPLACE_TIMEOUT_SECONDS = 2.0
REPLACE_DELAYS = (.02, .04, .08, .16, .20, .25, .25, .25, .25, .25, .25)
TERMINAL_STATES = frozenset(("completed", "failed", "cancelled"))


class JSONReplaceError(PermissionError):
    def __init__(self, path, pending):
        self.pending_path = pending
        super().__init__(f"JSON publication exceeded bounded sharing retry: {path}; complete pending JSON: {pending}")


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    # 이름을 바꾸기 전에 파일 핸들을 닫는다. Windows에서도 필수다.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        pending = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    deadline = time.monotonic() + REPLACE_TIMEOUT_SECONDS
    for attempt in range(len(REPLACE_DELAYS) + 1):
        try:
            pending.replace(path)
            return
        except PermissionError as error:
            remaining = deadline - time.monotonic()
            if attempt == len(REPLACE_DELAYS) or remaining <= 0:
                raise JSONReplaceError(path, pending) from error
            time.sleep(min(REPLACE_DELAYS[attempt], remaining))
    # 다른 입출력 오류는 즉시 전달하고 저장 대기 자료는 의도적으로 보존한다.


def publish_progress(directory, record):
    """작업자는 하나만 사용한다. 공개 파일명은 매번 새로 만들고 이후 변경하지 않는다."""
    folder = Path(directory) / "progress"
    folder.mkdir(exist_ok=True)
    existing = sorted(folder.glob("[0-9]????????.json"))
    number = int(existing[-1].stem) + 1 if existing else 1
    path = folder / f"{number:09d}.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, record)
    return path


def latest_progress(directory):
    """가장 최근의 완전한 기록을 반환한다. 외부 파일이 읽히지 않거나 불완전하면 건너뛴다."""
    folder = Path(directory) / "progress"
    try:
        paths = sorted(folder.glob("[0-9]????????.json"), reverse=True)
    except OSError:
        return None
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return path.name, value
        except (OSError, ValueError):
            continue
    return None


def write_terminal(directory, record):
    """전체 백업을 먼저 저장하며 run.json은 최종 기준 파일이다. 작업 중에는 읽기 프로세스가 두 파일을 열지 않는다. 외부 작업이 run.json을 재시도
    한도보다 오래 잡으면 run.final.json을 보존하고 오류를 생성 성공으로 바꾸지 않는다.
    """
    if record.get("status") not in TERMINAL_STATES:
        raise ValueError("A terminal state is required")
    directory = Path(directory)
    backup = directory / "run.final.json"
    if backup.exists():
        raise FileExistsError("Final record is immutable: " + str(backup))
    write_json(backup, record)
    write_json(directory / "run.json", record)


def finalize_stopped_run(directory, request_sha256, *, cancelled, error, returncode):
    """프로세스 종료를 기다린 뒤 부모가 복구한다. 작업자 저장과 경쟁하지 않는다. 강제 종료는 자식의 finally를 실행하지 못하므로 최근의 완전한 관찰 기록을
    보존하고 미완성 지표를 구분한다. raw.png만으로 생성 성공을 판단하지 않는다.
    """
    directory = Path(directory)
    canonical = directory / "run.json"
    backup = directory / "run.final.json"
    if backup.exists():
        final = json.loads(backup.read_text(encoding="utf-8"))
        if final.get("request_sha256") != request_sha256 or final.get("status") not in TERMINAL_STATES:
            raise ValueError("Final backup belongs to another/invalid run")
        try:
            current = json.loads(canonical.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = None
        if current != final:
            write_json(canonical, final)
        return final
    observed = latest_progress(directory)
    if observed:
        final = observed[1]
    else:
        try:
            final = json.loads(canonical.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            final = {}
    if final.get("request_sha256", request_sha256) != request_sha256:
        raise ValueError("Recovery snapshot belongs to another run")
    if final.get("status") not in TERMINAL_STATES:
        final.update(status="cancelled" if cancelled else "failed", error=error,
                     recovered_by_parent=True, metrics_incomplete=True, worker_returncode=returncode)
    final.update(request_sha256=request_sha256)
    raw = directory / "raw.png"
    if raw.is_file():
        from genai_lab.qwen_preservation import file_sha
        final["raw_sha256"] = file_sha(raw)
        final["raw_preserved"] = True
    write_terminal(directory, final)
    return final
