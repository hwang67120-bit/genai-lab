"""Single-writer JSON publication and immutable Qwen progress (no model imports).

Windows readers without FILE_SHARE_DELETE can block rename/replace. Never
poll the mutable final record while its writer is alive. Same-directory temp
publication keeps readers from observing partial JSON. Retry waits are bounded;
permanent filesystem failures stay errors and preserve the complete temp file.
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
    # Close this handle before rename (also essential on Windows).
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
    # Other I/O errors propagate immediately; pending data is intentionally retained.


def publish_progress(directory, record):
    """One worker only. Each published name is new, then immutable."""
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
    """Return latest complete snapshot; tolerate unreadable/partial external files."""
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
    """Complete backup first; run.json remains the canonical final file.

    No polling reader opens either while the worker runs. If run.json is held
    by an external reader beyond the retry budget, run.final.json survives and
    the error is NOT silently converted to successful generation.
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
    """Parent recovery ONLY after process wait: never race the worker's writes.

    Forced termination cannot execute the child's finally. Preserve the latest
    full observed snapshot and identify incomplete metrics explicitly. Do not
    infer generation success from raw.png alone.
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
