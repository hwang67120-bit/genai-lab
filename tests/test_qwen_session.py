"""Qwen 작업 프로세스 세션(D-180): seed만 다른 재시도만 같은 프로세스로, 요청마다 새로 만드는 것은 넘기지 않고, 실패·취소 뒤 상태를 버린다."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from genai_lab import qwen_pose_edit, qwen_pose_worker as worker
from genai_lab.qwen_preservation import file_sha, json_sha
from genai_lab.qwen_record_io import write_json
from genai_lab.qwen_session import ENV, QwenSession, session_enabled


def request(seed=42, **spec):
    return {"schema_version": 1, "task_kind": "free_generation", "free_spec": dict(dict(accelerator="", style_lora=""), **spec),
            "prompt": {"positive": "p", "negative": " "}, "images": ["a.png", "b.png"], "settings": {"python_executable": "py"},
            "seed": seed, "memory": {"reserved_limit_gib": 6.5}, "output_size": [896, 1184]}


def test_only_the_seed_may_differ_inside_a_session():
    assert worker.reuse_key(request(42)) == worker.reuse_key(request(43))
    changed = request(43)
    changed["prompt"] = {"positive": "other", "negative": " "}
    assert worker.reuse_key(changed) != worker.reuse_key(request(42))
    assert worker.session_reusable(request())
    assert not worker.session_reusable(request(accelerator="lightning8"))
    assert not worker.session_reusable(request(style_lora="anime_borrowed"))
    assert not worker.session_reusable(dict(request(), task_kind="outfit_edit"))


class FakeVae:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)


def kept_cache(tmp_path):
    cache = worker.SessionCache()
    cache.job = 1
    mi = tmp_path / "model_inputs"
    mi.mkdir()
    Image.new("RGB", (8, 8), "white").save(mi / "vl_picture1_8x8.png")
    pipe = SimpleNamespace(vae=FakeVae())
    record = dict(models=dict(files_checked=27), processor_inputs=[dict(sequence_length=5)], embedding_shapes=dict(positive=[1, 2, 3]),
                  vae_tiling=True)
    cache.keep(request(42), pipe, "tr", ("pe", "pm", "ne", "nm"), record, mi)
    return cache, pipe


def test_cache_keeps_the_pipeline_and_embeddings_only_and_moves_the_vae_off_the_gpu(tmp_path):
    cache, pipe = kept_cache(tmp_path)
    assert cache.ready() and cache.first_job == 1 and pipe.vae.devices == ["cpu"]
    assert cache.embeddings == ("pe", "pm", "ne", "nm") and cache.vae_tiling
    assert set(cache.cond_sha) == {"vl_picture1_8x8.png"}
    models = cache.models_record()
    assert models["files_rechecked"] is False and models["reused_from_job"] == 1 and models["files_checked"] == 27
    assert not hasattr(cache, "seed") and not hasattr(cache, "generator") and not hasattr(cache, "images")
    cache.clear()
    assert not cache.ready() and cache.embeddings is None and cache.first_job is None


def test_check_session_reuses_only_a_matching_request(tmp_path):
    assert worker.check_session(None, request()) is False
    empty = worker.SessionCache()
    assert worker.check_session(empty, request()) is False          # 첫 작업: 새로 불러온다
    cache, _ = kept_cache(tmp_path)
    assert worker.check_session(cache, request(43)) is True
    changed = request(43)
    changed["images"] = ["a.png", "c.png"]
    with pytest.raises(RuntimeError, match="재사용 조건 불일치"):
        worker.check_session(cache, changed)
    with pytest.raises(RuntimeError, match="세션 대상이 아닌"):
        worker.check_session(cache, request(43, accelerator="lightning8"))


def test_serve_runs_jobs_in_order_with_one_cache_and_stops_after_a_failure(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    for n in (1, 2, 3):
        write_json(control / f"job-{n:06d}.json", dict(request=str(tmp_path / f"r{n}" / "request.json")))
    seen = []

    def run(path, cache, session):
        seen.append((Path(path).parent.name, id(cache), session["job"]))
        return (0, "completed") if session["job"] == 1 else (1, "failed")
    code = worker.serve(control, sleep=lambda s: None, run=run)
    assert code == 1
    assert [s[0] for s in seen] == ["r1", "r2"] and seen[0][1] == seen[1][1]       # 3번 작업은 실패 뒤라 받지 않는다
    assert json.loads((control / "job-000001.done.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((control / "job-000002.done.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert not (control / "job-000003.done.json").exists()


def test_serve_ends_on_close_or_when_idle(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    (control / "close").touch()
    assert worker.serve(control, sleep=lambda s: None, run=lambda *a, **k: pytest.fail("no job")) == 0
    assert json.loads((control / "session-ended.json").read_text(encoding="utf-8"))["reason"] == "close"
    (control / "close").unlink()
    assert worker.serve(control, idle_seconds=0, sleep=lambda s: None, run=lambda *a, **k: pytest.fail("no job")) == 0
    assert json.loads((control / "session-ended.json").read_text(encoding="utf-8"))["reason"] == "idle"


def test_parent_identity_needs_the_same_pid_and_creation_time():
    import os
    import psutil
    me, created = os.getpid(), psutil.Process(os.getpid()).create_time()
    assert worker.parent_alive(me, created)
    assert not worker.parent_alive(me, created - 100)          # 같은 pid라도 다른 프로세스(pid 재사용)로 본다
    assert not worker.parent_alive(2 ** 31 - 2, created)       # 없는 pid
    assert worker.process_identity(2 ** 31 - 2) is None


def test_serve_refuses_to_start_without_its_parent_and_records_why(tmp_path):
    import os
    control = tmp_path / "control"
    control.mkdir()
    watched = []
    code = worker.serve(control, parent=(os.getpid(), 1.0), sleep=lambda s: None, run=lambda *a, **k: pytest.fail("no job"),
                        watch=lambda *a: watched.append(a))
    error = json.loads((control / "session-error.json").read_text(encoding="utf-8"))
    assert code == 2 and error["stage"] == "start" and "실행기 프로세스를 확인할 수 없습니다" in error["error"] and watched == []
    assert worker.serve(tmp_path / "missing", sleep=lambda s: None, run=lambda *a, **k: pytest.fail("no job")) == 2


def test_serve_watches_a_live_parent_before_taking_jobs(tmp_path):
    import os
    import psutil
    control = tmp_path / "control"
    control.mkdir()
    (control / "close").touch()
    watched = []
    parent = (os.getpid(), psutil.Process(os.getpid()).create_time())
    assert worker.serve(control, parent=parent, sleep=lambda s: None, run=lambda *a, **k: None, watch=lambda *a: watched.append(a)) == 0
    assert watched == [(parent[0], parent[1], control.resolve())]


def test_a_job_that_raises_is_recorded_as_failed_instead_of_hanging(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    write_json(control / "job-000001.json", dict(request=str(tmp_path / "r1" / "request.json")))

    def run(path, cache, session):
        raise SystemExit("기존 실행 결과를 덮어쓰거나 자동 재시도하지 않습니다.")
    assert worker.serve(control, sleep=lambda s: None, run=run) == 1
    assert json.loads((control / "job-000001.done.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert "덮어쓰거나" in json.loads((control / "session-error.json").read_text(encoding="utf-8"))["error"]
    (control / "job-000001.done.json").unlink()
    (control / "job-000001.json").write_text("{깨진 json", encoding="utf-8")
    assert worker.serve(control, sleep=lambda s: None, run=lambda *a, **k: pytest.fail("not reached")) == 1
    assert json.loads((control / "session-error.json").read_text(encoding="utf-8"))["error_type"] == "JSONDecodeError"


def test_watcher_records_and_exits_when_the_parent_is_gone(tmp_path):
    import os
    import threading
    exited = threading.Event()
    codes = []
    worker.watch_parent(os.getpid(), 1.0, tmp_path, interval=0.01, exit=lambda code: codes.append(code) or exited.set())
    assert exited.wait(5) and codes == [3]
    ended = json.loads((tmp_path / "session-ended.json").read_text(encoding="utf-8"))
    assert ended["reason"] == "parent_gone" and ended["parent_pid"] == os.getpid()


def test_session_arguments_are_parsed_strictly():
    control, parent, idle = worker.parse_session_args(["--session", "c", "--parent-pid", "12", "--parent-created", "3.5"])
    assert (control, parent, idle) == ("c", (12, 3.5), worker.SESSION_IDLE_SECONDS)
    assert worker.parse_session_args(["--session", "c", "--idle-seconds", "3"])[2] == 3.0
    for bad in (["--session", "c", "--parent-pid", "12"], ["--session", "c", "--other", "1"], ["--session", "c", "x"]):
        with pytest.raises(SystemExit):
            worker.parse_session_args(bad)


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command, self.kwargs = command, kwargs
        self.pid, self.returncode, self.stdin = 7, None, SimpleNamespace(closed=False, close=lambda: setattr(self.stdin, "closed", True))
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("x", timeout)
        return self.returncode

    def terminate(self):
        self.terminated, self.returncode = True, -1

    def kill(self):
        self.returncode = -9


def settings():
    return SimpleNamespace(python_executable="py.exe")


def test_session_starts_one_process_and_accepts_only_seed_changes(tmp_path):
    started = []
    session = QwenSession(settings(), tmp_path / "qwen-session", popen=lambda c, **k: started.append(FakeProcess(c, **k)) or started[-1])
    assert session.submit(tmp_path / "q1" / "request.json", request(42)) == 1
    assert session.submit(tmp_path / "q2" / "request.json", request(43)) == 2
    import os
    command = started[0].command
    assert len(started) == 1 and command[command.index("--session") + 1] == str(tmp_path / "qwen-session")
    assert int(command[command.index("--parent-pid") + 1]) == os.getpid()
    assert worker.parent_alive(os.getpid(), float(command[command.index("--parent-created") + 1]))
    assert started[0].kwargs["stdin"] is subprocess.DEVNULL and started[0].kwargs["shell"] is False   # 표준 입력을 열지 않는다(D-183)
    job2 = json.loads((tmp_path / "qwen-session" / "job-000002.json").read_text(encoding="utf-8"))
    assert job2["request"].endswith("request.json") and job2["request_sha256"] == json_sha(request(43))
    changed = request(43)
    changed["prompt"] = {"positive": "other", "negative": " "}
    assert not session.accepts(changed) and not session.accepts(request(43, style_lora="x"))
    session.kill()
    assert started[0].terminated and started[0].stdin.closed and session.closed and not session.accepts(request(44))


def test_close_asks_the_worker_to_stop_first(tmp_path):
    process = FakeProcess(["py"])
    session = QwenSession(settings(), tmp_path / "s", popen=lambda c, **k: process)
    session.submit(tmp_path / "q" / "request.json", request())
    process.wait = lambda timeout=None: setattr(process, "returncode", 0) or 0
    session.close()
    assert (tmp_path / "s" / "close").exists() and not process.terminated and process.returncode == 0


def test_session_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not session_enabled()
    monkeypatch.setenv(ENV, "1")
    assert session_enabled()


class WorkerSession:
    """작업 프로세스 대신: submit 때 그 요청 폴더에 결과·최종 기록·완료 표시를 쓴다."""
    def __init__(self, control, job=1, status="completed", sha_override=None):
        self.control, self.job, self.status, self.sha_override = Path(control), job, status, sha_override
        self.control.mkdir()
        self.process = SimpleNamespace(pid=9, poll=lambda: None)
        self.command, self.killed = ["py", "--session", str(control)], False

    def submit(self, path, req):
        directory = Path(path).parent
        Image.new("RGB", (896, 1184), "white").save(directory / "raw.png")
        final = dict(status=self.status, request_sha256=self.sha_override or json_sha(req), raw_sha256=file_sha(directory / "raw.png"),
                     session=dict(job=self.job))
        write_json(directory / "run.final.json", final)
        write_json(directory / "run.json", final)
        write_json(self.control / f"job-{self.job:06d}.done.json", dict(status=self.status))
        return self.job

    def alive(self):
        return True

    def kill(self):
        self.killed = True


def prepared(tmp_path):
    directory = tmp_path / "qwen"
    directory.mkdir()
    req = request()
    write_json(directory / "request.json", req)
    record = dict(status="starting", request_sha256=json_sha(req))
    return directory, req, record, SimpleNamespace(image_sha256="b" * 64)


def test_session_job_result_is_checked_like_a_new_process(tmp_path):
    directory, req, record, spec = prepared(tmp_path)
    product = qwen_pose_edit._run_in_session(req, directory, record, WorkerSession(tmp_path / "s"), spec, (768, 1024),
                                             cancelled=lambda: False, on_progress=lambda _: None)
    assert product.name == "product.png" and record["status"] == "awaiting_review" and record["session"]["job"] == 1
    assert "세션 작업 프로세스 기록" in (directory / "worker.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("make, message", [
    (lambda c: WorkerSession(c, status="failed"), "불완전하거나 다른 요청"),
    (lambda c: WorkerSession(c, sha_override="x" * 64), "불완전하거나 다른 요청"),
])
def test_failed_or_foreign_session_results_are_refused(tmp_path, make, message):
    directory, req, record, spec = prepared(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        qwen_pose_edit._run_in_session(req, directory, record, make(tmp_path / "s"), spec, (768, 1024),
                                       cancelled=lambda: False, on_progress=lambda _: None)


def test_cancel_kills_the_session(tmp_path):
    directory, req, record, spec = prepared(tmp_path)
    session = WorkerSession(tmp_path / "s")
    session.submit = lambda path, r: 1          # 작업 프로세스가 아직 끝내지 않은 상태
    with pytest.raises(RuntimeError, match="취소"):
        qwen_pose_edit._run_in_session(req, directory, record, session, spec, (768, 1024),
                                       cancelled=lambda: True, on_progress=lambda _: None)
    assert session.killed and (directory / "cancel.request").exists()


def test_missing_results_are_not_comparable_rather_than_mismatched(tmp_path, monkeypatch):
    import importlib.util
    monkeypatch.setenv("GENAI_QWEN_SESSION_REUSE", "0")       # 실행기가 가져올 때 켜는 값을 시험 뒤 되돌린다
    spec = importlib.util.spec_from_file_location("session_reuse_check", Path(__file__).resolve().parents[1] / "scripts/session_reuse_check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    (tmp_path / "new" / "outfit-edit" / "qwen").mkdir(parents=True)
    (tmp_path / "old" / "outfit-edit" / "qwen").mkdir(parents=True)
    (tmp_path / "old" / "free-generation" / "qwen").mkdir(parents=True)
    (tmp_path / "new" / "outfit-edit" / "qwen" / "raw.png").write_bytes(b"same")
    (tmp_path / "old" / "outfit-edit" / "qwen" / "raw.png").write_bytes(b"same")
    (tmp_path / "old" / "free-generation" / "qwen" / "raw.png").write_bytes(b"x")
    rows = check.compare(tmp_path / "new", tmp_path / "old")
    assert rows["outfit-edit/qwen/raw.png"]["same"] is True and rows["free-generation/qwen/raw.png"]["same"] is None
    summary = check.sha_summary(rows)
    assert summary.startswith("비교 불가") and "불일치" not in summary
    rows["outfit-edit/qwen/raw.png"]["same"] = False
    assert "불일치" in check.sha_summary(rows) and "비교 불가" in check.sha_summary(rows)
    assert check.sha_summary({}) == "비교 불가 — 비교할 기록 없음"
