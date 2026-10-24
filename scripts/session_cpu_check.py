"""세션 작업 프로세스 CPU 검증(D-183, GPU 없음 — CUDA 숨김). 실제 작업 프로세스를 띄워 확인하고, 경우마다 제한 시간을 둔다(넘으면 실패).

  C1 멈췄던 실제 요청(요청 38 ② 사본)이 같은 조건(세션·qwen-test-venv)에서 멈추지 않고 끝까지 가서 'CUDA 없음'으로 기록·종료되는가
  C2 한 세션에서 두 작업을 차례로 처리하는가(생성 대신 numpy·cv2를 불러오는 가짜 — 멈춤을 부른 가져오기를 그대로 거침)
  C3 부모(실행기)가 끝나면 자식이 끝나는가        C4 취소(QwenSession.kill)하면 끝나는가        C5 유휴 시간이 지나면 끝나는가
  C6 끝난 부모·생성 시각이 다른 pid를 살아 있다고 보지 않는가(pid 재사용 오인 방지)
  C7 시작 실패·잘못된 작업이 무응답 대신 기록(session-error.json·완료 표시)과 종료 코드로 남는가
사용: python scripts/session_cpu_check.py <작업 프로세스 python> <멈췄던 request.json> <출력 폴더(새 폴더)>
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
ENV.pop("GENAI_QWEN_SESSION_REUSE", None)
FAKE = r'''
import sys, genai_lab.qwen_pose_worker as w
def fake_generate(request, directory, record, cache=None):
    import genai_lab.qwen_free_generation   # numpy·cv2 네이티브 모듈 — 요청 38에서 멈춘 가져오기
    import numpy
    record["fake_generation"] = dict(numpy=numpy.__version__, cache_job=getattr(cache, "job", None))
    (directory / "raw.png").write_bytes(b"fake")
w.generate = fake_generate
sys.exit(w.main(sys.argv[1:]))
'''


def me():
    return os.getpid(), psutil.Process(os.getpid()).create_time()


def job(control, n, request_text):
    folder = control / f"job{n}"
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(request_text, encoding="utf-8")
    (control / f"job-{n:06d}.json").write_text(json.dumps({"request": str(folder / "request.json")}), encoding="utf-8")
    return folder


def start(python, control, parent, extra=(), fake=False, log_name="worker.log"):
    args = ["--session", str(control)] + (["--parent-pid", str(parent[0]), "--parent-created", repr(parent[1])] if parent else []) + list(extra)
    cmd = [python, "-c", FAKE] + args if fake else [python, "-m", "genai_lab.qwen_pose_worker"] + args
    log = (control.parent / f"{control.name}-{log_name}").open("w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(ROOT), env=ENV, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT), log


def wait(proc, limit):
    try:
        return proc.wait(timeout=limit), False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return proc.returncode, True


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def descendants_alive(pid):
    try:
        return [p.pid for p in psutil.Process(pid).children(recursive=True) if p.is_running()]
    except psutil.Error:
        return []


def main(argv):
    python, stalled, out = argv[0], Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True)
    results = {}

    def record(name, ok, **details):
        results[name] = dict(result="통과" if ok else "실패", **details)
        print(name, results[name]["result"], json.dumps(details, ensure_ascii=False, default=str)[:300], flush=True)

    # C1 멈췄던 실제 요청
    control = out / "C1"; control.mkdir()
    folder = control / "job1"; folder.mkdir()
    shutil.copy2(stalled, folder / "request.json")
    (control / "job-000001.json").write_text(json.dumps({"request": str(folder / "request.json")}), encoding="utf-8")
    t = time.time()
    proc, log = start(python, control, me())
    code, timeout = wait(proc, 900)
    log.close()
    run = read(folder / "run.json") or {}
    done = read(control / "job-000001.done.json")
    record("C1_stalled_request", not timeout and done is not None and run.get("status") == "failed" and "CUDA" in (run.get("error") or "")
           and code == 1, seconds=round(time.time() - t, 1), exit_code=code, timeout=timeout, run_status=run.get("status"),
           error=(run.get("error") or "")[:160], models_checked=bool(run.get("models")), done_marker=done, limit_seconds=900)

    # C2 한 세션에서 두 작업
    control = out / "C2"; control.mkdir()
    job(control, 1, json.dumps({"n": 1}))
    job(control, 2, json.dumps({"n": 2}))
    t = time.time()
    proc, log = start(python, control, me(), fake=True)
    deadline = time.time() + 180
    while time.time() < deadline and not (control / "job-000002.done.json").exists() and proc.poll() is None:
        time.sleep(0.5)
    (control / "close").touch()
    code, timeout = wait(proc, 30)
    log.close()
    runs = [read(control / f"job{n}" / "run.json") or {} for n in (1, 2)]
    dones = [read(control / f"job-{n:06d}.done.json") for n in (1, 2)]
    ended = read(control / "session-ended.json") or {}
    record("C2_two_jobs_in_order", all(d and d["status"] == "completed" for d in dones) and [r.get("session", {}).get("job") for r in runs] == [1, 2]
           and runs[0].get("session", {}).get("pid") == runs[1].get("session", {}).get("pid") and code == 0 and ended.get("reason") == "close"
           and not timeout, seconds=round(time.time() - t, 1), exit_code=code, timeout=timeout, dones=dones,
           jobs=[r.get("session", {}).get("job") for r in runs], pids=[r.get("session", {}).get("pid") for r in runs],
           numpy=[(r.get("fake_generation") or {}).get("numpy") for r in runs], ended=ended.get("reason"), limit_seconds=180)

    # C3 부모가 끝나면 자식도
    control = out / "C3"; control.mkdir()
    pidfile = out / "C3-child.pid"
    helper = (f"import os, subprocess, sys, time, psutil; me=(os.getpid(), psutil.Process(os.getpid()).create_time()); "
              f"p=subprocess.Popen([{python!r}, '-m', 'genai_lab.qwen_pose_worker', '--session', {str(control)!r}, '--parent-pid', str(me[0]), "
              f"'--parent-created', repr(me[1])], cwd={str(ROOT)!r}, stdin=subprocess.DEVNULL); "
              f"open({str(pidfile)!r}, 'w').write(str(p.pid)); time.sleep(600)")
    parent = subprocess.Popen([sys.executable, "-c", helper], env=ENV)
    deadline = time.time() + 60
    while time.time() < deadline and not pidfile.exists():
        time.sleep(0.2)
    time.sleep(8)                                       # 자식이 시작·감시를 걸 시간
    child_pid = int(pidfile.read_text()) if pidfile.exists() else None
    tree = [child_pid] + descendants_alive(child_pid) if child_pid else []
    parent.kill(); parent.wait()
    t = time.time()
    gone = False
    while time.time() - t < 30:
        if not any(psutil.pid_exists(p) and psutil.Process(p).is_running() for p in tree if p):
            gone = True
            break
        time.sleep(0.5)
    ended = read(control / "session-ended.json") or {}
    record("C3_parent_exit", gone and ended.get("reason") == "parent_gone", seconds_after_parent_exit=round(time.time() - t, 1),
           worker_tree=tree, ended=ended.get("reason"), limit_seconds=30)

    # C4 취소(QwenSession.kill) — 실제 실행기 클래스로, 멈췄던 요청을 처리하는 중에
    from genai_lab.qwen_session import QwenSession
    control = out / "C4" / "qwen-session"
    (out / "C4").mkdir()
    folder = out / "C4" / "qwen"; folder.mkdir()
    shutil.copy2(stalled, folder / "request.json")
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    old_env = dict(os.environ)
    os.environ.update(CUDA_VISIBLE_DEVICES="-1")
    session = QwenSession(type("S", (), {"python_executable": python})(), control)
    session.submit(folder / "request.json", request)
    os.environ.clear(); os.environ.update(old_env)
    time.sleep(10)
    cmdline = []
    try:
        procs = [psutil.Process(session.process.pid)] + psutil.Process(session.process.pid).children(recursive=True)
        cmdline = procs[-1].cmdline()
        tree = [p.pid for p in procs]
    except psutil.Error:
        tree = [session.process.pid]
    t = time.time()
    session.kill()
    gone = False
    while time.time() - t < 20:
        if not any(psutil.pid_exists(p) and psutil.Process(p).is_running() for p in tree):
            gone = True
            break
        time.sleep(0.5)
    record("C4_cancel", gone and "--parent-pid" in cmdline and session.process.stdin is None, seconds_after_kill=round(time.time() - t, 1),
           worker_tree=tree, parent_args_in_command="--parent-pid" in cmdline, stdin_pipe=session.process.stdin is not None, limit_seconds=20)

    # C5 유휴 시간 초과
    control = out / "C5"; control.mkdir()
    t = time.time()
    proc, log = start(python, control, me(), extra=["--idle-seconds", "3"])
    code, timeout = wait(proc, 30)
    log.close()
    ended = read(control / "session-ended.json") or {}
    record("C5_idle_timeout", code == 0 and not timeout and ended.get("reason") == "idle", seconds=round(time.time() - t, 1), exit_code=code,
           timeout=timeout, ended=ended.get("reason"), limit_seconds=30)

    # C6 pid 재사용 오인 방지: 생성 시각이 다른 살아 있는 pid / 이미 끝난 프로세스
    control = out / "C6a"; control.mkdir()
    pid, created = me()
    proc, log = start(python, control, (pid, created - 100))
    code_a, timeout_a = wait(proc, 30)
    log.close()
    error_a = read(control / "session-error.json") or {}
    short = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    dead = (short.pid, psutil.Process(short.pid).create_time())
    short.wait()
    control = out / "C6b"; control.mkdir()
    proc, log = start(python, control, dead)
    code_b, timeout_b = wait(proc, 30)
    log.close()
    error_b = read(control / "session-error.json") or {}
    record("C6_pid_reuse_not_alive", code_a == 2 and code_b == 2 and not timeout_a and not timeout_b and error_a.get("stage") == "start"
           and error_b.get("stage") == "start", wrong_created=dict(exit_code=code_a, error=error_a.get("error", "")[:120]),
           exited_process=dict(pid=dead[0], exit_code=code_b, error=error_b.get("error", "")[:120]), limit_seconds=30)

    # C7 시작 실패·잘못된 작업
    proc, log = start(python, out / "C7-missing-folder", me())          # 없는 세션 폴더
    code_a, timeout_a = wait(proc, 30)
    log.close()
    control = out / "C7b"; control.mkdir()
    (control / "job-000001.json").write_text("{깨진", encoding="utf-8")
    proc, log = start(python, control, me())
    code_b, timeout_b = wait(proc, 60)
    log.close()
    control_c = out / "C7c"; control_c.mkdir()
    folder = job(control_c, 1, json.dumps({"n": 1}))
    (folder / "run.json").write_text("{}", encoding="utf-8")            # 기존 결과 → 덮어쓰기 거부
    proc, log = start(python, control_c, me())
    code_c, timeout_c = wait(proc, 60)
    log.close()
    done_b, err_b = read(control / "job-000001.done.json"), read(control / "session-error.json") or {}
    done_c, err_c = read(control_c / "job-000001.done.json"), read(control_c / "session-error.json") or {}
    record("C7_failures_are_recorded", code_a == 2 and code_b == 1 and code_c == 1 and not (timeout_a or timeout_b or timeout_c)
           and (done_b or {}).get("status") == "failed" and err_b.get("error_type") == "JSONDecodeError"
           and (done_c or {}).get("status") == "failed" and "덮어쓰" in err_c.get("error", ""),
           missing_folder=dict(exit_code=code_a, log=(out / "C7-missing-folder-worker.log").read_text(encoding="utf-8", errors="replace")[-160:]),
           broken_job=dict(exit_code=code_b, done=done_b, error_type=err_b.get("error_type")),
           existing_output=dict(exit_code=code_c, done=done_c, error=err_c.get("error", "")[:100]), limit_seconds=60)

    summary = dict(python=python, stalled_request=str(stalled), cuda_visible_devices="-1",
                   passed=sum(r["result"] == "통과" for r in results.values()), total=len(results), cases=results)
    (out / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("통과", summary["passed"], "/", summary["total"])
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
