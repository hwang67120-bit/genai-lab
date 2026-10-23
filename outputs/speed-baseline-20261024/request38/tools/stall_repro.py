"""요청 38 멈춤 원인 확인(CPU만, GPU 없음). 세션 작업 프로세스를 잘못된 요청 한 건으로 띄워, 표준 입력 감시 스레드가 있을 때(A)와 없을 때(B)
첫 작업이 끝나는지(완료 표시 job-000001.done.json) 본다. 잘못된 요청이라 torch를 불러오기 전에 실패해야 정상이다.
사용: python stall_repro.py <작업 프로세스 python> <저장소 루트> <출력 폴더(새 폴더)>"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

python, repo, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
results = {}
for name, command in (
        ("A_watch_on", [python, "-m", "genai_lab.qwen_pose_worker", "--session", "{control}"]),
        ("B_watch_off", [python, "-c", "import sys; from genai_lab.qwen_pose_worker import serve; sys.exit(serve(sys.argv[1], watch=False))", "{control}"])):
    control = out / name
    request_dir = control / "job1"
    request_dir.mkdir(parents=True)
    (request_dir / "request.json").write_text(json.dumps({"task_kind": "free_generation", "schema_version": 1}), encoding="utf-8")
    (control / "job-000001.json").write_text(json.dumps({"request": str(request_dir / "request.json")}), encoding="utf-8")
    cmd = [c.replace("{control}", str(control)) for c in command]
    start = time.time()
    with (control / "worker.log").open("w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=str(repo), env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        done = control / "job-000001.done.json"
        while time.time() - start < 90 and not done.exists() and p.poll() is None:
            time.sleep(0.5)
        finished = done.exists()
        (control / "close").touch()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        p.stdin.close()
    run = request_dir / "run.json"
    results[name] = dict(job_done_marker=finished, seconds=round(time.time() - start, 1), returncode=p.returncode,
                         run_status=json.loads(run.read_text(encoding="utf-8")).get("status") if run.exists() else None,
                         log_tail=(control / "worker.log").read_text(encoding="utf-8", errors="replace")[-400:])
(out / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: {x: v[x] for x in ("job_done_marker", "seconds", "returncode", "run_status")} for k, v in results.items()}, ensure_ascii=False))
