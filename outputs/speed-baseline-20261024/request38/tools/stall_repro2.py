"""요청 38 멈춤 구간 CPU 재현: 멈춘 작업의 실제 request.json 사본을 세션 작업 프로세스(감시 스레드 켬)에 넣는다. CUDA를 숨겨 GPU를 쓰지 않는다.
멈추지 않으면 요청 검사·입력 그림 확인·모델 SHA 검사를 지나 'Qwen CUDA 장치가 없습니다'에서 실패해야 한다.
사용: python stall_repro2.py <작업 프로세스 python> <저장소 루트> <멈춘 작업 request.json> <출력 폴더(새 폴더)> [<기다릴 초>]"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

python, repo, source, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
limit = float(sys.argv[5]) if len(sys.argv) > 5 else 600
control = out / "control"
job = control / "job1"
job.mkdir(parents=True)
shutil.copy2(source, job / "request.json")
(control / "job-000001.json").write_text(json.dumps({"request": str(job / "request.json")}), encoding="utf-8")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
start = time.time()
samples = []
with (control / "worker.log").open("w", encoding="utf-8") as log:
    p = subprocess.Popen([python, "-m", "genai_lab.qwen_pose_worker", "--session", str(control)], cwd=str(repo), env=env,
                         stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
    import psutil
    done = control / "job-000001.done.json"
    while time.time() - start < limit and not done.exists() and p.poll() is None:
        try:
            procs = [psutil.Process(p.pid)] + psutil.Process(p.pid).children(recursive=True)
            samples.append(dict(t=round(time.time() - start, 1), cpu=round(sum(x.cpu_times().user + x.cpu_times().system for x in procs), 2),
                                rss_mib=round(max(x.memory_info().rss for x in procs) / 2**20, 1)))
        except psutil.Error:
            pass
        time.sleep(5)
    finished = done.exists()
    (control / "close").touch()
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    p.stdin.close()
run = job / "run.json"
record = json.loads(run.read_text(encoding="utf-8")) if run.exists() else {}
result = dict(job_done_marker=finished, seconds=round(time.time() - start, 1), returncode=p.returncode, run_status=record.get("status"),
              error=(record.get("error") or "")[:300], models_checked=bool(record.get("models")), samples=samples[-6:],
              log_tail=(control / "worker.log").read_text(encoding="utf-8", errors="replace")[-300:])
(out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("job_done_marker", "seconds", "returncode", "run_status", "error", "models_checked")}, ensure_ascii=False))
print("last samples", result["samples"])
