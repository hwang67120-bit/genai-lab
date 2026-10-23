"""요청 38 멈춤 위치 찾기(CPU만): stall_repro2와 같게 띄우되, 60초 뒤 모든 스레드의 호출 위치를 파일로 남긴다(faulthandler).
사용: python stall_repro3.py <작업 프로세스 python> <저장소 루트> <멈춘 작업 request.json> <출력 폴더(새 폴더)>"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

python, repo, source, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
control = out / "control"
job = control / "job1"
job.mkdir(parents=True)
shutil.copy2(source, job / "request.json")
(control / "job-000001.json").write_text(json.dumps({"request": str(job / "request.json")}), encoding="utf-8")
stacks = out / "stacks.txt"
code = ("import faulthandler, runpy, sys; f = open(sys.argv[1], 'w'); faulthandler.dump_traceback_later(60, repeat=False, file=f); "
        "sys.argv = ['qwen_pose_worker', '--session', sys.argv[2]]; runpy.run_module('genai_lab.qwen_pose_worker', run_name='__main__')")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
with (control / "worker.log").open("w", encoding="utf-8") as log:
    p = subprocess.Popen([python, "-c", code, str(stacks), str(control)], cwd=str(repo), env=env,
                         stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
    start = time.time()
    while time.time() - start < 80 and p.poll() is None and not (control / "job-000001.done.json").exists():
        time.sleep(1)
    (control / "close").touch()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    p.stdin.close()
print("done marker:", (control / "job-000001.done.json").exists(), "returncode:", p.returncode)
print(stacks.read_text(encoding="utf-8", errors="replace") if stacks.exists() else "(no stack dump)")
