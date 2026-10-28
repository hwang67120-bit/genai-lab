"""배포 후보 실행기(GenAILab.cmd) 확인(D-174): 격리 환경에서 실행기를 띄워 시작 판단 출력과 실제 프로세스 실행 파일을 확인한 뒤 끝낸다.

사용: python launcher_check.py <배포 후보> <가짜 사용자 폴더> <출력 json> [<기다릴 초>]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_isolated_check import environment  # noqa: E402


def main(argv):
    candidate, profile, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    seconds = float(argv[3]) if len(argv) > 3 else 40
    env = environment(candidate, profile)
    process = subprocess.Popen(["cmd.exe", "/c", str(candidate / "GenAILab.cmd")], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, cwd=str(profile))
    lines, decision, start = [], None, time.time()
    import threading
    def read():
        for raw in process.stdout:
            try:                      # 배포 런타임은 PYTHON* 환경 변수를 무시(._pth) → 파이프 출력은 Windows 기본 인코딩일 수 있다
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp949", errors="replace")
            lines.append(text.rstrip())
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    while time.time() - start < seconds and process.poll() is None:
        if any(l.startswith("시작 판단:") for l in lines):
            decision = next(l for l in lines if l.startswith("시작 판단:"))
            time.sleep(5)                                  # 창이 뜬 뒤 잠시
            break
        time.sleep(0.5)
    import psutil
    tree = []
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            try:
                tree.append(dict(pid=child.pid, exe=child.exe(), cmd=child.cmdline()[:3]))
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    alive = process.poll() is None
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    reader.join(timeout=5)
    result = dict(decision_line=decision, running_when_checked=alive, processes=tree, output=lines[-20:])
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(decision, "| 실행 중:", alive, "| 프로세스:", [p["exe"] for p in tree])


if __name__ == "__main__":
    main(sys.argv[1:])
