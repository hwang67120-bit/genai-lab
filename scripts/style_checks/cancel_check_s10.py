"""S10 취소 확인(D-224): FLUX 생성 도중 취소 → 작업 프로세스 트리가 끝나고 GPU 메모리가 돌아오는지. 이 PC, GPU.
사용: python cancel_check_s10.py <출력 폴더>
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
from genai_lab.style_redraw_pipeline import run_style_redraw

CHAR = Path(r"C:\Users\user\Downloads\수인캐릭터\51eed7ade89a8e6c31f1505081611335.jpg")


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


def workers():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*style_redraw_worker.py*' }).Count"],
                         capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def main():
    out = Path(sys.argv[1])
    directory = out / "run"
    flag = threading.Event()
    result = {}
    before = gpu_used_mib()
    thread = threading.Thread(target=lambda: result.update(o=run_style_redraw(CHAR, directory, scene="walking in a park at dusk", adult_confirmed=True,
                                                                              cancelled=flag.is_set)))
    t0 = time.time()
    thread.start()
    while not (directory / "progress.json").is_file() and thread.is_alive():   # FLUX 후보 생성이 시작될 때까지
        time.sleep(0.5)
    time.sleep(8)                                                               # 모델을 올리고 그리는 중
    during = gpu_used_mib()
    workers_during = workers()
    flag.set()
    t_cancel = time.time()
    thread.join(timeout=120)
    stopped_in = round(time.time() - t_cancel, 1)
    time.sleep(3)
    report = dict(state=result.get("o", {}).get("state"), stage=result.get("o", {}).get("stage"), stopped_in_seconds=stopped_in,
                  gpu_mib_before=before, gpu_mib_during=during, gpu_mib_after=gpu_used_mib(),
                  worker_processes_during=workers_during, worker_processes_after=workers(),
                  candidates_written=sorted(p.name for p in directory.glob("result-*.png")), total_seconds=round(time.time() - t0, 1))
    (out / "cancel-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
