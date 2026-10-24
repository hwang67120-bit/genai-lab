"""요청 38(D-180 M4): 작업 프로세스 재사용(세션)을 켠 사용자 흐름 한 줄(B2, 재시도 있음)을 돌리고 요청 37과 그림 SHA·시간·메모리를 대조한다.

개발 환경(개발 venv + qwen-test-venv, 요청 34와 같은 경로)에서 실행한다. Python·패키지는 바꾸지 않는다. 장면 seed만 기록값으로 고정.
메모리는 셋을 함께 남긴다(사용자 10/24: 예약량 차감으로 여유를 확정하지 않음):
  device   NVML 장치 전체 사용량(1초) — 다른 프로그램 포함
  process  Windows 'GPU Process Memory' 카운터의 프로세스별 전용·공유 사용량(약 2초, 새로 뜬 프로세스 포함) — CUDA 문맥 포함
  ram      psutil 이 실행기 아래 모든 프로세스의 RSS·명령줄(1초)
판정은 하지 않는다. 그림 SHA가 다르면 '불일치 — 원인 조사'로만 적는다(품질 실패·성공으로 바꾸지 않음).
그림이 만들어지지 않았으면 '비교 불가'다 — 불일치와 구분한다(D-183).
사용: python scripts/session_reuse_check.py <줄 목록 json> <줄 이름> <출력 폴더(새 폴더)> <요청 37 흐름 폴더>
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["GENAI_QWEN_SESSION_REUSE"] = "1"            # 이 실행에서만 세션을 켠다(제품 기본값은 꺼짐)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

COMPARE = ("outfit-edit/qwen/raw.png", "free-generation/qwen/raw.png", "free-generation/qwen-2/raw.png")
COUNTER_LOOP = r"""
$ErrorActionPreference = 'SilentlyContinue'
while ($true) {
  $t = [DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0
  $c = Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage'
  $rows = @{}
  foreach ($s in $c.CounterSamples) {
    if ($s.InstanceName -match 'pid_(\d+)_') { $p = $Matches[1]; if (-not $rows[$p]) { $rows[$p] = @{ dedicated = 0; shared = 0 } }
      if ($s.Path -like '*dedicated usage') { $rows[$p].dedicated += $s.CookedValue } else { $rows[$p].shared += $s.CookedValue } }
  }
  @{ t = $t; pids = $rows } | ConvertTo-Json -Compress -Depth 4 | Add-Content -Encoding utf8 -Path $env:GENAI_COUNTER_LOG
}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Sampler:
    def __init__(self, out):
        self.out, self.stop = out, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.counter = None

    def start(self):
        env = dict(os.environ, GENAI_COUNTER_LOG=str(self.out / "gpu-process-counters.jsonl"))
        self.counter = subprocess.Popen(["powershell", "-NoProfile", "-Command", COUNTER_LOOP], env=env,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.thread.start()

    def run(self):
        import psutil
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        me = psutil.Process()
        with (self.out / "samples.jsonl").open("a", encoding="utf-8") as f:
            while not self.stop.is_set():
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                procs = {}
                for p in [me] + me.children(recursive=True):
                    try:
                        procs[p.pid] = dict(rss=p.memory_info().rss, cmd=" ".join(p.cmdline())[-200:])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                f.write(json.dumps(dict(t=time.time(), device_used=info.used, device_total=info.total, procs=procs)) + "\n")
                f.flush()
                self.stop.wait(1.0)
        pynvml.nvmlShutdown()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.counter is not None:
            self.counter.terminate()


def role(cmd):
    if "--session" in cmd:
        return "qwen_session"
    if "qwen_pose_worker" in cmd:
        return "qwen_single"
    if "outfit_edit_worker" in cmd:
        return "segmentation"
    if "studio_analysis" in cmd:
        return "analysis"
    if "session_reuse_check" in cmd:
        return "studio"
    return "other"


def summarize_memory(out):
    samples = [json.loads(l) for l in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    counters = []
    path = out / "gpu-process-counters.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            try:
                counters.append(json.loads(line))
            except ValueError:
                pass
    roles, ram = {}, {}
    for s in samples:
        for pid, p in s["procs"].items():
            r = role(p["cmd"])
            roles[str(pid)] = r
            ram[r] = max(ram.get(r, 0), p["rss"])
    gpu = {}
    both = []
    for c in counters:
        alive = {roles.get(pid, "other_or_outside") for pid in (c.get("pids") or {})}
        for pid, v in (c.get("pids") or {}).items():
            r = roles.get(pid)
            if r is None:
                continue
            g = gpu.setdefault(r, dict(dedicated=0, shared=0))
            g["dedicated"], g["shared"] = max(g["dedicated"], v["dedicated"]), max(g["shared"], v["shared"])
        if {"qwen_session", "segmentation"} <= alive:
            both.append(sum(v["dedicated"] for pid, v in c["pids"].items() if roles.get(pid) in ("qwen_session", "segmentation")))
    gib = 2 ** 30
    return dict(device_used_peak_gib=round(max(s["device_used"] for s in samples) / gib, 3) if samples else None,
                device_used_before_gib=round(samples[0]["device_used"] / gib, 3) if samples else None,
                device_total_gib=round(samples[0]["device_total"] / gib, 3) if samples else None,
                process_gpu_peak_gib={r: {k: round(v / gib, 3) for k, v in g.items()} for r, g in gpu.items()},
                session_and_segmentation_together=dict(samples=len(both), dedicated_sum_peak_gib=round(max(both) / gib, 3) if both else None),
                ram_peak_gib={r: round(v / gib, 3) for r, v in ram.items()},
                sample_counts=dict(device_and_ram=len(samples), process_counters=len(counters)),
                note="device = 장치 전체(다른 프로그램 포함). process = Windows 카운터(CUDA 문맥 포함). PyTorch reserved는 작업 프로세스 memory.json에 따로 있다")


def compare(run_dir, baseline_dir):
    rows = {}
    for rel in COMPARE:
        new, old = Path(run_dir) / rel, Path(baseline_dir) / rel
        rows[rel] = dict(exists=new.exists(), baseline=str(old),
                         same=(sha(new) == sha(old)) if new.exists() and old.exists() else None,     # None = 비교 불가(그림 없음)
                         new_sha256=sha(new) if new.exists() else None, baseline_sha256=sha(old) if old.exists() else None)
    return rows


def sha_summary(rows):
    """그림 대조 요약: 같음 / 불일치(그림이 다름 — 원인 조사) / 비교 불가(그림이 없음). 품질 판정이 아니다(D-183)."""
    mismatched = [k for k, v in rows.items() if v["same"] is False]
    missing = [k for k, v in rows.items() if v["same"] is None]
    parts = []
    if mismatched:
        parts.append(f"불일치 — 원인 조사 필요(품질 실패·성공으로 처리하지 않음): {mismatched}")
    if missing:
        parts.append(f"비교 불가 — 그림이 만들어지지 않음: {missing}")
    return "모두 같음" if rows and not parts else " / ".join(parts) if parts else "비교 불가 — 비교할 기록 없음"


def worker_records(run_dir):
    out = {}
    for rel in COMPARE:
        run = Path(run_dir) / Path(rel).parent / "run.json"
        if run.exists():
            d = json.loads(run.read_text(encoding="utf-8"))
            out[rel] = dict(status=d.get("status"), seed=(d.get("call_arguments") or {}).get("seed"), seconds=d.get("seconds"),
                            stage_seconds=d.get("stage_seconds"), session=d.get("session"),
                            files_rechecked=(d.get("models") or {}).get("files_rechecked", True),
                            text_encode_reused=d.get("text_encode_reused"), pipeline_reused=d.get("pipeline_reused"),
                            session_memory_at_start=d.get("session_memory_at_start"),
                            peak_reserved_gib=max([s.get("stage_peak_reserved_gib", 0) for s in d.get("memory_stages") or []] or [None]))
    control = Path(run_dir) / "free-generation" / "qwen-session"
    done = sorted(control.glob("job-*.done.json")) if control.exists() else []
    return out, [json.loads(p.read_text(encoding="utf-8")) for p in done]


def main(argv):
    cases_path, case_name, out, baseline_flow = Path(argv[0]), argv[1], Path(argv[2]), Path(argv[3])
    out.mkdir(parents=True)
    report = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.executable, case=case_name, session_env="1")
    sampler = Sampler(out)
    sampler.start()
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        import product_flow_run as flow
        from gui_main import GenAILabWindow
        from genai_lab.scene_composer import compose_scene
        case = next(c for c in json.loads(cases_path.read_text(encoding="utf-8")) if c["case"] == case_name)
        if sha(case["character"]) != case["character_sha256"] or (case.get("garment") and sha(case["garment"]) != case["garment_sha256"]):
            raise SystemExit("입력 SHA가 요청 34·37 기록과 다릅니다")
        if compose_scene(case["scene_seed"])["text"] != case["scene_text"]:
            raise SystemExit("장면 문장이 기록과 다릅니다")
        window = GenAILabWindow()
        if window.studio.product is None:
            raise SystemExit("사용자 화면이 아닙니다")
        flow.OUT = out / "flow"
        flow.OUT.mkdir()
        window.studio.output_root = flow.OUT / "runs"
        window.studio.product.choose_scene = lambda seed=case["scene_seed"]: compose_scene(seed)     # 장면 seed 고정(이 실행 안에서만)
        flow.run_case(window, case)
        record = flow.status["cases"][case_name]
        report["flow"] = {k: record.get(k) for k in ("state", "seconds", "run", "saved", "errors")}
        baseline_run = next(d for d in (baseline_flow / "runs").iterdir() if (d / "outfit-edit").exists() == bool(case.get("garment")))
        report["baseline_run"] = str(baseline_run)
        report["image_sha_comparison"] = compare(record["run"], baseline_run)
        report["saved_same_as_request37"] = (record.get("saved") is not None
                                             and sha(record["saved"]) == sha(baseline_flow / f"{case_name}-saved.png"))
        report["workers"], report["session_jobs"] = worker_records(record["run"])
        window.close()
    except BaseException as error:  # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        report["tb"] = traceback.format_exc()[-3000:]
    finally:
        sampler.finish()
        report["memory"] = summarize_memory(out)
        report["sha_result"] = sha_summary(report.get("image_sha_comparison") or {})
        report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(json.dumps({k: report.get(k) for k in ("sha_result", "saved_same_as_request37", "error")}, ensure_ascii=False))
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
