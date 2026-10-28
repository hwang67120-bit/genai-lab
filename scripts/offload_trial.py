"""요청 39(D-185): 같은 코드·같은 런타임·같은 입력에서 블록 오프로드 설정만 바꿔 Qwen 작업 프로세스 한 번씩 돌리고 비교한다.

순서(기본): group_block1(기준) → group_block1_stream → group_block2 → group_block1(기준 다시 — 실행 사이 변동 확인).
- 요청은 기준 요청 사본에서 settings.offload만 바꾼다(실행기가 다른 차이가 없음을 확인). 단일 작업 경로 그대로(세션 아님).
- 경우마다 장치 전체(NVML)·프로세스별 GPU(Windows 카운터)·RAM(psutil)을 표본으로 남긴다(session_reuse_check와 같은 표본기).
- 실행 전 캐시 상태(가용 RAM)와 실행 사이 간격을 기록한다. 디스크 캐시를 비우거나 데우지 않는다.
- 멈춤·실패·시간 초과·메모리 한도 초과(memory_stopped 또는 PyTorch reserved > 6.5GiB)면 기록하고 **거기서 멈춘다**(설정을 완화하거나 다음 경우를 돌리지 않음).
- 기준(group_block1) raw가 기준 요청의 raw와 다르면 같은 조건이 아니므로 멈춘다. 측정 전용 설정의 SHA가 다르면 '불일치 — 원인 조사'로 적고 판정하지 않는다.
사용: python scripts/offload_trial.py <기준 request.json> <기준 raw.png> <작업 프로세스 python> <출력 폴더(새 폴더)> [--dry] [설정 이름 …]
"""
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import session_reuse_check as sampler_tools          # noqa: E402 — 같은 표본기·메모리 요약
os.environ.pop("GENAI_QWEN_SESSION_REUSE", None)     # 위 모듈이 켜는 세션 값은 이 측정과 무관 — 끈다(단일 작업 경로)

ORDER = ("group_block1", "group_block1_stream", "group_block2", "group_block1")
LIMIT_GIB = 6.5
RUN_TIMEOUT = 90 * 60


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diff_keys(a, b, prefix=""):
    keys = []
    for k in sorted(set(a) | set(b)):
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            keys += diff_keys(a[k], b[k], f"{prefix}{k}.")
        elif a.get(k) != b.get(k):
            keys.append(prefix + k)
    return keys


def worker_summary(directory):
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    steps = [s["seconds"] for s in run.get("steps") or []]
    stages = run.get("memory_stages") or []
    peak = max([s.get("stage_peak_reserved_gib", 0) for s in stages] or [0])
    step_peak = max([s.get("stage_peak_reserved_gib", 0) for s in stages if s.get("stage", "").startswith("step")] or [0])
    t_request, t_raw = (directory / "request.json").stat().st_mtime, (directory / "raw.png").stat().st_mtime if (directory / "raw.png").exists() else None
    return dict(status=run.get("status"), error=(run.get("error") or "")[:300], offload=run.get("offload"), seconds=run.get("seconds"),
                stage_seconds=run.get("stage_seconds"), steps=len(steps),
                step_mean=round(statistics.mean(steps[1:]), 3) if len(steps) > 1 else None,
                step_min=round(min(steps[1:]), 3) if len(steps) > 1 else None, step_max=round(max(steps[1:]), 3) if len(steps) > 1 else None,
                step0=round(steps[0], 3) if steps else None,
                before_worker_clock=round((t_raw - run["seconds"]) - t_request, 1) if t_raw and run.get("seconds") else None,
                peak_reserved_gib=peak, steps_peak_reserved_gib=step_peak,
                memory_stopped="memory_stopped" in json.dumps(run, ensure_ascii=False) or run.get("error_type") == "OutOfMemoryError",
                raw_sha256=sha(directory / "raw.png") if (directory / "raw.png").exists() else None)


def main(argv):
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    base_request, base_raw, python, out = Path(argv[0]), Path(argv[1]), argv[2], Path(argv[3])
    order = tuple(argv[4:]) or ORDER
    from genai_lab.child_env import child_environment
    from genai_lab.qwen_pose_settings import OFFLOAD_ARGS, QwenPoseSettings
    import psutil
    out.mkdir(parents=True)
    base = json.loads(base_request.read_text(encoding="utf-8"))
    base_sha = sha(base_raw)
    report = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), dry=dry, base_request=str(base_request), base_raw_sha256=base_sha,
                  worker_python=python, order=order, limit_gib=LIMIT_GIB, runs=[], stopped=None,
                  note="같은 코드·런타임·입력, settings.offload만 다름. 디스크 캐시는 조절하지 않음(실행 전 가용 RAM만 기록)")
    previous_end = None
    for index, name in enumerate(order, 1):
        if name not in OFFLOAD_ARGS:
            report["stopped"] = f"알 수 없는 설정 {name}"
            break
        request = copy.deepcopy(base)
        request["settings"]["offload"] = name
        QwenPoseSettings(**request["settings"])                       # 설정 검사(측정 전용 이름 포함)
        changed = diff_keys(base, request)
        if changed not in ([], ["settings.offload"]):
            report["stopped"] = f"요청 차이가 오프로드 밖에 있음: {changed}"
            break
        directory = out / f"{index:02d}-{name}"
        directory.mkdir()
        (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        memory = psutil.virtual_memory()
        entry = dict(index=index, offload=name, args=OFFLOAD_ARGS[name], request_changes=changed,
                     before=dict(available_ram_gib=round(memory.available / 2**30, 2), total_ram_gib=round(memory.total / 2**30, 2),
                                 seconds_since_previous_run=round(time.time() - previous_end, 1) if previous_end else None))
        report["runs"].append(entry)
        if dry:
            entry["state"] = "dry"
            continue
        sampler = sampler_tools.Sampler(directory)
        sampler.start()
        start = time.time()
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([python, "-m", "genai_lab.qwen_pose_worker", str(directory / "request.json")], cwd=str(ROOT),
                                    env=child_environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                code = proc.wait(timeout=RUN_TIMEOUT)
                timeout = False
            except subprocess.TimeoutExpired:
                proc.kill()
                code, timeout = proc.wait(), True
        sampler.finish()
        previous_end = time.time()
        entry.update(wall_seconds=round(previous_end - start, 1), exit_code=code, timeout=timeout,
                     worker=worker_summary(directory) if (directory / "run.json").exists() else None,
                     memory=sampler_tools.summarize_memory(directory))
        w = entry["worker"] or {}
        entry["sha_vs_base"] = (None if not w.get("raw_sha256") else w["raw_sha256"] == base_sha)
        problems = []
        if timeout:
            problems.append(f"시간 초과({RUN_TIMEOUT}초)")
        if code != 0 or w.get("status") != "completed":
            problems.append(f"실패(종료 코드 {code}, 상태 {w.get('status')}): {w.get('error', '')[:120]}")
        if w.get("memory_stopped") or (w.get("peak_reserved_gib") or 0) > LIMIT_GIB:
            problems.append(f"메모리 한도 초과(최고 reserved {w.get('peak_reserved_gib')}GiB)")
        if name == "group_block1" and entry["sha_vs_base"] is False:
            problems.append("기준 설정의 그림이 기준 요청의 그림과 다름 — 같은 조건이 아님")
        entry["state"] = "ok" if not problems else "stopped"
        entry["problems"] = problems
        if entry["sha_vs_base"] is False and name != "group_block1":
            entry["sha_note"] = "불일치 — 원인 조사(품질 실패·성공으로 판정하지 않음)"
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(index, name, entry["state"], json.dumps({k: w.get(k) for k in ("seconds", "step_mean", "peak_reserved_gib")}, ensure_ascii=False),
              "SHA 같음" if entry["sha_vs_base"] else "SHA 다름/없음", problems, flush=True)
        if problems:
            report["stopped"] = f"{index}-{name}: " + "; ".join(problems) + " — 설정을 완화하거나 다음 경우를 돌리지 않음"
            break
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("멈춤:" if report["stopped"] else "끝:", report["stopped"] or f"{len(report['runs'])}개 경우")
    return 0 if not report["stopped"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
