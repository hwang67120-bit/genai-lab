"""배포 후보 GPU 검증 P4·P5 진행(D-175). 이 스크립트는 진행만 하고, 확인 대상은 모두 배포 후보 런타임 프로세스다.

- 시작 전·끝난 뒤 배포 후보를 고정 목록과 대조한다(`freeze_candidate.py verify`). 시작 전 불일치면 아무것도 돌리지 않는다.
- 환경 변수는 처음부터 만든다(`run_isolated_check.environment`와 같음): PATH = 배포 후보 runtime-main + Windows 폴더, 가짜 사용자 폴더.
  GPU 단계는 CUDA_VISIBLE_DEVICES를 지정하지 않는다(--dry는 -1로 숨김). 띄운 셸의 GPU 관련 변수는 이름만 기록하고 넘기지 않는다.
- p5는 먼저 사용자 첫 실행 그대로(준비 화면 → 모델 폴더 선택 → 깊은 검사 → 기록) 진행한 뒤 그 기록으로 작업실을 연다.
- p5(실제 실행)는 --after-p4 <P4 출력 폴더>가 가리키는 P4가 예상 밖 결과 없이 끝났을 때만 시작한다(예정된 거절 재현은 통과).
사용: python run_candidate_gpu.py <p4|p5> <배포 후보> <고정 목록 json.gz> <모델 폴더> <출력 폴더(새 폴더)> <저장소 루트> [<줄 목록 json>]
      [--after-p4 <P4 출력 폴더>] [--dry]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_isolated_check import environment, run_phase  # noqa: E402

GPU_PREFIXES = ("CUDA_", "PYTORCH_", "NVIDIA_", "CUBLAS_", "CUDNN_")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(candidate, manifest, repo, out, label):
    done = subprocess.run([sys.executable, "-I", "-X", "utf8", str(repo / "packaging/freeze_candidate.py"), "verify", str(candidate), str(manifest)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / f"verify-{label}.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    print(f"verify-{label}", done.returncode, done.stdout.strip().splitlines()[0] if done.stdout.strip() else "", flush=True)
    return done.returncode == 0


def p4_gate(folder, manifest):
    """P5 시작 조건: 같은 고정 목록으로 돈 실제(dry 아님) P4가 끝났고, 예상 밖 결과가 없고, 끝난 뒤 후보가 그대로일 것."""
    try:
        run = json.loads((folder / "run-report.json").read_text(encoding="utf-8"))
        inner = json.loads((folder / "segment" / "inner-p4.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"P4 기록을 읽을 수 없음: {error}"
    problems = []
    if run.get("stage") != "p4" or run.get("dry") or inner.get("dry"):
        problems.append("실제 P4 실행 기록이 아님")
    if Path(run.get("manifest", "")).resolve() != manifest.resolve():
        problems.append("다른 고정 목록으로 돈 P4")
    if run.get("returncode") != 0 or not run.get("unchanged_after") or run.get("error") or inner.get("error"):
        problems.append("P4 실행기 오류 또는 끝난 뒤 후보가 달라짐")
    if not inner.get("p5_allowed"):
        problems.append(f"P4 예상 밖 결과: {inner.get('unexpected')}")
    return "; ".join(problems)


def main(argv):
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    after_p4 = None
    if "--after-p4" in argv:                       # P5는 통과한 P4 기록이 있어야 시작한다(2026-10-23 사용자)
        i = argv.index("--after-p4")
        after_p4 = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    stage, candidate, manifest, home, out, repo = argv[0], *(Path(a) for a in argv[1:6])
    cases = Path(argv[6]) if len(argv) > 6 else None
    if stage == "p5" and not dry:
        problem = "P4 기록 폴더(--after-p4)가 필요함" if after_p4 is None else p4_gate(after_p4, manifest)
        if problem:
            raise SystemExit(f"P5를 시작하지 않습니다 — {problem}")
    if out.exists():
        raise SystemExit(f"출력 폴더가 이미 있습니다(새 폴더로): {out}")
    out.mkdir(parents=True)
    report = dict(stage=stage, dry=dry, candidate=str(candidate), manifest=str(manifest), model_home=str(home),
                  started=time.strftime("%Y-%m-%d %H:%M:%S"),
                  launching_shell_gpu_vars=sorted(k for k in os.environ if k.upper().startswith(GPU_PREFIXES)))  # 이름만
    if not verify(candidate, manifest, repo, out, "before"):
        report["error"] = "배포 후보가 고정 목록과 다름 — 실행하지 않음"
        (out / "run-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        raise SystemExit(report["error"])
    harness = out / "harness"
    harness.mkdir()
    inner = harness / "candidate_gpu_inner.py"
    shutil.copy2(repo / "packaging/candidate_gpu_inner.py", inner)      # 실행한 안쪽 스크립트를 결과와 함께 보존
    report["inner_sha256"] = sha(inner)
    profile = out / "sandbox-profile"
    env = environment(candidate, profile)
    if not dry:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    python = str(candidate / "runtime-main" / "python.exe")
    start = time.time()
    if stage == "p5":
        report["setup"] = run_phase(candidate, "first", out / "setup", home, env)     # 첫 실행: 준비 화면 → 깊은 검사 → 기록
        if (report["setup"].get("record") or {}).get("status") != "files_verified":
            report["error"] = "준비 단계(깊은 검사)가 통과하지 않음 — 생성하지 않음"
        else:
            command = [python, str(inner), "p5", str(out / "flow"), str(cases)] + (["--dry"] if dry else [])
    elif stage == "p4":
        command = [python, str(inner), "p4", str(out / "segment"), str(home), str(repo)] + (["--dry"] if dry else [])
    else:
        raise SystemExit("usage: p4|p5 ...")
    if "error" not in report:
        done = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8 * 3600)
        (out / f"console-{stage}.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr, encoding="utf-8")
        report["returncode"] = done.returncode
    report["seconds"] = round(time.time() - start, 1)
    report["unchanged_after"] = verify(candidate, manifest, repo, out, "after")
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "run-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("stage", "dry", "returncode", "seconds", "unchanged_after", "error")}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
