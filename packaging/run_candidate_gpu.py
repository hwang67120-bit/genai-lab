"""배포 후보 GPU 검증 P4·P5 진행(D-175). 이 스크립트는 진행만 하고, 확인 대상은 모두 배포 후보 런타임 프로세스다.

- 시작 전·끝난 뒤 배포 후보를 고정 목록과 대조한다(`freeze_candidate.py verify`). 시작 전 불일치면 아무것도 돌리지 않는다.
- 환경 변수는 처음부터 만든다(`run_isolated_check.environment`와 같음): PATH = 배포 후보 runtime-main + Windows 폴더, 가짜 사용자 폴더.
  GPU 단계는 CUDA_VISIBLE_DEVICES를 지정하지 않는다(--dry는 -1로 숨김). 띄운 셸의 GPU 관련 변수는 이름만 기록하고 넘기지 않는다.
- p5는 먼저 사용자 첫 실행 그대로(준비 화면 → 모델 폴더 선택 → 깊은 검사 → 기록) 진행한 뒤 그 기록으로 작업실을 연다.
- 기록은 둘로 나눈다(2026-10-24 사용자): `result`(검증 결과) / `process`(띄운 프로세스마다 종료 코드, 모두 0이어야 정상 종료).
  결과가 좋아도 프로세스가 0이 아니면 정상으로 처리하지 않는다.
- p5(실제 실행)는 --after-p4 <P4 출력 폴더>가 가리키는 P4가 모두 '통과'일 때만 시작한다. '검토 필요'가 있으면
  --p4-review <검토 기록 json>에 그 건마다 실질적 차이가 없다는 근거가 적혀 있어야 하고, '예상 밖 실패'가 있으면 시작하지 않는다.
사용: python run_candidate_gpu.py <p4|p5> <배포 후보> <고정 목록 json.gz> <모델 폴더> <출력 폴더(새 폴더)> <저장소 루트> [<줄 목록 json>]
      [--after-p4 <P4 출력 폴더>] [--p4-review <검토 기록 json>] [--dry]
검토 기록: {"reviewer": "...", "rows": {"<건 이름>": {"resolution": "no_substantive_difference", "evidence": "본 자료와 이유"}}}
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
RESOLVED = "no_substantive_difference"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def option(argv, name):
    if name not in argv:
        return argv, None
    i = argv.index(name)
    return argv[:i] + argv[i + 2:], Path(argv[i + 1])


def verify(candidate, manifest, repo, out, label):
    done = subprocess.run([sys.executable, "-I", "-X", "utf8", str(repo / "packaging/freeze_candidate.py"), "verify", str(candidate), str(manifest)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / f"verify-{label}.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    print(f"verify-{label}", done.returncode, done.stdout.strip().splitlines()[0] if done.stdout.strip() else "", flush=True)
    return done.returncode == 0


def review_problems(review_path, rows):
    """'검토 필요' 건마다 사람이 남긴 근거가 있는지. 없거나 해결로 적지 않은 건은 그대로 미해결."""
    if not rows:
        return []
    if review_path is None:
        return [f"검토 필요 미해결(검토 기록 없음): {rows}"]
    try:
        review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"검토 기록을 읽을 수 없음: {error}"]
    entries = review.get("rows") or {}
    open_rows = [name for name in rows
                 if (entries.get(name) or {}).get("resolution") != RESOLVED or not str((entries.get(name) or {}).get("evidence", "")).strip()]
    problems = [f"검토 필요 미해결: {open_rows}"] if open_rows else []
    if not str(review.get("reviewer", "")).strip():
        problems.append("검토 기록에 검토자가 없음")
    return problems


def summarize_inner(stage, data):
    """안쪽 결과 기록의 요약. P5 줄 상태는 실제 흐름 기록(cases.<줄>.flow.state)에 있고, 건너뛴 줄·dry만 cases.<줄>.state에 있다
    (요청 37에서 Codex가 찾은 요약 누락 — 판정에는 영향 없었음)."""
    if stage == "p4":
        return {k: data.get(k) for k in ("outcomes", "unexpected", "review_needed", "p5_allowed", "p5_allowed_after_review", "error")
                if k in data}
    cases = {name: ((row or {}).get("flow") or {}).get("state") or (row or {}).get("state") for name, row in (data.get("cases") or {}).items()}
    return dict(cases=cases, **({"error": data["error"]} if data.get("error") else {}))


def p4_gate(folder, manifest, review_path=None):
    """P5 시작 조건: 같은 고정 목록으로 돈 실제(dry 아님) P4가 정상 종료했고, 예상 밖 실패가 없고,
    검토 필요 건은 모두 근거와 함께 해결됐고, 끝난 뒤 후보가 그대로일 것. 문제가 없으면 빈 문자열."""
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
    process = run.get("process") or {}
    if not process.get("normal_exit"):
        problems.append(f"P4 프로세스 정상 종료 아님: {process.get('exit_codes')}")
    if not run.get("unchanged_after") or run.get("error") or inner.get("error"):
        problems.append("P4 실행기 오류 또는 끝난 뒤 후보가 달라짐")
    if inner.get("unexpected"):
        problems.append(f"P4 예상 밖 실패: {inner.get('unexpected')}")
    elif not inner.get("p5_allowed_after_review"):
        problems.append("P4 결과가 완전하지 않음")
    else:
        problems += review_problems(review_path, inner.get("review_needed") or [])
    return "; ".join(problems)


def main(argv):
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    argv, after_p4 = option(argv, "--after-p4")         # P5는 통과한 P4 기록이 있어야 시작한다(2026-10-23·24 사용자)
    argv, p4_review = option(argv, "--p4-review")
    stage, candidate, manifest, home, out, repo = argv[0], *(Path(a) for a in argv[1:6])
    cases = Path(argv[6]) if len(argv) > 6 else None
    if stage == "p5" and not dry:
        problem = "P4 기록 폴더(--after-p4)가 필요함" if after_p4 is None else p4_gate(after_p4, manifest, p4_review)
        if problem:
            raise SystemExit(f"P5를 시작하지 않습니다 — {problem}")
    if out.exists():
        raise SystemExit(f"출력 폴더가 이미 있습니다(새 폴더로): {out}")
    out.mkdir(parents=True)
    report = dict(stage=stage, dry=dry, candidate=str(candidate), manifest=str(manifest), model_home=str(home),
                  after_p4=str(after_p4) if after_p4 else None, p4_review=str(p4_review) if p4_review else None,
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
    exit_codes = {}
    start = time.time()
    if stage == "p5":
        report["setup"] = run_phase(candidate, "first", out / "setup", home, env)     # 첫 실행: 준비 화면 → 깊은 검사 → 기록
        exit_codes["setup"] = report["setup"].get("process_returncode")
        if (report["setup"].get("record") or {}).get("status") != "files_verified" or exit_codes["setup"] != 0:
            report["error"] = "준비 단계(깊은 검사)가 통과하지 않았거나 프로세스가 정상 종료하지 않음 — 생성하지 않음"
        else:
            command = [python, str(inner), "p5", str(out / "flow"), str(cases)] + (["--dry"] if dry else [])
    elif stage == "p4":
        command = [python, str(inner), "p4", str(out / "segment"), str(home), str(repo)] + (["--dry"] if dry else [])
    else:
        raise SystemExit("usage: p4|p5 ...")
    if "error" not in report:
        done = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8 * 3600)
        (out / f"console-{stage}.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr, encoding="utf-8")
        exit_codes[stage] = done.returncode
    report["seconds"] = round(time.time() - start, 1)
    report["unchanged_after"] = verify(candidate, manifest, repo, out, "after")
    report["process"] = dict(exit_codes=exit_codes, normal_exit=bool(exit_codes) and all(c == 0 for c in exit_codes.values()),
                             note="결과 판정과 별개. 0이 아니면 결과가 좋아도 정상 종료로 처리하지 않는다")
    inner_report = out / ("segment" if stage == "p4" else "flow") / f"inner-{stage}{'-dry' if dry else ''}.json"
    report["result"] = (summarize_inner(stage, json.loads(inner_report.read_text(encoding="utf-8"))) if inner_report.is_file()
                        else dict(error="안쪽 결과 기록 없음"))
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "run-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("stage", "dry", "process", "result", "seconds", "unchanged_after", "error")},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
