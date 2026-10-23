"""배포 후보 격리 검증 진행(D-174, P2·P3). 이 스크립트는 진행만 하고, 확인 대상은 모두 배포 후보 런타임 프로세스다.

- 환경 변수는 처음부터 만든다: PATH = 배포 후보 runtime-main + Windows 폴더(시스템 Python·개발 venv 없음), 사용자 폴더(APPDATA 등)는 가짜 폴더.
- variant release: 배포용 python310._pth 그대로(기록 도구 없음) — 기능 확인.
- variant trace: 기록 도구를 쓰려고 ._pth에 'import site'를 임시로 넣고 sitecustomize를 런타임에 복사. 가짜 사용자 폴더라 사용자 site-packages는 없다.
  끝나면 ._pth와 sitecustomize를 원래대로 되돌린다.
- 모델 폴더: 공백·한글 경로로 이름을 옮겨 쓰고, '옮긴 폴더' 단계 뒤 원래 이름으로 되돌린다(같은 드라이브 안 이름 바꾸기, 내용은 그대로).
사용: python run_isolated_check.py <배포 후보 폴더> <모델 폴더(원래)> <WD 없는 모델 폴더> <출력 폴더> <저장소 루트>
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SYSTEM_KEEP = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
               "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "OS", "SYSTEMDRIVE", "COMPUTERNAME", "USERNAME")


def environment(candidate, profile, trace_log=None, data_root=None):
    env = {k: v for k, v in os.environ.items() if k.upper() in SYSTEM_KEEP}
    windows = os.environ.get("SYSTEMROOT", r"C:\Windows")
    env.update(PATH=os.pathsep.join([str(candidate / "runtime-main"), windows + r"\System32", windows, windows + r"\System32\Wbem"]),
               USERPROFILE=str(profile / "home"), HOMEDRIVE=str(profile)[:2], HOMEPATH=str(profile / "home")[2:],
               APPDATA=str(profile / "Roaming"), LOCALAPPDATA=str(profile / "Local"), TEMP=str(profile / "Temp"), TMP=str(profile / "Temp"),
               GENAI_APP_HOME=str(candidate), QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="-1")
    if data_root:
        env["GENAI_TRACE_DATA_ROOT"] = str(data_root)      # 대신 쓸 기존 결과(시험 입력)의 위치
        env["GENAI_TRACE_DOWNLOADS"] = str(Path.home() / "Downloads")   # 시험 입력 그림(사용자 원래 폴더, 읽기만)
    if trace_log:
        env.update(GENAI_OPEN_TRACE=str(trace_log), GENAI_OPEN_TRACE_MODULES="1", GENAI_OPEN_TRACE_NET="1",
                   GENAI_CHILD_ENV_EXTRA="GENAI_OPEN_TRACE,GENAI_OPEN_TRACE_MODULES,GENAI_OPEN_TRACE_NET")
    for key in ("home", "Roaming", "Local", "Temp"):
        (profile / key).mkdir(parents=True, exist_ok=True)
    return env


def run_phase(candidate, phase, out, home, env, picture=None):
    command = [str(candidate / "runtime-main" / "python.exe"), str(candidate / "app/scripts/product_file_trace/package_flow_check.py"),
               phase, str(out), str(home)] + ([str(picture)] if picture else [])
    out.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    (out / f"console-{phase}.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr, encoding="utf-8")
    report = out / f"report-{phase}.json"
    result = json.loads(report.read_text(encoding="utf-8")) if report.exists() else dict(error="보고 없음", returncode=done.returncode)
    print(phase, done.returncode, result.get("decision"), result.get("error", "ok"), flush=True)
    return result


def fabricate_missing_record(candidate, profile, no_wd_home, env):
    """'검증 기록은 있는데 파일이 빠진' 상황을 만들려고 WD 없는 폴더를 가리키는 시험용 기록을 직접 쓴다(가짜 사용자 폴더 안, 표시 포함)."""
    code = ("import json,sys; from genai_lab.model_setup import manifest_fingerprint; "
            "print(json.dumps(manifest_fingerprint()))")
    fingerprint = json.loads(subprocess.run([str(candidate / "runtime-main/python.exe"), "-c", code], env=env, capture_output=True,
                                            text=True, encoding="utf-8", check=True).stdout)
    record = dict(schema="model_setup/1", model_home=str(no_wd_home), status="files_verified", checked_at="fabricated-for-test",
                  run_id="fabricated", seconds=0, fingerprint=fingerprint, problem_count=0, message="", gpu_generation_verified=False,
                  note="시험용으로 직접 쓴 기록(D-174 missing 단계) — 실제 검사 결과가 아님")
    target = profile / "Roaming" / "GenAILab" / "model-setup.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def set_trace(candidate, repo, enabled):
    for kind in ("main", "pose"):
        runtime = candidate / f"runtime-{kind}"
        pth = runtime / "python310._pth"
        lines = [l for l in pth.read_text(encoding="utf-8").splitlines() if l.strip() and l.strip() != "import site"]
        if enabled:
            lines.append("import site")
            shutil.copy2(repo / "scripts/product_file_trace/sitecustomize.py", runtime / "Lib/site-packages/sitecustomize.py")
        else:
            (runtime / "Lib/site-packages/sitecustomize.py").unlink(missing_ok=True)
        pth.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv):
    candidate, original_home, no_wd, out, repo = (Path(a) for a in argv[:5])
    korean_parent = original_home.parent / "모델 폴더 (공백 한글)"
    korean_home = korean_parent / "model home 20261023"
    moved_home = korean_parent / "model home 20261023 (옮김)"
    picture = Path.home() / "Downloads" / "참조 이미지" / "86a72742408d9b4d04f5fb23fb0b93bd.jpg"
    summary = dict(candidate=str(candidate), korean_home=str(korean_home))
    korean_parent.mkdir(exist_ok=True)
    original_home.rename(korean_home)
    try:
        for variant in ("release", "trace"):
            set_trace(candidate, repo, variant == "trace")
            base = out / variant
            profile = out.parent / f"sandbox-profile-{variant}"
            trace_log = (base / "opens.jsonl") if variant == "trace" else None
            env = environment(candidate, profile, trace_log, data_root=repo)
            results = summary.setdefault(variant, {})
            results["first"] = run_phase(candidate, "first", base, korean_home, env)
            results["second"] = run_phase(candidate, "second", base, korean_home, env)
            korean_home.rename(moved_home)
            try:
                results["moved"] = run_phase(candidate, "moved", base, korean_home, env)
            finally:
                moved_home.rename(korean_home)
            profile_missing = out.parent / f"sandbox-profile-{variant}-missing"
            env_missing = environment(candidate, profile_missing, trace_log, data_root=repo)
            fabricate_missing_record(candidate, profile_missing, no_wd, env_missing)
            results["missing"] = run_phase(candidate, "missing", base, no_wd, env_missing, picture=picture)
    finally:
        set_trace(candidate, repo, False)
        korean_home.rename(original_home)
        korean_parent.rmdir()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
