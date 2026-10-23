"""최소 배포 후보 폴더 만들기(D-174, P1) — 내려받기 없이 이 PC에 이미 설치된 것을 복사만 한다(원본은 읽기만).

  <출력>/runtime-main/  Python 3.10 본체(시스템 설치본에서 필요한 파일만) + python310._pth + requirements-main.txt 패키지
  <출력>/runtime-pose/  같은 Python 본체 + python310._pth + requirements-pose.txt 패키지
  <출력>/app/           앱 코드(genai_lab, scripts, gui_main.py), configs, docs/FREE_GENERATION_CONTRACT.md
  <출력>/GenAILab.cmd   실행기(GENAI_APP_HOME을 설치 폴더로 두고 runtime-main으로 작업실 실행)

패키지는 각 배포의 설치 기록(RECORD)에 적힌 파일만 원래 환경 site-packages에서 복사한다(목록 밖 패키지는 넣지 않음).
최종 배포에서는 Python 본체를 공식 내장형 배포판으로 바꾼다(내려받기 승인 뒤). 이 후보는 P1~P3 검증용이다.
사용: python build_candidate.py <출력 폴더> <저장소 루트> <주 환경 site-packages> <관절 환경 site-packages> [<시스템 Python 폴더>]
"""
import csv
import json
import re
import shutil
import sys
from pathlib import Path

# "import site"는 넣지 않는다(D-174 P1에서 확인): 넣으면 사용자 폴더의 Python 패키지(%APPDATA%\\Python\\Python310\\site-packages)가
# 경로에 섞인다(이 PC: 다른 프로젝트의 torch 2.5.1). 공식 내장형 배포판의 기본값과 같다.
PTH = """Lib
DLLs
Lib\\site-packages
..\\app
"""
CORE_FILES = ("python.exe", "pythonw.exe", "python3.dll", "python310.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt")
LIB_SKIP = {"site-packages", "test", "idlelib", "tkinter", "turtledemo", "__pycache__", "ensurepip", "venv", "lib2to3"}
APP_DIRS = ("genai_lab", "scripts", "configs")
APP_FILES = ("gui_main.py", "run.py", "docs/FREE_GENERATION_CONTRACT.md")   # run.py: gui_main이 불러옴(D-174 P2에서 빠진 것 발견)
LAUNCHER = """@echo off
chcp 65001 >nul
rem GenAILab 실행기(배포 후보). 설치 폴더 안의 런타임만 쓴다.
set "GENAI_APP_HOME=%~dp0"
if "%GENAI_APP_HOME:~-1%"=="\\" set "GENAI_APP_HOME=%GENAI_APP_HOME:~0,-1%"
"%~dp0runtime-main\\python.exe" "%~dp0app\\gui_main.py" %*
"""


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(path):
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split("==")[0])
    return names


def copy_core(python_home, target):
    target.mkdir(parents=True)
    for name in CORE_FILES:
        shutil.copy2(python_home / name, target / name)
    shutil.copytree(python_home / "DLLs", target / "DLLs", ignore=shutil.ignore_patterns("*_d.pyd", "*_d.dll", "*.pdb"))
    shutil.copytree(python_home / "Lib", target / "Lib", ignore=lambda d, names: [n for n in names if n in LIB_SKIP])
    (target / "Lib" / "site-packages").mkdir()
    (target / "python310._pth").write_text(PTH, encoding="utf-8")


def copy_packages(site, requirements, target_site):
    dist_infos = {normalize(p.name.split("-")[0]): p for p in site.glob("*.dist-info")}
    copied, missing, files = [], [], 0
    for name in requirement_names(requirements):
        info = dist_infos.get(normalize(name))
        if info is None:
            missing.append(name)
            continue
        with open(info / "RECORD", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0]:
                    continue
                rel = Path(row[0])
                if rel.parts[0] == ".." or rel.is_absolute():       # Scripts 등 site-packages 밖은 넣지 않는다
                    continue
                source = site / rel
                if source.is_file():
                    dest = target_site / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copy2(source, dest)
                        files += 1
        copied.append(name)
    return dict(copied=len(copied), missing=missing, files=files)


def copy_app(repo, target):
    app = target / "app"
    for d in APP_DIRS:
        shutil.copytree(repo / d, app / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in APP_FILES:
        (app / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / f, app / f)


def main(argv):
    out, repo, main_site, pose_site = (Path(a) for a in argv[:4])
    python_home = Path(argv[4]) if len(argv) > 4 else Path(r"C:\Program Files\Python310")
    if out.exists():
        raise SystemExit(f"이미 있습니다(덮어쓰지 않음): {out}")
    report = dict(python_home=str(python_home), version_file=(python_home / "python.exe").stat().st_size)
    for kind, site, req in (("main", main_site, repo / "packaging/requirements-main.txt"),
                            ("pose", pose_site, repo / "packaging/requirements-pose.txt")):
        runtime = out / f"runtime-{kind}"
        copy_core(python_home, runtime)
        report[kind] = copy_packages(site, req, runtime / "Lib" / "site-packages")
        print(kind, report[kind], flush=True)
    copy_app(repo, out)
    (out / "GenAILab.cmd").write_text(LAUNCHER.replace("\n", "\r\n"), encoding="utf-8")
    (out / "build-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", out)


if __name__ == "__main__":
    main(sys.argv[1:])
