"""실행 환경 기록 요약(D-169): modules-*.json(실제 실행 프로세스) + popen 기록 → 프로세스별 Python·모듈 출처 환경·하위 프로세스 명령.

사용: python env_report.py <관찰 폴더> <출력 json>
"""
import json
import sys
from collections import Counter
from pathlib import Path

from env_probe import env_of


def main(argv):
    folder, out = Path(argv[0]), Path(argv[1])
    processes = []
    for f in sorted(folder.glob("modules-*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        envs = Counter(env_of(p) for p in d["modules"].values())
        foreign = sorted({p.split("site-packages")[0] for p in d["modules"].values()
                          if env_of(p) not in ("base_python", "repo") and env_of(d["executable"]) != env_of(p)})
        processes.append(dict(pid=d["pid"], argv=d["argv"][:3], executable=d["executable"], exe_env=env_of(d["executable"]),
                              base_prefix=d["base_prefix"], cwd=d["cwd"], pythonpath=d["pythonpath"],
                              module_count=len(d["modules"]), module_envs=dict(envs), modules_from_other_env=foreign,
                              path=d["path"]))
    popens = []
    log = folder / "opens.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("kind") == "popen":
                popens.append(r)
    out.write_text(json.dumps(dict(processes=processes, popens=popens), ensure_ascii=False, indent=1), encoding="utf-8")
    for p in processes:
        print(p["pid"], p["exe_env"], "|", " ".join(str(a) for a in p["argv"]), "|", p["module_envs"], "| 다른 환경:", p["modules_from_other_env"] or "없음")
    for r in popens:
        print("popen", r["pid"], "|", " ".join(r["args"][:4]) if isinstance(r["args"], list) else r["args"], "| cwd", r["cwd"], "| PYTHONPATH", r["env_pythonpath"])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main(sys.argv[1:])
