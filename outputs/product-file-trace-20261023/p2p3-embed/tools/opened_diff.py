"""두 trace 실행의 연 파일 집합 비교(후보 폴더·실행 폴더 이름 정규화)."""
import json, os, re, sys
from pathlib import Path

def norm(p):
    return os.path.normcase(os.path.normpath(str(p).replace("/", "\\")))

def opened(log, subs):
    out = set()
    for line in Path(log).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "path" in r and r.get("kind") not in ("net", "net-sample", "popen"):
            p = norm(r["path"])
            for a, b in subs:
                p = p.replace(norm(a), b)
            p = re.sub(r"[0-9a-f]{32}", "<run>", p)
            p = re.sub(r"\tmp[\w]{6,}", r"\<tmp>", p)
            out.add(p)
    return out

old = opened(sys.argv[1], [(sys.argv[2], "<CAND>"), (sys.argv[3], "<RUN>")])
new = opened(sys.argv[4], [(sys.argv[5], "<CAND>"), (sys.argv[6], "<RUN>")])
print("old", len(old), "new", len(new))
for p in sorted(old - new): print("  only_old", p)
for p in sorted(new - old): print("  only_new", p)
