"""기존 후보(run2)와 새 후보(embed)의 CPU 단계 결과 비교: 같은 상대 경로 파일의 SHA, 다르면 JSON 키 차이 요약."""
import hashlib, json, re, sys
from pathlib import Path

old_root, new_root = Path(sys.argv[1]), Path(sys.argv[2])
RUNID = re.compile(r"[0-9a-f]{32}")

def norm(rel):
    return RUNID.sub("<run>", rel)

def collect(root):
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            out.setdefault(norm(p.relative_to(root).as_posix()), []).append(p)
    return out

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def flat(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{prefix}{k}.")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], d

o, n = collect(old_root), collect(new_root)
same, diff, only_old, only_new = [], [], sorted(set(o) - set(n)), sorted(set(n) - set(o))
for k in sorted(set(o) & set(n)):
    if len(o[k]) != 1 or len(n[k]) != 1:
        diff.append((k, "multiple"))
        continue
    if sha(o[k][0]) == sha(n[k][0]):
        same.append(k)
    else:
        diff.append((k, o[k][0], n[k][0]))
print("same", len(same), "diff", len(diff), "only_old", len(only_old), "only_new", len(only_new))
for k in only_old[:15]: print("  only_old", k)
for k in only_new[:15]: print("  only_new", k)
for item in diff:
    k = item[0]
    if item[1] == "multiple":
        print("  multi", k); continue
    a, b = item[1], item[2]
    line = f"  diff {k}"
    if k.endswith(".json"):
        try:
            fa = dict(flat(json.loads(a.read_text(encoding="utf-8").replace("p2p3-20261023-run2", "p2p3-20261023-embed").replace("GenAILab-20261023\\\\", "GenAILab-20261023-embed\\\\"))))
            fb = dict(flat(json.loads(b.read_text(encoding="utf-8"))))
            keys = sorted(x for x in set(fa) | set(fb) if fa.get(x) != fb.get(x))
            short = sorted({RUNID.sub("<run>", re.sub(r"\.\d+(\.|$)", r".N\1", x)) for x in keys})
            line += f" | {len(keys)} keys: " + ", ".join(short[:12])
        except Exception as e:
            line += f" | json err {e}"
    print(line)
