"""두 배포 후보 목록(json.gz) 비교: 최상위 폴더별 같음/다름/한쪽만."""
import gzip, json, sys
from collections import Counter

def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)

old, new = load(sys.argv[1]), load(sys.argv[2])
of, nf = old["files"], new["files"]
stats = Counter()
examples = {}
for k in sorted(set(of) | set(nf)):
    top = k.split("/")[0]
    if k.startswith("runtime-") and "/Lib/site-packages/" in k:
        top += "/site-packages"
    if k in of and k in nf:
        kind = "same" if of[k] == nf[k] else "changed"
    elif k in of:
        kind = "old_only"
    else:
        kind = "new_only"
    stats[(top, kind)] += 1
    if kind != "same":
        examples.setdefault((top, kind), []).append(k)
out = dict(old=old["root_name"], new=new["root_name"], old_fingerprint=old["fingerprint"], new_fingerprint=new["fingerprint"],
           counts={f"{t} {k}": n for (t, k), n in sorted(stats.items())},
           examples={f"{t} {k}": v[:40] for (t, k), v in sorted(examples.items())})
print(json.dumps(out["counts"], ensure_ascii=False, indent=1))
for key, v in out["examples"].items():
    print(key, len(examples[tuple(key.split(" "))]), v[:12])
if len(sys.argv) > 3:
    open(sys.argv[3], "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
