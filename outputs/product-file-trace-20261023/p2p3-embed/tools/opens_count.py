import json, sys, collections
for log in sys.argv[1:]:
    c = collections.Counter(); kinds = collections.Counter()
    for line in open(log, encoding="utf-8"):
        r = json.loads(line); kinds[r.get("kind")] += 1
        if "path" in r: c[r["path"].split("\\")[-1] if "python310.zip" in r["path"] else "other"] += 1
    print(log[-60:], sum(kinds.values()), dict(kinds), "zip opens:", c.get("python310.zip", 0))
