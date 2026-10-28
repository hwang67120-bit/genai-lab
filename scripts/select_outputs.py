"""main PR용 결과 기록 선별(D-180): 문서(추적된 .md, outputs 밖)가 가리키는 최상위 결과 폴더는 통째로 넣고, 아무 문서도 가리키지 않는 폴더는 넣지 않는다.
원자료는 지우지 않는다 — 넣지 않는 폴더는 개발 브랜치와 디스크에 그대로 남는다. 사용: python select_outputs.py <keep 목록 출력> <제외 목록 출력>"""
import collections
import os
import re
import subprocess
import sys

tracked = [t for t in subprocess.run(["git", "ls-files", "outputs"], capture_output=True, text=True).stdout.split("\n") if t]
docs = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n")
        if f.endswith(".md") and not f.startswith("outputs/")]
pattern = re.compile(r"outputs[/\\]([^/\\\s)\]\"'`>|,]+)")
referenced = collections.Counter()
for d in docs:
    with open(d, encoding="utf-8", errors="replace") as f:
        for m in pattern.findall(f.read()):
            referenced[m.rstrip(".:;")] += 1
folders = collections.defaultdict(list)
for t in tracked:
    parts = t.split("/")
    folders[parts[1] if len(parts) > 2 else "(root) " + parts[1]].append(t)


def size(files):
    return sum(os.path.getsize(f) for f in files if os.path.exists(f))


keep, drop = [], []
for name, files in sorted(folders.items()):
    (keep if name in referenced else drop).append((name, files))
print("keep folders", len(keep), "files", sum(len(f) for _, f in keep), "MB", round(sum(size(f) for _, f in keep) / 2**20, 1))
print("drop folders", len(drop), "files", sum(len(f) for _, f in drop), "MB", round(sum(size(f) for _, f in drop) / 2**20, 1))
for name, files in drop:
    print("  drop", name, len(files), round(size(files) / 2**20, 2), "MB")
for name, files in sorted(keep, key=lambda k: -size(k[1]))[:6]:
    print("  keep-big", name, len(files), round(size(files) / 2**20, 2), "MB")
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n".join(t for _, files in keep for t in files) + "\n")
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write("\n".join(t for _, files in drop for t in files) + "\n")
