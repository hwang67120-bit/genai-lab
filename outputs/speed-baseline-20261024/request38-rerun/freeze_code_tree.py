"""요청 38 재실행의 실행 후보(개발 환경이 읽는 저장소 코드) 파일 목록·해시 고정과 대조. freeze_candidate.py와 같은 형식(json.gz).
  freeze <저장소 루트> <목록 json.gz>   genai_lab·scripts·configs·gui_main.py·run.py·docs/FREE_GENERATION_CONTRACT.md(__pycache__ 제외)
  verify <저장소 루트> <목록 json.gz>   다르면 종료 코드 1"""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

PARTS = ("genai_lab", "scripts", "configs", "gui_main.py", "run.py", "docs/FREE_GENERATION_CONTRACT.md")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(root):
    files = {}
    for part in PARTS:
        base = root / part
        items = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for p in items:
            if "__pycache__" not in p.parts:
                files[p.relative_to(root).as_posix()] = dict(size=p.stat().st_size, sha256=sha(p))
    return files


def fingerprint(files):
    return hashlib.sha256("\n".join(f"{k}\t{v['size']}\t{v['sha256']}" for k, v in sorted(files.items())).encode("utf-8")).hexdigest()


command, root, manifest = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
files = scan(root)
if command == "freeze":
    record = dict(schema="code_tree_manifest/1", root=str(root), parts=PARTS, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  file_count=len(files), fingerprint=fingerprint(files), files=files)
    with gzip.open(manifest, "wt", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    print("고정:", len(files), "파일, 지문", record["fingerprint"])
    sys.exit(0)
with gzip.open(manifest, "rt", encoding="utf-8") as f:
    expected = json.load(f)["files"]
missing, extra = sorted(set(expected) - set(files)), sorted(set(files) - set(expected))
changed = sorted(k for k in set(files) & set(expected) if files[k] != expected[k])
ok = not (missing or extra or changed)
print("대조:", "일치" if ok else "불일치", "| 지문", fingerprint(files), "| 빠짐", len(missing), "추가", len(extra), "바뀜", len(changed), missing[:5], extra[:5], changed[:5])
sys.exit(0 if ok else 1)
