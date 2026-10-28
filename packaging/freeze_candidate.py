"""배포 후보 파일 목록·해시 고정과 대조(D-175).

  freeze <배포 후보> <목록 json.gz>   모든 파일의 상대 경로·크기·SHA-256 + 전체 지문(정렬된 목록의 SHA-256)
  verify <배포 후보> <목록 json.gz>   목록과 다르면(빠짐·추가·크기·SHA) 종료 코드 1 — GPU 검증 전에 Codex가 돌린다
검증 도구(sitecustomize.py)를 넣은 상태는 고정 대상이 아니다 — verify는 그 파일을 '추가'로 보고한다.
"""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(root):
    root = Path(root)
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            files[p.relative_to(root).as_posix()] = dict(size=p.stat().st_size, sha256=sha(p))
    return files


def fingerprint(files):
    text = "\n".join(f"{k}\t{v['size']}\t{v['sha256']}" for k, v in sorted(files.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv):
    command, root, manifest = argv[0], Path(argv[1]), Path(argv[2])
    if command == "freeze":
        files = scan(root)
        record = dict(schema="candidate_manifest/1", root_name=root.name, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                      file_count=len(files), total_bytes=sum(v["size"] for v in files.values()), fingerprint=fingerprint(files), files=files)
        with gzip.open(manifest, "wt", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        print("고정:", record["file_count"], "파일", round(record["total_bytes"] / 2**30, 2), "GiB, 지문", record["fingerprint"])
        return 0
    if command == "verify":
        with gzip.open(manifest, "rt", encoding="utf-8") as f:
            record = json.load(f)
        files = scan(root)
        expected = record["files"]
        missing = sorted(set(expected) - set(files))
        extra = sorted(set(files) - set(expected))
        changed = sorted(k for k in set(files) & set(expected) if files[k] != expected[k])
        ok = not (missing or extra or changed)
        print("대조:", "일치" if ok else "불일치", "| 지문", fingerprint(files), "/ 고정", record["fingerprint"],
              "| 빠짐", len(missing), "추가", len(extra), "바뀜", len(changed))
        for name, items in (("빠짐", missing), ("추가", extra), ("바뀜", changed)):
            for k in items[:10]:
                print("  ", name, k)
        return 0 if ok else 1
    raise SystemExit("usage: freeze|verify <candidate> <manifest.json.gz>")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
