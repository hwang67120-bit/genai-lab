"""패키지 사용 근거 조사(CPU, D-171): 같은 폴더를 함께 쓰는 배포(opencv 두 종류, PySide6 분할 패키지)와 setuptools.

사용: <주 환경 python> package_evidence.py <출력 json> <modules json> [...]
- 실제로 불러온 파일이 어느 배포의 RECORD에 있고, 지금 디스크의 파일 해시가 어느 배포의 기록과 같은지(나중에 설치한 쪽이 덮어씀)를 본다.
- 이 조사는 제거 판단의 근거 자료다. 독립 실행 확인 전에는 목록에서 빼지 않는다.
"""
import base64
import hashlib
import importlib.metadata as md
import json
import sys
import sysconfig
from pathlib import Path

TARGETS = ("opencv-python", "opencv-contrib-python", "PySide6", "PySide6_Essentials", "PySide6_Addons", "PySide6_Pdf",
           "PySide6_WebEngine", "shiboken6", "setuptools")


def record_hash(path):
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def main(argv):
    out = Path(argv[0])
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    loaded = {}
    for src in argv[1:]:
        for name, origin in json.loads(Path(src).read_text(encoding="utf-8"))["modules"].items():
            try:
                p = Path(origin).resolve()
            except (OSError, ValueError):
                continue
            if site in p.parents:
                loaded.setdefault(str(p.relative_to(site)).replace("\\", "/"), set()).add(name)
    result = {}
    for dist_name in TARGETS:
        try:
            dist = md.distribution(dist_name)
        except md.PackageNotFoundError:
            result[dist_name] = dict(installed=False)
            continue
        files = {str(f).replace("\\", "/"): f for f in dist.files or ()}
        used = sorted(set(files) & set(loaded))
        matching, differing = [], []
        for rel in used:
            f = files[rel]
            if f.hash is None:
                continue
            actual = record_hash(site / rel)
            (matching if actual == f"{f.hash.mode}={f.hash.value}" else differing).append(rel)
        result[dist_name] = dict(installed=True, version=dist.version, files_in_record=len(files), loaded_files_in_record=len(used),
                                 loaded_files_hash_match=len(matching), loaded_files_hash_differ=len(differing),
                                 examples=used[:8], loaded_modules=sorted({m for r in used for m in loaded[r]})[:20])
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in result.items():
        print(k, {x: v.get(x) for x in ("version", "loaded_files_in_record", "loaded_files_hash_match", "loaded_files_hash_differ")})


if __name__ == "__main__":
    main(sys.argv[1:])
