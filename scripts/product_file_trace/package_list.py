"""제품 최소 패키지 목록 초안(D-170): 관찰한 프로세스들이 실제로 불러온 모듈 → 그 환경의 배포 패키지 이름·버전.

사용: <그 환경의 python> package_list.py <출력 json> <modules json> [<modules json> ...]
- modules json: sitecustomize의 modules-<pid>.json 또는 env_probe/worker_imports 결과(둘 다 "modules": {이름: 파일}).
- 이 Python의 site-packages 아래 파일만 센다(다른 환경·표준 라이브러리·저장소 코드는 뺀다).
- 실행 중 늦게 불러오는 모듈 중 관찰하지 않은 것은 빠질 수 있다 → 초안. 설치 시험(pip 의존성 해석)으로 보완한다.
"""
import importlib.metadata as md
import json
import sys
import sysconfig
from pathlib import Path


def main(argv):
    out, sources = Path(argv[0]), [Path(p) for p in argv[1:]]
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    tops, seen_in = {}, {}
    for src in sources:
        data = json.loads(src.read_text(encoding="utf-8"))
        for name, origin in data["modules"].items():
            try:
                path = Path(origin).resolve()
            except (OSError, ValueError):
                continue
            if site not in path.parents:
                continue
            top = path.relative_to(site).parts[0]
            top = top[:-3] if top.endswith(".py") else top.split(".")[0]
            tops.setdefault(top, set()).add(name.split(".")[0])
            seen_in.setdefault(top, set()).add(src.name)
    # 최상위 이름 → 배포 패키지: 각 배포의 RECORD 파일 목록에서 최상위 폴더·모듈 이름을 모은다(top_level.txt가 없는 배포도 잡힌다)
    owners = {}
    for dist in md.distributions():
        for f in dist.files or ():
            parts = Path(str(f)).parts
            if not parts or parts[0].endswith((".dist-info", ".egg-info", ".data")) or parts[0] in ("..", "__pycache__"):
                continue
            first = parts[0][:-3] if parts[0].endswith(".py") else parts[0].split(".")[0]
            owners.setdefault(first, set()).add(dist.metadata["Name"])
    dists, unmapped = {}, []
    for top, names in sorted(tops.items()):
        found = set()
        for n in names | {top}:
            found |= owners.get(n, set())
        if not found:
            unmapped.append(top)
        for d in found:
            dist = md.distribution(d)
            key = dist.metadata["Name"]
            entry = dists.setdefault(key, dict(version=dist.version, license=dist.metadata.get("License-Expression") or dist.metadata.get("License") or "",
                                                import_names=set(), observed_in=set()))
            entry["import_names"].add(top)
            entry["observed_in"] |= seen_in[top]
    record = dict(python=sys.executable, site_packages=str(site), version=sys.version.split()[0],
                  packages={k: dict(v, import_names=sorted(v["import_names"]), observed_in=sorted(v["observed_in"]),
                                    license=(v["license"] or "")[:80]) for k, v in sorted(dists.items(), key=lambda kv: kv[0].lower())},
                  unmapped_top_level=unmapped)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(record["packages"]), "packages", "| 배포 이름 못 찾음:", unmapped or "없음")


if __name__ == "__main__":
    main(sys.argv[1:])
