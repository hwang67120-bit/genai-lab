"""고정한 패키지 버전마다 PyPI 공개 정보(파일 목록)만 읽어 Windows용 설치 파일이 지원하는 Python 버전을 센다. 패키지는 내려받지 않는다."""
import json, re, sys, urllib.request

TARGETS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

def pins(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            out.append((name.strip(), version.split(";")[0].strip()))
    return out

def support(name, version):
    v = version.split("+")[0]
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{v}/json", timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        return None, f"조회 실패 {type(e).__name__}"
    ok = set()
    requires = data["info"].get("requires_python") or ""
    for f in data["urls"]:
        fn = f["filename"]
        if not fn.endswith(".whl"):
            continue
        py, abi, plat = fn[:-4].split("-")[-3:]
        if not (plat == "any" or "win_amd64" in plat):
            continue
        for tag in py.split("."):
            if tag in ("py3", "py2.py3") or abi == "none" and tag.startswith("py3"):
                ok.update(TARGETS)
            m = re.match(r"cp3(\d+)", tag)
            if m:
                minor = int(m.group(1))
                if abi == "abi3":
                    ok.update(t for t in TARGETS if int(t.split(".")[1]) >= minor)
                else:
                    ok.add(f"3.{minor}")
    sdist = any(f["packagetype"] == "sdist" for f in data["urls"])
    return sorted(ok, key=lambda t: int(t.split(".")[1])), f"requires_python={requires!r} sdist={sdist}"

for path in sys.argv[1:]:
    print("==", path)
    for name, version in pins(path):
        ok, note = support(name, version)
        missing = [t for t in TARGETS if ok is not None and t not in ok]
        print(f"{name}=={version} | win/any 휠: {','.join(ok) if ok else '-'} | 없음: {','.join(missing) or '-'} | {note}")
