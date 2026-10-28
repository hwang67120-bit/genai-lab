"""제품 패키지 목록(packaging/requirements-*.txt)과 실제 실행 환경의 설치 버전을 대조한다(CPU, D-170).

사용: <그 환경의 python> scripts/check_product_packages.py packaging/requirements-main.txt
목록에 있는데 없거나 버전이 다르면 실패(종료 코드 1). 목록에 없는 추가 설치는 개수만 알린다(개발 도구 등).
"""
import importlib.metadata as md
import re
import sys
from pathlib import Path

LINE = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)")


def read_requirements(path):
    pins = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE.match(line)
        if not match:
            raise ValueError(f"고정 버전(이름==버전)이 아닌 줄: {raw}")
        pins[match.group(1)] = match.group(2)
    return pins


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def check(path):
    pins = read_requirements(path)
    installed = {normalize(d.metadata["Name"]): d.version for d in md.distributions()}
    problems = []
    for name, version in pins.items():
        actual = installed.get(normalize(name))
        if actual is None:
            problems.append(f"없음: {name}=={version}")
        elif actual != version:
            problems.append(f"버전 다름: {name} 목록 {version} / 설치 {actual}")
    extra = sorted(set(installed) - {normalize(n) for n in pins})
    return pins, problems, extra


def main(argv):
    pins, problems, extra = check(argv[0])
    print(f"{sys.executable} | 목록 {len(pins)}개 | 문제 {len(problems)}개 | 목록 밖 설치 {len(extra)}개")
    for line in problems:
        print("  ", line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
