"""캐릭터 기억 관리(H1-B, genai_lab/character_memory.py). 작업실 확인 창 밖에서 보고·고치는 도구.

  python scripts/character_memory.py list
  python scripts/character_memory.py show <캐릭터 ID>
  python scripts/character_memory.py unlink <캐릭터 ID> <그림 경로 또는 SHA-256> --reason "잘못 연결"
  python scripts/character_memory.py remove <캐릭터 ID> --reason "..."        (removed/로 옮김, 되돌릴 수 있음)
  python scripts/character_memory.py migrate <그림 경로...> [--apply]       (기존 파일 경로 기록 → 캐릭터 등록. 기본은 미리 보기만)

--root <폴더>로 기억 위치를 바꾼다(기본: character_memory.default_root()).
이관은 그 파일 경로로 저장된 성별·부위·특징표가 모두 있을 때만 한다(사람이 확정한 값만 옮긴다). 경로 기록은 지우지 않는다.
경로 기록의 열쇠는 경로 문자열이라 같은 파일도 Z:\\ 와 \\\\192.168.0.109\\ 로 열면 다른 기록이다 — 작업실에서 열던 경로로 적는다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_lab.character_memory import CharacterMemory, CharacterMemoryError, image_sha256  # noqa: E402


def summary(record):
    roles = {}
    for entry in record["images"].values():
        roles[entry["role"]] = roles.get(entry["role"], 0) + 1
    return (f"{record['character_id']}  {record['name']}  그림 {roles}  승인 {len(record['approvals'])}  "
            f"해제 {len(record['unlinked'])}  기준 {record['canonical']['path']}")


def migrate(memory, paths, apply):
    from genai_lab.character_preferences import load_character_body_parts, load_character_features, load_character_gender
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            print(f"건너뜀(파일 없음): {raw}")
            continue
        record = dict(gender=load_character_gender(path), body_parts=load_character_body_parts(path), features=load_character_features(path))
        missing = [k for k, v in record.items() if v is None]
        found = memory.find_image(image_sha256(path))
        if found:
            print(f"건너뜀(이미 '{found[0]['name']}'에 연결): {raw}")
        elif missing:
            print(f"건너뜀(경로 기록에 확정값 없음: {', '.join(missing)}): {raw}")
        elif not apply:
            print(f"등록 예정: {raw} · 성별 {record['gender']} · 부위 {list(record['body_parts'])}")
        else:
            created = memory.create(path, migrated_from=dict(kind="path_preferences", path=str(path), via="scripts/character_memory.py"),
                                    **record)
            print(f"등록: {raw} → {created['character_id']}")
    if not apply:
        print("미리 보기만 했습니다. 등록하려면 --apply")


def main(argv=None):
    parser = argparse.ArgumentParser(description="캐릭터 기억 관리")
    parser.add_argument("--root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("character_id")
    unlink = commands.add_parser("unlink")
    unlink.add_argument("character_id")
    unlink.add_argument("image")
    unlink.add_argument("--reason", required=True)
    remove = commands.add_parser("remove")
    remove.add_argument("character_id")
    remove.add_argument("--reason", required=True)
    move = commands.add_parser("migrate")
    move.add_argument("paths", nargs="+")
    move.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    memory = CharacterMemory(args.root)
    try:
        if args.command == "list":
            records = memory.characters()
            for record in records:
                print(summary(record))
            print(f"저장된 캐릭터 {len(records)}명 · 위치 {memory.root}"
                  + ("" if records else " — 작업실 확인 창에서 시작을 누르면 등록됩니다"))
        elif args.command == "show":
            print(json.dumps(memory.get(args.character_id), ensure_ascii=False, indent=1))
        elif args.command == "unlink":
            sha = args.image if re.fullmatch(r"[0-9a-f]{64}", args.image) else image_sha256(args.image)
            memory.unlink(args.character_id, sha, reason=args.reason)
            print(f"해제: {sha}")
        elif args.command == "remove":
            print(f"옮김: {memory.remove(args.character_id, reason=args.reason)}")
        else:
            migrate(memory, args.paths, args.apply)
    except CharacterMemoryError as error:
        print(f"오류: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
