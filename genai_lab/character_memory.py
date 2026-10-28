"""캐릭터 기억(H1-B, docs/CHARACTER_MEMORY_REUSE_REVIEW.md): 같은 캐릭터의 확정 정체성(성별·부위·특징표)을 캐릭터 ID로 묶는다.

- 그림은 바이트 SHA-256으로만 찾는다. 비슷해 보이는 다른 캐릭터를 자동으로 찾지 않는다(그건 H2, 범위 밖).
- 그림을 캐릭터에 잇는 것은 사람이 확인한 경우뿐이다: 확인 창에서 고르고 '같은 캐릭터' 체크를 한 연결, 또는 사람이 결과 승인에서
  '같은 캐릭터입니다'를 체크하고 저장한 결과.
- 저장하는 것은 정체성(IDENTITY_KEYS)뿐이다. 의상·옷 참조·연출 요청·배경·자세는 저장하지 않으므로 연결 때문에 새 요청이 과거 값으로
  바뀌지 않는다.
- 기존 파일 경로 기록(character_preferences, QSettings)은 지우지 않는다. 처음 등록할 때 그 값을 출처와 함께 가져온다(이관).
- 생성 결과는 기준 원본(canonical)을 대체하지 않는다. 연결 해제는 기록을 남기고, 캐릭터 삭제는 removed/로 옮긴다(되돌릴 수 있다).
GUI·HTTP에 독립(Qt 없음).
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

SCHEMA = "character_memory/1"
IDENTITY_KEYS = ("gender", "body_parts", "features")
LINK_ROLES = ("linked_source", "approved_free", "approved_edit")       # canonical은 만들 때만
GENDERS = ("male", "female", "unspecified")
BODY_PARTS = ("ears", "tail", "wings")
ID_PATTERN = re.compile(r"[0-9a-f]{32}")
HISTORY = 10                    # 정체성 이전 값을 이만큼 남긴다(잘못 확정했을 때 되돌리기용)


class CharacterMemoryError(ValueError):
    """사람이 고쳐야 하는 기억 오류(이미 다른 캐릭터에 연결된 그림, 확인 없는 연결 등)."""


def default_root():
    override = os.environ.get("GENAI_CHARACTER_MEMORY")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA")
    return (Path(base) / "GenAILab" if base else Path.home() / ".local" / "share" / "genai-lab") / "character-memory"


def image_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now().isoformat(timespec="seconds")


def validate_identity(gender, body_parts, features):
    """정체성 세 값만 받는다(의상·요청은 받을 자리가 없다). features는 확인 창 형식({항목: [태그] | None}, 고유 형태 메모 포함)."""
    from genai_lab.character_features import DISTINCTIVE, TAG_ITEMS
    if gender not in GENDERS:
        raise CharacterMemoryError("성별 값 오류")
    parts = tuple(body_parts)
    if len(set(parts)) != len(parts) or not set(parts) <= set(BODY_PARTS):
        raise CharacterMemoryError("부위 값 오류")
    if (not isinstance(features, dict) or set(features) - {DISTINCTIVE} != set(TAG_ITEMS)
            or not all(v is None or (isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v)) for v in features.values())):
        raise CharacterMemoryError("특징표 값 오류")
    return dict(gender=gender, body_parts=[p for p in BODY_PARTS if p in parts],
                features={k: None if features.get(k) is None else list(features[k]) for k in (*TAG_ITEMS, DISTINCTIVE)})


class CharacterMemory:
    """root/characters/<id>.json 한 캐릭터 한 파일, root/log.jsonl 연결·해제 기록, root/removed/ 삭제한 캐릭터."""

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else default_root()

    # 파일

    def _path(self, character_id):
        if not isinstance(character_id, str) or not ID_PATTERN.fullmatch(character_id):
            raise CharacterMemoryError("캐릭터 ID 형식 오류")
        return self.root / "characters" / f"{character_id}.json"

    def get(self, character_id):
        path = self._path(character_id)
        if not path.is_file():
            raise CharacterMemoryError("저장된 캐릭터가 없습니다.")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("character_id") != character_id:
            raise CharacterMemoryError("캐릭터 기억 형식 오류")
        return record

    def _write(self, record):
        record["updated_at"] = now()
        path = self._path(record["character_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temporary, path)

    def _log(self, event, **values):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(at=now(), event=event, **values), ensure_ascii=False) + "\n")

    def characters(self):
        folder = self.root / "characters"
        if not folder.is_dir():
            return []
        records = [self.get(p.stem) for p in folder.glob("*.json") if ID_PATTERN.fullmatch(p.stem)]
        return sorted(records, key=lambda r: r["created_at"])

    # 조회(정확 일치만)

    def find_image(self, sha256):
        """그림 SHA가 연결된 캐릭터 → (기록, 그림 항목) 또는 None. 두 캐릭터에 있으면 오류(사람이 해제해야 함)."""
        found = [(r, r["images"][sha256]) for r in self.characters() if sha256 in r["images"]]
        if len(found) > 1:
            raise CharacterMemoryError("같은 그림이 여러 캐릭터에 연결돼 있습니다. 하나를 해제해 주세요: "
                                       + ", ".join(r["name"] for r, _ in found))
        return found[0] if found else None

    def identity(self, character_id):
        """확인 창 미리 채움용 정체성(성별·부위·특징표). 의상·요청은 들어 있지 않다."""
        identity = self.get(character_id)["identity"]
        return dict(gender=identity["gender"], body_parts=tuple(identity["body_parts"]), features=dict(identity["features"]))

    # 변경(모두 사람 확인 뒤에 부른다)

    def create(self, canonical_path, *, gender, body_parts, features, name=None, migrated_from=None):
        """새 캐릭터. canonical_path가 기준 원본이 되고 이후 자동으로 바뀌지 않는다."""
        sha = image_sha256(canonical_path)
        if self.find_image(sha):
            raise CharacterMemoryError("이 그림은 이미 다른 캐릭터에 연결돼 있습니다. 먼저 연결을 해제해 주세요.")
        character_id = uuid.uuid4().hex
        identity = dict(validate_identity(gender, body_parts, features), confirmed_at=now(), source_sha256=sha)
        record = dict(schema=SCHEMA, character_id=character_id, name=name or Path(canonical_path).stem, created_at=now(),
                      canonical=dict(sha256=sha, path=str(Path(canonical_path).resolve()), registered_at=now()),
                      images={sha: dict(role="canonical", path=str(Path(canonical_path).resolve()), linked_at=now(),
                                        evidence=dict(kind="created"))},
                      identity=identity, identity_history=[], approvals=[], unlinked=[],
                      migrated_from=[migrated_from] if migrated_from else [])
        self._write(record)
        self._log("create", character_id=character_id, sha256=sha, path=str(canonical_path), migrated_from=migrated_from)
        return record

    def link(self, character_id, path, *, user_confirmed, evidence, role="linked_source"):
        """그림을 이 캐릭터에 잇는다. user_confirmed=True(확인 창 체크 또는 결과 승인의 '같은 캐릭터')가 없으면 거부."""
        if user_confirmed is not True:
            raise CharacterMemoryError("같은 캐릭터인지 사람이 확인해야 연결할 수 있습니다.")
        if role not in LINK_ROLES:
            raise CharacterMemoryError("연결 종류 오류")
        sha = image_sha256(path)
        found = self.find_image(sha)
        if found and found[0]["character_id"] != character_id:
            raise CharacterMemoryError(f"이 그림은 '{found[0]['name']}'에 연결돼 있습니다. 먼저 연결을 해제해 주세요.")
        record = self.get(character_id)
        if sha in record["images"]:
            return record
        record["images"][sha] = dict(role=role, path=str(Path(path).resolve()), linked_at=now(), evidence=evidence)
        self._write(record)
        self._log("link", character_id=character_id, sha256=sha, path=str(path), role=role, evidence=evidence)
        return record

    def unlink(self, character_id, sha256, *, reason):
        """잘못 연결한 그림을 뗀다. 기준 원본은 뗄 수 없다(캐릭터 삭제를 쓴다). 승인 기록은 그 그림 것만 해제 표시."""
        record = self.get(character_id)
        entry = record["images"].get(sha256)
        if entry is None:
            raise CharacterMemoryError("이 캐릭터에 연결된 그림이 아닙니다.")
        if entry["role"] == "canonical":
            raise CharacterMemoryError("기준 원본은 연결 해제할 수 없습니다. 캐릭터를 삭제해 주세요.")
        del record["images"][sha256]
        record["unlinked"].append(dict(sha256=sha256, entry=entry, unlinked_at=now(), reason=reason))
        for approval in record["approvals"]:
            if approval["sha256"] == sha256:
                approval["unlinked"] = True
        self._write(record)
        self._log("unlink", character_id=character_id, sha256=sha256, reason=reason)
        return record

    def remove(self, character_id, *, reason):
        """캐릭터 삭제 = removed/로 옮김(파일 그대로라 되돌릴 수 있다)."""
        path = self._path(character_id)
        self.get(character_id)
        target = self.root / "removed" / f"{character_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
        self._log("remove", character_id=character_id, reason=reason, moved_to=str(target))
        return target

    def save_identity(self, character_id, *, gender, body_parts, features, source_sha256):
        """사람이 이번에 확인한 정체성으로 갱신. 이전 값은 identity_history에 남긴다."""
        record = self.get(character_id)
        identity = dict(validate_identity(gender, body_parts, features), confirmed_at=now(), source_sha256=source_sha256)
        if {k: identity[k] for k in IDENTITY_KEYS} == {k: record["identity"][k] for k in IDENTITY_KEYS}:
            return record
        record["identity_history"] = ([record["identity"]] + record["identity_history"])[:HISTORY]
        record["identity"] = identity
        self._write(record)
        self._log("identity", character_id=character_id, source_sha256=source_sha256)
        return record

    def register_approved(self, character_id, path, sidecar_path):
        """사람이 승인·저장한 결과를 이 캐릭터에 등록한다. 근거 = 같은 그림의 승인 기록(reviewer user, '같은 캐릭터' 체크)."""
        sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        checks = sidecar.get("checks") or {}
        mode = sidecar.get("mode")
        if (sidecar.get("reviewer") != "user" or checks.get("character") is not True or mode not in ("free_generation", "outfit_edit")
                or sidecar.get("sha256") != image_sha256(path)):
            raise CharacterMemoryError("같은 그림에 대한 사람의 '같은 캐릭터' 승인 기록이 없습니다.")
        steps = None
        try:
            trace = json.loads(Path(sidecar["trace"]).read_text(encoding="utf-8"))
            steps = 8 if (trace.get("experiment") or {}).get("lightning8") else 40
        except (OSError, KeyError, TypeError, ValueError):
            pass
        evidence = dict(kind="user_approval", sidecar=str(sidecar_path), checks=checks)
        record = self.link(character_id, path, user_confirmed=True, evidence=evidence,
                           role="approved_free" if mode == "free_generation" else "approved_edit")
        sha = sidecar["sha256"]
        if not any(a["sha256"] == sha for a in record["approvals"]):
            record["approvals"].append(dict(sha256=sha, path=str(Path(path).resolve()), sidecar=str(sidecar_path), mode=mode,
                                            source_run=sidecar.get("source_run"), variant=sidecar.get("variant"), steps=steps,
                                            approved_checks=checks, registered_at=now(),
                                            scope="사람 승인 체크 항목만 확인됨(전체 그림 크기에서). 세부 색·작은 장식은 판정 범위 밖일 수 있다"))
            self._write(record)
        return record

    def suggestion(self, path):
        """연결 안 된 그림에 대한 연결 제안(자동 연결 아님, 사람이 확인해야 함). 근거: 같은 그림의 자유 생성 승인 기록 →
        그 실행의 기준 그림 SHA가 연결된 캐릭터. 편집 승인 기록에는 원래 캐릭터 SHA가 없어 제안하지 않는다(직접 고르기)."""
        path = Path(path)
        sidecar_path = path.with_suffix(".review.json")
        if not sidecar_path.is_file():
            return None
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if (sidecar.get("reviewer") != "user" or (sidecar.get("checks") or {}).get("character") is not True
                    or sidecar.get("sha256") != image_sha256(path)):
                return None
            trace = json.loads(Path(sidecar["trace"]).read_text(encoding="utf-8"))
            base_sha = trace["input_facts"]["base"]["sha256"]
        except (OSError, KeyError, TypeError, ValueError):
            return None
        found = self.find_image(base_sha)
        if not found:
            return None
        return dict(character_id=found[0]["character_id"], name=found[0]["name"],
                    evidence=dict(kind="approved_export", sidecar=str(sidecar_path), base_sha256=base_sha))


def prefill(memory, path, path_record=None):
    """확인 창이 쓸 연결 정보와 미리 채움.
    path_record: 기존 파일 경로 기록 dict(gender, body_parts, features) — 값이 없는 칸은 None.
    반환 dict(sha256, linked, suggestion, others, identity, source). source = memory | path_record | None(자동 분석)."""
    sha = image_sha256(path)
    found = memory.find_image(sha)
    linked = dict(character_id=found[0]["character_id"], name=found[0]["name"], role=found[1]["role"],
                  canonical=found[0]["canonical"]["path"]) if found else None
    suggestion = None if linked else memory.suggestion(path)
    others = [dict(character_id=r["character_id"], name=r["name"], canonical=r["canonical"]["path"]) for r in memory.characters()
              if not linked or r["character_id"] != linked["character_id"]]
    if linked:
        return dict(sha256=sha, linked=linked, suggestion=None, others=others, identity=memory.identity(linked["character_id"]), source="memory")
    record = {k: v for k, v in (path_record or {}).items() if k in IDENTITY_KEYS and v is not None}
    return dict(sha256=sha, linked=None, suggestion=suggestion, others=others, identity=record or None,
                source="path_record" if record else None)


def apply_link(memory, path, link, *, gender, body_parts, features, path_record_used=False):
    """확인 창에서 고른 연결을 적용하고 이번에 확인한 정체성을 저장한다. 반환: 캐릭터 ID 또는 None(연결 안 함).
    link: dict(action=keep|link|create|unlink_create|unlink_none|none, character_id, confirmed, evidence)."""
    action = (link or {}).get("action", "none")
    sha = image_sha256(path)
    migrated = dict(kind="path_preferences", path=str(path)) if path_record_used else None
    if action == "none":
        return None
    if action in ("unlink_create", "unlink_none"):
        memory.unlink(link["character_id"], sha, reason="사용자가 확인 창에서 연결 해제")
        if action == "unlink_none":
            return None
        action = "create"
    if action == "create":
        return memory.create(path, gender=gender, body_parts=body_parts, features=features, migrated_from=migrated)["character_id"]
    if action == "link":
        memory.link(link["character_id"], path, user_confirmed=link.get("confirmed"), evidence=link.get("evidence") or dict(kind="user_choice"))
    elif action != "keep":
        raise CharacterMemoryError("연결 선택 오류")
    memory.save_identity(link["character_id"], gender=gender, body_parts=body_parts, features=features, source_sha256=sha)
    return link["character_id"]
