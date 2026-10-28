"""캐릭터 기억(H1-B): SHA 정확 조회·사람 확인 연결·해제·이관·정체성만 저장."""
import json

import pytest

from genai_lab.character_memory import CharacterMemory, CharacterMemoryError, apply_link, image_sha256, prefill


def features(**changes):
    items = dict(hair_length=["long_hair"], hair_style=["braid"], hair_color=["blonde_hair"], eye_color=["blue_eyes"],
                 head_accessories=[], skin=None, distinctive=None)
    items.update(changes)
    return items


def picture(path, data):
    path.write_bytes(data)
    return path


def test_create_and_find_by_bytes_not_path(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    original = picture(tmp_path / "a.png", b"original")
    record = memory.create(original, gender="male", body_parts=(), features=features())
    copy = picture(tmp_path / "other-folder-copy.png", b"original")
    found, entry = memory.find_image(image_sha256(copy))
    assert found["character_id"] == record["character_id"] and entry["role"] == "canonical"
    assert memory.find_image(image_sha256(picture(tmp_path / "b.png", b"different"))) is None
    assert memory.identity(record["character_id"]) == dict(gender="male", body_parts=(), features=features())


def test_identity_has_no_place_for_outfit_or_request(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    with pytest.raises(CharacterMemoryError):
        memory.create(picture(tmp_path / "a.png", b"a"), gender="male", body_parts=(),
                      features=dict(features(), outfit=["suit"]))
    record = memory.create(picture(tmp_path / "b.png", b"b"), gender="female", body_parts=("tail",), features=features())
    stored = json.loads((tmp_path / "memory" / "characters" / f"{record['character_id']}.json").read_text(encoding="utf-8"))
    assert set(stored["identity"]) == {"gender", "body_parts", "features", "confirmed_at", "source_sha256"}
    assert "request" not in json.dumps(stored["identity"]) and "garment" not in json.dumps(stored)


def test_link_needs_user_confirmation_and_one_owner(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    a = memory.create(picture(tmp_path / "a.png", b"a"), gender="male", body_parts=(), features=features())
    b = memory.create(picture(tmp_path / "b.png", b"b"), gender="female", body_parts=(), features=features())
    other = picture(tmp_path / "a-approved.png", b"a2")
    with pytest.raises(CharacterMemoryError):
        memory.link(a["character_id"], other, user_confirmed=False, evidence={})
    memory.link(a["character_id"], other, user_confirmed=True, evidence=dict(kind="user_choice"))
    with pytest.raises(CharacterMemoryError):
        memory.link(b["character_id"], other, user_confirmed=True, evidence=dict(kind="user_choice"))
    assert memory.find_image(image_sha256(other))[0]["character_id"] == a["character_id"]


def test_unlink_keeps_record_and_refuses_canonical(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    original = picture(tmp_path / "a.png", b"a")
    record = memory.create(original, gender="male", body_parts=(), features=features())
    wrong = picture(tmp_path / "wrong.png", b"someone else")
    memory.link(record["character_id"], wrong, user_confirmed=True, evidence=dict(kind="user_choice"))
    with pytest.raises(CharacterMemoryError):
        memory.unlink(record["character_id"], image_sha256(original), reason="x")
    after = memory.unlink(record["character_id"], image_sha256(wrong), reason="잘못 연결")
    assert image_sha256(wrong) not in after["images"] and after["unlinked"][0]["reason"] == "잘못 연결"
    assert memory.find_image(image_sha256(wrong)) is None
    log = [json.loads(line)["event"] for line in (tmp_path / "memory" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log == ["create", "link", "unlink"]


def test_remove_moves_file_so_it_can_be_restored(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    record = memory.create(picture(tmp_path / "a.png", b"a"), gender="male", body_parts=(), features=features())
    moved = memory.remove(record["character_id"], reason="테스트")
    assert moved.is_file() and memory.characters() == []


def test_save_identity_keeps_history(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    record = memory.create(picture(tmp_path / "a.png", b"a"), gender="male", body_parts=(), features=features())
    memory.save_identity(record["character_id"], gender="male", body_parts=(), features=features(hair_color=["pink_hair"]),
                         source_sha256="x")
    after = memory.get(record["character_id"])
    assert after["identity"]["features"]["hair_color"] == ["pink_hair"]
    assert after["identity_history"][0]["features"]["hair_color"] == ["blonde_hair"]


def approved_export(tmp_path, base, character=True, mode="free_generation"):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "trace.json").write_text(json.dumps(dict(input_facts=dict(base=dict(sha256=image_sha256(base))),
                                                    experiment=None)), encoding="utf-8")
    image = picture(tmp_path / f"saved-{mode}-{character}.png", b"approved " + mode.encode() + str(character).encode())
    sidecar = image.with_suffix(".review.json")
    sidecar.write_text(json.dumps(dict(mode=mode, sha256=image_sha256(image), reviewer="user", trace=str(run / "trace.json"),
                                       checks=dict(character=character, garment=True))), encoding="utf-8")
    return image, sidecar


def test_register_approved_needs_same_character_check(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    base = picture(tmp_path / "a.png", b"a")
    record = memory.create(base, gender="male", body_parts=(), features=features())
    image, sidecar = approved_export(tmp_path, base, character=False)
    with pytest.raises(CharacterMemoryError):
        memory.register_approved(record["character_id"], image, sidecar)
    image, sidecar = approved_export(tmp_path, base)
    after = memory.register_approved(record["character_id"], image, sidecar)
    assert after["images"][image_sha256(image)]["role"] == "approved_free"
    assert after["approvals"][0]["steps"] == 40 and after["approvals"][0]["approved_checks"]["character"] is True


def test_suggestion_from_older_export_is_only_a_suggestion(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    base = picture(tmp_path / "a.png", b"a")
    record = memory.create(base, gender="male", body_parts=(), features=features())
    image, _ = approved_export(tmp_path, base)
    info = prefill(memory, image)
    assert info["linked"] is None and info["suggestion"]["character_id"] == record["character_id"]
    assert info["identity"] is None and memory.find_image(image_sha256(image)) is None     # 제안만, 연결·미리 채움 없음


def test_prefill_order_memory_then_path_record(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    new = picture(tmp_path / "new.png", b"n")
    assert prefill(memory, new)["source"] is None
    old = dict(gender="female", body_parts=("tail",), features=None)
    info = prefill(memory, new, old)
    assert info["source"] == "path_record" and info["identity"] == dict(gender="female", body_parts=("tail",))
    record = memory.create(new, gender="male", body_parts=(), features=features())
    info = prefill(memory, new, old)
    assert info["source"] == "memory" and info["linked"]["character_id"] == record["character_id"]
    assert info["identity"]["gender"] == "male"


def test_apply_link_actions(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    base = picture(tmp_path / "a.png", b"a")
    identity = dict(gender="male", body_parts=(), features=features())
    cid = apply_link(memory, base, dict(action="create"), path_record_used=True, **identity)
    assert memory.get(cid)["migrated_from"][0]["kind"] == "path_preferences"
    second = picture(tmp_path / "a-other.png", b"a-other")
    with pytest.raises(CharacterMemoryError):
        apply_link(memory, second, dict(action="link", character_id=cid, confirmed=False), **identity)
    assert apply_link(memory, second, dict(action="link", character_id=cid, confirmed=True), **identity) == cid
    assert apply_link(memory, second, dict(action="unlink_none", character_id=cid), **identity) is None
    assert memory.find_image(image_sha256(second)) is None
    assert apply_link(memory, second, dict(action="none"), **identity) is None and len(memory.characters()) == 1
