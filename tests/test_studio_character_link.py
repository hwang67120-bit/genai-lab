"""확인 창의 캐릭터 연결(H1-B): 사람 확인 연결·해제, 사람이 고친 칸과 요청 문장은 덮어쓰지 않음, 승인 저장 등록."""
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QPlainTextEdit, QPushButton

from genai_lab import studio_controller as ui, studio_free_generation as free_ui
from genai_lab.character_memory import CharacterMemory, image_sha256
from genai_lab.character_features import parse_features_text


def features(**changes):
    items = dict(hair_length=["long_hair"], hair_style=["braid"], hair_color=["blonde_hair"], eye_color=["blue_eyes"],
                 head_accessories=[], skin=None, distinctive=None)
    items.update(changes)
    return items


def no_path_records(monkeypatch):
    for name in ("load_character_gender", "load_character_body_parts", "load_character_features"):
        monkeypatch.setattr(free_ui, name, lambda *a: None)


def open_free(monkeypatch, base, memory, interact):
    QApplication.instance() or QApplication([])
    no_path_records(monkeypatch)
    monkeypatch.setattr(QDialog, "exec", interact)
    return free_ui.confirm_free_inputs(None, dict(source=str(base), character_scores={}), memory)


def choose(dialog, prefix, action, character_id=None):
    combo = dialog.findChild(QComboBox, prefix + "_character_link")
    index = next(i for i in range(combo.count()) if combo.itemData(i)["action"] == action
                 and (character_id is None or combo.itemData(i).get("character_id") == character_id))
    combo.setCurrentIndex(index)


def test_linked_image_prefills_identity_but_not_request(tmp_path, monkeypatch):
    memory = CharacterMemory(tmp_path / "memory")
    base = tmp_path / "m3.png"
    base.write_bytes(b"m3")
    memory.create(base, gender="male", body_parts=("tail",), features=features(hair_color=["pink_hair"]))
    seen = {}
    def interact(dialog):
        seen["gender"] = dialog.findChild(QComboBox, "free_gender").currentData()
        seen["tail"] = dialog.findChild(QCheckBox, "free_part_tail").isChecked()
        seen["features"] = parse_features_text(dialog.findChild(QPlainTextEdit, "free_features").toPlainText())
        seen["request"] = dialog.findChild(QPlainTextEdit, "free_request").toPlainText()
        dialog.findChild(QPlainTextEdit, "free_request").setPlainText("sitting in a cafe")
        dialog.findChild(QCheckBox, "free_confirm").setChecked(True)
        return QDialog.DialogCode.Accepted
    result = open_free(monkeypatch, base, memory, interact)
    assert seen["gender"] == "male" and seen["tail"] and seen["features"]["hair_color"] == ["pink_hair"] and seen["request"] == ""
    assert result["character_link"]["action"] == "keep" and result["request_text"] == "sitting in a cafe"


def test_linking_to_another_character_needs_the_same_character_check(tmp_path, monkeypatch):
    memory = CharacterMemory(tmp_path / "memory")
    original = tmp_path / "original.png"
    original.write_bytes(b"original")
    record = memory.create(original, gender="female", body_parts=(), features=features(hair_color=["aqua_hair"]))
    other = tmp_path / "other-picture.png"
    other.write_bytes(b"other")
    seen = {}
    def interact(dialog):
        start = next(b for b in dialog.findChildren(QPushButton) if b.text() == "자유 생성 시작")
        dialog.findChild(QCheckBox, "free_confirm").setChecked(True)
        seen["default"] = dialog.findChild(QComboBox, "free_character_link").currentData()["action"]
        choose(dialog, "free", "link", record["character_id"])
        seen["features"] = parse_features_text(dialog.findChild(QPlainTextEdit, "free_features").toPlainText())
        seen["blocked"] = not start.isEnabled()
        dialog.findChild(QCheckBox, "free_character_link_confirm").setChecked(True)
        seen["allowed"] = start.isEnabled()
        return QDialog.DialogCode.Accepted
    result = open_free(monkeypatch, other, memory, interact)
    assert seen["default"] == "create" and seen["blocked"] and seen["allowed"]
    assert seen["features"]["hair_color"] == ["aqua_hair"]
    assert result["character_link"] == dict(action="link", character_id=record["character_id"], evidence=dict(kind="user_choice"),
                                            confirmed=True, path_record_used=False)


def test_user_edits_are_not_overwritten_by_link_prefill(tmp_path, monkeypatch):
    memory = CharacterMemory(tmp_path / "memory")
    original = tmp_path / "original.png"
    original.write_bytes(b"original")
    record = memory.create(original, gender="female", body_parts=(), features=features(hair_color=["aqua_hair"]))
    other = tmp_path / "other.png"
    other.write_bytes(b"other")
    typed = "머리 길이: short_hair\n머리 모양: 없음\n머리색: red_hair\n눈 색: 모름\n머리 장식: 없음\n피부: 모름\n고유 형태: 모름"
    seen = {}
    def interact(dialog):
        dialog.findChild(QPlainTextEdit, "free_features").setPlainText(typed)
        choose(dialog, "free", "link", record["character_id"])
        seen["features"] = dialog.findChild(QPlainTextEdit, "free_features").toPlainText()
        seen["gender"] = dialog.findChild(QComboBox, "free_gender").currentData()   # 고치지 않은 칸은 대상 값으로
        return QDialog.DialogCode.Rejected
    open_free(monkeypatch, other, memory, interact)
    assert seen["features"] == typed and seen["gender"] == "female"


def test_unlink_is_offered_for_linked_copies_not_for_the_original(tmp_path, monkeypatch):
    memory = CharacterMemory(tmp_path / "memory")
    original = tmp_path / "original.png"
    original.write_bytes(b"original")
    record = memory.create(original, gender="male", body_parts=(), features=features())
    linked = tmp_path / "linked.png"
    linked.write_bytes(b"linked")
    memory.link(record["character_id"], linked, user_confirmed=True, evidence=dict(kind="user_choice"))
    actions = {}
    def interact_for(key):
        def interact(dialog):
            combo = dialog.findChild(QComboBox, "free_character_link")
            actions[key] = [combo.itemData(i)["action"] for i in range(combo.count())]
            if key == "linked":
                choose(dialog, "free", "unlink_none")
                dialog.findChild(QCheckBox, "free_confirm").setChecked(True)
                return QDialog.DialogCode.Accepted
            return QDialog.DialogCode.Rejected
        return interact
    open_free(monkeypatch, original, memory, interact_for("original"))
    result = open_free(monkeypatch, linked, memory, interact_for("linked"))
    assert actions["original"] == ["keep"] and actions["linked"] == ["keep", "unlink_create", "unlink_none"]
    controller = ui.StudioController.__new__(ui.StudioController)
    controller.character_memory, controller.window = memory, None
    controller.apply_character_link(linked, result["character_link"], result["gender"], result["body_parts"], result["character_features"])
    assert controller.character_id is None and memory.find_image(image_sha256(linked)) is None
    assert memory.find_image(image_sha256(original))[0]["character_id"] == record["character_id"]


def test_controller_registers_only_user_approved_saves(tmp_path):
    memory = CharacterMemory(tmp_path / "memory")
    original = tmp_path / "original.png"
    original.write_bytes(b"original")
    controller = ui.StudioController.__new__(ui.StudioController)
    controller.character_memory, controller.window = memory, None
    controller.apply_character_link(original, dict(action="create"), "male", (), features())
    saved = tmp_path / "saved.png"
    saved.write_bytes(b"approved result")
    saved.with_suffix(".review.json").write_text(json.dumps(dict(mode="free_generation", sha256=image_sha256(saved), reviewer="user",
                                                                 trace=str(tmp_path / "missing-trace.json"),
                                                                 checks=dict(character=True))), encoding="utf-8")
    assert "기억했습니다" in controller.register_saved(saved)
    found, entry = memory.find_image(image_sha256(saved))
    assert found["character_id"] == controller.character_id and entry["role"] == "approved_free"
    controller.character_id = None
    assert controller.register_saved(saved) == ""


def test_edit_dialog_link_row_changes_identity_only(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    from PIL import Image
    memory = CharacterMemory(tmp_path / "memory")
    original = tmp_path / "original.png"
    original.write_bytes(b"original")
    record = memory.create(original, gender="male", body_parts=("ears",), features=features(hair_color=["white_hair"]))
    directory = tmp_path / "analysis"
    directory.mkdir()
    for name in ("character.png", "garment.png"):
        Image.new("RGB", (8, 8)).save(directory / name)
    other = tmp_path / "other.png"
    other.write_bytes(b"other")
    from genai_lab.character_memory import prefill
    info = prefill(memory, other)
    analysis = dict(directory=str(directory), character_scores={}, groups=dict(fixed=[], appearance=[], body=[]))
    seen = {}
    def interact(dialog):
        choose(dialog, "outfit_edit", "link", record["character_id"])
        seen["gender"] = dialog.findChild(QComboBox, "outfit_edit_gender").currentData()
        seen["ears"] = dialog.findChild(QCheckBox, "outfit_edit_part_ears").isChecked()
        dialog.findChild(QCheckBox, "outfit_edit_confirm").setChecked(True)
        seen["blocked"] = "같은 캐릭터" in dialog.findChild(ui.QLabel, "outfit_edit_hint").text()
        dialog.findChild(QCheckBox, "outfit_edit_character_link_confirm").setChecked(True)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", interact)
    result = ui.confirm_outfit_edit(None, analysis, "keep", link_info=info, memory=memory)
    assert seen == dict(gender="male", ears=True, blocked=True)
    assert result["features"]["hair_color"] == ["white_hair"] and result["character_link"]["confirmed"] is True
    assert set(result) == {"gender", "body_parts", "features", "character_link"}
