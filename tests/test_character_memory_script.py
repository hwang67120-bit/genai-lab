"""캐릭터 기억 관리 도구: 경로 기록 이관(미리 보기 → --apply), 해제, 삭제."""
import importlib.util
from pathlib import Path

from PySide6.QtCore import QSettings

from genai_lab import character_preferences
from genai_lab.character_memory import CharacterMemory, image_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "character_memory.py"
spec = importlib.util.spec_from_file_location("character_memory_script", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_migrate_previews_then_moves_only_fully_confirmed_records(tmp_path, monkeypatch, capsys):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    for name in ("load_character_gender", "load_character_body_parts", "load_character_features"):
        original = getattr(character_preferences, name)
        monkeypatch.setattr(character_preferences, name, lambda path, original=original: original(path, settings))
    full, partial = tmp_path / "full.png", tmp_path / "partial.png"
    full.write_bytes(b"full")
    partial.write_bytes(b"partial")
    items = dict(hair_length=["long_hair"], hair_style=[], hair_color=["white_hair"], eye_color=None, head_accessories=[], skin=None)
    character_preferences.save_character_gender(full, "male", settings)
    character_preferences.save_character_body_parts(full, (), settings)
    character_preferences.save_character_features(full, items, settings)
    character_preferences.save_character_gender(partial, "female", settings)
    root = tmp_path / "memory"
    assert tool.main(["--root", str(root), "migrate", str(full), str(partial)]) == 0
    assert CharacterMemory(root).characters() == [] and "등록 예정" in capsys.readouterr().out
    tool.main(["--root", str(root), "migrate", str(full), str(partial), "--apply"])
    out = capsys.readouterr().out
    records = CharacterMemory(root).characters()
    assert len(records) == 1 and records[0]["identity"]["gender"] == "male" and "확정값 없음" in out
    assert records[0]["migrated_from"][0]["kind"] == "path_preferences"
    assert character_preferences.load_character_gender(full) == "male"          # 경로 기록은 그대로


def test_unlink_and_remove_from_the_command_line(tmp_path, capsys):
    root = tmp_path / "memory"
    memory = CharacterMemory(root)
    original, wrong = tmp_path / "a.png", tmp_path / "wrong.png"
    original.write_bytes(b"a")
    wrong.write_bytes(b"wrong")
    features = dict(hair_length=None, hair_style=None, hair_color=None, eye_color=None, head_accessories=None, skin=None, distinctive=None)
    record = memory.create(original, gender="male", body_parts=(), features=features)
    memory.link(record["character_id"], wrong, user_confirmed=True, evidence=dict(kind="user_choice"))
    assert tool.main(["--root", str(root), "unlink", record["character_id"], str(wrong), "--reason", "다른 캐릭터"]) == 0
    assert memory.find_image(image_sha256(wrong)) is None
    assert tool.main(["--root", str(root), "unlink", record["character_id"], str(original), "--reason", "x"]) == 1   # 기준 원본은 거부
    assert tool.main(["--root", str(root), "remove", record["character_id"], "--reason", "정리"]) == 0
    assert memory.characters() == []


def test_list_says_when_nothing_is_saved(tmp_path, capsys):
    assert tool.main(["--root", str(tmp_path / "memory"), "list"]) == 0
    assert "저장된 캐릭터 0명" in capsys.readouterr().out


def test_no_command_means_list(tmp_path, capsys):
    assert tool.main(["--root", str(tmp_path / "memory")]) == 0
    assert "저장된 캐릭터 0명" in capsys.readouterr().out
