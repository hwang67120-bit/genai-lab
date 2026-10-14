from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialogButtonBox
from genai_lab.character_preferences import load_character_gender, save_character_gender
from genai_lab.clothing_reference_generation_review import ClothingReferenceGenerationDialog


def test_gender_persists_per_source_without_cross_character_leak(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    assert load_character_gender(a, settings) is None
    save_character_gender(a, "male", settings)
    reopened = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    assert load_character_gender(a, reopened) == "male"
    assert load_character_gender(b, reopened) is None
    save_character_gender(a, "unspecified", reopened)
    assert load_character_gender(a, reopened) == "unspecified"


def test_new_dialog_requires_choice_and_restores_explicit_choice():
    app = QApplication.instance() or QApplication([])
    dialog = ClothingReferenceGenerationDialog((768, 1344), ("jacket",))
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert dialog.character_gender is None and not ok.isEnabled()
    dialog.width_input.setValue(512)
    assert not ok.isEnabled()
    dialog.accept()
    assert dialog.result() == 0
    dialog.character_gender_input.setCurrentIndex(dialog.character_gender_input.findData("unspecified"))
    assert ok.isEnabled()
    dialog.close()
    restored = ClothingReferenceGenerationDialog((768, 1344), ("jacket",), previous_gender="male")
    assert restored.character_gender == "male"
    assert restored.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    restored.close()


def test_character_features_persist_per_source_and_reject_bad_shapes(tmp_path):
    from genai_lab.character_preferences import load_character_features, save_character_features
    import pytest
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    items = dict(hair_length=["long_hair"], hair_style=["twintails"], hair_color=["aqua_hair"], eye_color=["blue_eyes"],
                 head_accessories=[], skin=[])
    assert load_character_features(a, settings) is None
    save_character_features(a, items, settings)
    reopened = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    assert load_character_features(a, reopened) == items and load_character_features(b, reopened) is None
    with pytest.raises(ValueError):
        save_character_features(a, dict(hair_length=["x"]), settings)
