"""기준 이미지 파일별 사용자 성별·피부 톤 선택. 외형 분석으로 값을 만들지 않는다."""
import hashlib
import os
from pathlib import Path
from PySide6.QtCore import QSettings
from genai_lab.reference_tag_policy import validate_character_gender

SKIN_TONE_VALUES = ("dark", "tan", "unspecified")


def preference_key(source_path, kind="character_gender"):
    canonical = os.path.normcase(str(Path(source_path).resolve()))
    return kind + "/" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_character_gender(source_path, settings=None):
    store = settings if settings is not None else QSettings("GenAILab", "CharacterPreferences")
    value = store.value(preference_key(source_path))
    return value if value in ("male", "female", "unspecified") else None


def save_character_gender(source_path, gender, settings=None):
    validate_character_gender(gender)
    store = settings if settings is not None else QSettings("GenAILab", "CharacterPreferences")
    store.setValue(preference_key(source_path), gender)
    store.sync()
    if store.status() != QSettings.Status.NoError:
        raise OSError("캐릭터 성별 선택을 저장하지 못했습니다.")


def load_character_skin_tone(source_path, settings=None):
    """사용자가 이전에 확인한 값만 불러온다. 측정 추천값은 불러오지 않는다."""
    store = settings if settings is not None else QSettings("GenAILab", "CharacterPreferences")
    value = store.value(preference_key(source_path, "character_skin_tone"))
    return value if value in SKIN_TONE_VALUES else None


def save_character_skin_tone(source_path, value, settings=None):
    if value not in SKIN_TONE_VALUES:
        raise ValueError("피부 톤 선택값 오류")
    store = settings if settings is not None else QSettings("GenAILab", "CharacterPreferences")
    store.setValue(preference_key(source_path, "character_skin_tone"), value)
    store.sync()
    if store.status() != QSettings.Status.NoError:
        raise OSError("캐릭터 피부 톤 선택을 저장하지 못했습니다.")
