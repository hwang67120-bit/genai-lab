"""기준 이미지 파일별 사용자 성별·피부 톤 선택. 외형 분석으로 값을 만들지 않는다."""
import hashlib
import os
from pathlib import Path
from PySide6.QtCore import QSettings
from genai_lab.reference_tag_policy import validate_character_gender

SKIN_TONE_VALUES = ("dark", "tan", "unspecified")
BODY_PART_VALUES = ("ears", "tail", "wings")


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


def load_character_body_parts(source_path, settings=None):
    """사용자가 확인한 동물 귀·꼬리·날개(D-111). 확인한 적이 없으면 None, 없다고 확인했으면 ()."""
    store = settings if settings is not None else QSettings("GenAILab", "CharacterPreferences")
    value = store.value(preference_key(source_path, "character_body_parts"))
    if not isinstance(value, str):
        return None
    parts = tuple(p for p in value.split(",") if p)
    return parts if all(p in BODY_PART_VALUES for p in parts) else None


def save_character_body_parts(source_path, parts, settings=None):
    parts = tuple(parts)
    if any(p not in BODY_PART_VALUES for p in parts) or len(set(parts)) != len(parts):
        raise ValueError("캐릭터 부위 선택값 오류")
    store = settings if settings is not None else QSettings("GenAILab", "CharacterPreferences")
    store.setValue(preference_key(source_path, "character_body_parts"), ",".join(p for p in BODY_PART_VALUES if p in parts))
    store.sync()
    if store.status() != QSettings.Status.NoError:
        raise OSError("캐릭터 부위 선택을 저장하지 못했습니다.")


def load_character_features(source_path, settings=None):
    """사용자가 확정한 캐릭터 특징표(태그 항목, D-120). 확인한 적이 없으면 None."""
    import json
    from genai_lab.character_features import DISTINCTIVE, TAG_ITEMS
    store = settings if settings is not None else QSettings("GenAILab", "CharacterPreferences")
    value = store.value(preference_key(source_path, "character_features"))
    if not isinstance(value, str):
        return None
    try:
        items = json.loads(value)
    except ValueError:
        return None
    if (not isinstance(items, dict) or set(items) - {DISTINCTIVE} != set(TAG_ITEMS)
            or not all(v is None or isinstance(v, list) for v in items.values())):
        return None
    keys = (*TAG_ITEMS, DISTINCTIVE)          # 고유 형태 메모가 없던 저장값은 모름(None)
    return {k: None if items.get(k) is None else [str(x) for x in items[k]] for k in keys}   # None = 모름


def save_character_features(source_path, items, settings=None):
    import json
    from genai_lab.character_features import DISTINCTIVE, TAG_ITEMS
    if set(items) - {DISTINCTIVE} != set(TAG_ITEMS) or not all(v is None or (isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v))
                                                               for v in items.values()):
        raise ValueError("캐릭터 특징표 값 오류")
    store = settings if settings is not None else QSettings("GenAILab", "CharacterPreferences")
    keys = (*TAG_ITEMS, DISTINCTIVE)
    store.setValue(preference_key(source_path, "character_features"),
                   json.dumps({k: None if items.get(k) is None else list(items[k]) for k in keys}, ensure_ascii=False))
    store.sync()
    if store.status() != QSettings.Status.NoError:
        raise OSError("캐릭터 특징표를 저장하지 못했습니다.")


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
