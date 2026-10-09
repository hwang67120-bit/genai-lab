"""피부 톤 화면: 표시 여부 확인 → 저장값 선택 → 견본 안내 → 사용자 선택 반환."""
from PySide6.QtWidgets import QLabel, QComboBox, QHBoxLayout
from genai_lab.character_preferences import load_character_skin_tone
from genai_lab.skin_tone import default_skin_choice, observe_skin_tone, SkinToneChoice

CHOICES = (("어두운 피부 (dark skin)", "dark"),
           ("햇볕에 탄 피부 (tan)", "tan"), ("지정 안 함", "unspecified"))


def skin_tone_field(layout, analysis, settings, preferences=None):
    """확인 화면에 피부 톤 항목을 만들고, 현재 선택을 읽는 함수를 반환한다."""
    field = prepare_skin_field(analysis, settings)
    if field is None:
        return lambda: SkinToneChoice()

    default, trigger, observation = field
    saved = load_character_skin_tone(analysis["source"], preferences)
    combo = add_skin_choices(layout, saved or default)
    add_skin_observation(layout, observation, trigger, saved)

    def read_selection():
        measurement = {**observation, "saved_choice": saved, "trigger": trigger}
        return SkinToneChoice(combo.currentData(), measurement)

    return read_selection


def prepare_skin_field(analysis, settings):
    """태그 또는 측정 결과로 표시 여부만 정한다. 추천으로 선택값을 바꾸지 않는다."""
    tags = analysis["groups"]["appearance"]
    default = default_skin_choice(tags)
    observation = observe_skin_tone(analysis["directory"], settings)

    if default is not None:
        return default, "tags", observation
    if observation["status"] != "measured":
        return None
    if not observation["recommend_tan"]:
        return None
    return "unspecified", "measured_tan", observation


def add_skin_choices(layout, selected):
    layout.addWidget(QLabel("피부 톤 확인"))
    combo = QComboBox()
    combo.setObjectName("skin_tone_choice")
    for text, value in CHOICES:
        combo.addItem(text, value)
    combo.setCurrentIndex(combo.findData(selected))
    layout.addWidget(combo)
    return combo


def add_skin_observation(layout, observation, trigger, saved):
    row = QHBoxLayout()
    if observation["status"] == "measured":
        row.addWidget(skin_swatch(observation["rgb"]))

    notice = QLabel(skin_notice(observation, trigger, saved))
    notice.setObjectName("skin_tone_recommendation")
    notice.setWordWrap(True)
    row.addWidget(notice)
    layout.addLayout(row)


def skin_swatch(rgb):
    color = QLabel()
    color.setObjectName("skin_tone_swatch")
    color.setFixedSize(40, 25)
    color.setStyleSheet("background: rgb(%d,%d,%d); border: 1px solid #888;" % tuple(rgb))
    return color


def skin_notice(observation, trigger, saved):
    """측정 상태·항목 표시 이유·저장값을 안내 문장으로 바꾼다."""
    if observation["status"] != "measured":
        text = "볼 색을 측정하지 못했습니다. 추천 없이 직접 선택할 수 있습니다."
    else:
        text = measured_skin_notice(observation, trigger)
    if saved is not None:
        text = "이 캐릭터에 저장한 선택을 불러왔습니다. " + text
    return text


def measured_skin_notice(observation, trigger):
    suffix = "추천은 임시 기준이며, 선택값은 자동으로 바뀌지 않습니다."
    if trigger == "measured_tan":
        return ("원본 피부가 황갈색으로 측정됐습니다. 피부 태그가 없어 밝게 그려질 수 있습니다. tan 추천 · "
                + suffix)
    if observation["recommend_tan"]:
        return "밝은 황갈색: tan 추천 · " + suffix
    return "원본 양 볼에서 측정한 색입니다. " + suffix
