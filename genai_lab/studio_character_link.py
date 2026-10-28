"""확인 창의 '캐릭터 연결' 줄(H1-B, genai_lab/character_memory.py). 자유 생성·옷 바꾸기 확인 창이 같이 쓴다.

- 이미 연결된 그림: 저장된 캐릭터의 정체성으로 미리 채운다. 잘못 연결됐으면 이 줄에서 해제한다(기준 원본은 해제 불가).
- 연결 안 된 그림: 기본은 '새 캐릭터로 등록'. 다른 캐릭터에 잇는 것은 사람이 고르고 '같은 캐릭터' 체크를 해야만 시작할 수 있다.
  예전 승인 기록이 가리키는 캐릭터는 제안으로만 보인다(체크 필요).
- 미리 채움은 성별·부위·특징표에만 쓰고, 사람이 이미 고친 칸은 덮어쓰지 않는다. 요청 문장·기준 종류·옷은 건드리지 않는다.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel


SOURCE_LABELS = {"memory": "저장된 캐릭터 기억", "path_record": "이 파일에 저장된 값", None: "자동 분석"}


def link_options(info):
    """(표시 문장, 선택 값) 목록. 첫 항목이 기본값이다."""
    options = []
    linked = info.get("linked")
    if linked:
        options.append((f"저장된 캐릭터 '{linked['name']}'에 연결된 그림입니다", dict(action="keep", character_id=linked["character_id"])))
        if linked["role"] != "canonical":
            options.append(("연결 해제 — 이 그림을 새 캐릭터로 등록", dict(action="unlink_create", character_id=linked["character_id"])))
            options.append(("연결 해제 — 연결 없이 이번만 사용", dict(action="unlink_none", character_id=linked["character_id"])))
        return options
    suggestion = info.get("suggestion")
    if suggestion:
        options.append((f"'{suggestion['name']}'의 승인 결과로 기록된 그림입니다 — 같은 캐릭터로 연결",
                        dict(action="link", character_id=suggestion["character_id"], evidence=suggestion["evidence"])))
    options.append(("새 캐릭터로 등록" + (" (이 파일에 저장된 값을 가져옵니다)" if info.get("source") == "path_record" else ""),
                    dict(action="create")))
    for other in info.get("others") or ():
        if suggestion and other["character_id"] == suggestion["character_id"]:
            continue
        options.append((f"저장된 캐릭터 '{other['name']}'에 연결", dict(action="link", character_id=other["character_id"],
                                                                     evidence=dict(kind="user_choice"))))
    options.append(("연결하지 않음(이번만 사용, 기억에 저장하지 않음)", dict(action="none")))
    return options


class CharacterLinkRow:
    """확인 창에 붙는 연결 줄. on_identity(정체성 또는 None)는 연결 대상이 바뀔 때 불린다(None = 대상 없음 → 원래 미리 채움)."""

    def __init__(self, layout, info, memory, prefix, on_identity, on_change):
        self.info, self.memory, self.on_identity = info, memory, on_identity
        layout.addWidget(QLabel("캐릭터 연결(같은 캐릭터의 확정 특징을 다음에도 씁니다. 의상·요청은 저장하지 않습니다)"))
        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setObjectName(prefix + "_character_link")
        for label, value in link_options(info):
            self.combo.addItem(label, value)
        row.addWidget(self.combo, 1)
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(72, 96)
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.thumbnail)
        layout.addLayout(row)
        self.confirm = QCheckBox("이 그림은 위 캐릭터와 같은 캐릭터입니다(연결 확인)")
        self.confirm.setObjectName(prefix + "_character_link_confirm")
        layout.addWidget(self.confirm)
        self.note = QLabel()
        self.note.setObjectName(prefix + "_character_link_note")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.combo.currentIndexChanged.connect(self.changed)
        self.combo.currentIndexChanged.connect(on_change)
        self.confirm.toggled.connect(on_change)
        self.changed(apply=False)

    def target(self):
        value = self.combo.currentData() or {}
        return value.get("character_id") if value.get("action") in ("keep", "link") else None

    def changed(self, *_, apply=True):
        value = self.combo.currentData() or {}
        self.confirm.setVisible(value.get("action") == "link")
        target = self.target()
        canonical = next((c["canonical"] for c in [self.info.get("linked") or {}, *(self.info.get("others") or ())]
                          if c.get("character_id") == target), None) if target else None
        pixmap = QPixmap(canonical) if canonical and Path(canonical).is_file() else QPixmap()
        self.thumbnail.setPixmap(pixmap.scaled(72, 96, Qt.AspectRatioMode.KeepAspectRatio) if not pixmap.isNull() else pixmap)
        source = "memory" if target else self.info.get("source") if self.info.get("source") != "memory" else None
        self.note.setText("미리 채움: " + SOURCE_LABELS[source] + ". 사람이 고친 칸은 바꾸지 않습니다."
                          + (" 연결 대상의 기준 그림을 오른쪽에서 확인하고 체크해 주세요." if value.get("action") == "link" else "")
                          + (" 이 그림의 연결만 풀고 저장된 캐릭터는 그대로 둡니다." if value.get("action", "").startswith("unlink") else ""))
        if apply:
            self.on_identity(self.memory.identity(target) if target else None)

    def ready(self):
        return (self.combo.currentData() or {}).get("action") != "link" or self.confirm.isChecked()

    def choice(self):
        value = dict(self.combo.currentData() or dict(action="none"))
        if value["action"] == "link":
            value["confirmed"] = self.confirm.isChecked()
        return value


class Prefill:
    """정체성 칸(성별 콤보·부위 체크·특징표 글 상자)에 코드가 값을 넣고, 사람이 고친 칸은 다시 덮어쓰지 않는다."""

    def __init__(self, gender, part_boxes, features, features_text, fallback):
        self.gender, self.part_boxes, self.features, self.features_text, self.fallback = gender, part_boxes, features, features_text, fallback
        self.edited, self.setting = set(), False
        gender.currentIndexChanged.connect(lambda *_: self.mark("gender"))
        for box in part_boxes.values():
            box.toggled.connect(lambda *_: self.mark("body_parts"))
        features.textChanged.connect(lambda: self.mark("features"))

    def mark(self, key):
        if not self.setting:
            self.edited.add(key)

    def apply(self, identity):
        """identity: dict(gender, body_parts, features) 일부 또는 None(→ 처음 미리 채움 값)."""
        values = dict(self.fallback)
        values.update({k: v for k, v in (identity or {}).items() if v is not None})
        self.setting = True
        try:
            if "gender" not in self.edited and values.get("gender"):
                self.gender.setCurrentIndex(self.gender.findData(values["gender"]))
            if "body_parts" not in self.edited and values.get("body_parts") is not None:
                for key, box in self.part_boxes.items():
                    box.setChecked(key in values["body_parts"])
            if "features" not in self.edited and values.get("features") is not None:
                self.features.setPlainText(self.features_text(values["features"]))
        finally:
            self.setting = False
