"""작업실 자유 생성 입력·승인 확인창. 생성 함수는 free_generation_pipeline에 있다."""
from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPlainTextEdit, QCheckBox, QPushButton
from genai_lab.character_features import tag_features, features_text, parse_features_text, weak_note
from genai_lab.character_preferences import load_character_features, load_character_gender, load_character_body_parts
from genai_lab.character_verification import measure_result_scores
from genai_lab.free_generation import check_request
from genai_lab.outfit_support import support_summary, support_warnings


def analyze_base(path, runtime):
    """옷 참조를 요구하지 않고 선택한 기준 한 장만 CPU로 분석한다."""
    with Image.open(path) as image:
        scores = measure_result_scores(np.asarray(image.convert("RGB")), runtime) or {}
    return dict(source=str(path), character_scores=scores, groups=dict(fixed=[], appearance=[], body=[]))


def confirm_free_inputs(window, analysis):
    from genai_lab.studio_controller import picture, guessed_gender, guessed_body_parts
    source = Path(analysis["source"])
    dialog = QDialog(window)
    dialog.setWindowTitle("같은 캐릭터·의상으로 자유 생성")
    dialog.resize(650, 820)
    layout = QVBoxLayout(dialog)
    layout.addLayout(picture(source, "기준 이미지 · 이 의상을 유지합니다", 260, 260))
    kind = QComboBox()
    kind.setObjectName("free_base_kind")
    kind.addItem("원본 캔버스 (의상 교체 해결의 증거가 아님)", "original")
    kind.addItem("사용자가 승인·저장한 의상 편집본", "edited")
    if source.with_suffix(".review.json").is_file():
        kind.setCurrentIndex(1)
    layout.addWidget(kind)
    gender = QComboBox()
    gender.setObjectName("free_gender")
    for key, label in (("male", "남성"), ("female", "여성"), ("unspecified", "미지정")):
        gender.addItem(label, key)
    gender.setCurrentIndex(gender.findData(load_character_gender(source) or guessed_gender(analysis)))
    layout.addWidget(gender)
    part_boxes = {}
    saved = load_character_body_parts(source)
    chosen = guessed_body_parts(analysis) if saved is None else saved
    row = QHBoxLayout()
    for key, label in (("ears", "동물 귀"), ("tail", "꼬리"), ("wings", "날개")):
        box = QCheckBox(label)
        box.setObjectName("free_part_"+key)
        box.setChecked(key in chosen)
        part_boxes[key] = box
        row.addWidget(box)
    layout.addLayout(row)
    prefill = load_character_features(source)
    guessed = tag_features(analysis["character_scores"])
    layout.addWidget(QLabel("특징표: 자동 분석/저장값을 확인하고 수정해 주세요. '없음'은 없다고 확정, '모름'은 확인 불가로 기록합니다."))
    features = QPlainTextEdit()
    features.setObjectName("free_features")
    features.setPlainText(features_text({k: dict(values=v) for k, v in prefill.items()} if prefill else guessed))
    features.setFixedHeight(130)
    layout.addWidget(features)
    layout.addWidget(QLabel(weak_note(guessed)))
    request = QPlainTextEdit()
    request.setObjectName("free_request")
    request.setPlaceholderText("자세·구도·시점·배경·상황만 요청. 비워 두면 고정 목록에서 자세를 선택합니다.")
    request.setFixedHeight(65)
    layout.addWidget(request)
    support = QLabel()
    support.setWordWrap(True)
    layout.addWidget(support)
    confirm = QCheckBox("성인 캐릭터이며 특징표·부위와 연출 범위를 확인했습니다. 외형·의상을 바꾸지 않습니다.")
    confirm.setObjectName("free_confirm")
    layout.addWidget(confirm)
    note = QLabel("약 30분, 자동 재시도 한 번이면 약 1시간. 같은 기준의 머리 크롭도 참조하며 동일성 보장은 없습니다.")
    note.setWordWrap(True)
    layout.addWidget(note)
    hint = QLabel()
    hint.setWordWrap(True)
    layout.addWidget(hint)
    start = QPushButton("자유 생성 시작")
    back = QPushButton("취소")
    parsed = {}
    def refresh():
        parsed.clear()
        try:
            parsed["features"] = parse_features_text(features.toPlainText())
            checked = check_request(request.toPlainText())
            if not checked["ok"]:
                raise ValueError(checked["reason"])
            parsed["valid"] = True
            hint.setText("결과는 자동 승인하지 않습니다. 같은 캐릭터·의상·연출·심한 이상을 다시 확인합니다.")
        except ValueError as error:
            hint.setText(str(error))
        support.setText(support_summary(support_warnings(analysis["character_scores"], tuple(k for k,v in part_boxes.items() if v.isChecked())))
                        + " 요청한 앉은 자세는 허용됩니다. 경고는 입력 근거일 뿐입니다.")
        start.setEnabled(confirm.isChecked() and bool(parsed.get("valid")))
    for box in part_boxes.values():
        box.toggled.connect(refresh)
    confirm.toggled.connect(refresh)
    features.textChanged.connect(refresh)
    request.textChanged.connect(refresh)
    refresh()
    start.clicked.connect(dialog.accept)
    back.clicked.connect(dialog.reject)
    layout.addWidget(start)
    layout.addWidget(back)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    refresh()
    if not confirm.isChecked() or not parsed.get("valid"):
        return None
    return dict(base_kind=kind.currentData(), gender=gender.currentData(), body_parts=tuple(k for k,v in part_boxes.items() if v.isChecked()),
                character_features=parsed["features"], request_text=request.toPlainText(), confirmed=True, adult_confirmed=True)


def confirm_free_result(window, results):
    from genai_lab.studio_controller import picture
    dialog = QDialog(window)
    dialog.setWindowTitle("자유 생성 결과 확인")
    layout = QVBoxLayout(dialog)
    trace = __import__("json").loads(Path(results.outcome["trace"]).read_text(encoding="utf-8"))
    row = QHBoxLayout()
    row.addLayout(picture(Path(trace["input_facts"]["base"].get("snapshot") or trace["input_facts"]["base"]["path"]), "기준", 250, 350))
    row.addLayout(picture(results.current_path, "생성 결과", 250, 350))
    layout.addLayout(row)
    summary = QLabel("자동 특징 관찰: " + str(results.outcome["automatic_features"].get("summary", {})))
    summary.setWordWrap(True)
    layout.addWidget(summary)
    checks = {}
    for key, label in (("character", "같은 캐릭터입니다(얼굴·머리·성별·귀/꼬리/날개의 개수·형태·색·연결 확인)."),
                       ("garment", "기준과 같은 의상입니다(옷 종류·색·무늬·세부 확인)."),
                       ("request", "요청하거나 선택된 자세·구도·시점·배경·상황이 반영됐습니다."),
                       ("exposure", "심한 이상과 원치 않는 노출·아동 외형이 없습니다.")):
        box = QCheckBox(label)
        box.setObjectName("free_review_"+key)
        checks[key] = box
        layout.addWidget(box)
    layout.addWidget(QLabel("실제 연출 조건: " + next(c["text"] for c in trace["conditions"] if c["key"] == "request")))
    accept = QPushButton("이 결과 사용")
    accept.setEnabled(False)
    for box in checks.values():
        box.toggled.connect(lambda: accept.setEnabled(all(c.isChecked() for c in checks.values())))
    accept.clicked.connect(dialog.accept)
    back = QPushButton("다시 비교하기")
    back.clicked.connect(dialog.reject)
    layout.addWidget(accept)
    layout.addWidget(back)
    return {k:v.isChecked() for k,v in checks.items()} if dialog.exec() == QDialog.DialogCode.Accepted else None
