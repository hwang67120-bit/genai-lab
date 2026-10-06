"""Human tail selection and comparison only; generation is owned by StudioController."""
import math
import os
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget,
    QPushButton, QComboBox, QLineEdit, QCheckBox, QMessageBox, QFileDialog)
from genai_lab.qwen_tail_edit import (LIMITATIONS, SPIRAL_EXAMPLE, assemble_tail_prompt,
    crop_bytes, validate_box)
from genai_lab.qwen_pose_settings import QwenPoseSettings


def screen_box_to_source(start, end, image_rect, source_size):
    """Inverse of centered KeepAspectRatio display; outer crop edges use floor/ceil."""
    x, y, w, h = image_rect
    if w <= 0 or h <= 0:
        raise ValueError("이미지가 표시되지 않았습니다.")
    points = [(min(max((px-x)/w, 0), 1)*source_size[0],
               min(max((py-y)/h, 0), 1)*source_size[1]) for px, py in (start, end)]
    box = (math.floor(min(p[0] for p in points)), math.floor(min(p[1] for p in points)),
           math.ceil(max(p[0] for p in points)), math.ceil(max(p[1] for p in points)))
    validate_box(box, source_size)
    return box


class TailCropCanvas(QWidget):
    selectionChanged = Signal()

    def __init__(self, source, parent=None):
        super().__init__(parent)
        # Decode with the same pixel orientation as crop_bytes (no EXIF auto-rotation).
        import io
        with Image.open(source) as im, im.convert("RGBA") as rgba, Image.new("RGBA", im.size, "white") as white:
            with Image.alpha_composite(white, rgba) as rgb:
                data = io.BytesIO()
                rgb.save(data, format="PNG")
        self.pixmap = QPixmap()
        self.pixmap.loadFromData(data.getvalue())
        if self.pixmap.isNull():
            raise ValueError("원본 캐릭터를 표시할 수 없습니다.")
        self.box = None
        self.start = None
        self.end = None
        self.dragging = False
        self.setMinimumSize(350, 420)
        self.setAccessibleName("원본 캐릭터에서 꼬리 영역 드래그")

    def image_rect(self):
        scale = min(self.width()/self.pixmap.width(), self.height()/self.pixmap.height())
        w, h = self.pixmap.width()*scale, self.pixmap.height()*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        rect = self.image_rect()
        painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        painter.setPen(QPen(QColor("#df3d37"), 2))
        if self.dragging and self.start is not None and self.end is not None:
            painter.drawRect(QRectF(self.start, self.end).normalized().intersected(rect))
        elif self.box is not None:
            x0, y0, x1, y1 = self.box
            sx, sy = rect.width()/self.pixmap.width(), rect.height()/self.pixmap.height()
            painter.drawRect(QRectF(rect.x()+x0*sx, rect.y()+y0*sy, (x1-x0)*sx, (y1-y0)*sy))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.image_rect().contains(event.position()):
            self.dragging = True
            self.start = self.end = event.position()
            self.box = None
            self.selectionChanged.emit()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.dragging or event.button() != Qt.MouseButton.LeftButton:
            return
        self.dragging = False
        self.end = event.position()
        r = self.image_rect()
        try:
            self.box = screen_box_to_source((self.start.x(), self.start.y()), (self.end.x(), self.end.y()),
                (r.x(), r.y(), r.width(), r.height()), (self.pixmap.width(), self.pixmap.height()))
        except ValueError:
            self.box = None
        self.selectionChanged.emit()
        self.update()


class TailInputDialog(QDialog):
    def __init__(self, source, source_sha256, parent=None, *, settings_path=""):
        super().__init__(parent)
        self.source, self.source_sha256 = Path(source), source_sha256
        self.settings_path = settings_path or os.environ.get("GENAI_QWEN_SETTINGS", "")
        self.settings = None
        self.setWindowTitle("꼬리 고치기")
        self.resize(900, 790)
        layout = QVBoxLayout(self)
        notice = QLabel("원래 캐릭터에서 꼬리만 감싸고, 옷·다리는 최대한 빼 주세요.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        row = QHBoxLayout()
        self.canvas = TailCropCanvas(source)
        row.addWidget(self.canvas, 2)
        self.preview = QLabel("지정한 꼬리 영역")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(220, 220)
        row.addWidget(self.preview, 1)
        layout.addLayout(row, 1)
        self.pattern = QComboBox()
        self.pattern.setObjectName("tail_pattern")
        self.pattern.addItem("무늬를 선택해 주세요", None)
        self.pattern.addItem("무늬 있음", True)
        self.pattern.addItem("무늬 없음", False)
        layout.addWidget(self.pattern)
        self.tip = QLineEdit()
        self.tip.setObjectName("tail_tip")
        self.tip.setPlaceholderText("끝 모양 영어 한 문장 (선택, 비워도 됩니다)")
        layout.addWidget(self.tip)
        example = QLabel("예: " + SPIRAL_EXAMPLE)
        example.setWordWrap(True)
        layout.addWidget(example)
        from genai_lab.tail_recognition import settings_path
        self.recognition_settings_path = str(settings_path())
        self.recognition_enabled = QCheckBox("그림을 읽어 색·형태 설명 보강 (확인 후 전달)")
        self.recognition_enabled.setChecked(settings_path().is_file())
        layout.addWidget(self.recognition_enabled)
        self.recognition_enabled.toggled.connect(self.changed)
        recognition_config = QPushButton("이미지 인식 실행 환경 선택")
        recognition_config.clicked.connect(self.choose_recognition_settings)
        layout.addWidget(recognition_config)
        config = QPushButton("편집 실행 환경 설정 파일 선택")
        config.clicked.connect(self.choose_settings)
        layout.addWidget(config)
        self.confirm = QCheckBox("꼬리 영역과 무늬·끝 모양 설명을 확인했습니다.")
        layout.addWidget(self.confirm)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        limits = QLabel(LIMITATIONS)
        limits.setWordWrap(True)
        layout.addWidget(limits)
        buttons = QHBoxLayout()
        back = QPushButton("돌아가기")
        back.clicked.connect(self.reject)
        self.run = QPushButton("이 꼬리를 참고해서 고치기")
        self.run.setEnabled(False)
        self.run.clicked.connect(self.submit)
        buttons.addWidget(back)
        buttons.addWidget(self.run)
        layout.addLayout(buttons)
        self.canvas.selectionChanged.connect(self.changed)
        self.pattern.currentIndexChanged.connect(self.changed)
        self.tip.textChanged.connect(self.changed)
        self.confirm.toggled.connect(self.validate)
        self.validate()

    def choose_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Qwen 실행 환경 설정", self.settings_path, "JSON (*.json)")
        if path:
            self.settings_path = path
            self.changed()

    def choose_recognition_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "이미지 인식 실행 환경", self.recognition_settings_path, "JSON (*.json)")
        if path:
            self.recognition_settings_path = path
            self.recognition_enabled.setChecked(True)
            self.changed()

    def changed(self):
        self.confirm.setChecked(False)
        self.validate()

    def validate(self):
        try:
            if self.canvas.box is None:
                self.preview.clear()
                self.preview.setText("지정한 꼬리 영역")
                raise ValueError("원본 그림에서 꼬리 영역을 드래그해 주세요.")
            data = crop_bytes(self.source, self.canvas.box, self.source_sha256)
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            self.preview.setPixmap(pixmap.scaled(260, 300, Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))
            assemble_tail_prompt(self.pattern.currentData(), self.tip.text())
            if not self.settings_path:
                raise ValueError("처음 실행할 때 편집 실행 환경 설정 파일을 선택해 주세요.")
        except (ValueError, OSError) as error:
            self.run.setEnabled(False)
            self.hint.setText(str(error))
            return
        self.hint.setText("확인 후 실행하면 고치기 전 결과는 그대로 보관됩니다.")
        self.run.setEnabled(self.confirm.isChecked())

    def submit(self):
        self.validate()
        if not self.run.isEnabled():
            return
        try:
            self.settings = QwenPoseSettings.load(self.settings_path)
            if self.recognition_enabled.isChecked():
                from genai_lab.tail_recognition import RecognitionSettings
                RecognitionSettings.load(self.recognition_settings_path)
        except (ValueError, OSError, TypeError) as error:
            QMessageBox.warning(self, "실행 환경 확인", str(error))
            return
        self.accept()


def review_tail_result(window, source, before, after):
    from genai_lab.studio_controller import picture
    dialog = QDialog(window)
    dialog.setWindowTitle("꼬리 편집 결과 비교")
    dialog.resize(1030, 720)
    layout = QVBoxLayout(dialog)
    row = QHBoxLayout()
    for path, title in ((source, "원래 캐릭터"), (before, "고치기 전"), (after, "고친 뒤")):
        row.addLayout(picture(path, title, 315, 450))
    layout.addLayout(row)
    checks = {}
    for key, text in (("character", "캐릭터와 자세가 유지됐습니다."),
                      ("garment", "옷의 모양과 색이 유지됐습니다."),
                      ("exposure", "원치 않는 노출이나 잘못 그려진 부분이 없습니다."),
                      ("appendages", "꼬리·귀의 개수·종류·색이 원래 캐릭터와 같습니다.")):
        checks[key] = QCheckBox(text)
        layout.addWidget(checks[key])
    adopt = QPushButton("이 편집본 사용")
    adopt.setEnabled(False)
    for check in checks.values():
        check.toggled.connect(lambda: adopt.setEnabled(all(c.isChecked() for c in checks.values())))
    adopt.clicked.connect(dialog.accept)
    discard = QPushButton("고치기 전 결과 유지")
    discard.clicked.connect(dialog.reject)
    layout.addWidget(adopt)
    layout.addWidget(discard)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return {key: c.isChecked() for key, c in checks.items()}
    return None


def review_tail_recognition(window, spec, report):
    """No approval by inference: each visible observation can be excluded before confirmation."""
    from PySide6.QtWidgets import QScrollArea
    from genai_lab.studio_controller import picture
    from genai_lab.tail_recognition import FIELDS, LABELS, approve_observations
    dialog=QDialog(window)
    dialog.setWindowTitle("읽은 내용 확인 · 아직 편집하지 않았습니다")
    dialog.resize(940,780)
    layout=QVBoxLayout(dialog)
    notice=QLabel("그림에서 읽은 설명입니다. 틀린 항목은 체크를 빼 주세요. 가려져 확인할 수 없는 부분은 전달하지 않습니다. "
                 "설명을 전달해도 편집 결과가 반드시 보존되는 것은 아닙니다.")
    notice.setWordWrap(True)
    layout.addWidget(notice)
    images=QHBoxLayout()
    images.addLayout(picture(spec.image_path,"그대로 유지할 이미지",220,260))
    images.addLayout(picture(spec.crop_path,"참고할 꼬리",220,260))
    layout.addLayout(images)
    scroll=QScrollArea()
    scroll.setWidgetResizable(True)
    contents=QWidget(); fields_layout=QVBoxLayout(contents)
    checks={role:{} for role in FIELDS}
    for role in ("tail","basis"):
        fields_layout.addWidget(QLabel("꼬리에 반영할 특징" if role=="tail" else "현재 이미지에서 유지할 특징"))
        for field in FIELDS[role]:
            value=report["observations"][role]["fields"][field]
            omitted=(role=="tail" and ((field=="pattern" and not spec.pattern) or (field=="tip" and bool(spec.tip))))
            check=QCheckBox(LABELS[field])
            check.setObjectName("recognition_"+role+"_"+field)
            check.setChecked(value is not None and not omitted)
            check.setEnabled(value is not None and not omitted)
            fields_layout.addWidget(check)
            reason=report["observations"][role].get("unmeasured",{}).get(field,"모델이 확인하지 못함")
            text=QLabel((value or "확인 불가: "+reason)+(" — 직접 선택한 조건을 우선하므로 전달하지 않음" if omitted else ""))
            text.setWordWrap(True); fields_layout.addWidget(text)
            checks[role][field]=check
    scroll.setWidget(contents); layout.addWidget(scroll,1)
    confirmed=QCheckBox("선택한 설명을 확인했고 편집 지시문에 반영합니다.")
    confirmed.setObjectName("recognition_confirm")
    layout.addWidget(confirmed)
    start=QPushButton("확인한 설명으로 꼬리 고치기")
    start.setObjectName("recognition_apply")
    start.setEnabled(False)
    def changed():
        start.setEnabled(confirmed.isChecked() and any(c.isChecked() for role in checks.values() for c in role.values()))
    confirmed.toggled.connect(changed)
    for role in checks.values():
        for c in role.values(): c.toggled.connect(lambda: confirmed.setChecked(False))
    start.clicked.connect(dialog.accept)
    layout.addWidget(start)
    cancel=QPushButton("편집하지 않고 돌아가기")
    cancel.clicked.connect(dialog.reject); layout.addWidget(cancel)
    if dialog.exec()!=QDialog.DialogCode.Accepted: return None
    selected={role:[field for field,c in items.items() if c.isChecked()] for role,items in checks.items()}
    return approve_observations(report,selected)
