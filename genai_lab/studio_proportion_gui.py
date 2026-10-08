"""Head selection and review only. No model work or generation on the GUI thread."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox
from genai_lab.qwen_tail_gui import TailCropCanvas
from genai_lab.studio_proportion import restore_head


class HeadCanvas(TailCropCanvas):
    def __init__(self, source, nose, parent=None):
        super().__init__(source, parent)
        self.face_point = tuple(nose)
        self.setAccessibleName("머리 영역 드래그 · 얼굴 중심은 오른쪽 클릭")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self.image_rect().contains(event.position()):
            rect = self.image_rect()
            self.face_point = (round((event.position().x()-rect.x()) / rect.width() * self.pixmap.width()),
                               round((event.position().y()-rect.y()) / rect.height() * self.pixmap.height()))
            self.selectionChanged.emit()
            self.update()
        else:
            super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = self.image_rect()
        x = rect.x() + self.face_point[0] / self.pixmap.width() * rect.width()
        y = rect.y() + self.face_point[1] / self.pixmap.height() * rect.height()
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#1964d8"),2))
        painter.drawEllipse(int(x)-4,int(y)-4,8,8)


def select_head_region(window, pose):
    dialog = QDialog(window)
    dialog.setWindowTitle("원본 비율 참고 · 머리 범위 지정")
    dialog.resize(650,760)
    layout = QVBoxLayout(dialog)
    note = QLabel("머리카락과 얼굴을 사각형으로 감싸 주세요. 귀·뿔·옷이 섞이지 않았는지 확인합니다.\n"
                  "파란 점은 얼굴을 찾는 기준입니다. 얼굴 중심이 아니면 오른쪽 클릭으로 옮겨 주세요.")
    note.setWordWrap(True)
    layout.addWidget(note)
    canvas = HeadCanvas(pose["normalized_file"], pose["nose"])
    layout.addWidget(canvas,1)
    status = QLabel("아직 머리 범위를 선택하지 않았습니다.")
    status.setWordWrap(True)
    layout.addWidget(status)
    accept = QPushButton("이 범위로 윤곽 미리보기")
    accept.setEnabled(False)
    def changed():
        box = canvas.box
        point = canvas.face_point
        valid = bool(box and box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])
        accept.setEnabled(valid)
        status.setText("범위를 선택했습니다. 윤곽을 만든 뒤 다시 확인합니다." if valid
                       else "머리 사각형 안에 파란 얼굴 중심점이 있어야 합니다.")
    canvas.selectionChanged.connect(changed)
    accept.clicked.connect(dialog.accept)
    cancel = QPushButton("생성 취소")
    cancel.clicked.connect(dialog.reject)
    layout.addWidget(accept)
    layout.addWidget(cancel)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return {"box": canvas.box, "face_point": canvas.face_point}


def review_head_outline(window, pose, draft):
    from genai_lab.studio_controller import picture
    dialog = QDialog(window)
    dialog.setWindowTitle("원본 비율 참고 · 윤곽 확인")
    dialog.resize(850,700)
    layout = QVBoxLayout(dialog)
    note = QLabel("원본의 자세와 확인한 머리 윤곽으로 2단계 생성합니다. 생성량은 2배입니다.\n"
                  "현재 검증은 캐릭터 1명 기준이며, 다른 캐릭터의 비율 유지와 의상 색은 보장되지 않습니다.")
    note.setWordWrap(True)
    layout.addWidget(note)
    row = QHBoxLayout()
    head = restore_head(draft)
    for path, label in ((draft["preview_file"], "빨간 선: 확인할 머리 윤곽"),
                        (pose["overlay_file"], "원본 자세 검출 결과"), (head.contour_file, "생성에 전달할 머리 윤곽")):
        row.addLayout(picture(path,label,250,410))
    layout.addLayout(row)
    warning_names = {"K3":"팔·다리 길이가 비정상적으로 검출됐을 수 있습니다",
        "K4":"다리가 겹치거나 접힌 위치를 확인해 주세요", "K5":"좌우 길이가 크게 다르게 검출됐습니다",
        "K6":"몸통 폭이나 길이를 확인해 주세요", "K7":"두 다리가 겹쳐 검출됐을 수 있습니다",
        "GUESS":"일부 관절의 검출 확신이 낮습니다", "HIGH_ANGLE":"위에서 내려다보는 구도를 확인해 주세요"}
    warnings = [warning_names.get(code,"추가로 확인할 검출 결과가 있습니다") for code in pose["decision"]["warnings"]]
    if warnings:
        notice = QLabel("원본 자세 확인 필요: " + ", ".join(warnings) + " — 잘못 검출된 부위가 있으면 취소해 주세요.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
    error = draft.get("geometry_error")
    if error:
        notice = QLabel("이 윤곽으로는 생성할 수 없습니다: " + error)
        notice.setWordWrap(True)
        layout.addWidget(notice)
    check = QCheckBox("머리카락·얼굴 윤곽과 귀·뿔의 포함 범위, 원본 자세를 확인했습니다.")
    check.setObjectName("head_outline_confirmed")
    layout.addWidget(check)
    accept = QPushButton("확인한 비율로 4장 만들기")
    accept.setEnabled(False)
    check.toggled.connect(lambda checked: accept.setEnabled(checked and not error))
    accept.clicked.connect(dialog.accept)
    retry = QPushButton("머리 범위 다시 지정")
    retry.clicked.connect(lambda: dialog.done(2))
    cancel = QPushButton("생성 취소")
    cancel.clicked.connect(dialog.reject)
    for button in (accept,retry,cancel):
        layout.addWidget(button)
    answer = dialog.exec()
    return "confirm" if answer == QDialog.DialogCode.Accepted else "retry" if answer == 2 else None
