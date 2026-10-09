"""사용자 비교만 담당한다. 창을 닫으면 항상 원본을 유지한다."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QPushButton

NOTICE = "약 1분 이상 걸립니다 · 얼굴 참조로 캐릭터 세부를 유지하며 선명하게 다시 그립니다"


class FinishingComparisonDialog(QDialog):
    def __init__(self, before, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("고화질 마무리 전·후 비교")
        self.resize(790, 760)
        layout = QVBoxLayout(self)
        note = QLabel("눈색·머리 장식·체형을 비교해 주세요. 원본이 기본 선택이며, 선택 후에도 저장 전 확인이 필요합니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        images = QHBoxLayout()
        self.original = QRadioButton("마무리 전 사용 (원본)")
        self.finished = QRadioButton("마무리 후 사용")
        self.original.setChecked(True)
        box = info["finishing"].get("face_box")
        final_size = info["finishing"]["final_size"]
        for path, button in ((before, self.original), (info["product_file"], self.finished)):
            column = QVBoxLayout()
            pixmap = QPixmap(str(path))
            column.addWidget(self.image_label(pixmap, 350, 405))
            column.addWidget(QLabel("얼굴 확대"))
            if box:
                sx, sy = pixmap.width() / final_size[0], pixmap.height() / final_size[1]
                x0, y0, x1, y1 = [round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box)]
                column.addWidget(self.image_label(pixmap.copy(x0, y0, x1-x0, y1-y0), 250, 170))
            else:
                column.addWidget(QLabel("얼굴을 찾지 못해 확대 영역을 표시할 수 없습니다."))
            column.addWidget(button)
            images.addLayout(column)
        layout.addLayout(images)
        accept = QPushButton("선택한 이미지 사용")
        accept.clicked.connect(self.accept)
        back = QPushButton("원본 유지")
        back.clicked.connect(self.reject)
        layout.addWidget(accept)
        layout.addWidget(back)

    @staticmethod
    def image_label(pixmap, width, height):
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        return label


def choose_finished(parent, before, info):
    dialog = FinishingComparisonDialog(before, info, parent)
    return dialog.exec() == QDialog.DialogCode.Accepted and dialog.finished.isChecked()
