"""원본 크기와 독립된 복원·의상 생성 연산 해상도 선택."""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QSpinBox, QDialogButtonBox
from scripts.generation_inputs import resolve_inference_size


class GenerationResolutionDialog(QDialog):
    def __init__(self, source_size, previous_width=None, parent=None):
        super().__init__(parent)
        self.source_size = source_size
        self.setWindowTitle("신체 복원·의상 합성 해상도")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"원본: {source_size[0]}×{source_size[1]}px\n"
            "연산 가로 크기를 입력하세요. 세로는 원본 비율에 맞춰 8px 단위로 계산합니다.\n"
            "작게 생성하면 세부 표현이 줄어들 수 있습니다.\n"
            "결과는 원본 크기로 되돌린 뒤 보호 마스크로 합성합니다.\n"
            "해상도 변경 시 전체 화면 기준으로 연산하며 마스크 부분 확대는 사용하지 않습니다."
        ))
        self.width_input = QSpinBox()
        self.width_input.setRange(256, 2048)
        self.width_input.setSingleStep(8)
        self.width_input.setSuffix(" px (가로)")
        self.width_input.setValue(previous_width or source_size[0])
        layout.addWidget(self.width_input)
        self.preview = QLabel()
        layout.addWidget(self.preview)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("이 해상도로 진행")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.width_input.valueChanged.connect(self.refresh)
        self.refresh()

    @property
    def inference_width(self):
        return self.width_input.value()

    def refresh(self):
        try:
            w, h = resolve_inference_size(self.source_size, self.inference_width)
            ratio = w * h / (self.source_size[0] * self.source_size[1])
            self.preview.setText(
                f"실제 연산: {w}×{h}px | 원본 대비 픽셀 수 {ratio:.1%}\n"
                "시간·메모리 감소율은 픽셀 비율과 다릅니다. 큰 해상도는 메모리 부족 위험이 있습니다."
            )
            valid = True
        except ValueError as error:
            self.preview.setText(str(error))
            valid = False
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
