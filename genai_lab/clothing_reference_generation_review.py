"""Minimal reference-regeneration settings; no body mask approval."""

from PySide6.QtWidgets import QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QDialogButtonBox
from genai_lab.generation_resolution_review import GenerationResolutionDialog


class ClothingReferenceGenerationDialog(GenerationResolutionDialog):
    def __init__(self, source_size, tags, previous_width=None, parent=None, *, previous_gender=None):
        super().__init__(source_size, previous_width, parent)
        self.setWindowTitle('의상 디자인 참조 생성 설정')
        # Replace the restoration-specific explanation.
        self.layout().itemAt(0).widget().setText(
            '얼굴, 헤어, 의상 이미지를 분리된 영역별 참조로 전달합니다. 얼굴 픽셀을 고정하거나 덮어씌우지 않습니다.\n'
            '헤어 전용 참조는 승인 화면에서 별도로 확인합니다. 초기 이미지·신체 복원·자세 ControlNet은 실행하지 않습니다.\n'
            '픽셀 보존이나 단추·무늬의 정확한 복제를 보장하지 않습니다.\n'
            '생성 결과는 아래 연산 해상도로 저장합니다.'
        )
        label = QLabel('의상 디자인: ' + ', '.join(tags))
        label.setWordWrap(True)
        self.layout().insertWidget(1, label)
        self.candidate_count_input = QSpinBox()
        self.candidate_count_input.setRange(2, 100)
        self.candidate_count_input.setValue(2)
        self.layout().insertWidget(4, QLabel('후보 개수 (순차 생성, 개수에 따라 시간 증가):'))
        self.layout().insertWidget(5, self.candidate_count_input)
        self.identity_scale_input = QDoubleSpinBox()
        self.garment_scale_input = QDoubleSpinBox()
        for widget, value in ((self.identity_scale_input, 0.7), (self.garment_scale_input, 0.45)):
            widget.setRange(0.0, 1.0)
            widget.setSingleStep(0.05)
            widget.setDecimals(2)
            widget.setValue(value)
        self.layout().insertWidget(8, QLabel('얼굴 전용 참조 강도 (픽셀 고정 아님):'))
        self.layout().insertWidget(9, self.identity_scale_input)
        self.layout().insertWidget(10, QLabel('의상 이미지 참조 강도 (낮추면 의상 재현도 약해질 수 있음):'))
        self.layout().insertWidget(11, self.garment_scale_input)
        self.character_gender_input = QComboBox()
        self.character_gender_input.addItem('성별 조건을 선택하세요', None)
        self.character_gender_input.addItem('성별을 지정하지 않고 진행 (명시적 선택)', 'unspecified')
        self.character_gender_input.addItem('남성', 'male')
        self.character_gender_input.addItem('여성', 'female')
        self.character_gender_input.setToolTip(
            '외형으로 성별을 추정하지 않습니다. 지정값이 자동 분석 성별 태그보다 우선합니다. '
            '헤어·체형·의상은 자동 변경하지 않으며 생성 결과의 성별을 보장하지는 않습니다.')
        self.layout().insertWidget(12, QLabel('캐릭터 성별 (사용자 지정):'))
        self.layout().insertWidget(13, self.character_gender_input)
        if previous_gender in ('male', 'female', 'unspecified'):
            self.character_gender_input.setCurrentIndex(self.character_gender_input.findData(previous_gender))
        self.character_gender_input.currentIndexChanged.connect(self.refresh)
        self.garment_scale_input.setMinimum(0.05)
        self.refresh()

    def refresh(self):
        super().refresh()
        if hasattr(self, 'character_gender_input'):
            button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
            button.setEnabled(button.isEnabled() and self.character_gender is not None)

    def accept(self):
        if self.character_gender is None:
            return
        super().accept()

    @property
    def character_gender(self):
        return self.character_gender_input.currentData()
