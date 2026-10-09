"""기존 생성·승인 흐름의 화면 표시만 담당한다. 크기를 바꿔도 각 작업은 유지하고 입력 상세는 별도로 스크롤한다. 생성 설정·승인·모델 수명은 관리하지 않는다."""
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QCheckBox, QScrollArea,
    QSizePolicy, QSplitter, QToolButton, QVBoxLayout, QWidget,
)


class ImagePreview(QLabel):
    """원본 이미지를 보관해 크기 변경 때 작은 사본을 다시 확대하지 않게 한다."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._original = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def setPixmap(self, pixmap):
        self._original = QPixmap(pixmap)
        self._fit()

    def _fit(self):
        if not self._original.isNull():
            super().setPixmap(self._original.scaled(
                self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def clear(self):
        self._original = QPixmap()
        super().clear()

    def setText(self, text):
        self.clear()
        super().setText(text)


class ReferencePreview(ImagePreview):
    """파일 미리보기 크기를 제한한다. 미리보기 실패가 입력 검증을 바꾸지 않는다."""

    def __init__(self, title):
        super().__init__(title)
        self.setFixedSize(76, 90)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setObjectName("referencePreview")
        self.setAccessibleName(title + " 미리보기")
        self._key = None
        self._placeholder = title

    def show_path(self, path):
        try:
            stat = Path(path).stat() if path else None
            key = (str(path), stat.st_mtime_ns, stat.st_size) if stat else None
        except OSError:
            key = (str(path), None, None)
        if key == self._key:
            return
        self._key = key
        self.setToolTip(str(path) if path else "")
        if not path:
            self.setText(self._placeholder)
            return
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(QSize(240, 240), Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            self.setText("미리보기\n불가")
        else:
            self.setPixmap(QPixmap.fromImage(image))


def _label(text, role=None):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    if role:
        label.setObjectName(role)
    return label


def _button(text, action, *, enabled=True, primary=False):
    button = QPushButton(text)
    button.clicked.connect(action)
    button.setEnabled(enabled)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if primary:
        button.setObjectName("primaryAction")
    return button


def _card(title, subtitle=None):
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)
    layout.addWidget(_label(title, "sectionTitle"))
    if subtitle:
        layout.addWidget(_label(subtitle, "muted"))
    return frame, layout


def _reference_row(layout, preview, label, select, clear=None):
    row = QHBoxLayout()
    row.setSpacing(12)
    row.addWidget(preview)
    content = QVBoxLayout()
    content.addWidget(label)
    actions = QHBoxLayout()
    actions.addWidget(select)
    if clear:
        actions.addWidget(clear)
    content.addLayout(actions)
    row.addLayout(content, 1)
    layout.addLayout(row)


def build_studio(window, framing_options, clothing_mode):
    """기존 제어기 위젯 이름을 두 영역의 스크롤 가능한 작업실에 연결한다."""
    central = QWidget()
    central.setObjectName("studio")
    central.setFont(QFont("Malgun Gothic", 10))
    central.setStyleSheet(STUDIO_STYLE)
    window.setCentralWidget(central)
    window.setWindowTitle("GenAI Lab · 캐릭터 작업실")
    window.resize(1280, 860)
    window.setMinimumSize(880, 620)
    screen = window.screen()
    if screen:
        available = screen.availableGeometry()
        window.resize(min(1280, available.width()-40), min(860, available.height()-80))
    root = QVBoxLayout(central)
    root.setContentsMargins(24, 18, 24, 18)
    root.setSpacing(16)
    header = QHBoxLayout()
    heading = QVBoxLayout()
    heading.addWidget(_label("GenAI Lab", "brand"))
    heading.addWidget(_label("캐릭터 작업실", "windowTitle"))
    header.addLayout(heading, 1)
    header.addWidget(QLabel("캐릭터·옷 선택  →  만들기  →  결과 선택  →  저장"))
    root.addLayout(header)

    window.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
    window.workspace_splitter.setChildrenCollapsible(False)
    window.workspace_splitter.setHandleWidth(12)
    root.addWidget(window.workspace_splitter, 1)
    left = QWidget()
    left.setMinimumWidth(330)
    left.setMaximumWidth(520)
    left_layout = QVBoxLayout(left)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(12)
    window.input_scroll = QScrollArea()
    window.input_scroll.setWidgetResizable(True)
    window.input_scroll.setFrameShape(QFrame.Shape.NoFrame)
    window.input_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    inputs = QWidget()
    input_layout = QVBoxLayout(inputs)
    input_layout.setContentsMargins(0, 0, 4, 0)
    input_layout.setSpacing(12)

    card, box = _card("참조 이미지", "캐릭터와 적용할 의상을 선택해 주세요.")
    window.style_label = _label("캐릭터 · 선택되지 않음")
    window.style_button = _button("캐릭터 선택", lambda: window.select_image("style"))
    window.style_preview = ReferencePreview("캐릭터")
    _reference_row(box, window.style_preview, window.style_label, window.style_button)
    window.outfit_label = _label("의상 · 선택되지 않음")
    window.outfit_button = _button("의상 선택", lambda: window.select_image("outfit"))
    window.clear_outfit_button = _button("해제", window.clear_outfit_reference)
    window.clear_outfit_button.setToolTip("선택한 의상만 해제합니다.")
    window.outfit_preview = ReferencePreview("의상")
    _reference_row(box, window.outfit_preview, window.outfit_label,
                   window.outfit_button, window.clear_outfit_button)
    input_layout.addWidget(card)

    card, box = _card("생성 설정")
    box.addWidget(_label("전신 이미지 · 캐릭터와 옷을 함께 생성합니다.", "muted"))
    window.framing_combo = QComboBox()
    window.framing_combo.setAccessibleName("화면 범위")
    for framing_type, label in framing_options:
        window.framing_combo.addItem(label, framing_type.value)
    window.framing_combo.setParent(window)
    window.framing_combo.hide()
    window.studio_proportion_checkbox = QCheckBox("원본 비율 참고 (2단계 생성)")
    window.studio_proportion_checkbox.setObjectName("studio_proportion_checkbox")
    window.studio_proportion_checkbox.setChecked(False)
    box.addWidget(window.studio_proportion_checkbox)
    window.studio_shoulder_checkbox = QCheckBox("어깨 보정(마른 체형 각진 어깨 줄이기)")
    window.studio_shoulder_checkbox.setObjectName("studio_shoulder_checkbox")
    window.studio_shoulder_checkbox.setChecked(False)
    window.studio_shoulder_checkbox.setEnabled(False)
    window.studio_shoulder_checkbox.setToolTip("모델이 어깨를 넓게 그리는 경향을 줄입니다. 어깨가 넓은 캐릭터에는 효과가 없거나 좁아질 수 있습니다.")
    window.studio_proportion_checkbox.toggled.connect(window.studio_shoulder_checkbox.setEnabled)
    window.studio_proportion_checkbox.toggled.connect(lambda enabled: None if enabled else window.studio_shoulder_checkbox.setChecked(False))
    box.addWidget(window.studio_shoulder_checkbox)
    window.studio_mode_note = _label("자세를 따로 지정하지 않고 생성합니다. 자세 변경은 이번 범위에서 제외합니다.", "muted")
    box.addWidget(window.studio_mode_note)
    window.studio_proportion_checkbox.toggled.connect(lambda enabled: window.studio_mode_note.setText(
        "원본 자세와 확인한 머리 윤곽을 참고합니다. 생성량은 2배이며, 다른 캐릭터의 효과는 아직 미검증입니다."
        if enabled else "자세를 따로 지정하지 않고 생성합니다. 자세 변경은 이번 범위에서 제외합니다."))
    input_layout.addWidget(card)

    card, box = _card("완성 이미지로 이어서 작업", "필요할 때 선택하는 별도 기능입니다.")
    window.qwen_pose_button = _button("저장한 이미지 자세 바꾸기", window.open_qwen_pose_editor)
    window.qwen_pose_button.setToolTip("실행 설정, 승인한 완성 이미지, 자세 골격을 선택합니다.")
    # D-077: 처리 경로는 보존하되 이번 배포에서는 자세 편집을 표시하지 않는다.
    window.qwen_pose_button.setParent(window)
    window.qwen_pose_button.hide()
    window.external_candidate_button = _button("외부 결과 불러오기 (선택 기능)", window.import_external_candidate)
    box.addWidget(window.external_candidate_button)
    window.external_candidate_notice = _label(
        "외부 편집 결과는 로컬 생성 결과로 간주하지 않으며, 자동 게이트와 "
        "원본 픽셀 보존 검사는 미실행 상태로 기록됩니다.", "muted")
    window.external_candidate_button.setToolTip(window.external_candidate_notice.text())
    input_layout.addWidget(card)

    window.diagnostics_toggle = QToolButton()
    window.diagnostics_toggle.setText("실행 정보·진단")
    window.diagnostics_toggle.setCheckable(True)
    window.diagnostics_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    window.diagnostics_toggle.setArrowType(Qt.ArrowType.RightArrow)
    window.diagnostics_toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    input_layout.addWidget(window.diagnostics_toggle)
    window.diagnostics_panel, box = _card("실행 경로와 상세 진단")
    box.addWidget(window.external_candidate_notice)
    window.refinement_mode_combo = QComboBox()
    window.refinement_mode_combo.setAccessibleName("정밀화 모드")
    window.refinement_mode_combo.addItem("SDXL 국소 정밀화", "sdxl_local")
    window.refinement_mode_combo.setParent(window)
    window.refinement_mode_combo.hide()
    window.framing_help = _label(
        "실행 경로: CPU 입력 분석 → without_pose 1회 생성(4개 seed) → 사용자 확인 → 저장. "
        "자동 품질·노출 게이트는 미실행이며 사용자 확인을 별도로 기록합니다.", "muted")
    window.local_engine_status_label = _label("실행 엔진: SDXL 1회 생성 · 자세 미적용", "muted")
    window.pipeline_stage_label = _label("실행 단계: 입력 → 분석 → 확인 → 생성 → 결과 선택 → 저장", "muted")
    window.refinement_diagnostics_label = _label("실행 기록: outputs/studio-runs/ · 입력·생성·사용자 확인 기록", "muted")
    for label in (window.framing_help, window.local_engine_status_label,
                  window.pipeline_stage_label, window.refinement_diagnostics_label):
        box.addWidget(label)
    window.refinement_diagnostics_button = _button("정밀화 마스크·진단 보기", window.show_refinement_diagnostics, enabled=False)
    box.addWidget(window.refinement_diagnostics_button)
    window.refinement_diagnostic_paths = {}
    window.refinement_diagnostic_summary = ""

    # 비활성 자세 경로를 포함해 호환 위젯의 제어기 연결 규칙을 유지한다.
    window.body_comparison_button = _button("캐릭터 신체 비교 시작", window.start_character_body_comparison, enabled=False)
    window.body_comparison_button.setParent(window)
    window.body_comparison_button.hide()
    window.pose_label = _label("자세 참조: 선택하지 않음")
    window.pose_button = _button("자세 이미지 선택", lambda: window.select_image("pose"), enabled=not clothing_mode)
    window.clear_pose_button = _button("자세 선택 해제", window.clear_pose_reference, enabled=not clothing_mode)
    window.pose_estimation_button = _button("관절 추출 시작", window.start_pose_reference_estimation, enabled=False)
    window.pose_estimation_button.setParent(window)
    window.pose_estimation_button.hide()
    if clothing_mode:
        window.pose_label.setText("자세 변경은 향후 외부 연동 단계에서 제공합니다. 원본 비율 참고는 유지합니다.")
    box.addWidget(window.pose_label)
    for button in (window.pose_button, window.clear_pose_button):
        box.addWidget(button)
    input_layout.addWidget(window.diagnostics_panel)
    window.diagnostics_panel.hide()
    def toggle_diagnostics(checked):
        window.diagnostics_panel.setVisible(checked)
        window.diagnostics_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
    window.diagnostics_toggle.toggled.connect(toggle_diagnostics)
    input_layout.addStretch()
    window.input_scroll.setWidget(inputs)
    left_layout.addWidget(window.input_scroll, 1)
    window.generate_button = _button("이미지 생성 시작", window.start_generation, enabled=False, primary=True)
    window.generate_button.setMinimumHeight(46)
    window.generate_button.setToolTip("선택한 캐릭터와 옷으로 이미지 4장을 만들고 결과를 비교합니다.")
    left_layout.addWidget(window.generate_button)
    window.studio_cancel_button = QPushButton("현재 작업 취소")
    window.studio_cancel_button.hide()
    left_layout.addWidget(window.studio_cancel_button)
    left_layout.addWidget(_label("결과는 검토·승인한 뒤 저장할 수 있습니다.", "muted"))
    window.workspace_splitter.addWidget(left)

    right, result = _card("결과 미리보기")
    right.setMinimumWidth(420)
    window.candidate_preview = ImagePreview("아직 생성한 이미지가 없습니다.\n\n왼쪽에서 캐릭터와 의상을 선택한 뒤\n이미지 생성을 시작해 주세요.")
    window.candidate_preview.setObjectName("canvas")
    window.candidate_preview.setMinimumSize(240, 160)
    window.candidate_preview.setContentsMargins(16, 16, 16, 16)
    window.candidate_preview.setAccessibleName("생성 결과 미리보기")
    window.studio_candidate_combo = QComboBox()
    window.studio_candidate_combo.setAccessibleName("생성 결과 선택")
    window.studio_candidate_combo.hide()
    result.addWidget(window.studio_candidate_combo)
    result.addWidget(window.candidate_preview, 1)
    window.open_original_size_button = _button("원본 크기로 보기", window.show_candidate_original_size, enabled=False)
    window.studio_raw_button = QPushButton("원본(raw) 보기")
    window.studio_raw_button.setEnabled(False)
    window.studio_raw_button.hide()
    image_actions = QHBoxLayout()
    image_actions.addStretch(1)
    image_actions.addWidget(window.open_original_size_button)
    image_actions.addWidget(window.studio_raw_button)
    result.addLayout(image_actions)
    window.identity_report_label = _label("", "muted")
    window.identity_report_label.setWordWrap(True)
    window.identity_report_label.hide()
    window.identity_report_button = QPushButton("측정 자세히 보기")
    window.identity_report_button.hide()
    image_actions.addWidget(window.identity_report_button)  # 미리보기 동작과 같은 행에 둬 1024×700에서 미리보기 높이를 유지한다.
    result.addWidget(window.identity_report_label)
    window.studio_background_notice = _label("", "muted")
    window.studio_background_notice.setWordWrap(True)
    window.studio_background_notice.hide()
    result.addWidget(window.studio_background_notice)
    window.studio_finish_button = QPushButton("고화질 마무리")
    window.studio_finish_button.setObjectName("studio_finish_button")
    window.studio_finish_button.setEnabled(False)
    window.studio_finish_button.hide()
    window.studio_finish_notice = _label("", "muted")
    window.studio_finish_notice.setWordWrap(True)
    window.studio_finish_notice.hide()
    result.addWidget(window.studio_finish_notice)
    window.tail_edit_button = QPushButton("꼬리 고치기")
    window.tail_edit_button.setObjectName("tail_edit_button")
    window.tail_edit_button.hide()
    optional_actions = QHBoxLayout()
    optional_actions.addWidget(window.studio_finish_button)
    optional_actions.addWidget(window.tail_edit_button)
    result.addLayout(optional_actions)
    window.status_label = _label("상태: 캐릭터와 의상 이미지를 선택해 주세요.", "status")
    window.status_label.setAccessibleName("현재 작업 상태")
    window.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    result.addWidget(window.status_label)
    result.addWidget(_label("결과를 비교·승인하면 저장 버튼이 활성화됩니다.", "muted"))
    window.approve_candidate_button = _button("이 결과 확인", window.approve_candidate, enabled=False, primary=True)
    window.reject_candidate_button = _button("다른 결과 고르기", window.reject_candidate, enabled=False)
    window.save_candidate_button = _button("저장 위치 선택 후 저장", window.save_approved_candidate, enabled=False, primary=True)
    window.discard_candidate_button = _button("저장하지 않음", window.discard_approved_candidate, enabled=False)
    for buttons in ((window.approve_candidate_button, window.reject_candidate_button),
                    (window.save_candidate_button, window.discard_candidate_button)):
        row = QHBoxLayout()
        for button in buttons:
            row.addWidget(button, 1)
        result.addLayout(row)
    window.workspace_splitter.addWidget(right)
    window.workspace_splitter.setStretchFactor(0, 0)
    window.workspace_splitter.setStretchFactor(1, 1)
    window.workspace_splitter.setSizes([380, 820])


def refresh_reference_previews(window):
    window.style_preview.show_path(window.style_path)
    window.outfit_preview.show_path(window.selected_outfit_path)


STUDIO_STYLE = """
QWidget#studio { background: #f3f5f8; color: #243148; }
QWidget#studio QLabel { color: #243148; background: transparent; }
QWidget#studio QLabel#brand { color: #4166b7; font-size: 12px; font-weight: 700; }
QWidget#studio QLabel#windowTitle { font-size: 24px; font-weight: 700; }
QWidget#studio QLabel#sectionTitle { font-size: 15px; font-weight: 700; }
QWidget#studio QLabel#muted { color: #617086; font-size: 12px; }
QWidget#studio QFrame#card { background: white; border: 1px solid #dce2eb; border-radius: 12px; }
QWidget#studio QLabel#canvas { background: #edf0f5; border: 1px solid #e1e6ed; border-radius: 10px; color: #68758a; }
QWidget#studio QLabel#referencePreview { background: #f0f3f8; border: 1px dashed #c8d2e2; border-radius: 8px; color: #758297; }
QWidget#studio QLabel#status { background: #edf3ff; color: #294c87; padding: 12px; border-radius: 8px; }
QWidget#studio QPushButton, QWidget#studio QToolButton { background: white; color: #33445f; border: 1px solid #cdd6e4; border-radius: 7px; padding: 8px 12px; }
QWidget#studio QPushButton:hover, QWidget#studio QToolButton:hover { background: #edf3ff; border-color: #8faadb; }
QWidget#studio QPushButton:pressed { background: #dce8ff; }
QWidget#studio QPushButton#primaryAction { background: #365ec2; border-color: #365ec2; color: white; font-weight: 600; }
QWidget#studio QPushButton#primaryAction:hover { background: #284daa; }
QWidget#studio QPushButton:disabled, QWidget#studio QPushButton#primaryAction:disabled { background: #edf0f5; border-color: #e2e7ef; color: #8b96a8; }
QWidget#studio QPushButton:focus, QWidget#studio QToolButton:focus, QWidget#studio QComboBox:focus { border: 2px solid #365ec2; }
QWidget#studio QComboBox { background: white; color: #243148; border: 1px solid #cdd6e4; border-radius: 6px; padding: 8px; }
QWidget#studio QCheckBox { color: #243148; background: transparent; }
QWidget#studio QCheckBox:disabled { color: #8b96a8; }
QWidget#studio QScrollArea, QWidget#studio QScrollArea > QWidget > QWidget { background: transparent; }
QWidget#studio QSplitter::handle { background: #f3f5f8; }
QWidget#studio QSplitter::handle:hover { background: #dce6f8; }
"""
