"""머리 영역의 사용자 확인과 선택 후보 작업 연결. 자동 승인은 없다."""
from pathlib import Path
import traceback
import uuid
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from genai_lab.onepass_generation import OnePassCancelled
from genai_lab.proportion_inputs import require
from genai_lab.studio_head_paste import selected_request, prepare_hair, confirm_hair, apply_head, head_product_info
from genai_lab.studio_generation import StudioRuntime

NOTICE = "원본 머리 영역을 확인한 뒤 한 장에 적용합니다 · 원래 결과는 보존합니다"
LIMITS = "코트·정면 캐릭터에서 확인했습니다. 모자·후드·옆모습은 미검증이며 테두리나 작은 조각이 남을 수 있습니다."


class HairConfirmationDialog(QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("원본 머리 영역 확인")
        self.resize(980, 520)
        layout = QVBoxLayout(self)
        label = QLabel("초록 영역에 옮길 머리카락과 귀·장식이 포함됐는지 확인해 주세요. 아니면 원래 결과를 유지합니다.")
        label.setWordWrap(True)
        layout.addWidget(label)
        row = QHBoxLayout()
        for title, path in (("원본", info["original_file"]), ("자동 분할", info["automatic_preview"]),
                            ("확인할 영역", info["corrected_preview"])):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            image = QLabel()
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setPixmap(QPixmap(str(path)).scaled(290, 350, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            column.addWidget(image)
            row.addLayout(column)
        layout.addLayout(row)
        warning = QLabel(LIMITS)
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self.yes = QPushButton("맞아요 · 이 영역으로 붙이기")
        self.no = QPushButton("아니에요 · 원래 결과 유지")
        self.yes.clicked.connect(self.accept)
        self.no.clicked.connect(self.reject)
        layout.addWidget(self.yes)
        layout.addWidget(self.no)


def review_hair(parent, info):
    return HairConfirmationDialog(info, parent).exec() == QDialog.DialogCode.Accepted


class HeadPasteController:
    def __init__(self, studio):
        self.studio = studio
        self.context = None
        button = getattr(studio.window, "studio_head_paste_button", None)
        if button is not None:
            button.clicked.connect(self.start)

    def refresh(self, ready):
        button = getattr(self.studio.window, "studio_head_paste_button", None)
        if button is None:
            return
        results = self.studio.results
        button.setVisible(bool(results and results.two_pass))
        reason = results.head_paste_unavailable_reason if results else ""
        button.setEnabled(bool(ready and results and not reason))
        button.setToolTip(reason or NOTICE + " · " + LIMITS)

    def start(self):
        studio = self.studio
        if studio.results is None or studio.task is not None or studio.confirming:
            return
        if studio.results.head_paste_unavailable_reason:
            return
        if any(value is not None and hasattr(value, "isRunning") and value.isRunning()
               for name, value in vars(studio.window).items() if name.endswith("thread")):
            QMessageBox.information(studio.window, "실행 중", "현재 작업이 끝난 뒤 머리를 붙여 주세요.")
            return
        try:
            runtime = studio.runtime or StudioRuntime.from_environment()
            request = selected_request(studio.results, runtime)
            self.context = dict(selected=studio.results.selected, before_sha256=studio.results.verify_current(),
                request=request, runtime=runtime,
                directory=studio.results.batch.directory / "head-pastes" / uuid.uuid4().hex)
            studio.release_tail_predecessor()
            cache = studio.output_root / "hair-confirmations"
            studio.launch(lambda cancel, progress: prepare_hair(request, cache, runtime, cancelled=cancel, progress=progress),
                          self.preview_ready, on_error=self.failed)
        except Exception as error:
            self.failed(error, traceback.format_exc())

    def check_context(self):
        results = self.studio.results
        require(self.context is not None and results is not None
                and results.selected == self.context["selected"]
                and results.verify_current() == self.context["before_sha256"], "머리 작업 중 선택한 기준 결과가 변경됐습니다.")

    def preview_ready(self, info):
        studio = self.studio
        self.check_context()
        studio.confirming = True
        studio.controls()
        try:
            accepted = bool(info.get("remembered")) or review_hair(studio.window, info)
            if not info.get("remembered"):
                confirm_hair(info, accepted)
        finally:
            studio.confirming = False
            studio.controls()
        if not accepted:
            self.context = None
            studio.window.status_label.setText("상태: 머리 영역을 승인하지 않았습니다 · 원래 결과 유지")
            return
        context = self.context
        studio.release_tail_predecessor()
        studio.launch(lambda cancel, progress: apply_head(context["request"], info, context["directory"], context["runtime"],
            cancelled=cancel, progress=progress), self.completed, on_error=self.failed)

    def completed(self, info):
        self.check_context()
        studio = self.studio
        verified = head_product_info(self.context["directory"], studio.results.verify_original())
        require(verified == info, "머리 결과 기록이 변경됐습니다.")
        studio.results.add_head_candidate(self.context["directory"])
        combo = studio.window.studio_candidate_combo
        combo.blockSignals(True)
        combo.addItem(f"결과 {self.context['selected']+1} · 원본 머리 붙이기")
        combo.setCurrentIndex(studio.results.selected)
        combo.blockSignals(False)
        studio.select(studio.results.selected)
        studio.window.status_label.setText("상태: 머리 붙이기 후보 추가 · 원래 결과와 비교한 뒤 저장할 결과를 확인해 주세요.")
        self.context = None

    def failed(self, error, details=""):
        self.context = None
        studio = self.studio
        if studio.results is not None:
            studio.window.candidate_preview.setPixmap(QPixmap(str(studio.results.current_path)))
        studio.window.status_label.setText("상태: 머리 붙이기 취소 · 원래 결과 유지" if isinstance(error, OnePassCancelled)
                                          else "상태: 머리 붙이기 실패 · 원래 결과 유지")
        if not isinstance(error, OnePassCancelled):
            QMessageBox.warning(studio.window, "머리를 붙이지 못했습니다", str(error))
