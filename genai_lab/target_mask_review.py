"""기존 SAM2 선택 UI를 재사용하는 기준 캐릭터 의상·특수 영역 승인 창."""

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from genai_lab.clothing_reference import (
    NormalizedClothingSource, combine_clothing_mask_candidates,
)
from genai_lab.target_masks import approve_target_masks


def fit_review_dialog(dialog: QDialog) -> None:
    """기존 선택 UI의 내용을 스크롤하고 마지막 버튼 행은 고정한다."""
    old_layout = dialog.layout()
    footer_item = old_layout.takeAt(old_layout.count() - 1)
    content = QWidget()
    content.setLayout(old_layout)
    root = QVBoxLayout(dialog)
    scroll = QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    root.addWidget(scroll)
    if footer_item.layout() is not None:
        root.addLayout(footer_item.layout())
    elif footer_item.widget() is not None:
        root.addWidget(footer_item.widget())
    available = dialog.screen().availableGeometry()
    dialog.resize(min(1100, available.width() - 60), min(800, available.height() - 60))


class TargetMaskReviewDialog(QDialog):
    """역할별 후보는 사용자가 선택하며 실행 중 재진입과 닫기를 막는다."""

    def __init__(self, source, settings, region_dialog_type, mask_dialog_type,
                 worker_type, pixmap_factory, parent=None):
        super().__init__(parent)
        self.setWindowTitle("기준 캐릭터: 교체할 기존 의상 / 보호할 특수 영역")
        image = source.convert("RGB")
        self.source = NormalizedClothingSource(
            image=image, source_name="generated_candidate", source_format="PNG",
            source_mode="RGB", source_size_bytes=0,
            original_width=image.width, original_height=image.height,
            normalized_width=image.width, normalized_height=image.height,
        )
        self.settings = settings
        self.region_dialog_type = region_dialog_type
        self.mask_dialog_type = mask_dialog_type
        self.worker_type = worker_type
        self.pixmap_factory = pixmap_factory
        self.clothing_mask = Image.new("L", image.size, 0)
        self.protection_mask = Image.new("L", image.size, 0)
        self.protection_reviewed = False
        self.approved_masks = None
        self._thread = None
        self._worker = None
        self._result = None
        self._error = ""
        self._role = "clothing"
        layout = QVBoxLayout(self)
        guide = QLabel(
            "지금 보이는 생성 후보에서 교체할 기존 옷을 선택하세요. "
            "신발·다리 의상까지 바꾸려면 그 부위도 추가하세요. "
            "꼬리·귀 등은 별도 보호 영역으로 선택하며, 없는 경우 '특수 보호 없음'을 누르세요. "
            "사각형은 위치 안내일 뿐이며 실제 SAM2 마스크 후보를 직접 확인합니다."
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)
        preview_row = QHBoxLayout()
        self.previews = []
        for title in ("기준 캐릭터", "빨강: 교체할 기존 의상", "파랑: 보호할 특수 영역"):
            column = QVBoxLayout()
            caption = QLabel(title)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(caption)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(label)
            preview_row.addLayout(column)
            self.previews.append(label)
        layout.addLayout(preview_row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.clothing_button = QPushButton("교체할 기존 의상 선택")
        self.protection_button = QPushButton("꼬리·귀 등 보호 영역 선택")
        self.none_button = QPushButton("특수 보호 없음")
        self.approve_button = QPushButton("두 영역 승인 후 계속")
        self.cancel_button = QPushButton("취소")
        for button in (self.clothing_button, self.protection_button, self.none_button,
                       self.approve_button, self.cancel_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.clothing_button.clicked.connect(lambda: self._select_region("clothing"))
        self.protection_button.clicked.connect(lambda: self._select_region("protection"))
        self.none_button.clicked.connect(self._no_special_protection)
        self.approve_button.clicked.connect(self._approve)
        self.cancel_button.clicked.connect(self.reject)
        fit_review_dialog(self)
        self._refresh()

    def _refresh(self):
        required = np.asarray(self.clothing_mask) >= 128
        protected = np.asarray(self.protection_mask) >= 128
        conflicts = int(np.count_nonzero(required & protected))
        for label, title, mask in zip(
            self.previews,
            ("기준 캐릭터", "빨강: 교체 의상", "파랑: 보호 영역"),
            (None, required, protected),
        ):
            rgb = np.asarray(self.source.image).copy()
            if mask is not None:
                color = (255, 60, 60) if mask is required else (60, 120, 255)
                rgb[mask] = (rgb[mask].astype(np.float32) * 0.4 + np.asarray(color) * 0.6).astype(np.uint8)
            with Image.fromarray(rgb) as preview:
                label.setPixmap(self.pixmap_factory(preview).scaled(
                    280, 340, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            label.setToolTip(title)
        self.status.setText(
            f"교체 의상={np.count_nonzero(required):,}px, "
            f"특수 보호={np.count_nonzero(protected):,}px, 충돌={conflicts:,}px. "
            + ("겹친 영역의 마스크 후보를 다시 선택하세요." if conflicts else
               "마스크 밖의 기존 의상은 그대로 유지됩니다.")
        )
        self.approve_button.setEnabled(
            self._thread is None and bool(np.any(required))
            and self.protection_reviewed and conflicts == 0
        )

    @Slot()
    def _no_special_protection(self):
        self.protection_mask.paste(0, (0, 0, *self.protection_mask.size))
        self.protection_reviewed = True
        self._refresh()

    def _set_busy(self, busy):
        for button in (self.clothing_button, self.protection_button,
                       self.none_button, self.cancel_button):
            button.setEnabled(not busy)
        self.approve_button.setEnabled(False)

    def _select_region(self, role):
        if self._thread is not None:
            return
        title = "교체할 기존 의상" if role == "clothing" else "보호할 꼬리·귀·소품"
        dialog = self.region_dialog_type(self.source, None, "", self)
        dialog.setWindowTitle(f"생성된 기준 캐릭터에서 {title} 선택")
        fit_review_dialog(dialog)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._role, self._result, self._error = role, None, ""
        self._set_busy(True)
        self.status.setText(f"{title}: SAM2 후보 생성 중…")
        self._thread = QThread(self)
        self._worker = self.worker_type(self.source, dialog.selected_candidates, self.settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.completed.connect(self._capture_result, Qt.ConnectionType.QueuedConnection)
        self._worker.failed.connect(self._capture_error, Qt.ConnectionType.QueuedConnection)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished, Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    @Slot(object)
    def _capture_result(self, result):
        self._result = result

    @Slot(str, str)
    def _capture_error(self, message, details):
        self._error = f"{message}\n{details}"

    @Slot()
    def _on_thread_finished(self):
        # finished may precede native thread-local cleanup. Join before opening
        # another modal GUI, then defer it to a fresh GUI event-loop turn.
        if self._thread is None:
            return
        if not self._thread.wait(1000):
            QTimer.singleShot(0, self._on_thread_finished)
            return
        QTimer.singleShot(0, self._finish_extraction)

    @Slot()
    def _finish_extraction(self):
        finished_thread = self._thread
        self._thread, self._worker = None, None
        if finished_thread is not None:
            finished_thread.deleteLater()
        result, self._result = self._result, None
        self._set_busy(False)
        if result is None:
            QMessageBox.warning(self, "기준 캐릭터 마스크 추출 실패", self._error)
            self._refresh()
            return
        retry = False
        try:
            dialog = self.mask_dialog_type(self.source, result, self)
            dialog.setWindowTitle(
                "기존 의상 SAM2 후보 선택" if self._role == "clothing"
                else "특수 보호 SAM2 후보 선택"
            )
            fit_review_dialog(dialog)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                combined = combine_clothing_mask_candidates(
                    dialog.selected_candidates, self.source.image.size,
                )
                if self._role == "clothing":
                    self.clothing_mask.close()
                    self.clothing_mask = combined.mask_image
                else:
                    self.protection_mask.close()
                    self.protection_mask = combined.mask_image
                    self.protection_reviewed = True
            retry = dialog.retry_region_selection
        except Exception as error:
            QMessageBox.warning(self, "마스크 선택 오류", str(error))
        finally:
            for group in result.region_groups:
                for candidate in group.candidates:
                    candidate.mask_image.close()
        self._refresh()
        if retry:
            QTimer.singleShot(0, lambda: self._select_region(self._role))

    @Slot()
    def _approve(self):
        if self._thread is not None or not self.protection_reviewed:
            return
        try:
            self.approved_masks = approve_target_masks(
                self.source.image, self.clothing_mask, self.protection_mask,
            )
        except ValueError as error:
            QMessageBox.warning(self, "마스크 승인 불가", str(error))
            return
        self.accept()

    def reject(self):
        if self._thread is None:
            super().reject()

    def closeEvent(self, event):
        if self._thread is not None:
            event.ignore()
        else:
            super().closeEvent(event)

    def close_images(self):
        self.source.image.close()
        self.clothing_mask.close()
        self.protection_mask.close()
        if self.approved_masks is not None:
            self.approved_masks.close()
            self.approved_masks = None
