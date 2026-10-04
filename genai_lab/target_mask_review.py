"""기존 SAM2 선택 UI를 재사용하는 기준 캐릭터 의상·특수 영역 승인 창."""

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from genai_lab.clothing_reference import (
    NormalizedClothingSource, combine_clothing_mask_candidates,
)
from genai_lab.target_masks import approve_target_masks, accumulate_mask
from genai_lab.target_mask_repair import (
    LocalMaskRepairCandidate,
    create_region_mask,
    merge_approved_mask_repair,
    propose_local_mask_repair,
)


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
        self.excluded_mask = Image.new("L", image.size, 0)
        self.protection_reviewed = False
        self.approved_masks = None
        self._thread = None
        self._worker = None
        self._result = None
        self._error = ""
        self._role = "clothing"
        layout = QVBoxLayout(self)
        guide = QLabel(
            "기본은 자동 의상 후보입니다. 그대로 다음 단계에서 결과를 확인하거나, "
            "누락된 옷깃·소매·신발만 추가하고 잘못 선택된 손 등은 제외하세요. "
            "추가/제외 선택은 누적되며 같은 픽셀에서는 마지막 추가·제외 작업이 우선합니다. "
            "꼬리·귀 보호는 자동으로 추가하지 않습니다. 필요할 때만 별도 선택하세요. "
            "사각형은 위치 안내일 뿐이며 실제 SAM2 마스크 후보를 직접 확인합니다."
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)
        self.auto_checkbox = QCheckBox("자동 의상 후보를 기본으로 사용 (다음 검토 화면에서 확인)")
        self.auto_checkbox.setChecked(True)
        layout.addWidget(self.auto_checkbox)
        preview_row = QHBoxLayout()
        self.previews = []
        for title in ("기준 캐릭터", "빨강: 사용자 추가", "파랑: 명시적 보호", "주황: 사용자 제외"):
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
        buttons = QGridLayout()
        self.clothing_button = QPushButton("의상 제거 영역 추가")
        self.protection_button = QPushButton("꼬리·귀 등 보호 영역 선택")
        self.none_button = QPushButton("특수 보호 없음")
        self.repair_button = QPushButton("소매·손목 작은 누락 보정")
        self.exclude_button = QPushButton("제거에서 제외할 영역 선택")
        self.reset_button = QPushButton("추가·제외 선택 초기화")
        self.approve_button = QPushButton("설정 승인 후 자동 마스크 검토")
        self.cancel_button = QPushButton("취소")
        for index, button in enumerate((
            self.clothing_button, self.protection_button, self.none_button,
            self.exclude_button, self.reset_button, self.repair_button,
            self.approve_button, self.cancel_button,
        )):
            buttons.addWidget(button, index // 3, index % 3)
        layout.addLayout(buttons)
        self.clothing_button.clicked.connect(lambda: self._select_region("clothing"))
        self.protection_button.clicked.connect(lambda: self._select_region("protection"))
        self.exclude_button.clicked.connect(lambda: self._select_region("exclude"))
        self.reset_button.clicked.connect(self._reset_corrections)
        self.auto_checkbox.toggled.connect(self._refresh)
        self.none_button.clicked.connect(self._no_special_protection)
        self.repair_button.clicked.connect(self._review_local_repair)
        self.approve_button.clicked.connect(self._approve)
        self.cancel_button.clicked.connect(self.reject)
        fit_review_dialog(self)
        self._refresh()

    def _refresh(self):
        required = np.asarray(self.clothing_mask) >= 128
        protected = np.asarray(self.protection_mask) >= 128
        excluded = np.asarray(self.excluded_mask) >= 128
        conflicts = int(np.count_nonzero(required & protected))
        for label, title, mask in zip(
            self.previews,
            ("기준 캐릭터", "빨강: 추가 영역", "파랑: 보호 영역", "주황: 제외 영역"),
            (None, required, protected, excluded),
        ):
            rgb = np.asarray(self.source.image).copy()
            if mask is not None:
                color = ((255, 60, 60) if mask is required else
                         (255, 150, 40) if mask is excluded else (60, 120, 255))
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
            f"제외={np.count_nonzero(excluded):,}px, 자동 기본={self.auto_checkbox.isChecked()}. "
            + ("겹친 영역의 마스크 후보를 다시 선택하세요." if conflicts else
               "자동 결과와 실제 변경 범위는 다음 화면에서 승인합니다.")
        )
        self.approve_button.setEnabled(
            self._thread is None and (bool(np.any(required)) or self.auto_checkbox.isChecked())
            and self.protection_reviewed and conflicts == 0
        )
        self.repair_button.setEnabled(
            self._thread is None
            and bool(np.any(required))
            and self.protection_reviewed
        )

    @Slot()
    def _reset_corrections(self):
        for mask in (self.clothing_mask, self.excluded_mask):
            mask.paste(0, (0, 0, *mask.size))
        self._refresh()

    @Slot()
    def _no_special_protection(self):
        self.protection_mask.paste(0, (0, 0, *self.protection_mask.size))
        self.protection_reviewed = True
        self._refresh()

    def _set_busy(self, busy):
        for button in (self.clothing_button, self.protection_button,
                       self.none_button, self.repair_button, self.cancel_button,
                       self.exclude_button, self.reset_button, self.auto_checkbox):
            button.setEnabled(not busy)
        self.approve_button.setEnabled(False)

    def _select_region(self, role):
        if self._thread is not None:
            return
        title = ("추가할 기존 의상" if role == "clothing" else
                 "제거에서 제외할 손·피부 등" if role == "exclude" else "보호할 귀·꼬리·소품")
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

    @Slot()
    def _review_local_repair(self):
        if (
            self._thread is not None
            or not self.protection_reviewed
            or not np.any(np.asarray(self.clothing_mask) >= 128)
        ):
            return
        region_dialog = self.region_dialog_type(self.source, None, "", self)
        region_dialog.setWindowTitle("작은 누락을 찾을 소매·손목 영역 선택")
        fit_review_dialog(region_dialog)
        if region_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        boxes = tuple(
            candidate.box_xyxy
            for candidate in region_dialog.selected_candidates
        )
        selected_region = create_region_mask(self.source.image.size, boxes)
        repair_candidate = None
        local_keep = accumulate_mask(self.protection_mask, self.excluded_mask)
        try:
            repair_candidate = propose_local_mask_repair(
                self.clothing_mask,
                selected_region,
                local_keep,
            )
            review_dialog = LocalMaskRepairReviewDialog(
                self.source.image,
                self.clothing_mask,
                repair_candidate,
                self.pixmap_factory,
                self,
            )
            if review_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            merged = merge_approved_mask_repair(
                self.clothing_mask,
                repair_candidate.mask,
                local_keep,
            )
            self.clothing_mask.close()
            self.clothing_mask = merged
            self._refresh()
        except ValueError as error:
            QMessageBox.warning(self, "작은 누락 보정 불가", str(error))
        finally:
            selected_region.close()
            local_keep.close()
            if repair_candidate is not None:
                repair_candidate.close()

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
            dialog.setWindowTitle("SAM2 후보 확인: " + self._role)
            fit_review_dialog(dialog)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                combined = combine_clothing_mask_candidates(
                    dialog.selected_candidates, self.source.image.size,
                )
                try:
                    self._apply_selection(self._role, combined.mask_image)
                finally:
                    combined.mask_image.close()
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

    @Slot(str, object)
    def _apply_selection(self, role, new_mask):
        attribute = {"clothing": "clothing_mask", "exclude": "excluded_mask",
                     "protection": "protection_mask"}[role]
        previous = getattr(self, attribute)
        merged = accumulate_mask(previous, new_mask)
        previous.close()
        setattr(self, attribute, merged)
        if role in ("clothing", "exclude"):
            opposite = self.excluded_mask if role == "clothing" else self.clothing_mask
            with new_mask.convert("L") as gray:
                selected = Image.fromarray((np.asarray(gray) >= 128).astype(np.uint8) * 255)
            try:
                opposite.paste(0, mask=selected)
            finally:
                selected.close()
        else:
            self.protection_reviewed = True
        self._refresh()

    @Slot()
    def _approve(self):
        if self._thread is not None or not self.protection_reviewed:
            return
        try:
            self.approved_masks = approve_target_masks(
                self.source.image, self.clothing_mask, self.protection_mask,
                use_automatic_base=self.auto_checkbox.isChecked(),
                excluded_mask=self.excluded_mask,
                layered_priority=True,
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
        self.excluded_mask.close()
        if self.approved_masks is not None:
            self.approved_masks.close()
            self.approved_masks = None


class LocalMaskRepairReviewDialog(QDialog):
    """작은 누락 후보를 실제 마스크에 합치기 전에 공개한다."""

    def __init__(
        self,
        source: Image.Image,
        clothing_mask: Image.Image,
        candidate: LocalMaskRepairCandidate,
        pixmap_factory,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("소매·손목 작은 누락 후보 확인")
        layout = QVBoxLayout(self)
        guide = QLabel(
            f"후보={candidate.candidate_pixel_count:,}px/"
            f"{candidate.candidate_component_count:,}개, "
            f"닫기 반경={candidate.closing_radius_pixels}px, "
            f"조각 최대={candidate.maximum_component_area_pixels}px. "
            "노란색 후보만 기존 의상 마스크에 추가됩니다."
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)
        previews = QHBoxLayout()
        with (
            source.convert("RGB") as source_image,
            clothing_mask.convert("L") as clothing_image,
            candidate.mask.convert("L") as repair_image,
        ):
            source_array = np.asarray(source_image).copy()
            clothing = np.asarray(clothing_image).copy() >= 128
            repair = np.asarray(repair_image).copy() >= 128
        for title, overlay, color in (
            ("기존 교체 마스크", clothing, (255, 60, 60)),
            ("추가할 작은 누락 후보", repair, (255, 215, 0)),
            ("병합 후 예상", clothing | repair, (255, 90, 30)),
        ):
            column = QVBoxLayout()
            caption = QLabel(title)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(caption)
            rendered = source_array.copy()
            rendered[overlay] = (
                rendered[overlay].astype(np.float32) * 0.4
                + np.asarray(color) * 0.6
            ).astype(np.uint8)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            with Image.fromarray(rendered) as preview:
                label.setPixmap(pixmap_factory(preview).scaled(
                    300, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            column.addWidget(label)
            previews.addLayout(column)
        layout.addLayout(previews)
        buttons = QHBoxLayout()
        approve = QPushButton("보정 후보 추가")
        approve.setEnabled(candidate.candidate_pixel_count > 0)
        cancel = QPushButton("추가하지 않음")
        approve.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(approve)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        fit_review_dialog(self)
