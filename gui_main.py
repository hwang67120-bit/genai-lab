import gc
import sys
import os
import traceback
from math import ceil, floor
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from PIL import Image
import torch
from PySide6.QtCore import QObject, QPoint, QRect, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox,
    QCheckBox, QDialog, QScrollArea, QGridLayout, QGroupBox,
)

from run import (
    check_environment,
    configure_system_certificates,
    load_yaml,
    validate_config,
)
from genai_lab.model import prepare_pipeline
from genai_lab.generator import (
    apply_clothing_to_generated_candidate,
    generate_character_candidate,
)
from genai_lab.guardrails import (
    GuardSeverity,
    create_human_agnostic_guard_decision,
)
from genai_lab.body_comparison import (
    CharacterBodyComparisonCandidate,
    CharacterBodyComparisonSettings,
    ConfirmedCharacterBodyComparison,
    execute_character_body_comparison,
)
from genai_lab.catvton_preflight import (
    CatVTONInputSnapshot,
    CatVTONPreflightCandidate,
    CatVTONPreflightSettings,
    create_catvton_input_snapshot,
    execute_catvton_preflight,
)
from genai_lab.target_masks import ApprovedTargetMasks
from genai_lab.target_mask_review import TargetMaskReviewDialog
from genai_lab.clothing import (
    CatVTONLocalSettings,
    CharacterAgnosticApprovedInput,
    ClothingCategory,
    ClothingReferenceInput,
    find_catvton_clothing_type,
    prepare_catvton_clothing_condition_image,
)
from genai_lab.clothing_reference import (
    ClothingCombinedMaskCandidate,
    ClothingDetectionSettings,
    ClothingDesignAnalysisResult,
    ClothingDesignSummary,
    ClothingExtractionCandidate,
    ClothingMaskExtractionResult,
    ClothingMaskExtractionSettings,
    ClothingMaskReviewCandidate,
    ClothingPixelExtractionSettings,
    ClothingRegionCandidate,
    ClothingRegionDetectionResult,
    ClothingSourceInput,
    NormalizedClothingSource,
    combine_clothing_mask_candidates,
    create_manual_clothing_region,
    detect_clothing_regions,
    extract_clothing_pixels,
    load_and_normalize_clothing_source,
    extract_clothing_mask_candidates,
    measure_clothing_region,
)
from genai_lab.clothing_analysis import (
    ClothingDesignAnalysisSettings,
    analyze_clothing_design,
)
from genai_lab.request import (
    CharacterFramingType,
    CharacterGenerationInput,
    CharacterGenerationRequest,
    CharacterGenerationSettings,
    load_reference_image_as_rgb,
    prepare_character_generation_request,
)
from genai_lab.reference import (
    ApprovedReferenceImage,
    ReferenceImageEnhancementCandidate,
    ReferenceImagePreparationResult,
    ReferenceImageQualityStatus,
    approve_enhanced_reference_image,
    approve_original_reference_image,
    prepare_reference_image_for_review,
)
from genai_lab.pose_reference import (
    PoseReferenceApprovedInput,
    PoseReferenceReviewCandidate,
    PoseReferenceValidationError,
    approve_pose_reference_candidate,
    load_pose_reference_candidate,
)
from genai_lab.pose_estimation import (
    PoseEstimationApprovedInput,
    PoseEstimationReviewCandidate,
    PoseReferenceEstimationError,
    PoseReferenceEstimationSettings,
    approve_pose_estimation_candidate,
    execute_pose_reference_estimation,
    prepare_pose_control_input,
)
from genai_lab.pose_fallback import (
    PoseFallbackError,
    PoseFallbackSettings,
    SavedApprovedPose,
    evaluate_pose_quality,
    load_default_approved_pose,
    save_default_approved_pose,
)
from genai_lab.garment_landmarks import extract_garment_mask_landmarks
from genai_lab.character_target_landmarks import (
    extract_character_target_landmarks,
)
from genai_lab.garment_component_matching import (
    GarmentComponentMatchResult,
    propose_garment_component_matches,
)
from genai_lab.garment_warp_review import (
    GarmentWarpApprovedInput,
    GarmentWarpReviewCandidate,
    approve_garment_tps_warp_review,
    create_garment_tps_warp_review,
)
from genai_lab.garment_lineart import (
    GarmentLineartApprovedInput,
    GarmentLineartReviewCandidate,
    approve_garment_lineart_review,
    create_garment_lineart_review,
)
from genai_lab.garment_inpaint import (
    GarmentInpaintProgress,
    GarmentInpaintReviewCandidate,
    GarmentInpaintSettings,
    approve_garment_inpaint_review,
    execute_garment_inpaint,
)

from genai_lab.result import (
    CharacterGenerationCandidate,
    save_approved_character_candidate,
)
from genai_lab.run_log import (
    GenerationRunLog,
    create_generation_run_log,
    find_recovery_action,
)
from genai_lab.workflow import (
    GenerationWorkflowContext,
    GenerationWorkflowStage,
)


FRAMING_OPTIONS = (
    (CharacterFramingType.FULL_BODY, "전신 (머리부터 발끝까지)"),
    (CharacterFramingType.UPPER_BODY, "상반신 (허리 위)"),
    (CharacterFramingType.FACE, "얼굴 중심 (어깨 위)"),
)


CLOTHING_OPTIONS = (
    (ClothingCategory.TOP, "상의"),
    (ClothingCategory.BOTTOM, "하의"),
    (ClothingCategory.DRESS, "드레스"),
    (ClothingCategory.FULL_BODY_OUTFIT, "전신 의상"),
)



def create_pil_image_pixmap(image) -> QPixmap:
    """PIL 이미지를 파일 저장 없이 Qt 화면 이미지로 변환한다."""
    rgb_image = image.convert("RGB")
    try:
        image_bytes = rgb_image.tobytes("raw", "RGB")
        qt_image = QImage(
            image_bytes,
            rgb_image.width,
            rgb_image.height,
            rgb_image.width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qt_image)
    finally:
        rgb_image.close()


class ClothingRegionDetectionWorker(QObject):
    """의상 원본 정규화와 Grounding DINO 위치 탐지를 GUI 밖에서 실행한다."""

    status_changed = Signal(str)
    completed = Signal(object, object, str)
    failed = Signal(str, str)

    def __init__(
        self,
        image_path: Path,
        settings: ClothingDetectionSettings,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.settings = settings

    @Slot()
    def run(self) -> None:
        normalized_source = None
        try:
            self.status_changed.emit("의상 이미지 정규화 중 (1/2)")
            normalized_source = load_and_normalize_clothing_source(
                ClothingSourceInput(image_path=self.image_path)
            )
            self.status_changed.emit("의상 영역 자동 탐지 중 (2/2)")
            try:
                detection_result = detect_clothing_regions(
                    normalized_source,
                    self.settings,
                )
                self.completed.emit(normalized_source, detection_result, "")
            except Exception as detection_error:
                self.completed.emit(
                    normalized_source,
                    None,
                    (
                        f"자동 탐지 실패: {detection_error}\n"
                        "수동 영역 선택으로 전환합니다.\n\n"
                        f"{traceback.format_exc()}"
                    ),
                )
        except Exception as error:
            if normalized_source is not None:
                normalized_source.image.close()
            self.failed.emit(str(error), traceback.format_exc())


class ClothingRegionCanvas(QLabel):
    """의상 이미지 위에서 사각형 최대 8개와 새 드래그 영역을 표시한다."""

    def __init__(self, source_image, parent=None) -> None:
        super().__init__(parent)
        self.source_size = source_image.size
        preview_pixmap = create_pil_image_pixmap(source_image).scaled(
            680,
            600,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(preview_pixmap)
        self.setFixedSize(preview_pixmap.size())
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selection_origin = QPoint()
        self.is_selecting = False
        self.region_display_rects: list[QRect] = []
        self.pending_display_rect: QRect | None = None
        self.active_region_index = -1

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.selection_origin = event.position().toPoint()
        self.is_selecting = True
        self.pending_display_rect = QRect(
            self.selection_origin,
            self.selection_origin,
        )
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self.is_selecting:
            return
        self.pending_display_rect = QRect(
            self.selection_origin,
            event.position().toPoint(),
        ).normalized().intersected(self.rect())
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.mouseMoveEvent(event)
        self.is_selecting = False
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            for region_index, region_rect in enumerate(self.region_display_rects):
                is_active = region_index == self.active_region_index
                pen = QPen(
                    QColor("#00e5ff" if is_active else "#00c853"),
                    3 if is_active else 2,
                )
                painter.setPen(pen)
                painter.drawRect(region_rect)
                painter.drawText(
                    region_rect.topLeft() + QPoint(4, 16),
                    str(region_index + 1),
                )
            if self.pending_display_rect is not None:
                pending_pen = QPen(QColor("#ffca28"), 2)
                pending_pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pending_pen)
                painter.drawRect(self.pending_display_rect)
        finally:
            painter.end()

    def set_region_boxes(
        self,
        boxes_xyxy: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        """원본 이미지 좌표 목록을 화면 미리보기 좌표로 표시한다."""
        self.region_display_rects = [
            self.image_box_to_display_rect(box_xyxy) for box_xyxy in boxes_xyxy
        ]
        self.active_region_index = 0 if self.region_display_rects else -1
        self.pending_display_rect = None
        self.update()

    def image_box_to_display_rect(
        self,
        box_xyxy: tuple[int, int, int, int],
    ) -> QRect:
        source_width, source_height = self.source_size
        x1, y1, x2, y2 = box_xyxy
        return QRect(
            floor(x1 * self.width() / source_width),
            floor(y1 * self.height() / source_height),
            max(1, ceil((x2 - x1) * self.width() / source_width)),
            max(1, ceil((y2 - y1) * self.height() / source_height)),
        ).intersected(self.rect())

    def display_rect_to_image_box(
        self,
        display_rect: QRect | None,
    ) -> tuple[int, int, int, int] | None:
        """화면 사각형을 원본 이미지 픽셀 좌표로 변환한다."""
        if display_rect is None or display_rect.width() < 2 or display_rect.height() < 2:
            return None
        source_width, source_height = self.source_size
        x1 = floor(display_rect.left() * source_width / self.width())
        y1 = floor(display_rect.top() * source_height / self.height())
        x2 = ceil((display_rect.right() + 1) * source_width / self.width())
        y2 = ceil((display_rect.bottom() + 1) * source_height / self.height())
        return (
            max(0, x1),
            max(0, y1),
            min(source_width, x2),
            min(source_height, y2),
        )

    def pending_image_box(self) -> tuple[int, int, int, int] | None:
        return self.display_rect_to_image_box(self.pending_display_rect)

    def region_image_boxes(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            image_box
            for image_box in (
                self.display_rect_to_image_box(region_rect)
                for region_rect in self.region_display_rects
            )
            if image_box is not None
        )

    def add_pending_region(self) -> bool:
        if self.pending_display_rect is None:
            return False
        self.region_display_rects.append(QRect(self.pending_display_rect))
        self.active_region_index = len(self.region_display_rects) - 1
        self.pending_display_rect = None
        self.update()
        return True

    def replace_active_region(self) -> bool:
        if (
            self.pending_display_rect is None
            or not 0 <= self.active_region_index < len(self.region_display_rects)
        ):
            return False
        self.region_display_rects[self.active_region_index] = QRect(
            self.pending_display_rect
        )
        self.pending_display_rect = None
        self.update()
        return True

    def delete_active_region(self) -> bool:
        if not 0 <= self.active_region_index < len(self.region_display_rects):
            return False
        del self.region_display_rects[self.active_region_index]
        self.active_region_index = min(
            self.active_region_index,
            len(self.region_display_rects) - 1,
        )
        self.update()
        return True

    def set_active_region(self, region_index: int) -> None:
        self.active_region_index = (
            region_index if 0 <= region_index < len(self.region_display_rects) else -1
        )
        self.update()


class ClothingRegionReviewDialog(QDialog):
    """자동 영역과 사용자가 추가한 영역 최대 8개를 확인한다."""

    MAXIMUM_REGION_COUNT = 8

    def __init__(
        self,
        normalized_source: NormalizedClothingSource,
        detection_result: ClothingRegionDetectionResult | None,
        detection_warning: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("복수 의상 영역 확인")
        self.resize(760, 850)
        self.selected_candidates: tuple[ClothingRegionCandidate, ...] = ()
        self.normalized_source = normalized_source
        self.automatic_candidate = (
            detection_result.selected_candidate
            if detection_result is not None
            else None
        )

        layout = QVBoxLayout(self)
        guide_label = QLabel(
            "초록색·하늘색 사각형은 SAM2 위치 안내입니다. "
            "추가할 부위를 드래그한 뒤 '새 영역 추가'를 누르세요. "
            "기존 영역의 위치나 크기를 바꾸려면 목록에서 고르고 새로 드래그한 뒤 "
            "'선택 영역 교체'를 누르세요. 최대 8개입니다."
        )
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        self.canvas = ClothingRegionCanvas(normalized_source.image, self)
        initial_boxes = (
            (self.automatic_candidate.box_xyxy,)
            if self.automatic_candidate is not None
            else ()
        )
        self.canvas.set_region_boxes(initial_boxes)
        layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        self.region_combo = QComboBox()
        self.region_combo.currentIndexChanged.connect(
            self.canvas.set_active_region
        )
        layout.addWidget(self.region_combo)
        self.refresh_region_list()

        if self.automatic_candidate is not None and detection_result is not None:
            measurement = measure_clothing_region(
                self.automatic_candidate,
                normalized_source.image.size,
            )
            info_text = (
                f"자동 후보={len(detection_result.candidates)}개, "
                f"초기 채택=1개, 신뢰도={measurement.confidence_percent:.1f}%, "
                f"점유율={measurement.area_ratio_percent:.1f}%, "
                f"처리 시간={detection_result.elapsed_seconds:.2f}초"
            )
        else:
            info_text = "자동 후보=0개. 영역을 1개 이상 직접 추가해야 합니다."
        if detection_warning:
            info_text += "\n" + detection_warning.split("\n\n", 1)[0]
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        edit_button_layout = QHBoxLayout()
        add_button = QPushButton("새 영역 추가")
        replace_button = QPushButton("선택 영역 교체")
        delete_button = QPushButton("선택 영역 삭제")
        add_button.clicked.connect(self.add_drawn_region)
        replace_button.clicked.connect(self.replace_selected_region)
        delete_button.clicked.connect(self.delete_selected_region)
        edit_button_layout.addWidget(add_button)
        edit_button_layout.addWidget(replace_button)
        edit_button_layout.addWidget(delete_button)
        layout.addLayout(edit_button_layout)

        decision_button_layout = QHBoxLayout()
        approve_button = QPushButton("전체 영역 승인")
        cancel_button = QPushButton("의상 선택 취소")
        approve_button.clicked.connect(self.accept_selected_regions)
        cancel_button.clicked.connect(self.reject)
        decision_button_layout.addWidget(approve_button)
        decision_button_layout.addWidget(cancel_button)
        layout.addLayout(decision_button_layout)

    def refresh_region_list(self) -> None:
        active_index = self.canvas.active_region_index
        self.region_combo.blockSignals(True)
        self.region_combo.clear()
        for region_number, box_xyxy in enumerate(
            self.canvas.region_image_boxes(),
            start=1,
        ):
            x1, y1, x2, y2 = box_xyxy
            self.region_combo.addItem(
                f"영역 {region_number}/{len(self.canvas.region_display_rects)} - "
                f"{x2 - x1}x{y2 - y1}px, 좌표={box_xyxy}"
            )
        if self.region_combo.count() > 0:
            self.region_combo.setCurrentIndex(
                min(max(active_index, 0), self.region_combo.count() - 1)
            )
        self.region_combo.blockSignals(False)
        self.canvas.set_active_region(self.region_combo.currentIndex())

    def validate_pending_region(self) -> bool:
        if self.canvas.pending_image_box() is not None:
            return True
        QMessageBox.warning(
            self,
            "새 영역 필요",
            "이미지 위에서 가로와 세로가 각각 2픽셀 이상인 영역을 드래그하세요.",
        )
        return False

    @Slot()
    def add_drawn_region(self) -> None:
        if len(self.canvas.region_display_rects) >= self.MAXIMUM_REGION_COUNT:
            QMessageBox.warning(
                self,
                "영역 수 초과",
                f"의상 영역은 최대 {self.MAXIMUM_REGION_COUNT}개까지 추가할 수 있습니다.",
            )
            return
        if not self.validate_pending_region():
            return
        self.canvas.add_pending_region()
        self.refresh_region_list()

    @Slot()
    def replace_selected_region(self) -> None:
        if self.region_combo.currentIndex() < 0:
            QMessageBox.warning(self, "선택 영역 없음", "교체할 영역이 없습니다.")
            return
        if not self.validate_pending_region():
            return
        self.canvas.set_active_region(self.region_combo.currentIndex())
        self.canvas.replace_active_region()
        self.refresh_region_list()

    @Slot()
    def delete_selected_region(self) -> None:
        self.canvas.set_active_region(self.region_combo.currentIndex())
        if not self.canvas.delete_active_region():
            QMessageBox.warning(self, "선택 영역 없음", "삭제할 영역이 없습니다.")
            return
        self.refresh_region_list()

    @Slot()
    def accept_selected_regions(self) -> None:
        selected_boxes = self.canvas.region_image_boxes()
        if not 1 <= len(selected_boxes) <= self.MAXIMUM_REGION_COUNT:
            QMessageBox.warning(
                self,
                "의상 영역 필요",
                f"의상 영역을 1개~{self.MAXIMUM_REGION_COUNT}개 지정하세요.",
            )
            return

        selected_candidates: list[ClothingRegionCandidate] = []
        try:
            for region_index, selected_box in enumerate(selected_boxes):
                if (
                    region_index == 0
                    and self.automatic_candidate is not None
                    and selected_box == self.automatic_candidate.box_xyxy
                ):
                    selected_candidates.append(self.automatic_candidate)
                else:
                    selected_candidates.append(
                        create_manual_clothing_region(
                            self.normalized_source.image.size,
                            selected_box,
                        )
                    )
        except ValueError as error:
            QMessageBox.warning(self, "의상 영역 오류", str(error))
            return

        self.selected_candidates = tuple(selected_candidates)
        self.accept()


class ClothingMaskExtractionWorker(QObject):
    """SAM2 의상 마스크 후보 생성을 GUI 밖에서 실행한다."""

    status_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        normalized_source: NormalizedClothingSource,
        approved_regions: tuple[ClothingRegionCandidate, ...],
        settings: ClothingMaskExtractionSettings,
    ) -> None:
        super().__init__()
        self.normalized_source = normalized_source
        self.approved_regions = approved_regions
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            self.status_changed.emit("SAM2 의상 마스크 후보 생성 중...")
            configure_system_certificates()
            extraction_result = extract_clothing_mask_candidates(
                normalized_source=self.normalized_source,
                approved_regions=self.approved_regions,
                settings=self.settings,
            )
            self.completed.emit(extraction_result)
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())


class ClothingDesignAnalysisWorker(QObject):
    """WD14 의상 태그 분석을 GUI 밖에서 최대 1개 실행한다."""

    status_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        extraction_candidate: ClothingExtractionCandidate,
        settings: ClothingDesignAnalysisSettings,
    ) -> None:
        super().__init__()
        self.extraction_candidate = extraction_candidate
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            self.status_changed.emit(
                "WD14 의상 디자인 분석 중... 최초 실행은 379MB 모델을 받습니다."
            )
            configure_system_certificates()
            analysis_result = analyze_clothing_design(
                self.extraction_candidate,
                self.settings,
            )
            self.completed.emit(analysis_result)
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())


class CharacterBodyComparisonWorker(QObject):
    """SCHP·DensePose·DWPose 신체 비교를 GUI 밖에서 최대 1개 실행한다."""

    status_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        character_image: Image.Image,
        clothing_type: str,
        settings: CharacterBodyComparisonSettings,
        clothing_reference_input: ClothingReferenceInput,
        preflight_settings: CatVTONPreflightSettings,
        approved_target_masks: ApprovedTargetMasks | None = None,
    ) -> None:
        super().__init__()
        self.character_image = character_image.copy()
        self.clothing_type = clothing_type
        self.settings = settings
        self.clothing_reference_input = clothing_reference_input
        self.preflight_settings = preflight_settings
        self.approved_target_masks = (
            approved_target_masks.copy() if approved_target_masks is not None else None
        )

    @Slot()
    def run(self) -> None:
        comparison_candidate = None
        clothing_condition = None
        input_snapshot = None
        preflight_candidate = None
        try:
            self.status_changed.emit(
                "SCHP·DensePose 신체 영역과 DWPose 관절 18개 분석 중..."
            )
            comparison_candidate = execute_character_body_comparison(
                self.character_image,
                self.clothing_type,
                self.settings,
                approved_target_masks=self.approved_target_masks,
            )
            self.status_changed.emit(
                "CatVTON 실제 모델 입력 Preflight 변환 중..."
            )
            clothing_condition = prepare_catvton_clothing_condition_image(
                self.clothing_reference_input
            )
            input_snapshot = create_catvton_input_snapshot(
                person_image=comparison_candidate.source_image,
                approved_change_mask=(
                    comparison_candidate.mask_refinement.safe_change_mask
                ),
                clothing_condition_image=clothing_condition.image,
                identity_protection_mask=(
                    comparison_candidate.mask_refinement.identity_protection_mask
                ),
                expanded_foreground_mask=(
                    comparison_candidate.mask_refinement.expanded_foreground_mask
                ),
            )
            preflight_candidate = execute_catvton_preflight(
                person_image=comparison_candidate.source_image,
                approved_change_mask=(
                    comparison_candidate.mask_refinement.safe_change_mask
                ),
                clothing_condition_image=clothing_condition.image,
                identity_protection_mask=(
                    comparison_candidate.mask_refinement.identity_protection_mask
                ),
                expanded_foreground_mask=(
                    comparison_candidate.mask_refinement.expanded_foreground_mask
                ),
                settings=self.preflight_settings,
            )
            self.completed.emit(
                (comparison_candidate, input_snapshot, preflight_candidate)
            )
            comparison_candidate = None
            input_snapshot = None
            preflight_candidate = None
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())
        finally:
            if comparison_candidate is not None:
                comparison_candidate.close()
            if preflight_candidate is not None:
                preflight_candidate.close()
            if input_snapshot is not None:
                input_snapshot.close()
            if clothing_condition is not None:
                clothing_condition.image.close()
            if self.clothing_reference_input.approved_image is not None:
                self.clothing_reference_input.approved_image.close()
            self.character_image.close()
            if self.approved_target_masks is not None:
                self.approved_target_masks.close()


class PoseReferenceEstimationWorker(QObject):
    """승인 자세 이미지의 DWPose 관절 추출을 GUI 밖에서 1회 실행한다."""

    status_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        approved_pose_reference: PoseReferenceApprovedInput,
        settings: PoseReferenceEstimationSettings,
    ) -> None:
        super().__init__()
        self.approved_pose_reference = PoseReferenceApprovedInput(
            source_path=approved_pose_reference.source_path,
            image=approved_pose_reference.image.copy(),
            image_format=approved_pose_reference.image_format,
            width=approved_pose_reference.width,
            height=approved_pose_reference.height,
            pixel_count=approved_pose_reference.pixel_count,
            aspect_ratio=approved_pose_reference.aspect_ratio,
            file_size_bytes=approved_pose_reference.file_size_bytes,
        )
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            self.status_changed.emit(
                "승인 자세 이미지에서 DWPose 관절 18개 추출 중..."
            )
            configure_system_certificates()
            estimation_candidate = execute_pose_reference_estimation(
                self.approved_pose_reference,
                self.settings,
            )
            self.completed.emit(estimation_candidate)
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())
        finally:
            self.approved_pose_reference.close()


class ClothingMaskReviewDialog(QDialog):
    """의상 영역별 SAM2 후보를 공개하고 각 영역에서 1개씩 선택한다."""

    def __init__(
        self,
        normalized_source: NormalizedClothingSource,
        extraction_result: ClothingMaskExtractionResult,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SAM2 의상 마스크 후보 확인")
        self.resize(1050, 760)
        self.normalized_source = normalized_source
        self.extraction_result = extraction_result
        self.selected_candidates: tuple[ClothingMaskReviewCandidate, ...] = ()
        self.selected_candidates_by_region: dict[
            int,
            ClothingMaskReviewCandidate,
        ] = {}
        self.retry_region_selection = False

        layout = QVBoxLayout(self)
        guide_label = QLabel(
            "각 의상 영역에서 SAM2 후보 1개를 선택하세요. "
            "이 화면은 마스크만 만들며 원본 RGB 픽셀은 아직 추출하지 않습니다."
        )
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        self.region_combo = QComboBox()
        for region_group in extraction_result.region_groups:
            self.region_combo.addItem(
                f"영역 {region_group.region_number}/"
                f"{len(extraction_result.region_groups)} - "
                f"좌표={region_group.approved_region.box_xyxy}",
                region_group.region_number,
            )
        self.region_combo.currentIndexChanged.connect(self.change_region)
        layout.addWidget(self.region_combo)

        self.candidate_combo = QComboBox()
        self.candidate_combo.currentIndexChanged.connect(
            self.show_selected_candidate
        )
        layout.addWidget(self.candidate_combo)

        preview_layout = QHBoxLayout()
        self.source_preview = self.create_preview_label("정규화 원본")
        self.mask_preview = self.create_preview_label("흑백 마스크")
        self.overlay_preview = self.create_preview_label("원본 위 50% 겹쳐보기")
        preview_layout.addWidget(self.source_preview)
        preview_layout.addWidget(self.mask_preview)
        preview_layout.addWidget(self.overlay_preview)
        layout.addLayout(preview_layout)

        self.measurement_label = QLabel()
        self.measurement_label.setWordWrap(True)
        layout.addWidget(self.measurement_label)

        self.selection_progress_label = QLabel()
        layout.addWidget(self.selection_progress_label)

        button_layout = QHBoxLayout()
        select_button = QPushButton("현재 영역 마스크 선택")
        complete_button = QPushButton("선택 완료하고 마스크 합치기")
        retry_button = QPushButton("의상 위치 다시 선택")
        cancel_button = QPushButton("의상 사용 취소")
        select_button.clicked.connect(self.select_current_candidate)
        complete_button.clicked.connect(self.accept_selected_candidates)
        retry_button.clicked.connect(self.request_region_reselection)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(select_button)
        button_layout.addWidget(complete_button)
        button_layout.addWidget(retry_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.change_region(0)
        self.update_selection_progress()

    @staticmethod
    def create_preview_label(title: str) -> QLabel:
        preview_label = QLabel(title)
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setMinimumSize(300, 520)
        preview_label.setStyleSheet(
            "border: 1px solid #777; background-color: #202020; color: #dddddd;"
        )
        return preview_label

    @Slot(int)
    def show_selected_candidate(self, candidate_index: int) -> None:
        """선택한 후보의 세 중간 이미지와 측정 수치를 표시한다."""
        region_index = self.region_combo.currentIndex()
        if not 0 <= region_index < len(self.extraction_result.region_groups):
            return
        region_group = self.extraction_result.region_groups[region_index]
        if not 0 <= candidate_index < len(region_group.candidates):
            return
        candidate = region_group.candidates[candidate_index]
        overlay_image = create_clothing_mask_overlay_image(
            self.normalized_source.image,
            candidate.mask_image,
        )
        try:
            self.set_preview_image(
                self.source_preview,
                self.normalized_source.image,
            )
            self.set_preview_image(
                self.mask_preview,
                candidate.mask_image,
            )
            self.set_preview_image(
                self.overlay_preview,
                overlay_image,
            )
        finally:
            overlay_image.close()

        self.measurement_label.setText(
            f"모델={self.extraction_result.model_id}, "
            f"구성={self.extraction_result.source_model_type}→"
            f"{self.extraction_result.runtime_model_type}, "
            f"영역={region_group.region_number}/"
            f"{len(self.extraction_result.region_groups)}, "
            f"후보={candidate.candidate_number}/"
            f"{len(region_group.candidates)}, "
            f"모델 점수={candidate.model_score * 100.0:.1f}%, "
            f"선택 픽셀={candidate.selected_pixel_count:,}개, "
            f"사각형 내부 점유율="
            f"{candidate.region_coverage_percent:.1f}%, "
            f"분리 영역={candidate.connected_region_count}개, "
            f"경계 접촉 픽셀="
            f"{candidate.boundary_touch_pixel_count:,}개, "
            f"처리 시간={self.extraction_result.elapsed_seconds:.2f}초"
        )

    @staticmethod
    def set_preview_image(preview_label: QLabel, image) -> None:
        preview_pixmap = create_pil_image_pixmap(image).scaled(
            300,
            520,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        preview_label.setPixmap(preview_pixmap)

    @Slot(int)
    def change_region(self, region_index: int) -> None:
        """영역을 바꾸면 그 영역에서 생성된 후보 최대 3개를 표시한다."""
        if not 0 <= region_index < len(self.extraction_result.region_groups):
            return
        region_group = self.extraction_result.region_groups[region_index]
        selected_candidate = self.selected_candidates_by_region.get(
            region_group.region_number
        )
        self.candidate_combo.blockSignals(True)
        self.candidate_combo.clear()
        selected_index = 0
        for candidate_index, candidate in enumerate(region_group.candidates):
            self.candidate_combo.addItem(
                f"후보 {candidate.candidate_number}/"
                f"{len(region_group.candidates)} - "
                f"모델 점수 {candidate.model_score * 100.0:.1f}%",
                candidate.candidate_number,
            )
            if candidate is selected_candidate:
                selected_index = candidate_index
        self.candidate_combo.setCurrentIndex(selected_index)
        self.candidate_combo.blockSignals(False)
        self.show_selected_candidate(selected_index)

    @Slot()
    def select_current_candidate(self) -> None:
        region_index = self.region_combo.currentIndex()
        candidate_index = self.candidate_combo.currentIndex()
        if not 0 <= region_index < len(self.extraction_result.region_groups):
            return
        region_group = self.extraction_result.region_groups[region_index]
        if not 0 <= candidate_index < len(region_group.candidates):
            QMessageBox.warning(
                self,
                "마스크 후보 없음",
                "선택할 수 있는 SAM2 마스크 후보가 없습니다.",
            )
            return
        self.selected_candidates_by_region[region_group.region_number] = (
            region_group.candidates[candidate_index]
        )
        self.update_selection_progress()
        for next_index, next_group in enumerate(
            self.extraction_result.region_groups
        ):
            if next_group.region_number not in self.selected_candidates_by_region:
                self.region_combo.setCurrentIndex(next_index)
                return

    def update_selection_progress(self) -> None:
        self.selection_progress_label.setText(
            f"선택 완료={len(self.selected_candidates_by_region)}/"
            f"{len(self.extraction_result.region_groups)}개 영역"
        )

    @Slot()
    def accept_selected_candidates(self) -> None:
        missing_region_numbers = [
            region_group.region_number
            for region_group in self.extraction_result.region_groups
            if region_group.region_number not in self.selected_candidates_by_region
        ]
        if missing_region_numbers:
            QMessageBox.warning(
                self,
                "마스크 선택 미완료",
                "다음 영역의 마스크를 선택하세요: "
                + ", ".join(str(number) for number in missing_region_numbers),
            )
            return
        self.selected_candidates = tuple(
            self.selected_candidates_by_region[region_group.region_number]
            for region_group in self.extraction_result.region_groups
        )
        self.accept()


    @Slot()
    def request_region_reselection(self) -> None:
        """현재 후보를 버리고 의상 위치 선택 단계로 돌아가도록 표시한다."""
        self.retry_region_selection = True
        self.reject()


class ClothingCombinedMaskReviewDialog(QDialog):
    """영역별 선택 마스크를 합친 최종 마스크의 사용자 승인을 받는다."""

    def __init__(
        self,
        normalized_source: NormalizedClothingSource,
        combined_mask: ClothingCombinedMaskCandidate,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("합친 의상 마스크 최종 확인")
        self.resize(1050, 760)
        self.approved = False
        self.retry_region_selection = False

        layout = QVBoxLayout(self)
        guide_label = QLabel(
            "영역별 선택 마스크를 합친 결과입니다. "
            "상의·하의·신발 등 필요한 부위가 모두 포함됐는지 확인하세요."
        )
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        preview_layout = QHBoxLayout()
        source_preview = ClothingMaskReviewDialog.create_preview_label("정규화 원본")
        mask_preview = ClothingMaskReviewDialog.create_preview_label("합친 흑백 마스크")
        overlay_preview = ClothingMaskReviewDialog.create_preview_label(
            "원본 위 50% 겹쳐보기"
        )
        overlay_image = create_clothing_mask_overlay_image(
            normalized_source.image,
            combined_mask.mask_image,
        )
        try:
            ClothingMaskReviewDialog.set_preview_image(
                source_preview,
                normalized_source.image,
            )
            ClothingMaskReviewDialog.set_preview_image(
                mask_preview,
                combined_mask.mask_image,
            )
            ClothingMaskReviewDialog.set_preview_image(
                overlay_preview,
                overlay_image,
            )
        finally:
            overlay_image.close()
        preview_layout.addWidget(source_preview)
        preview_layout.addWidget(mask_preview)
        preview_layout.addWidget(overlay_preview)
        layout.addLayout(preview_layout)

        measurement_label = QLabel(
            f"합친 영역={combined_mask.source_region_count}개, "
            f"선택 픽셀={combined_mask.selected_pixel_count:,}개, "
            f"분리 영역={combined_mask.connected_region_count}개, "
            f"경계 접촉 픽셀={combined_mask.boundary_touch_pixel_count:,}개"
        )
        measurement_label.setWordWrap(True)
        layout.addWidget(measurement_label)

        button_layout = QHBoxLayout()
        approve_button = QPushButton("합친 마스크 승인")
        retry_button = QPushButton("의상 영역 다시 선택")
        cancel_button = QPushButton("의상 사용 취소")
        approve_button.clicked.connect(self.approve_combined_mask)
        retry_button.clicked.connect(self.request_region_reselection)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(approve_button)
        button_layout.addWidget(retry_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    @Slot()
    def approve_combined_mask(self) -> None:
        self.approved = True
        self.accept()

    @Slot()
    def request_region_reselection(self) -> None:
        self.retry_region_selection = True
        self.reject()


class ClothingPixelExtractionReviewDialog(QDialog):
    """원본 RGB 추출본과 자동 공백 처리 수치를 공개하고 승인을 받는다."""

    def __init__(
        self,
        normalized_source: NormalizedClothingSource,
        approved_mask: ClothingCombinedMaskCandidate,
        extraction_candidate: ClothingExtractionCandidate,
        extraction_settings: ClothingPixelExtractionSettings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("원본 의상 픽셀 추출 확인")
        self.resize(1320, 780)
        self.approved = False
        self.retry_mask_selection = False

        layout = QVBoxLayout(self)
        guide_label = QLabel(
            "AI가 의상을 다시 그린 결과가 아닙니다. 승인한 마스크를 알파로 사용해 "
            "원본 RGB 픽셀을 그대로 꺼낸 결과입니다. 자동으로 메운 공백과 "
            "빠진 의상 부위가 없는지 확인하세요."
        )
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        preview_layout = QHBoxLayout()
        source_preview = ClothingMaskReviewDialog.create_preview_label(
            "정규화 원본"
        )
        approved_mask_preview = ClothingMaskReviewDialog.create_preview_label(
            "사용자 승인 마스크"
        )
        repaired_mask_preview = ClothingMaskReviewDialog.create_preview_label(
            "공백 검사 후 알파 마스크"
        )
        extracted_preview = ClothingMaskReviewDialog.create_preview_label(
            "투명 의상 흰 배경 확인"
        )
        white_preview = create_white_background_clothing_preview(
            extraction_candidate
        )
        try:
            ClothingMaskReviewDialog.set_preview_image(
                source_preview,
                normalized_source.image,
            )
            ClothingMaskReviewDialog.set_preview_image(
                approved_mask_preview,
                approved_mask.mask_image,
            )
            ClothingMaskReviewDialog.set_preview_image(
                repaired_mask_preview,
                extraction_candidate.clothing_mask,
            )
            ClothingMaskReviewDialog.set_preview_image(
                extracted_preview,
                white_preview,
            )
        finally:
            white_preview.close()
        preview_layout.addWidget(source_preview)
        preview_layout.addWidget(approved_mask_preview)
        preview_layout.addWidget(repaired_mask_preview)
        preview_layout.addWidget(extracted_preview)
        layout.addLayout(preview_layout)

        x1, y1, x2, y2 = extraction_candidate.preview_crop_box
        measurement_label = QLabel(
            f"추출 알파 픽셀={extraction_candidate.selected_alpha_pixel_count:,}개, "
            f"반투명 경계 픽셀={extraction_candidate.soft_edge_pixel_count:,}개, "
            f"내부 공백={extraction_candidate.enclosed_hole_count}개, "
            f"복원 공백={extraction_candidate.filled_hole_count}개/"
            f"{extraction_candidate.filled_hole_pixel_count:,}픽셀, "
            f"보류 공백={extraction_candidate.skipped_hole_count}개, "
            f"원본 RGB 변경={extraction_candidate.changed_rgb_pixel_count}개, "
            f"원본 보존율="
            f"{extraction_candidate.original_pixel_preservation_percent:.3f}%, "
            f"미리보기 범위=({x1}, {y1})~({x2}, {y2}), "
            f"공백 최대={extraction_settings.maximum_hole_area_pixels:,}픽셀과 "
            f"전체의 {extraction_settings.maximum_hole_area_ratio * 100.0:.3f}% 중 "
            "작은 값, "
            f"RGB 거리≤{extraction_settings.maximum_rgb_distance:.1f}, "
            f"흰 의상 명도≥{extraction_settings.white_clothing_luminance:.1f}, "
            f"명도 차이≤"
            f"{extraction_settings.maximum_white_luminance_difference:.1f}"
        )
        measurement_label.setWordWrap(True)
        layout.addWidget(measurement_label)

        button_layout = QHBoxLayout()
        approve_button = QPushButton("픽셀 추출 승인")
        retry_button = QPushButton("마스크 다시 선택")
        cancel_button = QPushButton("의상 사용 취소")
        approve_button.clicked.connect(self.approve_extraction)
        retry_button.clicked.connect(self.request_mask_reselection)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(approve_button)
        button_layout.addWidget(retry_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    @Slot()
    def approve_extraction(self) -> None:
        self.approved = True
        self.accept()

    @Slot()
    def request_mask_reselection(self) -> None:
        self.retry_mask_selection = True
        self.reject()


class ClothingDesignAnalysisReviewDialog(QDialog):
    """WD14 일반 태그와 점수를 보여주고 사용자가 포함 여부를 정한다."""

    def __init__(
        self,
        extraction_candidate: ClothingExtractionCandidate,
        analysis_result: ClothingDesignAnalysisResult,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("WD14 의상 디자인 분석 확인")
        self.resize(900, 780)
        self.approved = False
        self.approved_tag_names: tuple[str, ...] = ()
        self.tag_checkboxes: list[tuple[str, QCheckBox]] = []

        layout = QVBoxLayout(self)
        guide_label = QLabel(
            "WD14 점수는 사실 판정이 아니라 후보입니다. 의상에 실제로 보이는 "
            "항목만 체크하고 잘못된 태그는 체크를 해제하세요. 등급 태그와 "
            "캐릭터 이름은 화면에 표시하지 않습니다."
        )
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        preview_label = ClothingMaskReviewDialog.create_preview_label(
            "분석에 사용한 추출 의상"
        )
        white_preview = create_white_background_clothing_preview(
            extraction_candidate
        )
        try:
            ClothingMaskReviewDialog.set_preview_image(
                preview_label,
                white_preview,
            )
        finally:
            white_preview.close()
        layout.addWidget(preview_label)

        measurement_label = QLabel(
            f"모델={analysis_result.model_id}, "
            f"장치={analysis_result.execution_provider}, "
            f"입력={analysis_result.input_width}x{analysis_result.input_height}"
            f"→{analysis_result.model_input_size}x"
            f"{analysis_result.model_input_size}, "
            f"전체 라벨={analysis_result.total_label_count:,}개, "
            f"일반 라벨={analysis_result.general_label_count:,}개, "
            f"제외 등급={analysis_result.excluded_rating_label_count}개, "
            f"제외 캐릭터={analysis_result.excluded_character_label_count:,}개, "
            f"기준={analysis_result.score_threshold * 100.0:.1f}%, "
            f"표시 후보={len(analysis_result.tag_candidates)}개, "
            f"처리 시간={analysis_result.elapsed_seconds:.2f}초"
        )
        measurement_label.setWordWrap(True)
        layout.addWidget(measurement_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        tag_container = QWidget()
        tag_layout = QVBoxLayout(tag_container)
        if analysis_result.tag_candidates:
            for tag_candidate in analysis_result.tag_candidates:
                checkbox = QCheckBox(
                    f"{tag_candidate.display_name} "
                    f"({tag_candidate.score * 100.0:.1f}%)"
                )
                checkbox.setChecked(True)
                tag_layout.addWidget(checkbox)
                self.tag_checkboxes.append(
                    (tag_candidate.tag_name, checkbox)
                )
        else:
            empty_label = QLabel(
                "35.0% 이상인 일반 태그가 0개입니다. "
                "태그 없이 승인하면 확인 불가 항목으로 기록됩니다."
            )
            empty_label.setWordWrap(True)
            tag_layout.addWidget(empty_label)
        scroll_area.setWidget(tag_container)
        layout.addWidget(scroll_area)

        button_layout = QHBoxLayout()
        approve_button = QPushButton("선택 태그 승인")
        cancel_button = QPushButton("의상 사용 취소")
        approve_button.clicked.connect(self.approve_selected_tags)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(approve_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    @Slot()
    def approve_selected_tags(self) -> None:
        self.approved_tag_names = tuple(
            tag_name
            for tag_name, checkbox in self.tag_checkboxes
            if checkbox.isChecked()
        )
        self.approved = True
        self.accept()


class CharacterBodyComparisonReviewDialog(QDialog):
    """신체 보호 영역과 마스크 보정 중간 결과를 스크롤로 보여준다."""

    def __init__(
        self,
        comparison_candidate: CharacterBodyComparisonCandidate,
        input_snapshot: CatVTONInputSnapshot,
        preflight_candidate: CatVTONPreflightCandidate,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("캐릭터 의상 변경 마스크 확인")
        available_geometry = self.screen().availableGeometry()
        self.resize(
            min(1400, max(700, available_geometry.width() - 80)),
            min(900, max(500, available_geometry.height() - 80)),
        )
        self.approved = False

        layout = QVBoxLayout(self)
        guide_label = QLabel(
            "isnet-anime 캐릭터 외곽, DensePose와 신체 보호 영역은 AI 추정 "
            "후보입니다. 마스크 보정 뒤 "
            "Clothing Region Erasure와 Inpainting Mask Neutralization으로 만든 "
            "Human-Agnostic Image를 확인하세요. 다음 Inpaint에 전달하는 것은 "
            "10번 Human-Agnostic과 8번 승인 변경 마스크입니다. "
            "13~21번은 CatVTON 호환 전처리 진단이며 현재 Inpaint 마스크를 대체하지 않습니다. "
            "승인 전에는 의상 생성을 실행하지 않습니다."
        )
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)

        mask_refinement = comparison_candidate.mask_refinement
        foreground_candidate = comparison_candidate.foreground_candidate
        agnostic_candidate = comparison_candidate.human_agnostic_candidate
        removal_verification = (
            comparison_candidate.clothing_removal_verification
        )
        self.guard_decision = create_human_agnostic_guard_decision(
            remaining_clothing_pixel_count=(
                removal_verification.remaining_clothing_pixel_count
            ),
            removal_percent=removal_verification.removal_percent,
            removal_status=removal_verification.status,
            protected_clothing_overlap_pixel_count=removal_verification.protected_overlap_pixel_count,
            input_change_mask_pixel_count=input_snapshot.change_mask_pixel_count,
            input_protected_overlap_pixel_count=(
                input_snapshot.protected_overlap_pixel_count
            ),
            input_outside_foreground_pixel_count=(
                input_snapshot.outside_foreground_pixel_count
            ),
            processed_mask_pixel_count=(
                preflight_candidate.processed_mask_pixel_count
            ),
            model_mask_pixel_count=preflight_candidate.model_mask_pixel_count,
            soft_overlap_pixel_count=(
                preflight_candidate.soft_overlap_pixel_count
            ),
            hard_overlap_pixel_count=(
                preflight_candidate.hard_overlap_pixel_count
            ),
            final_protected_overlap_pixel_count=(
                preflight_candidate.protected_overlap_pixel_count
            ),
            final_outside_foreground_pixel_count=(
                preflight_candidate.outside_foreground_pixel_count
            ),
        )

        input_group = QGroupBox("A. CatVTON 전처리 전 입력 5개")
        input_group_layout = QVBoxLayout(input_group)
        input_preview_grid = QGridLayout()
        input_preview_items = (
            ("입력 1. 기준 캐릭터 원본", input_snapshot.person_image),
            (
                "입력 2. 실제 변경 마스크 원천",
                input_snapshot.approved_change_mask,
            ),
            (
                "입력 3. 추출된 참조 의상",
                input_snapshot.clothing_condition_image,
            ),
            (
                "입력 4. 변경 금지 보호 영역",
                input_snapshot.identity_protection_mask,
            ),
            (
                "입력 5. 캐릭터 외곽 허용 범위",
                input_snapshot.expanded_foreground_mask,
            ),
        )
        for preview_index, (preview_title, preview_image) in enumerate(
            input_preview_items
        ):
            preview_label = ClothingMaskReviewDialog.create_preview_label(
                preview_title
            )
            preview_label.setMinimumSize(300, 320)
            ClothingMaskReviewDialog.set_preview_image(
                preview_label,
                preview_image,
            )
            input_preview_grid.addWidget(
                preview_label,
                preview_index // 4,
                preview_index % 4,
            )
        input_group_layout.addLayout(input_preview_grid)
        input_measurement_label = QLabel(
            f"캐릭터 좌표={input_snapshot.person_image.width}x"
            f"{input_snapshot.person_image.height}, "
            f"참조 의상 크기="
            f"{input_snapshot.clothing_condition_image.width}x"
            f"{input_snapshot.clothing_condition_image.height}, "
            f"변경 마스크={input_snapshot.change_mask_pixel_count:,}px, "
            f"보호 영역 침범="
            f"{input_snapshot.protected_overlap_pixel_count:,}px, "
            f"캐릭터 외곽 밖 침범="
            f"{input_snapshot.outside_foreground_pixel_count:,}px, "
            f"입력 판정={input_snapshot.reason_ko}"
        )
        input_measurement_label.setWordWrap(True)
        input_group_layout.addWidget(input_measurement_label)
        input_hash_label = QLabel(
            f"입력 SHA-256: 캐릭터={input_snapshot.person_sha256}, "
            f"변경 마스크={input_snapshot.change_mask_sha256}, "
            f"참조 의상={input_snapshot.clothing_sha256}"
        )
        input_hash_label.setWordWrap(True)
        input_group_layout.addWidget(input_hash_label)
        content_layout.addWidget(input_group)

        processing_group = QGroupBox("B. 실제 적용 마스크와 호환 전처리 진단 구분")
        processing_group_layout = QVBoxLayout(processing_group)
        preview_grid = QGridLayout()
        preview_items = (
            ("1. 캐릭터 원본", comparison_candidate.source_image),
            ("2. 캐릭터 전체 외곽", mask_refinement.character_foreground_mask),
            ("3. 캐릭터 외곽 확장", mask_refinement.expanded_foreground_mask),
            ("4. DensePose 신체 표면", comparison_candidate.densepose_preview_image),
            (f"5. 실제 교체 영역 원천: {comparison_candidate.mask_source}", mask_refinement.raw_mask),
            ("6. 구멍 닫기 후", mask_refinement.closed_mask),
            ("7. 의상 외곽 팽창 후", mask_refinement.expanded_mask),
            ("8. 캐릭터 외곽·보호 제한 후", mask_refinement.safe_change_mask),
            ("9. 신체 보호 영역", mask_refinement.identity_protection_mask),
            (
                "10. Human-Agnostic Image 승인 후보",
                agnostic_candidate.neutralized_image,
            ),
            (
                "11. 캐릭터 외곽 밖 SCHP 오탐",
                removal_verification.outside_foreground_mask,
            ),
            (
                "12. 캐릭터 외곽 안의 기존 의상 잔여",
                removal_verification.remaining_clothing_mask,
            ),
            ("13. 실제 처리 크기 Person 입력", preflight_candidate.processed_person_image),
            ("14. resize·이분화 후 승인 마스크", preflight_candidate.processed_binary_mask),
            (
                "15. blur 직후 원본 마스크",
                preflight_candidate.raw_blurred_mask_image,
            ),
            ("16. 1~127 약한 침범", preflight_candidate.soft_overlap_mask),
            ("17. 128~255 강한 침범", preflight_candidate.hard_overlap_mask),
            (
                "18. 금지 영역 제한 후 실제 model_mask",
                preflight_candidate.model_mask_image,
            ),
            ("19. padding 후 참조 의상 입력", preflight_candidate.processed_clothing_image),
            ("20. 최종 model_mask의 보호 영역 침범", preflight_candidate.protected_overlap_mask),
            ("21. 최종 model_mask의 캐릭터 외곽 밖 침범", preflight_candidate.outside_foreground_mask),
        )
        if removal_verification.protected_conflict_mask is not None:
            preview_items += (("22. 교체 의상·보호 충돌 (재선택 필요)",
                               removal_verification.protected_conflict_mask),)
        if comparison_candidate.automatic_change_mask is not None:
            preview_items += (("23. AutoMasker 진단 후보 (적용 안 함)",
                               comparison_candidate.automatic_change_mask),)
        for preview_index, (preview_title, preview_image) in enumerate(
            preview_items
        ):
            preview_label = ClothingMaskReviewDialog.create_preview_label(
                preview_title
            )
            preview_label.setMinimumSize(300, 320)
            ClothingMaskReviewDialog.set_preview_image(
                preview_label,
                preview_image,
            )
            preview_grid.addWidget(
                preview_label,
                preview_index // 4,
                preview_index % 4,
            )
        processing_group_layout.addLayout(preview_grid)

        measurement_label = QLabel(
            f"모델={comparison_candidate.model_ids}, "
            f"캐릭터 외곽={foreground_candidate.foreground_pixel_count:,}px "
            f"({foreground_candidate.foreground_percent:.3f}%), "
            f"외곽 AI 시간={foreground_candidate.elapsed_seconds:.2f}초, "
            f"외곽 팽창={mask_refinement.foreground_expansion_pixels}px, "
            f"캐릭터 외부에서 제거="
            f"{mask_refinement.outside_foreground_rejected_pixel_count:,}px, "
            f"닫기 반경={mask_refinement.closing_radius_pixels}px, "
            f"팽창 반경={mask_refinement.expansion_radius_pixels}px, "
            f"팽창 중 보호 영역 접촉="
            f"{mask_refinement.attempted_protected_overlap_pixels:,}px, "
            f"최종 변경 영역={mask_refinement.safe_change_pixel_count:,}px "
            f"({mask_refinement.safe_change_percent:.3f}%), "
            f"중립 RGB={agnostic_candidate.neutral_rgb}, "
            f"중립화={agnostic_candidate.neutralized_pixel_count:,}px "
            f"({agnostic_candidate.neutralized_percent:.3f}%), "
            f"원본 마스크 포함률="
            f"{agnostic_candidate.raw_mask_coverage_percent:.3f}%, "
            f"기존 의상 탐지="
            f"{removal_verification.detected_clothing_pixel_count:,}px, "
            f"캐릭터 외곽 밖 SCHP 오탐="
            f"{removal_verification.outside_foreground_pixel_count:,}px "
            f"({removal_verification.outside_foreground_percent:.3f}%), "
            f"신체 보호 겹침="
            f"{removal_verification.protected_overlap_pixel_count:,}px, "
            f"제거 검증 대상="
            f"{removal_verification.verifiable_clothing_pixel_count:,}px, "
            f"검증 대상 포함="
            f"{removal_verification.removed_clothing_pixel_count:,}px, "
            f"캐릭터 외곽 안 잔여="
            f"{removal_verification.remaining_clothing_pixel_count:,}px, "
            f"기존 의상 제거율="
            f"{removal_verification.removal_percent if removal_verification.removal_percent is not None else '계산 불가'}, "
            f"범위 판정={removal_verification.status}: {removal_verification.reason_ko}, "
            f"중립화 마스크 밖 변경="
            f"{agnostic_candidate.changed_pixel_count_outside_mask:,}px, "
            f"Preflight 크기={preflight_candidate.width}x{preflight_candidate.height}, "
            f"이분화 마스크={preflight_candidate.processed_mask_pixel_count:,}px, "
            f"blur={preflight_candidate.blur_factor}, "
            f"실제 model_mask={preflight_candidate.model_mask_pixel_count:,}px, "
            f"약한 침범(1~127)="
            f"{preflight_candidate.soft_overlap_pixel_count:,}px, "
            f"강한 침범(128~255)="
            f"{preflight_candidate.hard_overlap_pixel_count:,}px, "
            f"금지 영역에서 제거="
            f"{preflight_candidate.removed_pixel_count:,}px, "
            f"보호 영역 침범={preflight_candidate.protected_overlap_pixel_count:,}px, "
            f"캐릭터 외곽 밖 침범={preflight_candidate.outside_foreground_pixel_count:,}px, "
            f"Preflight 판정={preflight_candidate.reason_ko}, "
            f"처리 시간={comparison_candidate.elapsed_seconds:.2f}초"
        )
        measurement_label.setWordWrap(True)
        processing_group_layout.addWidget(measurement_label)
        content_layout.addWidget(processing_group)

        guard_group = QGroupBox("C. Human-Agnostic 승인 가드레일 판정")
        guard_group_layout = QVBoxLayout(guard_group)
        severity_labels = {
            GuardSeverity.PASS: "PASS",
            GuardSeverity.WARNING: "WARNING",
            GuardSeverity.BLOCK: "BLOCK",
        }
        for guard_result in self.guard_decision.results:
            measured_text = (
                "계산 불가"
                if guard_result.measured_value is None
                else f"{guard_result.measured_value:,}{guard_result.unit}"
            )
            threshold_text = (
                "계산 불가"
                if guard_result.threshold_value is None
                else f"{guard_result.threshold_value:,}{guard_result.unit}"
            )
            corrected_text = (
                "해당 없음"
                if guard_result.corrected_value is None
                else f"{guard_result.corrected_value:,}{guard_result.unit}"
            )
            guard_label = QLabel(
                f"[{severity_labels[guard_result.severity]}] "
                f"{guard_result.code} | 측정={measured_text}, "
                f"기준={threshold_text}, 보정 후={corrected_text} | "
                f"{guard_result.message_ko}"
            )
            guard_label.setWordWrap(True)
            guard_group_layout.addWidget(guard_label)

        guard_summary_label = QLabel(
            f"가드레일 합계: PASS={len(self.guard_decision.passed_results)}개, "
            f"WARNING={len(self.guard_decision.warning_results)}개, "
            f"BLOCK={len(self.guard_decision.blocking_results)}개, "
            "최종 승인 가능="
            f"{'YES' if self.guard_decision.approval_enabled else 'NO'}"
        )
        guard_summary_label.setWordWrap(True)
        guard_group_layout.addWidget(guard_summary_label)
        content_layout.addWidget(guard_group)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)

        button_layout = QHBoxLayout()
        approve_button = QPushButton("Human-Agnostic Image와 변경 영역 승인")
        approve_button.setEnabled(self.guard_decision.approval_enabled)
        if not self.guard_decision.approval_enabled:
            approve_button.setToolTip(
                "\n".join(
                    f"[{result.code}] {result.message_ko} "
                    f"복구: {result.recovery_action_ko}"
                    for result in self.guard_decision.blocking_results
                )
            )
        elif self.guard_decision.warning_results:
            approve_button.setToolTip(
                "보정 전 경고가 있지만 최종 BLOCK은 0개입니다. "
                "표시된 보정 결과를 확인한 뒤 승인할 수 있습니다."
            )
        reject_button = QPushButton("거절하고 다시 확인")
        approve_button.clicked.connect(self.approve_comparison)
        reject_button.clicked.connect(self.reject)
        button_layout.addWidget(approve_button)
        button_layout.addWidget(reject_button)
        layout.addLayout(button_layout)

    @Slot()
    def approve_comparison(self) -> None:
        if not self.guard_decision.approval_enabled:
            blocking_details = "\n".join(
                f"- [{result.code}] {result.message_ko}\n"
                f"  복구: {result.recovery_action_ko}"
                for result in self.guard_decision.blocking_results
            )
            QMessageBox.warning(
                self,
                "Human-Agnostic 승인 차단",
                "최종 BLOCK 항목이 있어 실행할 수 없습니다.\n"
                f"{blocking_details}",
            )
            return
        self.approved = True
        self.accept()


def create_white_background_clothing_preview(
    extraction_candidate: ClothingExtractionCandidate,
):
    """자동 잘라낸 투명 의상을 흰 배경에 합성해 GUI 확인본을 만든다."""
    cropped_rgba = extraction_candidate.extracted_image.crop(
        extraction_candidate.preview_crop_box
    )
    white_background = Image.new(
        "RGBA",
        cropped_rgba.size,
        (255, 255, 255, 255),
    )
    try:
        return Image.alpha_composite(
            white_background,
            cropped_rgba,
        ).convert("RGB")
    finally:
        cropped_rgba.close()
        white_background.close()


def create_clothing_mask_overlay_image(
    source_image,
    mask_image,
):
    """정규화 원본 위에 선택 마스크를 빨간색 50% 투명도로 겹친다."""
    source_rgba = source_image.convert("RGBA")
    red_layer = Image.new("RGBA", source_rgba.size, (255, 0, 0, 0))
    mask_alpha = mask_image.convert("L").point(
        lambda pixel: 128 if pixel > 0 else 0
    )
    red_layer.putalpha(mask_alpha)
    try:
        return Image.alpha_composite(source_rgba, red_layer).convert("RGB")
    finally:
        source_rgba.close()
        red_layer.close()
        mask_alpha.close()


class ReferencePreparationWorker(QObject):
    """참조 이미지 화질 검사와 필요한 확대 복원을 화면 밖에서 처리한다."""

    status_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)

    def __init__(self, reference_path: Path, config: dict):
        super().__init__()
        self.reference_path = reference_path
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            self.status_changed.emit("참조 이미지 화질 확인 중...")
            reference_image = load_reference_image_as_rgb(self.reference_path)
            generation_config = self.config["generation"]
            quality_config = self.config["reference_quality"]
            preparation_result = prepare_reference_image_for_review(
                reference_image=reference_image,
                target_width=int(generation_config["width"]),
                target_height=int(generation_config["height"]),
                quality_config=quality_config,
            )
            reference_image.close()
            self.completed.emit(preparation_result)
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())


class ReferenceEnhancementDialog(QDialog):
    """원본과 확대 복원본을 나란히 보여주고 사용자의 선택을 받는다."""

    def __init__(
        self,
        enhancement_candidate: ReferenceImageEnhancementCandidate,
        parent=None,
    ):
        super().__init__(parent)
        self.use_enhanced_image = False
        self.setWindowTitle("참조 이미지 보정 확인")
        self.resize(900, 700)
        layout = QVBoxLayout(self)
        report = enhancement_candidate.quality_report
        explanation = QLabel(
            f"{report.reason_ko}\n"
            f"원본={report.width}x{report.height}, "
            f"선명도 점수={report.sharpness_score:.1f}"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        comparison_layout = QHBoxLayout()
        for title, image in (
            ("원본", enhancement_candidate.original_image),
            ("확대·복원 후보", enhancement_candidate.enhanced_image),
        ):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setPixmap(
                create_pil_image_pixmap(image).scaled(
                    400,
                    520,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            column.addWidget(image_label)
            comparison_layout.addLayout(column)
        layout.addLayout(comparison_layout)

        button_layout = QHBoxLayout()
        approve_button = QPushButton("보정 이미지 승인하고 생성")
        reject_button = QPushButton("보정 이미지 거절")
        approve_button.clicked.connect(self.approve_enhanced_image)
        reject_button.clicked.connect(self.reject)
        button_layout.addWidget(approve_button)
        button_layout.addWidget(reject_button)
        layout.addLayout(button_layout)

    @Slot()
    def approve_enhanced_image(self) -> None:
        self.use_enhanced_image = True
        self.accept()


class PoseReferenceApprovalDialog(QDialog):
    """자세 참조 원본과 입력 수치를 보여주고 사용자 승인을 받는다."""

    def __init__(
        self,
        review_candidate: PoseReferenceReviewCandidate,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.is_approved = False
        self.setWindowTitle("자세 참조 이미지 확인")
        self.resize(720, 760)
        layout = QVBoxLayout(self)

        file_size_kib = review_candidate.file_size_bytes / 1024.0
        information_label = QLabel(
            "아직 관절 추출이나 이미지 생성에는 사용하지 않습니다.\n"
            f"파일={review_candidate.source_path.name}, "
            f"형식={review_candidate.image_format}, "
            f"크기={review_candidate.width}x{review_candidate.height}px, "
            f"전체={review_candidate.pixel_count:,}px, "
            f"가로/세로={review_candidate.aspect_ratio:.4f}, "
            f"파일 크기={file_size_kib:.1f}KiB"
        )
        information_label.setWordWrap(True)
        layout.addWidget(information_label)

        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setPixmap(
            create_pil_image_pixmap(review_candidate.image).scaled(
                620,
                600,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addWidget(preview_label)

        button_layout = QHBoxLayout()
        approve_button = QPushButton("자세 참조 승인")
        reject_button = QPushButton("거절")
        approve_button.clicked.connect(self.approve_pose_reference)
        reject_button.clicked.connect(self.reject)
        button_layout.addWidget(approve_button)
        button_layout.addWidget(reject_button)
        layout.addLayout(button_layout)

    @Slot()
    def approve_pose_reference(self) -> None:
        """승인 상태만 기록하고 다음 관절 추출은 실행하지 않는다."""
        self.is_approved = True
        self.accept()


class PoseEstimationApprovalDialog(QDialog):
    """자세 원본·관절 확인본·ControlNet 뼈대 지도의 승인을 받는다."""

    def __init__(
        self,
        review_candidate: PoseEstimationReviewCandidate,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.is_approved = False
        self.setWindowTitle("DWPose 관절 추출 확인")
        self.resize(1120, 760)
        layout = QVBoxLayout(self)
        information_label = QLabel(
            f"관절={review_candidate.detected_joint_count}/18개, "
            f"누락={review_candidate.missing_joint_count}/18개, "
            f"기준={review_candidate.minimum_pose_confidence * 100.0:.1f}%, "
            f"시간={review_candidate.elapsed_seconds:.2f}초\n"
            "초록 선·점은 기준을 통과한 관절입니다. 승인 전 ControlNet 호출은 0회입니다."
        )
        information_label.setWordWrap(True)
        layout.addWidget(information_label)

        comparison_layout = QHBoxLayout()
        for title, image in (
            ("1. 자세 원본", review_candidate.source_image),
            ("2. 원본 위 관절 확인", review_candidate.overlay_image),
            ("3. ControlNet용 뼈대 지도", review_candidate.control_map_image),
        ):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setPixmap(
                create_pil_image_pixmap(image).scaled(
                    340,
                    580,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            column.addWidget(image_label)
            comparison_layout.addLayout(column)
        layout.addLayout(comparison_layout)

        button_layout = QHBoxLayout()
        approve_button = QPushButton("관절과 뼈대 지도 승인")
        reject_button = QPushButton("거절하고 자세 다시 선택")
        approve_button.clicked.connect(self.approve_estimation)
        reject_button.clicked.connect(self.reject)
        button_layout.addWidget(approve_button)
        button_layout.addWidget(reject_button)
        layout.addLayout(button_layout)

    @Slot()
    def approve_estimation(self) -> None:
        """뼈대 지도 승인 상태만 기록하고 ControlNet은 실행하지 않는다."""
        self.is_approved = True
        self.accept()


class PoseFallbackApprovalDialog(QDialog):
    """DWPose 실패 원본과 저장된 승인 자세를 비교해 재승인을 받는다."""

    def __init__(
        self,
        failed_source_image: Image.Image,
        saved_pose: SavedApprovedPose,
        failure_reason: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.is_approved = False
        self.setWindowTitle("저장된 자세 폴백 확인")
        self.resize(1080, 760)
        layout = QVBoxLayout(self)
        information_label = QLabel(
            "요청한 자세는 DWPose 품질 기준을 통과하지 못했습니다.\n"
            f"실패 원인={failure_reason}\n"
            f"폴백 자세 ID={saved_pose.pose_id}, "
            f"관절={saved_pose.approved_pose.detected_joint_count}/18개, "
            f"기준={saved_pose.approved_pose.minimum_pose_confidence * 100.0:.1f}%, "
            f"SHA-256={saved_pose.control_map_sha256[:12]}, "
            f"저장 시각={saved_pose.saved_at}\n"
            "승인 전 ControlNet 호출은 0회이며, 승인하면 요청 자세 대신 "
            "저장된 자세가 적용됩니다."
        )
        information_label.setWordWrap(True)
        layout.addWidget(information_label)

        comparison_layout = QHBoxLayout()
        for title, image in (
            ("1. 추출 실패 자세 원본", failed_source_image),
            ("2. 저장된 승인 자세 원본", saved_pose.source_preview_image),
            ("3. 저장된 ControlNet 뼈대", saved_pose.approved_pose.control_map_image),
        ):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setPixmap(
                create_pil_image_pixmap(image).scaled(
                    330,
                    560,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            column.addWidget(image_label)
            comparison_layout.addLayout(column)
        layout.addLayout(comparison_layout)

        button_layout = QHBoxLayout()
        approve_button = QPushButton("저장된 자세를 적용하고 계속")
        reject_button = QPushButton("거절하고 자세 다시 선택")
        approve_button.clicked.connect(self.approve_fallback)
        reject_button.clicked.connect(self.reject)
        button_layout.addWidget(approve_button)
        button_layout.addWidget(reject_button)
        layout.addLayout(button_layout)

    @Slot()
    def approve_fallback(self) -> None:
        self.is_approved = True
        self.accept()


class DetailCorrectionComparisonDialog(QDialog):
    """보정 전후 후보를 비교하고 최종 후보로 사용할 이미지를 선택한다."""

    def __init__(
        self,
        character_candidate: CharacterGenerationCandidate,
        parent=None,
    ):
        super().__init__(parent)
        self.use_corrected_image = True
        self.setWindowTitle("얼굴·손 보정 결과 비교")
        self.resize(900, 700)
        layout = QVBoxLayout(self)
        summary = QLabel(
            f"얼굴 탐지={character_candidate.detected_face_count}, "
            f"손 탐지={character_candidate.detected_hand_count}, "
            f"보정 영역={character_candidate.corrected_region_count}\n"
            "보정 영역 밖의 픽셀은 원본 후보 그대로 유지했습니다."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        if character_candidate.detail_verification_warning_ko:
            warning = QLabel(
                "확인 필요: "
                f"{character_candidate.detail_verification_warning_ko}"
            )
            warning.setWordWrap(True)
            layout.addWidget(warning)

        comparison_layout = QHBoxLayout()
        images = (
            ("보정 전 후보", character_candidate.original_generated_image),
            ("얼굴·손 보정 후보", character_candidate.image),
        )
        for title, image in images:
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setPixmap(
                create_pil_image_pixmap(image).scaled(
                    400,
                    520,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            column.addWidget(image_label)
            comparison_layout.addLayout(column)
        layout.addLayout(comparison_layout)

        button_layout = QHBoxLayout()
        corrected_button = QPushButton("보정본을 최종 후보로 사용")
        original_button = QPushButton("보정 전 이미지를 최종 후보로 사용")
        corrected_button.clicked.connect(self.choose_corrected_image)
        original_button.clicked.connect(self.choose_original_image)
        button_layout.addWidget(corrected_button)
        button_layout.addWidget(original_button)
        layout.addLayout(button_layout)

    @Slot()
    def choose_corrected_image(self) -> None:
        self.use_corrected_image = True
        self.accept()

    @Slot()
    def choose_original_image(self) -> None:
        self.use_corrected_image = False
        self.accept()


class ClothingTryOnComparisonDialog(QDialog):
    """의상 합성 전후와 변경 허용 마스크를 보여주고 사용자가 선택한다."""

    def __init__(
        self,
        character_candidate: CharacterGenerationCandidate,
        parent=None,
    ):
        super().__init__(parent)
        self.use_clothing_image = True
        self.setWindowTitle("의상 참조 합성 결과 비교")
        self.resize(1180, 820)
        layout = QVBoxLayout(self)
        metrics = character_candidate.clothing_effect_metrics
        metrics_text = "효과 측정값 없음"
        if metrics is not None:
            metrics_text = (
                "원시 model_mask 안 변경="
                f"{metrics.raw_changed_inside_model_mask:,}px, "
                "최종 승인 영역 안 변경="
                f"{metrics.final_changed_inside_approved_mask:,}px, "
                "보호 합성 제거="
                f"{metrics.discarded_by_protection_pixels:,}px, "
                "승인 영역 밖 변경="
                f"{metrics.mask_leakage_pixels:,}px, "
                "RGB L1 평균="
                f"{metrics.mean_rgb_l1_inside:.4f}, "
                f"no_effect={metrics.no_effect}"
            )
        summary = QLabel(
            f"의상 참조={character_candidate.clothing_reference_name}, "
            f"종류={character_candidate.clothing_category}\n"
            "마스크의 흰색 영역 안에서만 의상 합성 결과를 사용했습니다.\n"
            f"{metrics_text}"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        if character_candidate.clothing_verification_warning_ko:
            warning = QLabel(
                character_candidate.clothing_verification_warning_ko
            )
            warning.setWordWrap(True)
            layout.addWidget(warning)

        comparison_layout = QGridLayout()
        comparison_images = (
            ("의상 적용 전", character_candidate.before_clothing_image),
            ("CatVTON 원시 출력", character_candidate.raw_clothing_try_on_image),
            ("의상 변경 허용 영역", character_candidate.clothing_change_mask),
            ("최종 보호 합성", character_candidate.image),
            ("변경 차이맵 (4배)", character_candidate.clothing_difference_image),
        )
        visible_index = 0
        for title, image in comparison_images:
            if image is None:
                continue
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setPixmap(
                create_pil_image_pixmap(image).scaled(
                    300,
                    330,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            column.addWidget(image_label)
            comparison_layout.addLayout(
                column,
                visible_index // 3,
                visible_index % 3,
            )
            visible_index += 1
        layout.addLayout(comparison_layout)

        button_layout = QHBoxLayout()
        clothing_button = QPushButton("의상 적용본 사용")
        original_button = QPushButton("의상 적용 전 후보 사용")
        if metrics is not None and metrics.no_effect:
            clothing_button.setEnabled(False)
            clothing_button.setToolTip(
                "최종 승인 영역 안 변경이 0px이므로 적용본을 승인할 수 없습니다."
            )
        clothing_button.clicked.connect(self.choose_clothing_image)
        original_button.clicked.connect(self.choose_original_image)
        button_layout.addWidget(clothing_button)
        button_layout.addWidget(original_button)
        layout.addLayout(button_layout)

    @Slot()
    def choose_clothing_image(self) -> None:
        self.use_clothing_image = True
        self.accept()

    @Slot()
    def choose_original_image(self) -> None:
        self.use_clothing_image = False
        self.accept()



def add_image_review_grid(
    parent_layout: QVBoxLayout,
    preview_items: tuple[tuple[str, Image.Image], ...],
    columns: int = 3,
) -> None:
    """검토 이미지를 3열 스크롤 그리드에 배치한다."""
    grid = QGridLayout()
    for index, (title, image) in enumerate(preview_items):
        label = ClothingMaskReviewDialog.create_preview_label(title)
        label.setMinimumSize(280, 300)
        ClothingMaskReviewDialog.set_preview_image(label, image)
        grid.addWidget(label, index // columns, index % columns)
    parent_layout.addLayout(grid)


class GarmentTpsReviewDialog(QDialog):
    """TPS 조각 대응·좌표·승인 마스크 제한 결과 6개를 공개한다."""

    def __init__(
        self,
        candidate: GarmentWarpReviewCandidate,
        source_garment: Image.Image,
        component_matches: GarmentComponentMatchResult,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("의상 조각 대응과 TPS 좌표 승인")
        available = self.screen().availableGeometry()
        self.resize(
            min(1180, max(760, available.width() - 80)),
            min(860, max(560, available.height() - 80)),
        )
        self.approved = False
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        add_image_review_grid(
            content_layout,
            (
                ("1. 추출 의상 원본", source_garment),
                ("2. 승인 변경 마스크", candidate.approved_mask_preview),
                ("3. TPS 원본 합성", candidate.raw_combined_rgba),
                ("4. 승인 영역 밖 알파", candidate.outside_approved_mask),
                ("5. 영역 제한 TPS", candidate.protected_combined_rgba),
                ("6. 캐릭터 위 좌표 오버레이", candidate.overlay_preview),
            ),
        )
        proposal_lines = ", ".join(
            f"조각 {proposal.source_component_index}→{proposal.target_slot.value}"
            f"({proposal.rule_fit_score:.3f})"
            for proposal in component_matches.proposals
        )
        metrics = QLabel(
            f"조각={candidate.component_count}개, "
            f"모호={candidate.ambiguous_component_count}개, "
            f"공유 슬롯={candidate.shared_target_slot_component_count}개, "
            f"조각 겹침={candidate.component_hard_overlap_pixels:,}px, "
            f"승인 밖 soft={candidate.outside_soft_alpha_pixels:,}px, "
            f"승인 밖 hard={candidate.outside_hard_alpha_pixels:,}px, "
            f"보호 후 승인 밖={candidate.protected_outside_alpha_pixels:,}px, "
            f"자동 저장={candidate.automatic_save_count}개, "
            f"시간={candidate.elapsed_seconds:.3f}초\n대응: {proposal_lines}"
        )
        metrics.setWordWrap(True)
        content_layout.addWidget(metrics)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        buttons = QHBoxLayout()
        approve = QPushButton("조각 대응과 TPS 승인")
        reject = QPushButton("거절하고 중지")
        approve.setEnabled(
            candidate.component_count >= 1
            and candidate.protected_outside_alpha_pixels == 0
            and candidate.automatic_save_count == 0
        )
        approve.clicked.connect(self._approve)
        reject.clicked.connect(self.reject)
        buttons.addWidget(approve)
        buttons.addWidget(reject)
        layout.addLayout(buttons)

    @Slot()
    def _approve(self) -> None:
        self.approved = True
        self.accept()


class GarmentLineartReviewDialog(QDialog):
    """TPS 의상에서 추출한 Lineart 제어 입력 6개를 공개한다."""

    def __init__(
        self,
        candidate: GarmentLineartReviewCandidate,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("의상 Lineart 제어 입력 승인")
        available = self.screen().availableGeometry()
        self.resize(
            min(1180, max(760, available.width() - 80)),
            min(860, max(560, available.height() - 80)),
        )
        self.approved = False
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        add_image_review_grid(
            content_layout,
            (
                ("1. 흰 배경 TPS 의상", candidate.white_background_garment_preview),
                ("2. 외곽선", candidate.outer_boundary_mask),
                ("3. 내부 디테일", candidate.internal_detail_mask),
                ("4. 결합 선 마스크", candidate.combined_edge_mask),
                ("5. ControlNet 입력", candidate.control_image),
                ("6. 선 오버레이", candidate.overlay_preview),
            ),
        )
        metrics = QLabel(
            f"의상 알파={candidate.visible_alpha_pixels:,}px, "
            f"외곽선={candidate.raw_outer_boundary_pixels:,}px, "
            f"내부선={candidate.raw_internal_detail_pixels:,}px, "
            f"최종선={candidate.total_edge_pixels:,}px, "
            f"밀도={candidate.edge_density_percent:.3f}%, "
            f"승인 밖 원시선={candidate.raw_edge_pixels_outside_approved_mask:,}px, "
            f"보호 후 승인 밖={candidate.protected_edge_pixels_outside_approved_mask:,}px, "
            f"자동 저장={candidate.automatic_save_count}개, "
            f"시간={candidate.elapsed_seconds:.3f}초"
        )
        metrics.setWordWrap(True)
        content_layout.addWidget(metrics)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        buttons = QHBoxLayout()
        approve = QPushButton("Lineart 입력 승인")
        reject = QPushButton("거절하고 중지")
        approve.setEnabled(
            candidate.total_edge_pixels >= candidate.settings.minimum_edge_pixels
            and candidate.protected_edge_pixels_outside_approved_mask == 0
            and candidate.alpha_consistency_mismatch_pixels == 0
            and candidate.automatic_save_count == 0
        )
        approve.clicked.connect(self._approve)
        reject.clicked.connect(self.reject)
        buttons.addWidget(approve)
        buttons.addWidget(reject)
        layout.addLayout(buttons)

    @Slot()
    def _approve(self) -> None:
        self.approved = True
        self.accept()


class GarmentInpaintReviewDialog(QDialog):
    """생성 입력·결과 8개와 중립색 잔여 진단을 공개한다."""

    def __init__(
        self,
        candidate: GarmentInpaintReviewCandidate,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("2D 의상 Inpaint 결과 승인")
        available = self.screen().availableGeometry()
        self.resize(
            min(1180, max(760, available.width() - 80)),
            min(860, max(560, available.height() - 80)),
        )
        self.approved = False
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        add_image_review_grid(
            content_layout,
            (
                ("1. 기준 캐릭터", candidate.base_character_preview),
                ("2. Human-Agnostic 시작 이미지", candidate.human_agnostic_preview),
                ("3. 승인 Inpaint 마스크", candidate.approved_mask_preview),
                ("4. 승인 의상 원본", candidate.garment_reference_preview),
                (
                    "5. IP-Adapter Plus 실제 참조 보드",
                    candidate.garment_reference_board_preview,
                ),
                ("6. 모델 원시 출력", candidate.raw_inpaint_output),
                ("7. 승인 영역 보호 출력", candidate.protected_output),
                ("8. 기준 대비 차이 ×4", candidate.difference_preview),
            ),
        )
        residual = candidate.neutral_residual
        if residual is not None:
            add_image_review_grid(content_layout, (
                ("9. 중립색 잔여 의심 영역 (흰색, 경고 전용)", residual.mask),
            ))
            percent = (
                "계산 불가" if residual.suspected_percent is None
                else f"{residual.suspected_percent:.2f}%"
            )
            warning = QLabel(
                f"중립색 잔여 후보={residual.suspected_pixel_count:,}/"
                f"{residual.evaluated_pixel_count:,}px ({percent}). "
                "시작 이미지와 결과가 모두 중립색 근처인 영역입니다. "
                "실제 회색 의상일 수도 있으므로 생성 실패로 단정하거나 승인을 차단하지 않습니다."
            )
            warning.setWordWrap(True)
            content_layout.addWidget(warning)
        metrics = QLabel(
            f"하드 마스크={candidate.inpaint_mask_pixels:,}px, "
            f"소프트 마스크={candidate.inpaint_soft_mask_pixels:,}px, "
            f"원시 출력 내부 변경={candidate.raw_changed_inside_mask_pixels:,}px, "
            f"Human-Agnostic 대비 내부 변경="
            f"{candidate.raw_changed_from_initial_inside_mask_pixels:,}px, "
            f"보호 출력 내부 변경={candidate.protected_changed_inside_mask_pixels:,}px, "
            f"보호 출력 외부 변경={candidate.protected_changed_outside_mask_pixels:,}px, "
            f"평균 RGB L1={candidate.mean_rgb_l1_inside_mask:.3f}, "
            f"의상 조각={candidate.garment_retained_component_count}/"
            f"{candidate.garment_source_component_count}개, "
            f"참조 보드 점유={candidate.garment_board_occupied_pixel_count:,}px, "
            f"자동 저장={candidate.automatic_save_count}개, "
            f"벤치마크 파일={candidate.benchmark_file_count}개, "
            f"벤치마크 경로={candidate.benchmark_directory}, "
            f"파이프라인 로딩="
            f"{candidate.execution_metrics.pipeline_load_seconds:.3f}초, "
            f"IP-Adapter 로딩="
            f"{candidate.execution_metrics.ip_adapter_load_seconds:.3f}초, "
            f"Diffusion="
            f"{candidate.execution_metrics.diffusion_seconds:.3f}초, "
            f"출력 저장="
            f"{candidate.execution_metrics.output_save_seconds:.3f}초, "
            f"실행기 전체="
            f"{candidate.execution_metrics.runner_total_seconds:.3f}초, "
            f"부모 전체="
            f"{candidate.execution_metrics.parent_total_seconds:.3f}초, "
            f"진행 이벤트="
            f"{candidate.execution_metrics.progress_event_count}개, "
            f"잘못된 이벤트="
            f"{candidate.execution_metrics.invalid_progress_event_count}개, "
            f"Heartbeat="
            f"{candidate.execution_metrics.heartbeat_count}회, "
            f"기존 전체 시간={candidate.elapsed_seconds:.3f}초"
        )
        metrics.setWordWrap(True)
        content_layout.addWidget(metrics)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        buttons = QHBoxLayout()
        approve = QPushButton("2D 의상 합성 승인")
        reject = QPushButton("거절하고 중지")
        approve.setEnabled(
            candidate.protected_changed_inside_mask_pixels > 0
            and candidate.raw_changed_from_initial_inside_mask_pixels > 0
            and candidate.protected_changed_outside_mask_pixels == 0
            and candidate.automatic_save_count == 0
        )
        approve.clicked.connect(self._approve)
        reject.clicked.connect(self.reject)
        buttons.addWidget(approve)
        buttons.addWidget(reject)
        layout.addLayout(buttons)

    @Slot()
    def _approve(self) -> None:
        self.approved = True
        self.accept()


class GarmentGeometryWorker(QObject):
    """승인된 입력 복사본으로 좌표 추출·조각 대응·TPS 후보를 만든다."""

    status_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        garment_rgba: Image.Image,
        base_character: Image.Image,
        approved_change_mask: Image.Image,
        approved_pose: PoseEstimationApprovedInput,
        clothing_category: ClothingCategory,
    ) -> None:
        super().__init__()
        self.garment_rgba = garment_rgba.copy()
        self.base_character = base_character.copy()
        self.approved_change_mask = approved_change_mask.copy()
        self.approved_pose = PoseEstimationApprovedInput(
            control_map_image=approved_pose.control_map_image.copy(),
            joint_coordinates=approved_pose.joint_coordinates,
            detected_joint_count=approved_pose.detected_joint_count,
            missing_joint_count=approved_pose.missing_joint_count,
            minimum_pose_confidence=approved_pose.minimum_pose_confidence,
            model_ids=approved_pose.model_ids,
        )
        self.clothing_category = clothing_category

    @Slot()
    def run(self) -> None:
        prepared_pose = None
        try:
            self.status_changed.emit("의상 조각·캐릭터 목표 좌표·TPS 계산 중...")
            source_mask = self.garment_rgba.getchannel("A")
            try:
                source_landmarks = extract_garment_mask_landmarks(source_mask)
            finally:
                source_mask.close()
            prepared_pose = prepare_pose_control_input(
                self.approved_pose,
                self.base_character.width,
                self.base_character.height,
            )
            target_landmarks = extract_character_target_landmarks(
                self.approved_pose.joint_coordinates,
                prepared_pose,
                self.approved_change_mask,
                self.clothing_category,
            )
            component_matches = propose_garment_component_matches(
                source_landmarks,
                self.clothing_category,
            )
            review = create_garment_tps_warp_review(
                self.garment_rgba,
                self.base_character,
                self.approved_change_mask,
                source_landmarks,
                target_landmarks,
                component_matches,
            )
            self.completed.emit((review, component_matches))
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())
        finally:
            if prepared_pose is not None:
                prepared_pose.close()
            self.garment_rgba.close()
            self.base_character.close()
            self.approved_change_mask.close()
            self.approved_pose.close()


def build_garment_inpaint_prompts(
    base_prompt: str,
    base_negative_prompt: str,
    design_tags: tuple[str, ...],
) -> tuple[str, str]:
    """앱이 넣은 기존 의상 유지 토큰만 제외하고 새 의상 조건을 만든다."""
    blocked_positive = {"matching outfit and colors"}
    blocked_negative = {"different outfit", "mismatched colors"}

    def split_tokens(text: str) -> list[str]:
        return [token.strip() for token in text.split(",") if token.strip()]

    def unique_tokens(tokens: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            key = token.casefold()
            if key not in seen:
                seen.add(key)
                result.append(token)
        return result

    base_positive_tokens = [
        token for token in split_tokens(base_prompt)
        if token.casefold() not in blocked_positive
    ]
    garment_priority_tokens = [
        tag.strip() for tag in design_tags if tag.strip()
    ]
    garment_priority_tokens.extend((
        "reference garment",
        "preserve garment color pattern seams accessories",
    ))
    positive_tokens = garment_priority_tokens + base_positive_tokens
    negative_tokens = [
        token for token in split_tokens(base_negative_prompt)
        if token.casefold() not in blocked_negative
    ]
    return (
        ", ".join(unique_tokens(positive_tokens)),
        ", ".join(unique_tokens(negative_tokens)),
    )


class GarmentInpaintWorker(QObject):
    """Human-Agnostic 승인본을 별도 2D Inpaint 프로세스에 전달한다."""

    status_changed = Signal(str)
    progress_changed = Signal(object)
    completed = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        base_character: Image.Image,
        approved_human_agnostic_image: Image.Image,
        approved_change_mask: Image.Image,
        garment_reference: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        settings: GarmentInpaintSettings,
    ) -> None:
        super().__init__()
        self.base_character = base_character.copy()
        self.approved_human_agnostic_image = (
            approved_human_agnostic_image.copy()
        )
        self.approved_change_mask = approved_change_mask.copy()
        self.garment_reference = garment_reference.copy()
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            self.status_changed.emit(
                "SDXL Inpaint + IP-Adapter Plus 의상 생성 중..."
            )
            result = execute_garment_inpaint(
                self.base_character,
                self.approved_human_agnostic_image,
                self.approved_change_mask,
                self.garment_reference,
                self.prompt,
                self.negative_prompt,
                self.seed,
                self.settings,
                progress_callback=self.progress_changed.emit,
            )
            self.completed.emit(result)
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())
        finally:
            self.base_character.close()
            self.approved_human_agnostic_image.close()
            self.approved_change_mask.close()
            self.garment_reference.close()


class GenerationWorker(QObject):
    """모델 로딩과 이미지 생성을 화면 실행 흐름 밖에서 처리한다."""

    status_changed = Signal(str)
    completed = Signal(object, object)
    failed = Signal(str, str, object)

    def __init__(
        self,
        config,
        generation_request: CharacterGenerationRequest,
        current_dir,
        run_log: GenerationRunLog,
        clothing_reference_input: ClothingReferenceInput | None,
        catvton_settings: CatVTONLocalSettings | None,
        approved_agnostic_input: CharacterAgnosticApprovedInput | None,
        approved_pose_estimation: PoseEstimationApprovedInput | None,
        existing_candidate: CharacterGenerationCandidate | None = None,
        pipeline=None,
    ):
        super().__init__()
        self.config = config
        self.generation_request = generation_request
        self.current_dir = current_dir
        self.run_log = run_log
        self.pipeline = pipeline
        self.clothing_reference_input = clothing_reference_input
        self.catvton_settings = catvton_settings
        self.approved_agnostic_input = approved_agnostic_input
        self.approved_pose_estimation = (
            PoseEstimationApprovedInput(
                control_map_image=(
                    approved_pose_estimation.control_map_image.copy()
                ),
                joint_coordinates=approved_pose_estimation.joint_coordinates,
                detected_joint_count=(
                    approved_pose_estimation.detected_joint_count
                ),
                missing_joint_count=approved_pose_estimation.missing_joint_count,
                minimum_pose_confidence=(
                    approved_pose_estimation.minimum_pose_confidence
                ),
                model_ids=approved_pose_estimation.model_ids,
            )
            if approved_pose_estimation is not None
            else None
        )
        self.existing_candidate = existing_candidate

    @Slot()
    def run(self):
        execution_started_at = perf_counter()
        try:
            self.run_log.write_stage("환경 검사", "GPU와 필수 도구 확인 시작")
            configure_system_certificates()
            environment = check_environment()
            gpu_memory_gb = environment["vram_bytes"] / (1024**3)
            self.run_log.write_stage(
                "환경 검사",
                f"GPU={environment['gpu']}, GPU 메모리={gpu_memory_gb:.1f}GB",
            )
            pose_control_enabled = (
                self.existing_candidate is None
                and self.approved_pose_estimation is not None
            )
            if self.pipeline is None and self.existing_candidate is None:
                model_started_at = perf_counter()
                self.status_changed.emit("모델과 참조 그림 장치 준비 중...")
                self.run_log.write_stage("모델 준비", "모델 불러오기 시작")
                self.pipeline = prepare_pipeline(
                    self.config,
                    pose_control_enabled=pose_control_enabled,
                )
                self.run_log.write_stage(
                    "모델 준비",
                    f"완료, 소요 시간={perf_counter() - model_started_at:.1f}초",
                )
            elif self.pipeline is not None:
                self.run_log.write_stage("모델 준비", "메모리에 있는 모델 재사용")
            else:
                self.run_log.write_stage(
                    "모델 준비",
                    "승인된 기존 후보의 의상 적용만 실행하므로 베이스 모델 로딩 생략",
                )

            generation_started_at = perf_counter()
            if self.existing_candidate is None:
                self.status_changed.emit("기준 후보 이미지 생성 중...")
                self.run_log.write_stage("이미지 생성", "기준 후보 1번 생성 시작")
                character_candidate = generate_character_candidate(
                    self.pipeline,
                    self.config,
                    self.generation_request,
                    self.current_dir,
                    self.run_log,
                    approved_pose_estimation=self.approved_pose_estimation,
                )
            else:
                if (
                    self.clothing_reference_input is None
                    or self.catvton_settings is None
                    or self.approved_agnostic_input is None
                ):
                    raise ValueError("기존 후보 의상 적용 입력이 완성되지 않았습니다.")
                self.status_changed.emit("승인된 의상을 기준 후보에 적용 중...")
                if hasattr(self.pipeline, "maybe_free_model_hooks"):
                    self.pipeline.maybe_free_model_hooks()
                import torch

                torch.cuda.empty_cache()
                character_candidate = apply_clothing_to_generated_candidate(
                    base_candidate=self.existing_candidate,
                    clothing_reference_input=self.clothing_reference_input,
                    catvton_settings=self.catvton_settings,
                    approved_agnostic_input=self.approved_agnostic_input,
                    run_log=self.run_log,
                )
            self.run_log.write_stage(
                "이미지 생성",
                f"완료, 소요 시간={perf_counter() - generation_started_at:.1f}초",
            )
            self.run_log.write_stage(
                "실행 완료",
                (
                    f"전체 소요 시간={perf_counter() - execution_started_at:.1f}초, "
                    "후보 파일 저장 없음, GUI 메모리 전달 완료"
                ),
            )
            self.completed.emit(character_candidate, self.pipeline)
        except Exception as error:
            details = traceback.format_exc()
            self.run_log.write_failure(
                "이미지 생성 실행",
                error,
                find_recovery_action(error),
            )
            details_with_log = (
                f"로그 파일: {self.run_log.file_path}\n\n{details}"
            )
            self.failed.emit(str(error), details_with_log, self.pipeline)
        finally:
            self.run_log.close()
            if self.approved_agnostic_input is not None:
                self.approved_agnostic_input.close()
                self.approved_agnostic_input = None
            if self.approved_pose_estimation is not None:
                self.approved_pose_estimation.close()
                self.approved_pose_estimation = None


class GenAILabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenAI Lab - 캐릭터 후보 이미지 생성기")
        self.resize(800, 900)

        # 백엔드 파이프라인 및 설정 저장 변수
        self.pipeline = None
        self.config = None
        self.worker = None
        self.worker_thread = None
        self.garment_inpaint_start_deferred = False
        self.reference_worker = None
        self.reference_worker_thread = None
        self.outfit_worker = None
        self.outfit_worker_thread = None
        self.mask_worker = None
        self.mask_worker_thread = None
        self.design_worker = None
        self.design_worker_thread = None
        self.body_comparison_worker = None
        self.body_comparison_worker_thread = None
        self.pose_estimation_worker = None
        self.pose_estimation_worker_thread = None
        self.garment_geometry_worker = None
        self.garment_geometry_worker_thread = None
        self.garment_inpaint_worker = None
        self.garment_inpaint_worker_thread = None
        self.confirmed_character_body_comparison: ConfirmedCharacterBodyComparison | None = None
        self.body_comparison_clothing_category: ClothingCategory | None = None
        self.pending_clothing_source: NormalizedClothingSource | None = None
        self.selected_clothing_mask: ClothingCombinedMaskCandidate | None = None
        self.clothing_mask_result: ClothingMaskExtractionResult | None = None
        self.pending_clothing_extraction: ClothingExtractionCandidate | None = None
        self.clothing_design_result: ClothingDesignAnalysisResult | None = None
        self.confirmed_clothing_design: ClothingDesignSummary | None = None
        self.pending_outfit_path: Path | None = None
        self.clothing_region_candidates: tuple[ClothingRegionCandidate, ...] = ()
        self.clothing_source_size: tuple[int, int] | None = None
        self.clothing_region_measurements = ()
        self.approved_reference_image: ApprovedReferenceImage | None = None
        self.approved_pose_reference: PoseReferenceApprovedInput | None = None
        self.approved_pose_estimation: PoseEstimationApprovedInput | None = None
        self.pending_character_candidate: CharacterGenerationCandidate | None = None
        self.pending_clothing_base_candidate: CharacterGenerationCandidate | None = None
        self.approved_garment_warp: GarmentWarpApprovedInput | None = None
        self.approved_garment_lineart: GarmentLineartApprovedInput | None = None
        self.candidate_is_approved = False
        self.workflow_context: GenerationWorkflowContext | None = None
        self.approval_dialog_open = False
        self.selected_outfit_path: Path | None = None
        self.selected_pose_path: Path | None = None

        # 현재 구현에서 실제 사용하는 입력은 캐릭터 기준 이미지 하나다.
        self.style_path = None
        self.outfit_path = None

        user_profile = os.environ.get('USERPROFILE', '')
        self.default_get_dir = os.path.join(user_profile, "Downloads") if user_profile else ""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 1. 원본 캐릭터 기준 이미지 선택 UI
        style_layout = QHBoxLayout()
        self.style_label = QLabel("1. 원본 캐릭터 기준 이미지: 선택되지 않음")
        self.style_button = QPushButton("이미지 선택")
        self.style_button.clicked.connect(lambda: self.select_image("style"))
        style_layout.addWidget(self.style_label)
        style_layout.addWidget(self.style_button)
        layout.addLayout(style_layout)

        # 의상 참조는 CatVTON 별도 환경으로 전달하고 보호 마스크로 제한한다.
        outfit_layout = QHBoxLayout()
        self.outfit_label = QLabel("2. 의상 참조: 선택하지 않음")
        self.outfit_button = QPushButton("의상 이미지 선택")
        self.outfit_button.clicked.connect(lambda: self.select_image("outfit"))
        self.clear_outfit_button = QPushButton("의상 선택 해제")
        self.clear_outfit_button.clicked.connect(self.clear_outfit_reference)
        outfit_layout.addWidget(self.clear_outfit_button)
        outfit_layout.addWidget(self.outfit_label)
        outfit_layout.addWidget(self.outfit_button)
        layout.addLayout(outfit_layout)
        clothing_type_layout = QHBoxLayout()
        clothing_type_layout.addWidget(QLabel("의상 종류:"))
        self.clothing_category_combo = QComboBox()
        for clothing_category, label in CLOTHING_OPTIONS:
            self.clothing_category_combo.addItem(label, clothing_category.value)
        clothing_type_layout.addWidget(self.clothing_category_combo)
        layout.addLayout(clothing_type_layout)
        self.body_comparison_button = QPushButton("캐릭터 신체 비교 시작")
        self.body_comparison_button.setEnabled(False)
        clothing_type_layout.addWidget(self.body_comparison_button)
        self.body_comparison_button.clicked.connect(
            self.start_character_body_comparison
        )
        self.body_comparison_button.setVisible(False)
        self.clothing_category_combo.currentIndexChanged.connect(
            self.invalidate_character_body_comparison
        )

        pose_layout = QHBoxLayout()
        self.pose_label = QLabel("3. 자세 참조: 선택하지 않음")
        self.pose_button = QPushButton("자세 이미지 선택")
        self.pose_button.clicked.connect(lambda: self.select_image("pose"))
        self.pose_estimation_button = QPushButton("관절 추출 시작")
        self.pose_estimation_button.setEnabled(False)
        self.pose_estimation_button.clicked.connect(
            self.start_pose_reference_estimation
        )
        self.pose_estimation_button.setVisible(False)
        self.clear_pose_button = QPushButton("자세 선택 해제")
        self.clear_pose_button.clicked.connect(self.clear_pose_reference)
        pose_layout.addWidget(self.pose_label)
        pose_layout.addWidget(self.pose_button)
        pose_layout.addWidget(self.pose_estimation_button)
        pose_layout.addWidget(self.clear_pose_button)
        layout.addLayout(pose_layout)

        framing_layout = QHBoxLayout()
        framing_label = QLabel("4. 화면 범위:")
        self.framing_combo = QComboBox()
        for framing_type, label in FRAMING_OPTIONS:
            self.framing_combo.addItem(label, framing_type.value)
        self.framing_combo.setCurrentIndex(0)
        framing_layout.addWidget(framing_label)
        framing_layout.addWidget(self.framing_combo)
        layout.addLayout(framing_layout)

        self.framing_help = QLabel(
            "화면 범위를 선택합니다. 의상 합성 결과는 "
            "얼굴·신체·배경 보호 검사를 통과해야 합니다."
        )
        self.framing_help.setWordWrap(True)
        layout.addWidget(self.framing_help)
        self.generate_button = QPushButton("전체 이미지 생성 시작")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.start_generation)
        layout.addWidget(self.generate_button)

        self.status_label = QLabel("상태: 캐릭터 기준 이미지를 선택해 주세요")
        layout.addWidget(self.status_label)

        self.candidate_preview = QLabel("생성 후보가 여기에 표시됩니다.")
        self.candidate_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.candidate_preview.setMinimumSize(420, 480)
        self.candidate_preview.setStyleSheet(
            "border: 1px solid #777; background-color: #202020; color: #dddddd;"
        )
        layout.addWidget(self.candidate_preview)

        self.open_original_size_button = QPushButton("원본 크기로 보기")
        self.open_original_size_button.setEnabled(False)
        self.open_original_size_button.clicked.connect(self.show_candidate_original_size)
        layout.addWidget(self.open_original_size_button)

        candidate_decision_layout = QHBoxLayout()
        self.approve_candidate_button = QPushButton("후보 승인")
        self.reject_candidate_button = QPushButton("후보 거절")
        self.approve_candidate_button.setEnabled(False)
        self.reject_candidate_button.setEnabled(False)
        self.approve_candidate_button.clicked.connect(self.approve_candidate)
        self.reject_candidate_button.clicked.connect(self.reject_candidate)
        candidate_decision_layout.addWidget(self.approve_candidate_button)
        candidate_decision_layout.addWidget(self.reject_candidate_button)
        layout.addLayout(candidate_decision_layout)

        save_decision_layout = QHBoxLayout()
        self.save_candidate_button = QPushButton("승인 결과 저장")
        self.discard_candidate_button = QPushButton("저장하지 않음")
        self.save_candidate_button.setEnabled(False)
        self.discard_candidate_button.setEnabled(False)
        self.save_candidate_button.clicked.connect(self.save_approved_candidate)
        self.discard_candidate_button.clicked.connect(self.discard_approved_candidate)
        save_decision_layout.addWidget(self.save_candidate_button)
        save_decision_layout.addWidget(self.discard_candidate_button)
        layout.addLayout(save_decision_layout)

    def select_image(self, target_type):
        if self.workflow_context is not None and self.workflow_context.active:
            QMessageBox.information(
                self,
                "자동 작업 실행 중",
                "현재 자동 작업이 끝나거나 실패한 뒤 입력 이미지를 변경하세요.",
            )
            return
        if (
            self.reference_worker_thread is not None
            and self.reference_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "현재 참조 이미지의 화질을 확인하고 있습니다.",
            )
            return

        if (
            self.outfit_worker_thread is not None
            and self.outfit_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "현재 의상 영역을 자동 탐지하고 있습니다.",
            )
            return
        if (
            self.mask_worker_thread is not None
            and self.mask_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "현재 SAM2 의상 마스크 후보를 만들고 있습니다.",
            )
            return
        if (
            self.design_worker_thread is not None
            and self.design_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "현재 WD14 의상 디자인을 분석하고 있습니다.",
            )
            return
        if (
            self.body_comparison_worker_thread is not None
            and self.body_comparison_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "현재 캐릭터 신체와 관절을 분석하고 있습니다.",
            )
            return
        if self.pending_character_candidate is not None:
            QMessageBox.information(
                self,
                "안내",
                "현재 후보의 승인 또는 거절을 먼저 선택하세요.",
            )
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "참조 이미지 선택", self.default_get_dir, "이미지 파일 (*.png *.jpg *.jpeg)"
        )
        if file_path:
            file_name = os.path.basename(file_path)
            if target_type == "style":
                self.style_path = file_path
                self.release_approved_reference_image()
                self.release_pending_clothing_base_candidate()
                self.release_confirmed_character_body_comparison()
                self.style_label.setText(
                    f"1. 원본 캐릭터 기준 이미지: {file_name} (등록)"
                )

            elif target_type == "outfit":
                self.release_pending_clothing_base_candidate()
                self.release_clothing_mask_state()
                self.selected_outfit_path = Path(file_path)
                self.outfit_path = None
                self.pending_outfit_path = None
                self.clothing_region_candidates = ()
                self.clothing_source_size = None
                self.clothing_region_measurements = ()
                self.outfit_label.setText(
                    f"2. 의상 참조: {file_name} (등록)"
                )

            elif target_type == "pose":
                self.release_approved_pose_reference()
                self.release_approved_pose_estimation()
                self.selected_pose_path = Path(file_path)
                self.pose_label.setText(f"3. 자세 참조: {file_name} (등록)")

            self.workflow_context = None
            self.update_input_ready_status()

    def update_input_ready_status(self) -> None:
        """등록된 입력 수와 전체 자동 실행 가능 여부를 표시한다."""
        registered_input_count = sum(
            (
                self.style_path is not None,
                self.selected_outfit_path is not None,
                self.selected_pose_path is not None,
            )
        )
        self.generate_button.setEnabled(self.style_path is not None)
        self.generate_button.setText("전체 이미지 생성 시작")
        self.status_label.setText(
            "상태: 입력 등록 "
            f"{registered_input_count}/3개 - "
            + (
                "전체 이미지 생성 시작 버튼을 누르세요."
                if self.style_path is not None
                else "캐릭터 기준 이미지는 필수입니다."
            )
        )

    def review_pose_reference(self, image_path: Path) -> None:
        """자세 원본과 수치를 공개하고 승인된 복사본만 보관한다."""
        try:
            review_candidate = load_pose_reference_candidate(image_path)
        except PoseReferenceValidationError as error:
            self.status_label.setText(
                f"상태: 자세 참조 입력 실패 ({error})"
            )
            QMessageBox.critical(self, "자세 참조 입력 실패", str(error))
            return

        dialog = PoseReferenceApprovalDialog(review_candidate, self)
        self.execute_approval_dialog(dialog)
        if dialog.is_approved:
            approved_pose_reference = approve_pose_reference_candidate(
                review_candidate
            )
            self.release_approved_pose_reference()
            self.release_approved_pose_estimation()
            self.approved_pose_reference = approved_pose_reference
            self.pose_estimation_button.setEnabled(True)
            self.pose_label.setText(
                "3. 자세 참조: "
                f"{image_path.name} (승인, "
                f"{approved_pose_reference.width}x"
                f"{approved_pose_reference.height}px)"
            )
            self.status_label.setText(
                "상태: 자세 참조 입력 승인 완료 - "
                "DWPose 관절 추출을 자동으로 시작합니다."
            )
            if self.workflow_context is not None and self.workflow_context.active:
                self.start_pose_reference_estimation()
        else:
            self.pause_generation_workflow(
                GenerationWorkflowStage.POSE_ESTIMATING,
                "자세 참조 입력 승인이 취소되었습니다.",
            )
        review_candidate.close()

    @Slot()
    def clear_pose_reference(self) -> None:
        """승인 자세 입력을 저장하지 않고 메모리에서 해제한다."""
        if self.pending_character_candidate is not None:
            QMessageBox.information(self, "안내", "현재 후보를 먼저 판단하세요.")
            return
        self.release_approved_garment_inputs()
        self.release_approved_pose_reference()
        self.release_approved_pose_estimation()
        self.selected_pose_path = None
        self.workflow_context = None
        self.pose_estimation_button.setEnabled(False)
        self.pose_label.setText("3. 자세 참조: 선택하지 않음")
        self.update_input_ready_status()

    def release_approved_pose_reference(self) -> None:
        """이전 승인 자세 이미지의 메모리 복사본을 해제한다."""
        if self.approved_pose_reference is not None:
            self.approved_pose_reference.close()
            self.approved_pose_reference = None

    def release_approved_pose_estimation(self) -> None:
        """이전 승인 뼈대 지도 복사본을 메모리에서 해제한다."""
        self.release_approved_garment_inputs()
        if self.approved_pose_estimation is not None:
            self.approved_pose_estimation.close()
            self.approved_pose_estimation = None

    def get_pose_fallback_settings(self) -> PoseFallbackSettings:
        """설정 파일의 저장 자세 폴백 수치를 하나의 계약 객체로 만든다."""
        current_dir = Path(__file__).resolve().parent
        if self.config is None:
            self.config = load_yaml(current_dir / "configs" / "animagine.yaml")
        fallback_config = self.config.get("pose_fallback", {})
        library_root = Path(str(fallback_config.get(
            "library_root",
            "D:/genai-cache/genai-lab/approved-poses",
        )))
        if not library_root.is_absolute():
            library_root = current_dir / library_root
        return PoseFallbackSettings(
            library_root=library_root,
            enabled=bool(fallback_config.get("enabled", True)),
            default_pose_id=str(fallback_config.get(
                "default_pose_id", "last-approved"
            )),
            minimum_detected_joint_count=int(fallback_config.get(
                "minimum_detected_joint_count", 8
            )),
            require_shoulder=bool(fallback_config.get(
                "require_shoulder", True
            )),
            require_hip=bool(fallback_config.get("require_hip", True)),
            require_knee=bool(fallback_config.get("require_knee", True)),
            require_ankle=bool(fallback_config.get("require_ankle", True)),
            require_user_approval=bool(fallback_config.get(
                "require_user_approval", True
            )),
        )

    def offer_saved_pose_fallback(
        self,
        failure_reason: str,
        failed_source_image: Image.Image,
    ) -> tuple[str, str]:
        """검증된 마지막 승인 자세를 공개하고 승인되면 현재 자세로 교체한다."""
        settings = self.get_pose_fallback_settings()
        if not settings.enabled:
            return ("unavailable", "저장 자세 폴백이 설정에서 비활성화됨")
        try:
            saved_pose = load_default_approved_pose(settings)
        except PoseFallbackError as error:
            return ("unavailable", str(error))

        try:
            dialog = PoseFallbackApprovalDialog(
                failed_source_image=failed_source_image,
                saved_pose=saved_pose,
                failure_reason=failure_reason,
                parent=self,
            )
            self.execute_approval_dialog(dialog)
            if not dialog.is_approved:
                return ("rejected", "사용자가 저장 자세 폴백을 거절함")
            self.release_approved_pose_estimation()
            self.approved_pose_estimation = saved_pose.copy_approved_pose()
            self.pose_estimation_button.setEnabled(True)
            self.pose_label.setText(
                "3. 자세 참조: 저장 자세 승인 "
                f"{saved_pose.approved_pose.detected_joint_count}/18개"
            )
            self.status_label.setText(
                "상태: 저장 자세 폴백 승인 완료 - "
                f"ID={saved_pose.pose_id}, "
                f"관절={saved_pose.approved_pose.detected_joint_count}/18개, "
                f"SHA-256={saved_pose.control_map_sha256[:12]}, "
                "ControlNet 기준 후보 생성을 자동으로 이어갑니다."
            )
            return ("approved", "")
        finally:
            saved_pose.close()

    def save_last_approved_pose(
        self,
        approved_pose: PoseEstimationApprovedInput,
        source_preview_image: Image.Image,
    ) -> tuple[bool, str]:
        """현재 사용자 승인 자세를 다음 실패에 사용할 단일 슬롯에 저장한다."""
        settings = self.get_pose_fallback_settings()
        if not settings.enabled:
            return (False, "저장 자세 폴백이 설정에서 비활성화됨")
        try:
            digest = save_default_approved_pose(
                approved_pose=approved_pose,
                source_preview_image=source_preview_image,
                settings=settings,
            )
        except PoseFallbackError as error:
            return (False, str(error))
        return (True, digest)

    @Slot()
    def start_pose_reference_estimation(self) -> None:
        """승인 자세를 DWPose CPU 작업으로 1회 전달한다."""
        if (
            self.pose_estimation_worker_thread is not None
            and self.pose_estimation_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self, "안내", "현재 자세 관절을 추출하고 있습니다."
            )
            return
        if self.approved_pose_reference is None:
            QMessageBox.warning(
                self, "자세 입력 오류", "자세 이미지를 먼저 승인하세요."
            )
            return

        current_dir = Path(__file__).resolve().parent
        if self.config is None:
            self.config = load_yaml(current_dir / "configs" / "animagine.yaml")
        pose_config = self.config.get("pose_reference_estimation", {})
        runner_path = Path(str(pose_config.get(
            "runner_path", "scripts/pose_reference_runner.py"
        )))
        if not runner_path.is_absolute():
            runner_path = current_dir / runner_path
        settings = PoseReferenceEstimationSettings(
            python_executable=Path(str(pose_config.get(
                "python_executable",
                "D:/genai-cache/catvton-venv/Scripts/python.exe",
            ))),
            runner_path=runner_path,
            temporary_root=Path(str(pose_config.get(
                "temporary_root", "D:/genai-cache/temp/pose-reference"
            ))),
            cache_dir=Path(str(pose_config.get(
                "cache_dir", "D:/genai-cache/huggingface"
            ))),
            timeout_seconds=int(pose_config.get("timeout_seconds", 600)),
            pose_device=str(pose_config.get("pose_device", "cpu")),
            minimum_pose_confidence=float(
                pose_config.get("minimum_pose_confidence", 0.30)
            ),
        )
        self.release_approved_pose_estimation()
        self.pose_estimation_button.setEnabled(False)
        self.status_label.setText("상태: DWPose 관절 추출 준비 중...")
        self.pose_estimation_worker_thread = QThread(self)
        self.pose_estimation_worker = PoseReferenceEstimationWorker(
            self.approved_pose_reference,
            settings,
        )
        self.pose_estimation_worker.moveToThread(
            self.pose_estimation_worker_thread
        )
        self.pose_estimation_worker_thread.started.connect(
            self.pose_estimation_worker.run
        )
        self.pose_estimation_worker.status_changed.connect(
            self.show_worker_status
        )
        self.pose_estimation_worker.completed.connect(
            self.pose_reference_estimation_completed
        )
        self.pose_estimation_worker.failed.connect(
            self.pose_reference_estimation_failed
        )
        self.pose_estimation_worker.completed.connect(
            self.pose_estimation_worker_thread.quit
        )
        self.pose_estimation_worker.failed.connect(
            self.pose_estimation_worker_thread.quit
        )
        self.pose_estimation_worker_thread.finished.connect(
            self.pose_estimation_worker.deleteLater
        )
        self.pose_estimation_worker_thread.finished.connect(
            self.pose_estimation_worker_thread.deleteLater
        )
        self.pose_estimation_worker_thread.finished.connect(
            self.clear_pose_estimation_worker
        )
        self.pose_estimation_worker_thread.finished.connect(
            self.resume_generation_workflow
        )
        self.pose_estimation_worker_thread.start()

    @Slot(object)
    def pose_reference_estimation_completed(
        self,
        review_candidate: PoseEstimationReviewCandidate,
    ) -> None:
        """DWPose 중간 결과 3개를 공개하고 사용자 승인을 받는다."""
        try:
            fallback_settings = self.get_pose_fallback_settings()
            quality = evaluate_pose_quality(
                review_candidate,
                fallback_settings,
            )
            if not quality.accepted:
                failure_reason = "; ".join(quality.rejection_reasons)
                fallback_status, fallback_details = (
                    self.offer_saved_pose_fallback(
                        failure_reason=failure_reason,
                        failed_source_image=review_candidate.source_image,
                    )
                )
                if fallback_status == "approved":
                    return
                self.pose_estimation_button.setEnabled(True)
                self.pause_generation_workflow(
                    GenerationWorkflowStage.POSE_ESTIMATING,
                    "DWPose 자세 품질 미달 및 저장 자세 미적용",
                )
                if fallback_status == "unavailable":
                    QMessageBox.warning(
                        self,
                        "저장 자세 폴백 불가",
                        f"요청 자세 실패: {failure_reason}\n\n"
                        f"폴백 불가: {fallback_details}\n\n"
                        "품질 기준을 통과한 자세를 1회 승인하면 이후 "
                        "마지막 승인 자세로 폴백할 수 있습니다.",
                    )
                return

            dialog = PoseEstimationApprovalDialog(review_candidate, self)
            self.execute_approval_dialog(dialog)
            if not dialog.is_approved:
                self.pose_estimation_button.setEnabled(True)
                self.pause_generation_workflow(
                    GenerationWorkflowStage.POSE_ESTIMATING,
                    "DWPose 관절 결과 승인이 취소되었습니다.",
                )
                return
            self.release_approved_pose_estimation()
            self.approved_pose_estimation = approve_pose_estimation_candidate(
                review_candidate
            )
            pose_saved, save_details = self.save_last_approved_pose(
                approved_pose=self.approved_pose_estimation,
                source_preview_image=review_candidate.source_image,
            )
            self.pose_estimation_button.setEnabled(True)
            self.pose_label.setText(
                "3. 자세 참조: 관절 승인 "
                f"{review_candidate.detected_joint_count}/18개"
            )
            self.status_label.setText(
                "상태: DWPose 관절 승인 완료 - "
                f"탐지={review_candidate.detected_joint_count}/18개, "
                f"누락={review_candidate.missing_joint_count}/18개, "
                f"시간={review_candidate.elapsed_seconds:.2f}초, "
                f"폴백 저장={'완료' if pose_saved else '실패'}, "
                f"저장 근거={save_details[:12] if pose_saved else save_details}, "
                "ControlNet 기준 후보 생성을 자동으로 이어갑니다."
            )
        finally:
            review_candidate.close()

    @Slot(str, str)
    def pose_reference_estimation_failed(
        self,
        message: str,
        details: str,
    ) -> None:
        """DWPose 실패 원인과 재시도 행동을 표시한다."""
        self.pose_estimation_button.setEnabled(True)
        if self.approved_pose_reference is not None:
            fallback_status, fallback_details = self.offer_saved_pose_fallback(
                failure_reason=message,
                failed_source_image=self.approved_pose_reference.image,
            )
            if fallback_status == "approved":
                return
            if fallback_status == "rejected":
                self.pause_generation_workflow(
                    GenerationWorkflowStage.POSE_ESTIMATING,
                    "저장 자세 폴백이 사용자에게 거절되었습니다.",
                )
                return
        else:
            fallback_details = "승인 자세 원본이 메모리에 없음"
        self.status_label.setText(f"상태: DWPose 관절 추출 실패 ({message})")
        self.pause_generation_workflow(
            GenerationWorkflowStage.POSE_ESTIMATING,
            f"DWPose 관절 추출 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("DWPose 관절 추출 실패")
        dialog.setText(message)
        dialog.setInformativeText(
            "승인 자세 이미지는 유지합니다. 다른 자세를 선택하거나 다시 실행하세요.\n"
            f"저장 자세 폴백 불가 사유: {fallback_details}"
        )
        dialog.setDetailedText(details)
        dialog.exec()

    @Slot()
    def clear_pose_estimation_worker(self) -> None:
        """종료된 DWPose 자세 작업 객체를 해제한다."""
        self.pose_estimation_worker = None
        self.pose_estimation_worker_thread = None

    def start_outfit_region_preparation(self, image_path: Path) -> None:
        """의상 이미지를 정규화하고 자동 위치 탐지를 별도 작업으로 시작한다."""
        if (
            self.outfit_worker_thread is not None
            and self.outfit_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "의상 영역을 자동 탐지하고 있습니다.",
            )
            return

        self.release_clothing_mask_state()
        self.pending_outfit_path = image_path
        self.outfit_path = None
        self.clothing_region_candidates = ()
        self.clothing_source_size = None
        self.clothing_region_measurements = ()
        self.generate_button.setEnabled(False)
        self.outfit_label.setText(
            f"2. 의상 참조: {image_path.name} (영역 확인 중)"
        )
        self.status_label.setText("상태: 의상 영역 자동 탐지 준비 중...")

        current_dir = Path(__file__).resolve().parent
        if self.config is None:
            self.config = load_yaml(current_dir / "configs" / "animagine.yaml")
        detection_config = self.config.get("clothing_preparation", {})
        detection_settings = ClothingDetectionSettings(
            model_id=str(
                detection_config.get(
                    "detector_model_id",
                    "IDEA-Research/grounding-dino-tiny",
                )
            ),
            cache_dir=Path(
                str(
                    detection_config.get(
                        "cache_dir",
                        "D:/genai-cache/huggingface",
                    )
                )
            ),
            inference_device=str(
                detection_config.get("inference_device", "cpu")
            ),
            box_threshold=float(
                detection_config.get("box_threshold", 0.30)
            ),
            text_threshold=float(
                detection_config.get("text_threshold", 0.25)
            ),
            minimum_area_ratio=float(
                detection_config.get("minimum_area_ratio", 0.02)
            ),
            maximum_area_ratio=float(
                detection_config.get("maximum_area_ratio", 0.95)
            ),
        )

        self.outfit_worker_thread = QThread(self)
        self.outfit_worker = ClothingRegionDetectionWorker(
            image_path=image_path,
            settings=detection_settings,
        )
        self.outfit_worker.moveToThread(self.outfit_worker_thread)
        self.outfit_worker_thread.started.connect(self.outfit_worker.run)
        self.outfit_worker.status_changed.connect(self.show_worker_status)
        self.outfit_worker.completed.connect(
            self.outfit_region_preparation_completed
        )
        self.outfit_worker.failed.connect(self.outfit_region_preparation_failed)
        self.outfit_worker.completed.connect(self.outfit_worker_thread.quit)
        self.outfit_worker.failed.connect(self.outfit_worker_thread.quit)
        self.outfit_worker_thread.finished.connect(
            self.outfit_worker.deleteLater
        )
        self.outfit_worker_thread.finished.connect(
            self.outfit_worker_thread.deleteLater
        )
        self.outfit_worker_thread.finished.connect(self.clear_outfit_worker)
        self.outfit_worker_thread.start()

    @Slot(object, object, str)
    def outfit_region_preparation_completed(
        self,
        normalized_source: NormalizedClothingSource,
        detection_result: ClothingRegionDetectionResult | None,
        detection_warning: str,
    ) -> None:
        """승인된 의상 위치를 SAM2 마스크 후보 생성 단계에 전달한다."""
        review_dialog = ClothingRegionReviewDialog(
            normalized_source=normalized_source,
            detection_result=detection_result,
            detection_warning=detection_warning,
            parent=self,
        )
        self.execute_approval_dialog(review_dialog)
        selected_candidates = review_dialog.selected_candidates
        if not selected_candidates or self.pending_outfit_path is None:
            normalized_source.image.close()
            self.pending_outfit_path = None
            self.outfit_path = None
            self.clothing_region_candidates = ()
            self.clothing_source_size = None
            self.clothing_region_measurements = ()
            self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
            self.status_label.setText("상태: 의상 영역 승인이 취소되었습니다.")
            self.pause_generation_workflow(
                GenerationWorkflowStage.CLOTHING_MASKING,
                "의상 영역 승인이 취소되었습니다.",
            )
            self.generate_button.setEnabled(
                self.approved_reference_image is not None
            )
            return

        measurements = tuple(
            measure_clothing_region(candidate, normalized_source.image.size)
            for candidate in selected_candidates
        )
        self.outfit_path = str(self.pending_outfit_path)
        self.clothing_region_candidates = selected_candidates
        self.clothing_source_size = normalized_source.image.size
        self.clothing_region_measurements = measurements
        self.pending_clothing_source = normalized_source
        self.pending_outfit_path = None
        self.generate_button.setEnabled(False)
        self.outfit_label.setText(
            f"2. 의상 참조: {Path(self.outfit_path).name} "
            f"(영역 {len(selected_candidates)}개, 마스크 생성 중)"
        )
        self.start_clothing_mask_extraction(
            normalized_source,
            selected_candidates,
        )

    def read_clothing_mask_settings(self) -> ClothingMaskExtractionSettings:
        """참조 의상과 기준 캐릭터 선택이 같은 SAM2 설정을 사용한다."""
        current_dir = Path(__file__).resolve().parent
        if self.config is None:
            self.config = load_yaml(current_dir / "configs" / "animagine.yaml")
        mask_config = self.config.get("clothing_mask_extraction", {})
        return ClothingMaskExtractionSettings(
            model_id=str(
                mask_config.get("model_id", "facebook/sam2.1-hiera-tiny")
            ),
            cache_dir=Path(
                str(mask_config.get("cache_dir", "D:/genai-cache/huggingface"))
            ),
            inference_device=str(mask_config.get("inference_device", "cpu")),
            maximum_candidate_count=int(
                mask_config.get("maximum_candidate_count", 3)
            ),
            maximum_region_count=int(
                mask_config.get("maximum_region_count", 8)
            ),
            alpha_empty_probability=float(
                mask_config.get("alpha_empty_probability", 0.01)
            ),
            alpha_solid_probability=float(
                mask_config.get("alpha_solid_probability", 0.99)
            ),
        )

    def start_clothing_mask_extraction(
        self,
        normalized_source: NormalizedClothingSource,
        approved_regions: tuple[ClothingRegionCandidate, ...],
    ) -> None:
        """승인 영역 최대 8개를 입력으로 SAM2 후보 생성을 시작한다."""
        mask_settings = self.read_clothing_mask_settings()
        self.mask_worker_thread = QThread(self)
        self.mask_worker = ClothingMaskExtractionWorker(
            normalized_source=normalized_source,
            approved_regions=approved_regions,
            settings=mask_settings,
        )
        self.mask_worker.moveToThread(self.mask_worker_thread)
        self.mask_worker_thread.started.connect(self.mask_worker.run)
        self.mask_worker.status_changed.connect(self.show_worker_status)
        self.mask_worker.completed.connect(self.clothing_mask_extraction_completed)
        self.mask_worker.failed.connect(self.clothing_mask_extraction_failed)
        self.mask_worker.completed.connect(self.mask_worker_thread.quit)
        self.mask_worker.failed.connect(self.mask_worker_thread.quit)
        self.mask_worker_thread.finished.connect(self.mask_worker.deleteLater)
        self.mask_worker_thread.finished.connect(
            self.mask_worker_thread.deleteLater
        )
        self.mask_worker_thread.finished.connect(self.clear_mask_worker)
        self.mask_worker_thread.start()

    @Slot(object)
    def clothing_mask_extraction_completed(
        self,
        extraction_result: ClothingMaskExtractionResult,
    ) -> None:
        """영역별 SAM2 후보를 받고 합친 마스크의 최종 승인을 받는다."""
        normalized_source = self.pending_clothing_source
        if normalized_source is None:
            self.close_extraction_result_masks(extraction_result)
            return

        review_dialog = ClothingMaskReviewDialog(
            normalized_source=normalized_source,
            extraction_result=extraction_result,
            parent=self,
        )
        self.execute_approval_dialog(review_dialog)
        selected_candidates = review_dialog.selected_candidates
        if not selected_candidates:
            retry_requested = review_dialog.retry_region_selection
            self.release_clothing_mask_state(extraction_result)
            self.outfit_path = None
            self.clothing_region_candidates = ()
            self.clothing_source_size = None
            self.clothing_region_measurements = ()
            self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
            self.generate_button.setEnabled(
                self.approved_reference_image is not None
            )
            if retry_requested:
                self.status_label.setText(
                    "상태: 의상 위치 재선택 요청 - 의상 이미지를 다시 선택하세요."
                )
            else:
                self.status_label.setText("상태: 의상 마스크 선택 취소")
            self.pause_generation_workflow(
                GenerationWorkflowStage.CLOTHING_MASKING,
                "의상 마스크 후보 선택이 완료되지 않았습니다.",
            )
            return

        combined_mask = combine_clothing_mask_candidates(
            selected_candidates,
            normalized_source.image.size,
        )
        combined_review_dialog = ClothingCombinedMaskReviewDialog(
            normalized_source=normalized_source,
            combined_mask=combined_mask,
            parent=self,
        )
        self.execute_approval_dialog(combined_review_dialog)
        if not combined_review_dialog.approved:
            retry_requested = combined_review_dialog.retry_region_selection
            combined_mask.mask_image.close()
            self.release_clothing_mask_state(extraction_result)
            self.outfit_path = None
            self.clothing_region_candidates = ()
            self.clothing_source_size = None
            self.clothing_region_measurements = ()
            self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
            self.generate_button.setEnabled(
                self.approved_reference_image is not None
            )
            self.status_label.setText(
                "상태: 의상 영역 재선택 요청 - 의상 이미지를 다시 선택하세요."
                if retry_requested
                else "상태: 합친 의상 마스크 승인 취소"
            )
            self.pause_generation_workflow(
                GenerationWorkflowStage.CLOTHING_MASKING,
                "합친 의상 마스크 승인이 완료되지 않았습니다.",
            )
            return

        pixel_config = self.config.get("clothing_pixel_extraction", {})
        pixel_settings = ClothingPixelExtractionSettings(
            maximum_hole_area_pixels=int(
                pixel_config.get("maximum_hole_area_pixels", 4096)
            ),
            maximum_hole_area_ratio=float(
                pixel_config.get("maximum_hole_area_ratio", 0.0025)
            ),
            maximum_rgb_distance=float(
                pixel_config.get("maximum_rgb_distance", 36.0)
            ),
            white_clothing_luminance=float(
                pixel_config.get("white_clothing_luminance", 200.0)
            ),
            maximum_white_luminance_difference=float(
                pixel_config.get(
                    "maximum_white_luminance_difference",
                    48.0,
                )
            ),
        )
        try:
            pixel_extraction_candidate = extract_clothing_pixels(
                normalized_source,
                combined_mask,
                pixel_settings,
            )
        except Exception as error:
            combined_mask.mask_image.close()
            self.release_clothing_mask_state(extraction_result)
            self.outfit_path = None
            self.clothing_region_candidates = ()
            self.clothing_source_size = None
            self.clothing_region_measurements = ()
            self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
            self.generate_button.setEnabled(
                self.approved_reference_image is not None
            )
            self.status_label.setText(
                f"상태: 원본 의상 픽셀 추출 실패 ({error})"
            )
            QMessageBox.critical(
                self,
                "원본 의상 픽셀 추출 실패",
                str(error),
            )
            self.pause_generation_workflow(
                GenerationWorkflowStage.CLOTHING_MASKING,
                f"원본 의상 픽셀 추출 실패: {error}",
            )
            return

        pixel_review_dialog = ClothingPixelExtractionReviewDialog(
            normalized_source=normalized_source,
            approved_mask=combined_mask,
            extraction_candidate=pixel_extraction_candidate,
            extraction_settings=pixel_settings,
            parent=self,
        )
        self.execute_approval_dialog(pixel_review_dialog)
        if not pixel_review_dialog.approved:
            retry_requested = pixel_review_dialog.retry_mask_selection
            pixel_extraction_candidate.extracted_image.close()
            pixel_extraction_candidate.clothing_mask.close()
            combined_mask.mask_image.close()
            self.release_clothing_mask_state(extraction_result)
            self.outfit_path = None
            self.clothing_region_candidates = ()
            self.clothing_source_size = None
            self.clothing_region_measurements = ()
            self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
            self.generate_button.setEnabled(
                self.approved_reference_image is not None
            )
            self.status_label.setText(
                "상태: 의상 마스크 재선택 요청 - 의상 이미지를 다시 선택하세요."
                if retry_requested
                else "상태: 원본 의상 픽셀 추출 승인 취소"
            )
            self.pause_generation_workflow(
                GenerationWorkflowStage.CLOTHING_MASKING,
                "원본 의상 픽셀 추출 승인이 완료되지 않았습니다.",
            )
            return

        selected_candidate_ids = {id(candidate) for candidate in selected_candidates}
        selected_region_groups = []
        for region_group in extraction_result.region_groups:
            selected_candidate = next(
                candidate
                for candidate in region_group.candidates
                if id(candidate) in selected_candidate_ids
            )
            for candidate in region_group.candidates:
                if candidate is not selected_candidate:
                    candidate.mask_image.close()
            selected_region_groups.append(
                replace(region_group, candidates=(selected_candidate,))
            )

        self.selected_clothing_mask = combined_mask
        self.pending_clothing_extraction = pixel_extraction_candidate
        self.clothing_mask_result = replace(
            extraction_result,
            region_groups=tuple(selected_region_groups),
        )
        self.outfit_label.setText(
            f"2. 의상 참조: {Path(self.outfit_path).name} "
            f"(원본 픽셀 추출 승인, RGB 보존 "
            f"{pixel_extraction_candidate.original_pixel_preservation_percent:.3f}%)"
        )
        self.generate_button.setEnabled(False)
        self.status_label.setText(
            "상태: 원본 의상 픽셀 추출 승인 완료 - "
            f"영역={combined_mask.source_region_count}개, "
            f"알파 픽셀="
            f"{pixel_extraction_candidate.selected_alpha_pixel_count:,}개, "
            f"반투명 경계="
            f"{pixel_extraction_candidate.soft_edge_pixel_count:,}개, "
            f"공백 복원="
            f"{pixel_extraction_candidate.filled_hole_count}개/"
            f"{pixel_extraction_candidate.filled_hole_pixel_count:,}픽셀, "
            "WD14 디자인 분석을 시작합니다."
        )
        self.start_clothing_design_analysis()

    @Slot(str, str)
    def clothing_mask_extraction_failed(
        self,
        message: str,
        details: str,
    ) -> None:
        """SAM2 실패 원인과 상세 로그를 표시하고 의상 생성을 차단한다."""
        self.release_clothing_mask_state()
        self.outfit_path = None
        self.clothing_region_candidates = ()
        self.clothing_source_size = None
        self.clothing_region_measurements = ()
        self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
        self.generate_button.setEnabled(
            self.approved_reference_image is not None
        )
        self.status_label.setText(f"상태: SAM2 의상 마스크 생성 실패 ({message})")
        self.pause_generation_workflow(
            GenerationWorkflowStage.CLOTHING_MASKING,
            f"SAM2 의상 마스크 생성 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("SAM2 의상 마스크 생성 실패")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    def release_clothing_mask_state(
        self,
        extraction_result: ClothingMaskExtractionResult | None = None,
    ) -> None:
        """보관 중인 의상 원본·마스크·픽셀 추출 이미지를 메모리에서 해제한다."""
        self.release_approved_garment_inputs()
        result_to_release = extraction_result or self.clothing_mask_result
        if result_to_release is not None:
            self.close_extraction_result_masks(result_to_release)
        if self.selected_clothing_mask is not None:
            self.selected_clothing_mask.mask_image.close()
        if self.pending_clothing_extraction is not None:
            self.pending_clothing_extraction.extracted_image.close()
            self.pending_clothing_extraction.clothing_mask.close()
        if self.pending_clothing_source is not None:
            self.pending_clothing_source.image.close()
        self.pending_clothing_source = None
        self.selected_clothing_mask = None
        self.clothing_mask_result = None
        self.pending_clothing_extraction = None
        self.clothing_design_result = None
        self.confirmed_clothing_design = None
        self.release_confirmed_character_body_comparison()
        self.body_comparison_clothing_category = None
        self.body_comparison_button.setEnabled(False)

    @staticmethod
    def close_extraction_result_masks(
        extraction_result: ClothingMaskExtractionResult,
    ) -> None:
        """SAM2 영역별 후보 이미지 전부를 닫는다."""
        for region_group in extraction_result.region_groups:
            for candidate in region_group.candidates:
                candidate.mask_image.close()

    @Slot()
    def clear_mask_worker(self) -> None:
        """종료된 SAM2 작업 객체를 해제한다."""
        self.mask_worker = None
        self.mask_worker_thread = None

    def start_clothing_design_analysis(self) -> None:
        """승인된 픽셀 추출본을 WD14 CPU 작업으로 최대 1회 전달한다."""
        extraction_candidate = self.pending_clothing_extraction
        if extraction_candidate is None:
            QMessageBox.critical(
                self,
                "WD14 입력 오류",
                "승인된 원본 의상 픽셀 추출본이 없습니다.",
            )
            return
        if (
            self.design_worker_thread is not None
            and self.design_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "WD14 의상 디자인을 이미 분석하고 있습니다.",
            )
            return

        design_config = self.config.get("clothing_design_analysis", {})
        design_settings = ClothingDesignAnalysisSettings(
            model_id=str(
                design_config.get(
                    "model_id",
                    "SmilingWolf/wd-vit-tagger-v3",
                )
            ),
            cache_dir=Path(
                str(
                    design_config.get(
                        "cache_dir",
                        "D:/genai-cache/huggingface",
                    )
                )
            ),
            model_filename=str(
                design_config.get("model_filename", "model.onnx")
            ),
            label_filename=str(
                design_config.get(
                    "label_filename",
                    "selected_tags.csv",
                )
            ),
            execution_provider=str(
                design_config.get(
                    "execution_provider",
                    "CPUExecutionProvider",
                )
            ),
            score_threshold=float(
                design_config.get("score_threshold", 0.35)
            ),
            maximum_tag_count=int(
                design_config.get("maximum_tag_count", 30)
            ),
        )
        self.generate_button.setEnabled(False)
        self.design_worker_thread = QThread(self)
        self.design_worker = ClothingDesignAnalysisWorker(
            extraction_candidate,
            design_settings,
        )
        self.design_worker.moveToThread(self.design_worker_thread)
        self.design_worker_thread.started.connect(self.design_worker.run)
        self.design_worker.status_changed.connect(self.show_worker_status)
        self.design_worker.completed.connect(
            self.clothing_design_analysis_completed
        )
        self.design_worker.failed.connect(
            self.clothing_design_analysis_failed
        )
        self.design_worker.completed.connect(self.design_worker_thread.quit)
        self.design_worker.failed.connect(self.design_worker_thread.quit)
        self.design_worker_thread.finished.connect(
            self.design_worker.deleteLater
        )
        self.design_worker_thread.finished.connect(
            self.design_worker_thread.deleteLater
        )
        self.design_worker_thread.finished.connect(self.clear_design_worker)
        self.design_worker_thread.finished.connect(
            self.resume_generation_workflow
        )
        self.design_worker_thread.start()

    @Slot(object)
    def clothing_design_analysis_completed(
        self,
        analysis_result: ClothingDesignAnalysisResult,
    ) -> None:
        """WD14 일반 태그를 공개하고 사용자 포함·제외 결과를 확정한다."""
        extraction_candidate = self.pending_clothing_extraction
        if extraction_candidate is None:
            return
        review_dialog = ClothingDesignAnalysisReviewDialog(
            extraction_candidate=extraction_candidate,
            analysis_result=analysis_result,
            parent=self,
        )
        self.execute_approval_dialog(review_dialog)
        if not review_dialog.approved:
            self.outfit_path = None
            self.pending_outfit_path = None
            self.release_clothing_mask_state()
            self.clothing_region_candidates = ()
            self.clothing_source_size = None
            self.clothing_region_measurements = ()
            self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
            self.generate_button.setEnabled(
                self.approved_reference_image is not None
            )
            self.status_label.setText("상태: WD14 의상 분석 승인 취소")
            self.pause_generation_workflow(
                GenerationWorkflowStage.CLOTHING_ANALYZING,
                "WD14 의상 디자인 분석 승인이 취소되었습니다.",
            )
            return

        approved_tag_names = review_dialog.approved_tag_names
        self.clothing_design_result = analysis_result
        self.confirmed_clothing_design = ClothingDesignSummary(
            dominant_rgb_colors=(),
            design_tags=approved_tag_names,
            unknown_details=(
                ()
                if approved_tag_names
                else ("WD14 승인 태그 0개",)
            ),
        )
        self.outfit_label.setText(
            f"2. 의상 참조: {Path(self.outfit_path).name} "
            f"(WD14 태그 {len(approved_tag_names)}개 승인)"
        )
        self.generate_button.setEnabled(
            self.approved_reference_image is not None
            and not (
                self.workflow_context is not None
                and self.workflow_context.active
            )
        )
        self.release_confirmed_character_body_comparison()
        self.body_comparison_clothing_category = None
        self.body_comparison_button.setEnabled(False)
        self.status_label.setText(
            "상태: WD14 의상 디자인 분석 승인 완료 - "
            f"후보={len(analysis_result.tag_candidates)}개, "
            f"승인={len(approved_tag_names)}개, "
            f"기준={analysis_result.score_threshold * 100.0:.1f}%, "
            f"시간={analysis_result.elapsed_seconds:.2f}초, "
            "다음 자동 단계를 준비합니다."
        )

    @Slot(str, str)
    def clothing_design_analysis_failed(
        self,
        message: str,
        details: str,
    ) -> None:
        """WD14 실패 원인과 재시도 행동을 표시하고 CatVTON을 차단한다."""
        self.generate_button.setEnabled(False)
        self.status_label.setText(f"상태: WD14 의상 디자인 분석 실패 ({message})")
        self.pause_generation_workflow(
            GenerationWorkflowStage.CLOTHING_ANALYZING,
            f"WD14 의상 디자인 분석 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("WD14 의상 디자인 분석 실패")
        dialog.setText(message)
        dialog.setInformativeText(
            "승인된 추출 의상은 메모리에 유지합니다. "
            "의상 이미지를 다시 선택하면 처음부터 재시도할 수 있습니다."
        )
        dialog.setDetailedText(details)
        dialog.exec()

    @Slot()
    def clear_design_worker(self) -> None:
        """종료된 WD14 작업 객체를 해제한다."""
        self.design_worker = None
        self.design_worker_thread = None

    def release_confirmed_character_body_comparison(self) -> None:
        """승인된 중립화 이미지와 변경 마스크를 메모리에서 해제한다."""
        self.release_approved_garment_inputs()
        if self.confirmed_character_body_comparison is not None:
            self.confirmed_character_body_comparison.close()
        self.confirmed_character_body_comparison = None

    def release_approved_garment_inputs(self) -> None:
        """TPS와 Lineart 승인 입력 4개 이미지를 메모리에서 해제한다."""
        if self.approved_garment_lineart is not None:
            self.approved_garment_lineart.close()
        self.approved_garment_lineart = None
        if self.approved_garment_warp is not None:
            self.approved_garment_warp.close()
        self.approved_garment_warp = None

    def release_pending_clothing_base_candidate(self) -> None:
        """의상 적용 전 메모리에 보관한 기준 후보를 해제한다."""
        candidate = self.pending_clothing_base_candidate
        if candidate is None:
            return
        candidate.image.close()
        if candidate.original_generated_image is not None:
            candidate.original_generated_image.close()
        if candidate.before_clothing_image is not None:
            candidate.before_clothing_image.close()
        if candidate.clothing_change_mask is not None:
            candidate.clothing_change_mask.close()
        if candidate.raw_clothing_try_on_image is not None:
            candidate.raw_clothing_try_on_image.close()
        if candidate.clothing_difference_image is not None:
            candidate.clothing_difference_image.close()
        self.pending_clothing_base_candidate = None

    @Slot()
    def invalidate_character_body_comparison(self) -> None:
        """의상 종류가 바뀌면 이전 신체 비교 승인을 무효화한다."""
        self.release_approved_garment_inputs()
        self.release_confirmed_character_body_comparison()
        self.body_comparison_clothing_category = None
        self.generate_button.setEnabled(
            self.confirmed_clothing_design is not None
            and self.approved_reference_image is not None
        )
        self.body_comparison_button.setEnabled(
            self.pending_clothing_base_candidate is not None
        )

    @Slot()
    def review_target_character_masks(
        self, source: Image.Image,
    ) -> ApprovedTargetMasks | None:
        """기준 후보의 실제 교체 영역과 특수 보호를 승인받는다."""
        dialog = TargetMaskReviewDialog(
            source, self.read_clothing_mask_settings(),
            ClothingRegionReviewDialog, ClothingMaskReviewDialog,
            ClothingMaskExtractionWorker, create_pil_image_pixmap, self,
        )
        try:
            accepted = self.execute_approval_dialog(dialog)
            if accepted == QDialog.DialogCode.Accepted and dialog.approved_masks:
                return dialog.approved_masks.copy()
            return None
        finally:
            dialog.close_images()

    def start_character_body_comparison(self) -> None:
        """승인 캐릭터에 SCHP·DensePose·DWPose 신체 비교를 시작한다."""
        if self.approval_dialog_open:
            return
        if (
            self.body_comparison_worker_thread is not None
            and self.body_comparison_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "현재 캐릭터 신체와 관절을 분석하고 있습니다.",
            )
            return
        if self.pending_clothing_base_candidate is None:
            QMessageBox.warning(
                self,
                "신체 비교 입력 오류",
                "의상 적용 전 기준 후보 이미지를 먼저 생성하세요.",
            )
            return
        if self.confirmed_clothing_design is None or self.outfit_path is None:
            QMessageBox.warning(
                self,
                "신체 비교 입력 오류",
                "의상 픽셀 추출과 WD14 태그 승인을 먼저 완료하세요.",
            )
            return
        if self.pending_clothing_extraction is None:
            QMessageBox.warning(
                self,
                "신체 비교 입력 오류",
                "승인된 의상 픽셀 추출본이 없습니다.",
            )
            return

        current_dir = Path(__file__).resolve().parent
        if self.config is None:
            self.config = load_yaml(current_dir / "configs" / "animagine.yaml")
        body_config = self.config.get("character_body_comparison", {})
        runner_path = Path(str(body_config.get(
            "runner_path",
            "scripts/body_comparison_runner.py",
        )))
        if not runner_path.is_absolute():
            runner_path = current_dir / runner_path
        clothing_category = ClothingCategory(
            self.clothing_category_combo.currentData()
        )
        clothing_type = find_catvton_clothing_type(clothing_category)
        comparison_settings = CharacterBodyComparisonSettings(
            python_executable=Path(str(body_config.get(
                "python_executable",
                "D:/genai-cache/catvton-venv/Scripts/python.exe",
            ))),
            repository_path=Path(str(body_config.get(
                "repository_path",
                "D:/genai-cache/tools/CatVTON",
            ))),
            runner_path=runner_path,
            temporary_root=Path(str(body_config.get(
                "temporary_root",
                "D:/genai-cache/temp/body-comparison",
            ))),
            cache_dir=Path(str(body_config.get(
                "cache_dir",
                "D:/genai-cache/huggingface",
            ))),
            width=int(body_config.get("width", 576)),
            height=int(body_config.get("height", 1024)),
            timeout_seconds=int(body_config.get("timeout_seconds", 1800)),
            mask_expansion_ratio=float(
                body_config.get("mask_expansion_ratio", 0.01)
            ),
            minimum_mask_expansion_pixels=int(
                body_config.get("minimum_mask_expansion_pixels", 5)
            ),
            maximum_mask_expansion_pixels=int(
                body_config.get("maximum_mask_expansion_pixels", 15)
            ),
            mask_closing_radius_pixels=int(
                body_config.get("mask_closing_radius_pixels", 2)
            ),
            foreground_model_id=str(
                body_config.get("foreground_model_id", "isnet-anime")
            ),
            foreground_expansion_pixels=int(
                body_config.get("foreground_expansion_pixels", 15)
            ),
        )

        clothing_config = self.config.get("clothing_try_on", {})
        preflight_settings = CatVTONPreflightSettings(
            python_executable=Path(str(clothing_config["python_executable"])),
            repository_path=Path(str(clothing_config["repository_path"])),
            runner_path=current_dir / "scripts" / "catvton_preflight_runner.py",
            temporary_root=Path(str(clothing_config["temporary_root"])),
            cache_dir=Path(str(clothing_config["cache_dir"])),
            width=int(clothing_config["width"]),
            height=int(clothing_config["height"]),
            timeout_seconds=int(clothing_config["timeout_seconds"]),
            mask_blur_factor=int(clothing_config.get("mask_blur_factor", 9)),
        )
        approved_target_masks = self.review_target_character_masks(
            self.pending_clothing_base_candidate.image,
        )
        if approved_target_masks is None:
            self.pause_generation_workflow(
                GenerationWorkflowStage.BODY_MASKING,
                "기존 의상·특수 보호 선택을 취소했습니다. 다시 시도할 수 있습니다.",
            )
            return
        preflight_clothing_input = ClothingReferenceInput(
            image_path=Path(self.outfit_path),
            category=clothing_category,
            approved_image=(
                self.pending_clothing_extraction.extracted_image.copy()
            ),
        )

        self.release_confirmed_character_body_comparison()
        self.body_comparison_clothing_category = clothing_category
        self.generate_button.setEnabled(False)
        self.body_comparison_button.setEnabled(False)
        self.body_comparison_worker_thread = QThread(self)
        self.body_comparison_worker = CharacterBodyComparisonWorker(
            character_image=self.pending_clothing_base_candidate.image,
            clothing_type=clothing_type,
            settings=comparison_settings,
            clothing_reference_input=preflight_clothing_input,
            preflight_settings=preflight_settings,
            approved_target_masks=approved_target_masks,
        )
        approved_target_masks.close()
        self.body_comparison_worker.moveToThread(
            self.body_comparison_worker_thread
        )
        self.body_comparison_worker_thread.started.connect(
            self.body_comparison_worker.run
        )
        self.body_comparison_worker.status_changed.connect(
            self.show_worker_status
        )
        self.body_comparison_worker.completed.connect(
            self.character_body_comparison_completed
        )
        self.body_comparison_worker.failed.connect(
            self.character_body_comparison_failed
        )
        self.body_comparison_worker.completed.connect(
            self.body_comparison_worker_thread.quit
        )
        self.body_comparison_worker.failed.connect(
            self.body_comparison_worker_thread.quit
        )
        self.body_comparison_worker_thread.finished.connect(
            self.body_comparison_worker.deleteLater
        )
        self.body_comparison_worker_thread.finished.connect(
            self.body_comparison_worker_thread.deleteLater
        )
        self.body_comparison_worker_thread.finished.connect(
            self.clear_body_comparison_worker
        )
        self.body_comparison_worker_thread.finished.connect(
            self.resume_generation_workflow
        )
        self.body_comparison_worker_thread.start()

    @Slot(object)
    def character_body_comparison_completed(
        self,
        completed_result: object,
    ) -> None:
        """신체 마스크와 CatVTON 실제 전처리 입력을 공개하고 승인받는다."""
        comparison_candidate, input_snapshot, preflight_candidate = completed_result
        comparison_dialog = CharacterBodyComparisonReviewDialog(
            comparison_candidate,
            input_snapshot,
            preflight_candidate,
            self,
        )
        try:
            self.execute_approval_dialog(comparison_dialog)
            if not comparison_dialog.approved:
                self.release_confirmed_character_body_comparison()
                self.generate_button.setEnabled(False)
                self.body_comparison_button.setEnabled(True)
                self.pause_generation_workflow(
                    GenerationWorkflowStage.BODY_MASKING,
                    "Human-Agnostic Image와 변경 영역 승인이 취소되었습니다.",
                )
                return

            mask_refinement = comparison_candidate.mask_refinement
            agnostic_candidate = comparison_candidate.human_agnostic_candidate
            self.release_confirmed_character_body_comparison()
            self.confirmed_character_body_comparison = (
                ConfirmedCharacterBodyComparison(
                    clothing_type=find_catvton_clothing_type(
                        self.body_comparison_clothing_category
                    ),
                    approved_human_agnostic_image=(
                        agnostic_candidate.neutralized_image.copy()
                    ),
                    approved_change_mask=mask_refinement.safe_change_mask.copy(),
                    approved_model_mask=(
                        preflight_candidate.model_mask_image.copy()
                    ),
                    neutral_rgb=agnostic_candidate.neutral_rgb,
                    neutralized_pixel_count=agnostic_candidate.neutralized_pixel_count,
                    neutralized_percent=agnostic_candidate.neutralized_percent,
                    raw_mask_coverage_percent=agnostic_candidate.raw_mask_coverage_percent,
                    outside_foreground_pixel_count=(
                        comparison_candidate.clothing_removal_verification
                        .outside_foreground_pixel_count
                    ),
                    outside_foreground_percent=(
                        comparison_candidate.clothing_removal_verification
                        .outside_foreground_percent
                    ),
                    remaining_clothing_pixel_count=(
                        comparison_candidate.clothing_removal_verification
                        .remaining_clothing_pixel_count
                    ),
                    clothing_removal_percent=(
                        comparison_candidate.clothing_removal_verification
                        .removal_percent
                    ),
                    changed_pixel_count_outside_mask=agnostic_candidate.changed_pixel_count_outside_mask,
                    expansion_radius_pixels=(
                        mask_refinement.expansion_radius_pixels
                    ),
                    closing_radius_pixels=mask_refinement.closing_radius_pixels,
                    attempted_protected_overlap_pixels=(
                        mask_refinement.attempted_protected_overlap_pixels
                    ),
                    safe_change_pixel_count=(
                        mask_refinement.safe_change_pixel_count
                    ),
                    safe_change_percent=mask_refinement.safe_change_percent,
                    model_ids=comparison_candidate.model_ids,
                    preflight_person_sha256=preflight_candidate.person_sha256,
                    preflight_binary_mask_sha256=(
                        preflight_candidate.binary_mask_sha256
                    ),
                    preflight_model_mask_sha256=(
                        preflight_candidate.model_mask_sha256
                    ),
                    preflight_clothing_sha256=(
                        preflight_candidate.clothing_sha256
                    ),
                    preflight_protected_overlap_pixel_count=(
                        preflight_candidate.protected_overlap_pixel_count
                    ),
                    preflight_outside_foreground_pixel_count=(
                        preflight_candidate.outside_foreground_pixel_count
                    ),
                    preflight_soft_overlap_pixel_count=(
                        preflight_candidate.soft_overlap_pixel_count
                    ),
                    preflight_hard_overlap_pixel_count=(
                        preflight_candidate.hard_overlap_pixel_count
                    ),
                    preflight_removed_pixel_count=(
                        preflight_candidate.removed_pixel_count
                    ),
                )
            )
            self.generate_button.setEnabled(True)
            self.body_comparison_button.setEnabled(True)
            self.status_label.setText(
                "상태: Human-Agnostic Image 승인 완료 - "
                f"닫기={mask_refinement.closing_radius_pixels}px, "
                f"팽창={mask_refinement.expansion_radius_pixels}px, "
                f"변경={mask_refinement.safe_change_pixel_count:,}px "
                f"({mask_refinement.safe_change_percent:.3f}%), "
                f"원본 마스크 포함률="
                f"{agnostic_candidate.raw_mask_coverage_percent:.3f}%, "
                f"외곽 밖 SCHP 오탐="
                f"{comparison_candidate.clothing_removal_verification.outside_foreground_pixel_count:,}px "
                f"({comparison_candidate.clothing_removal_verification.outside_foreground_percent:.3f}%), "
                f"기존 의상 잔여="
                f"{comparison_candidate.clothing_removal_verification.remaining_clothing_pixel_count:,}px, "
                f"기존 의상 제거율="
                f"{comparison_candidate.clothing_removal_verification.removal_percent:.3f}%, "
                f"영역 밖 변경={agnostic_candidate.changed_pixel_count_outside_mask}px"
            )
        finally:
            comparison_candidate.close()
            input_snapshot.close()
            preflight_candidate.close()

    @Slot(str, str)
    def character_body_comparison_failed(
        self,
        message: str,
        details: str,
    ) -> None:
        """신체 비교 실패 원인과 재시도 행동을 표시한다."""
        self.release_confirmed_character_body_comparison()
        self.generate_button.setEnabled(False)
        self.body_comparison_button.setEnabled(True)
        self.status_label.setText(f"상태: 캐릭터 신체 비교 실패 ({message})")
        self.pause_generation_workflow(
            GenerationWorkflowStage.BODY_MASKING,
            f"캐릭터 신체 비교 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("캐릭터 신체 비교 실패")
        dialog.setText(message)
        dialog.setInformativeText(
            "의상과 캐릭터 승인은 유지합니다. 원인을 확인한 뒤 다시 실행하세요."
        )
        dialog.setDetailedText(details)
        dialog.exec()

    @Slot()
    def clear_body_comparison_worker(self) -> None:
        """종료된 신체 비교 작업 객체를 해제한다."""
        self.body_comparison_worker = None
        self.body_comparison_worker_thread = None

    def start_garment_geometry_preparation(self) -> None:
        """승인 4종 입력으로 조각 대응과 TPS 검토 후보를 1회 만든다."""
        if (
            self.garment_geometry_worker_thread is not None
            and self.garment_geometry_worker_thread.isRunning()
        ):
            return
        if (
            self.pending_clothing_base_candidate is None
            or self.pending_clothing_extraction is None
            or self.confirmed_character_body_comparison is None
        ):
            self.pause_generation_workflow(
                GenerationWorkflowStage.GARMENT_GEOMETRY,
                "TPS 입력이 없습니다: 기준 후보·추출 의상·승인 변경 마스크가 필요합니다.",
            )
            return
        if self.approved_pose_estimation is None:
            self.pause_generation_workflow(
                GenerationWorkflowStage.GARMENT_GEOMETRY,
                "의상 좌표를 추측하지 않습니다. DWPose 자세 승인이 필요합니다.",
            )
            return
        self.release_approved_garment_inputs()
        clothing_category = ClothingCategory(
            self.clothing_category_combo.currentData()
        )
        self.garment_geometry_worker_thread = QThread(self)
        self.garment_geometry_worker = GarmentGeometryWorker(
            self.pending_clothing_extraction.extracted_image,
            self.pending_clothing_base_candidate.image,
            self.confirmed_character_body_comparison.approved_change_mask,
            self.approved_pose_estimation,
            clothing_category,
        )
        self.garment_geometry_worker.moveToThread(
            self.garment_geometry_worker_thread
        )
        self.garment_geometry_worker_thread.started.connect(
            self.garment_geometry_worker.run
        )
        self.garment_geometry_worker.status_changed.connect(
            self.show_worker_status
        )
        self.garment_geometry_worker.completed.connect(
            self.garment_geometry_completed
        )
        self.garment_geometry_worker.failed.connect(
            self.garment_geometry_failed
        )
        self.garment_geometry_worker.completed.connect(
            self.garment_geometry_worker_thread.quit
        )
        self.garment_geometry_worker.failed.connect(
            self.garment_geometry_worker_thread.quit
        )
        self.garment_geometry_worker_thread.finished.connect(
            self.garment_geometry_worker.deleteLater
        )
        self.garment_geometry_worker_thread.finished.connect(
            self.garment_geometry_worker_thread.deleteLater
        )
        self.garment_geometry_worker_thread.finished.connect(
            self.clear_garment_geometry_worker
        )
        self.garment_geometry_worker_thread.finished.connect(
            self.resume_generation_workflow
        )
        self.garment_geometry_worker_thread.start()

    @Slot(object)
    def garment_geometry_completed(self, completed_result: object) -> None:
        review_candidate, component_matches = completed_result
        dialog = GarmentTpsReviewDialog(
            review_candidate,
            self.pending_clothing_extraction.extracted_image,
            component_matches,
            self,
        )
        try:
            self.execute_approval_dialog(dialog)
            if not dialog.approved:
                self.release_approved_garment_inputs()
                self.pause_generation_workflow(
                    GenerationWorkflowStage.GARMENT_GEOMETRY,
                    "사용자가 조각 대응 또는 TPS 좌표 후보를 거절했습니다.",
                )
                return
            self.approved_garment_warp = approve_garment_tps_warp_review(
                review_candidate
            )
            self.status_label.setText(
                "상태: 진단용 TPS 의상 워핑 - "
                f"TPS 승인, 조각={review_candidate.component_count}개, "
                f"승인 밖={review_candidate.protected_outside_alpha_pixels:,}px"
            )
        except Exception as error:
            self.release_approved_garment_inputs()
            self.pause_generation_workflow(
                GenerationWorkflowStage.GARMENT_GEOMETRY,
                f"TPS 승인 실패: {error}",
            )
            QMessageBox.critical(self, "TPS 승인 실패", str(error))
        finally:
            review_candidate.close()

    @Slot(str, str)
    def garment_geometry_failed(self, message: str, details: str) -> None:
        self.release_approved_garment_inputs()
        self.pause_generation_workflow(
            GenerationWorkflowStage.GARMENT_GEOMETRY,
            f"의상 좌표/TPS 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("의상 좌표/TPS 실패")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    @Slot()
    def clear_garment_geometry_worker(self) -> None:
        self.garment_geometry_worker = None
        self.garment_geometry_worker_thread = None

    def review_garment_lineart(self) -> None:
        """TPS 승인본에서 만든 6개 Lineart 중간 자료를 승인받는다."""
        if (
            self.approved_garment_warp is None
            or self.confirmed_character_body_comparison is None
        ):
            self.pause_generation_workflow(
                GenerationWorkflowStage.GARMENT_LINEART,
                "Lineart 입력이 없습니다: TPS와 변경 마스크 승인이 필요합니다.",
            )
            return
        review_candidate = None
        try:
            review_candidate = create_garment_lineart_review(
                self.approved_garment_warp,
                self.confirmed_character_body_comparison.approved_change_mask,
            )
            dialog = GarmentLineartReviewDialog(review_candidate, self)
            self.execute_approval_dialog(dialog)
            if not dialog.approved:
                self.pause_generation_workflow(
                    GenerationWorkflowStage.GARMENT_LINEART,
                    "사용자가 Lineart 제어 입력을 거절했습니다.",
                )
                return
            self.approved_garment_lineart = approve_garment_lineart_review(
                review_candidate
            )
            self.status_label.setText(
                "상태: 진단용 Garment Lineart - "
                f"Lineart 승인, 선={review_candidate.total_edge_pixels:,}px, "
                f"승인 밖={review_candidate.protected_edge_pixels_outside_approved_mask:,}px"
            )
        except Exception as error:
            self.pause_generation_workflow(
                GenerationWorkflowStage.GARMENT_LINEART,
                f"Lineart 생성/승인 실패: {error}",
            )
            QMessageBox.critical(self, "Lineart 실패", str(error))
        finally:
            if review_candidate is not None:
                review_candidate.close()

    def create_garment_inpaint_settings(self) -> GarmentInpaintSettings:
        """YAML 수치를 경로 해석 뒤 Stage 5 설정 계약으로 변환한다."""
        current_dir = Path(__file__).resolve().parent
        if self.config is None:
            self.config = load_yaml(current_dir / "configs" / "animagine.yaml")
        section = self.config.get("garment_inpaint", {})

        def resolved_path(key: str, default: str) -> Path:
            path = Path(str(section.get(key, default)))
            return path if path.is_absolute() else current_dir / path

        return GarmentInpaintSettings(
            python_executable=resolved_path(
                "python_executable",
                "D:/genai-cache/venv/Scripts/python.exe",
            ),
            runner_path=resolved_path(
                "runner_path",
                "scripts/garment_inpaint_runner.py",
            ),
            temporary_root=resolved_path(
                "temporary_root",
                "D:/genai-cache/temp/garment-inpaint",
            ),
            benchmark_root=resolved_path(
                "benchmark_root",
                "outputs/debug_benchmark",
            ),
            cache_dir=resolved_path(
                "cache_dir",
                "D:/genai-cache/huggingface",
            ),
            base_model_id=str(section.get(
                "base_model_id",
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            )),
            model_variant=str(section.get("model_variant", "fp16")),
            adapter_repository=str(
                section.get("adapter_repository", "h94/IP-Adapter")
            ),
            adapter_subfolder=str(section.get("adapter_subfolder", "sdxl_models")),
            adapter_weight=str(
                section.get(
                    "adapter_weight",
                    "ip-adapter-plus_sdxl_vit-h.safetensors",
                )
            ),
            adapter_image_encoder_subfolder=str(
                section.get(
                    "adapter_image_encoder_subfolder",
                    "models/image_encoder",
                )
            ),
            strength=float(section.get("strength", 0.90)),
            inference_steps=int(section.get("inference_steps", 28)),
            guidance_scale=float(section.get("guidance_scale", 5.5)),
            ip_adapter_scale=float(section.get("ip_adapter_scale", 0.80)),
            padding_mask_crop=int(section.get("padding_mask_crop", 64)),
            mask_threshold=int(section.get("mask_threshold", 128)),
            garment_board_size=int(section.get("garment_board_size", 1024)),
            garment_board_outer_padding=int(
                section.get("garment_board_outer_padding", 32)
            ),
            garment_board_cell_padding=int(
                section.get("garment_board_cell_padding", 16)
            ),
            garment_board_minimum_component_pixels=int(
                section.get("garment_board_minimum_component_pixels", 16)
            ),
            garment_board_maximum_components=int(
                section.get("garment_board_maximum_components", 8)
            ),
            timeout_seconds=int(section.get("timeout_seconds", 1800)),
            dtype=str(section.get("dtype", "float16")),
        )

    def start_garment_inpaint(self) -> None:
        """Human-Agnostic 승인본과 의상 참조로 GPU 작업을 1회 시작한다."""
        if (
            self.garment_inpaint_worker_thread is not None
            and self.garment_inpaint_worker_thread.isRunning()
        ):
            return
        if self.worker_thread is not None and self.worker_thread.isRunning():
            if not self.garment_inpaint_start_deferred:
                self.garment_inpaint_start_deferred = True
                self.status_label.setText(
                    "상태: 7/8 시작 전 Step 5 작업자 종료 대기 중..."
                )
                QTimer.singleShot(
                    100,
                    self.continue_garment_inpaint_after_generation_worker,
                )
            return
        if (
            self.pending_clothing_base_candidate is None
            or self.pending_clothing_extraction is None
            or self.confirmed_character_body_comparison is None
        ):
            self.pause_generation_workflow(
                GenerationWorkflowStage.CLOTHING_COMPOSITING,
                "2D Inpaint 승인 입력 4종 중 하나 이상이 없습니다.",
            )
            return
        base = self.pending_clothing_base_candidate
        tags = (
            self.confirmed_clothing_design.design_tags
            if self.confirmed_clothing_design is not None
            else ()
        )
        prompt, negative_prompt = build_garment_inpaint_prompts(
            base.prompt,
            base.negative_prompt,
            tuple(tags),
        )
        release_metrics = self.release_step5_pipeline()
        settings = replace(
            self.create_garment_inpaint_settings(),
            neutral_rgb=self.confirmed_character_body_comparison.neutral_rgb,
            step5_vram_before_allocated_mib=(
                release_metrics["before_allocated_mib"]
            ),
            step5_vram_after_allocated_mib=(
                release_metrics["after_allocated_mib"]
            ),
            step5_vram_after_reserved_mib=(
                release_metrics["after_reserved_mib"]
            ),
        )
        self.status_label.setText(
            "상태: 7/8 Step 5 모델 해제 완료 - "
            f"할당 {release_metrics['before_allocated_mib']:.1f}→"
            f"{release_metrics['after_allocated_mib']:.1f}MiB, "
            f"예약 {release_metrics['after_reserved_mib']:.1f}MiB"
        )
        self.garment_inpaint_worker_thread = QThread(self)
        self.garment_inpaint_worker = GarmentInpaintWorker(
            base.image,
            self.confirmed_character_body_comparison.approved_human_agnostic_image,
            self.confirmed_character_body_comparison.approved_change_mask,
            self.pending_clothing_extraction.extracted_image,
            prompt,
            negative_prompt,
            base.seed,
            settings,
        )
        self.garment_inpaint_worker.moveToThread(
            self.garment_inpaint_worker_thread
        )
        self.garment_inpaint_worker_thread.started.connect(
            self.garment_inpaint_worker.run
        )
        self.garment_inpaint_worker.status_changed.connect(
            self.show_worker_status
        )
        self.garment_inpaint_worker.progress_changed.connect(
            self.show_garment_inpaint_progress
        )
        self.garment_inpaint_worker.completed.connect(
            self.garment_inpaint_completed
        )
        self.garment_inpaint_worker.failed.connect(self.garment_inpaint_failed)
        self.garment_inpaint_worker.completed.connect(
            self.garment_inpaint_worker_thread.quit
        )
        self.garment_inpaint_worker.failed.connect(
            self.garment_inpaint_worker_thread.quit
        )
        self.garment_inpaint_worker_thread.finished.connect(
            self.garment_inpaint_worker.deleteLater
        )
        self.garment_inpaint_worker_thread.finished.connect(
            self.garment_inpaint_worker_thread.deleteLater
        )
        self.garment_inpaint_worker_thread.finished.connect(
            self.clear_garment_inpaint_worker
        )
        self.garment_inpaint_worker_thread.start()

    @Slot()
    def continue_garment_inpaint_after_generation_worker(self) -> None:
        """Step 5 Worker 참조가 정리된 뒤 Step 9 시작을 한 번만 재개한다."""
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QTimer.singleShot(
                100,
                self.continue_garment_inpaint_after_generation_worker,
            )
            return
        self.garment_inpaint_start_deferred = False
        self.start_garment_inpaint()

    def release_step5_pipeline(self) -> dict[str, float]:
        """Step 9 전에 GUI와 Worker가 보유한 Step 5 모델 참조를 해제한다."""
        cuda_available = torch.cuda.is_available()
        before_allocated = (
            torch.cuda.memory_allocated() if cuda_available else 0
        )
        pipeline = self.pipeline
        self.pipeline = None

        if (
            self.worker is not None
            and getattr(self.worker, "pipeline", None) is pipeline
        ):
            self.worker.pipeline = None

        if pipeline is not None:
            if hasattr(pipeline, "maybe_free_model_hooks"):
                pipeline.maybe_free_model_hooks()
            if hasattr(pipeline, "remove_all_hooks"):
                pipeline.remove_all_hooks()
            del pipeline

        gc.collect()
        if cuda_available:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            after_allocated = torch.cuda.memory_allocated()
            after_reserved = torch.cuda.memory_reserved()
        else:
            after_allocated = 0
            after_reserved = 0

        mib = 1024**2
        return {
            "before_allocated_mib": before_allocated / mib,
            "after_allocated_mib": after_allocated / mib,
            "after_reserved_mib": after_reserved / mib,
        }

    @Slot(object)
    def garment_inpaint_completed(
        self,
        review_candidate: GarmentInpaintReviewCandidate,
    ) -> None:
        dialog = GarmentInpaintReviewDialog(review_candidate, self)
        try:
            self.execute_approval_dialog(dialog)
            if not dialog.approved:
                self.pause_generation_workflow(
                    GenerationWorkflowStage.CLOTHING_COMPOSITING,
                    "사용자가 2D 의상 Inpaint 결과를 거절했습니다.",
                )
                return
            approved = approve_garment_inpaint_review(review_candidate)
            base = self.pending_clothing_base_candidate
            if base is None:
                approved.close()
                raise RuntimeError("Inpaint 승인 시 기준 후보가 없습니다.")
            change_mask = (
                self.confirmed_character_body_comparison.approved_change_mask.copy()
            )
            final_candidate = replace(
                base,
                image=approved.image,
                before_clothing_image=None,
                clothing_change_mask=change_mask,
                clothing_reference_name=(
                    Path(self.outfit_path).name if self.outfit_path else None
                ),
                clothing_category=self.clothing_category_combo.currentData(),
                clothing_try_on_status="completed_2d_inpaint",
                clothing_verification_warning_ko=None,
                raw_clothing_try_on_image=None,
                clothing_difference_image=None,
                clothing_effect_metrics=None,
            )
            self.release_pending_clothing_base_candidate()
            self.release_approved_garment_inputs()
            self.release_confirmed_character_body_comparison()
            self.pending_character_candidate = final_candidate
            self.candidate_is_approved = False
            self.show_character_candidate(final_candidate)
            self.approve_candidate_button.setEnabled(True)
            self.reject_candidate_button.setEnabled(True)
            self.open_original_size_button.setEnabled(True)
            self.move_generation_workflow(
                GenerationWorkflowStage.FINAL_REVIEW,
                "2D 의상 후보 최종 승인 대기 - 자동 저장 0개",
            )
        except Exception as error:
            self.pause_generation_workflow(
                GenerationWorkflowStage.CLOTHING_COMPOSITING,
                f"2D Inpaint 승인 실패: {error}",
            )
            QMessageBox.critical(self, "2D Inpaint 승인 실패", str(error))
        finally:
            review_candidate.close()

    @Slot(str, str)
    def garment_inpaint_failed(self, message: str, details: str) -> None:
        self.pause_generation_workflow(
            GenerationWorkflowStage.CLOTHING_COMPOSITING,
            f"2D 의상 Inpaint 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("2D 의상 Inpaint 실패")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    @Slot()
    def clear_garment_inpaint_worker(self) -> None:
        self.garment_inpaint_worker = None
        self.garment_inpaint_worker_thread = None

    @Slot(str, str)
    def outfit_region_preparation_failed(
        self,
        message: str,
        details: str,
    ) -> None:
        """의상 이미지 정규화 실패를 표시하고 생성 입력에서 제외한다."""
        self.pending_outfit_path = None
        self.outfit_path = None
        self.clothing_region_candidates = ()
        self.clothing_source_size = None
        self.clothing_region_measurements = ()
        self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
        self.generate_button.setEnabled(
            self.approved_reference_image is not None
        )
        self.pause_generation_workflow(
            GenerationWorkflowStage.CLOTHING_MASKING,
            f"의상 이미지 준비 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("의상 이미지 준비 실패")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    @Slot()
    def clear_outfit_worker(self) -> None:
        """종료된 의상 탐지 작업 객체를 해제한다."""
        self.outfit_worker = None
        self.outfit_worker_thread = None


    def start_reference_preparation(self, reference_path: Path) -> None:
        """참조 이미지 화질 검사와 필요한 확대 복원을 별도 작업으로 시작한다."""
        if (
            self.reference_worker_thread is not None
            and self.reference_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "참조 이미지 화질을 확인하고 있습니다.",
            )
            return

        self.release_approved_reference_image()
        self.generate_button.setEnabled(False)
        self.status_label.setText("상태: 참조 이미지 화질 확인 준비 중...")
        try:
            current_dir = Path(__file__).resolve().parent
            self.config = load_yaml(current_dir / "configs" / "animagine.yaml")
            self.config["style"]["enabled"] = True
            self.config["style"]["reference_image"] = str(reference_path)
            validate_config(self.config)
        except Exception as error:
            self.status_label.setText(f"상태: 참조 이미지 설정 오류 ({error})")
            QMessageBox.critical(self, "참조 이미지 설정 오류", str(error))
            self.pause_generation_workflow(
                GenerationWorkflowStage.REFERENCE_PREPARING,
                f"참조 이미지 설정 오류: {error}",
            )
            return
        self.reference_worker_thread = QThread(self)
        self.reference_worker = ReferencePreparationWorker(
            reference_path,
            self.config,
        )
        self.reference_worker.moveToThread(self.reference_worker_thread)
        self.reference_worker_thread.started.connect(self.reference_worker.run)
        self.reference_worker.status_changed.connect(self.show_worker_status)
        self.reference_worker.completed.connect(
            self.reference_preparation_completed
        )
        self.reference_worker.failed.connect(self.reference_preparation_failed)
        self.reference_worker.completed.connect(self.reference_worker_thread.quit)
        self.reference_worker.failed.connect(self.reference_worker_thread.quit)
        self.reference_worker_thread.finished.connect(
            self.reference_worker.deleteLater
        )
        self.reference_worker_thread.finished.connect(
            self.reference_worker_thread.deleteLater
        )
        self.reference_worker_thread.finished.connect(
            self.clear_reference_worker
        )
        self.reference_worker_thread.finished.connect(
            self.resume_generation_workflow
        )
        self.reference_worker_thread.start()

    @Slot()
    def clear_outfit_reference(self) -> None:
        """선택 의상 입력만 해제하고 캐릭터 기준 이미지는 유지한다."""
        if (
            self.outfit_worker_thread is not None
            and self.outfit_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self, "안내", "의상 영역 자동 탐지가 끝난 뒤 해제하세요."
            )
            return
        if (
            self.mask_worker_thread is not None
            and self.mask_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self, "안내", "SAM2 의상 마스크 생성이 끝난 뒤 해제하세요."
            )
            return
        if (
            self.design_worker_thread is not None
            and self.design_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "WD14 의상 디자인 분석이 끝난 뒤 해제하세요.",
            )
            return
        if (
            self.body_comparison_worker_thread is not None
            and self.body_comparison_worker_thread.isRunning()
        ):
            QMessageBox.information(
                self,
                "안내",
                "캐릭터 신체 비교가 끝난 뒤 의상 참조를 해제하세요.",
            )
            return
        if self.pending_character_candidate is not None:
            QMessageBox.information(self, "안내", "현재 후보를 먼저 판단하세요.")
            return
        self.release_pending_clothing_base_candidate()
        self.selected_outfit_path = None
        self.workflow_context = None
        self.outfit_path = None
        self.pending_outfit_path = None
        self.release_clothing_mask_state()
        self.clothing_region_candidates = ()
        self.clothing_source_size = None
        self.clothing_region_measurements = ()
        self.generate_button.setEnabled(self.approved_reference_image is not None)
        self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
        self.update_input_ready_status()

    @Slot(object)
    def reference_preparation_completed(
        self,
        preparation_result: ReferenceImagePreparationResult,
    ) -> None:
        """화질 검사 결과를 표시하고 보정본이 있으면 사용자 승인을 받는다."""
        quality_report = preparation_result.quality_report
        enhancement_candidate = preparation_result.enhancement_candidate
        source_name = Path(self.style_path).name

        if enhancement_candidate is None:
            self.approved_reference_image = approve_original_reference_image(
                preparation_result.original_image,
                source_name,
                quality_report,
            )
            preparation_result.original_image.close()
            self.generate_button.setEnabled(
                self.outfit_path is None
                or self.confirmed_clothing_design is not None
            )
            self.body_comparison_button.setEnabled(False)
            self.status_label.setText(
                "상태: 원본 참조 이미지 사용 가능 - "
                + (
                    "기준 후보 생성 필요"
                    if self.outfit_path
                    else "생성 준비 완료"
                )
            )
            return

        comparison_dialog = ReferenceEnhancementDialog(
            enhancement_candidate,
            self,
        )
        self.execute_approval_dialog(comparison_dialog)
        if comparison_dialog.use_enhanced_image:
            self.approved_reference_image = approve_enhanced_reference_image(
                enhancement_candidate,
                source_name,
            )
            self.generate_button.setEnabled(
                self.outfit_path is None
                or self.confirmed_clothing_design is not None
            )
            self.body_comparison_button.setEnabled(False)
            self.status_label.setText(
                "상태: 보정 참조 이미지 승인 - "
                + (
                    "기준 후보 생성 필요"
                    if self.outfit_path
                    else "생성 준비 완료"
                )
            )
        else:
            self.approved_reference_image = None
            self.generate_button.setEnabled(False)
            self.pause_generation_workflow(
                GenerationWorkflowStage.REFERENCE_PREPARING,
                "보정 이미지 승인이 취소되었습니다.",
            )

        preparation_result.original_image.close()
        enhancement_candidate.original_image.close()
        enhancement_candidate.enhanced_image.close()

    @Slot(str, str)
    def reference_preparation_failed(self, message: str, details: str) -> None:
        """참조 이미지 준비 실패를 표시하고 생성을 막는다."""
        self.release_approved_reference_image()
        self.generate_button.setEnabled(False)
        self.status_label.setText(f"상태: 참조 이미지 준비 실패 ({message})")
        self.pause_generation_workflow(
            GenerationWorkflowStage.REFERENCE_PREPARING,
            f"참조 이미지 준비 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("참조 이미지 준비 실패")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    def release_approved_reference_image(self) -> None:
        """이전 참조 승인 이미지를 메모리에서 해제한다."""
        if self.approved_reference_image is not None:
            self.approved_reference_image.image.close()
        self.approved_reference_image = None
        self.release_pending_clothing_base_candidate()
        self.release_confirmed_character_body_comparison()
        self.body_comparison_clothing_category = None
        self.body_comparison_button.setEnabled(False)

    @Slot()
    def clear_reference_worker(self) -> None:
        """종료된 참조 이미지 준비 작업 객체를 해제한다."""
        self.reference_worker = None
        self.reference_worker_thread = None

    def start_generation(self) -> None:
        """등록 입력으로 자동 실행을 시작하거나 실패 단계부터 재시도한다."""
        if self.pending_character_candidate is not None:
            QMessageBox.information(
                self,
                "안내",
                "현재 후보의 승인 또는 거절을 먼저 선택하세요.",
            )
            return
        if not self.style_path:
            QMessageBox.warning(self, "입력 오류", "캐릭터 기준 이미지를 선택하세요.")
            return

        if self.workflow_context is None:
            self.workflow_context = GenerationWorkflowContext(
                character_image_path=Path(self.style_path),
                clothing_image_path=self.selected_outfit_path,
                pose_image_path=self.selected_pose_path,
            )
        elif self.workflow_context.current_stage is GenerationWorkflowStage.FAILED:
            self.workflow_context.retry()
        elif self.workflow_context.active:
            QMessageBox.information(
                self,
                "자동 작업 실행 중",
                "현재 단계가 끝나면 다음 단계가 자동으로 시작됩니다.",
            )
            return
        else:
            self.workflow_context = GenerationWorkflowContext(
                character_image_path=Path(self.style_path),
                clothing_image_path=self.selected_outfit_path,
                pose_image_path=self.selected_pose_path,
            )

        self.set_workflow_input_buttons_enabled(False)
        self.generate_button.setEnabled(False)
        self.advance_generation_workflow()

    def execute_approval_dialog(self, dialog: QDialog) -> int:
        """승인 창을 동시에 1개만 열고 닫힌 뒤 자동 흐름을 1회 재개한다."""
        if self.approval_dialog_open:
            return int(QDialog.DialogCode.Rejected)
        self.approval_dialog_open = True
        try:
            return int(dialog.exec())
        finally:
            self.approval_dialog_open = False
            QTimer.singleShot(0, self.resume_generation_workflow)

    def advance_generation_workflow(self) -> None:
        """완료된 승인 상태를 확인하고 다음 작업 하나만 자동 시작한다."""
        if self.approval_dialog_open:
            return
        workflow_context = self.workflow_context
        if workflow_context is None:
            return
        if any(
            thread is not None and thread.isRunning()
            for thread in (
                self.reference_worker_thread,
                self.outfit_worker_thread,
                self.mask_worker_thread,
                self.design_worker_thread,
                self.pose_estimation_worker_thread,
                self.body_comparison_worker_thread,
                self.garment_geometry_worker_thread,
                self.garment_inpaint_worker_thread,
                self.worker_thread,
            )
        ):
            return

        if self.approved_reference_image is None:
            self.move_generation_workflow(
                GenerationWorkflowStage.REFERENCE_PREPARING,
                "캐릭터 기준 이미지 화질 확인",
            )
            self.start_reference_preparation(
                workflow_context.character_image_path
            )
            return

        if (
            workflow_context.clothing_image_path is not None
            and self.confirmed_clothing_design is None
        ):
            if self.pending_clothing_extraction is not None:
                self.move_generation_workflow(
                    GenerationWorkflowStage.CLOTHING_ANALYZING,
                    "WD14 의상 디자인 분석",
                )
                self.start_clothing_design_analysis()
            else:
                self.move_generation_workflow(
                    GenerationWorkflowStage.CLOTHING_MASKING,
                    "의상 영역 탐지와 SAM2 마스크 생성",
                )
                self.start_outfit_region_preparation(
                    workflow_context.clothing_image_path
                )
            return

        if (
            workflow_context.pose_image_path is not None
            and self.approved_pose_estimation is None
        ):
            self.move_generation_workflow(
                GenerationWorkflowStage.POSE_ESTIMATING,
                "DWPose 관절 18개 추출",
            )
            if self.approved_pose_reference is None:
                self.review_pose_reference(workflow_context.pose_image_path)
            else:
                self.start_pose_reference_estimation()
            return

        if self.pending_clothing_base_candidate is not None:
            if self.confirmed_character_body_comparison is None:
                self.move_generation_workflow(
                    GenerationWorkflowStage.BODY_MASKING,
                    "생성 후보 신체와 기존 의상 마스크 추출",
                )
                self.start_character_body_comparison()
            else:
                self.move_generation_workflow(
                    GenerationWorkflowStage.CLOTHING_COMPOSITING,
                    "Human-Agnostic + SDXL Inpaint + IP-Adapter Plus 의상 생성",
                )
                self.start_garment_inpaint()
            return

        self.move_generation_workflow(
            GenerationWorkflowStage.BASE_GENERATING,
            "ControlNet 기준 후보 생성",
        )
        self._start_model_generation()

    @Slot()
    def resume_generation_workflow(self) -> None:
        """Worker가 완전히 종료된 뒤 보관된 승인 상태에서 다음 단계를 찾는다."""
        if (
            self.approval_dialog_open
            or self.workflow_context is None
            or not self.workflow_context.active
        ):
            return
        self.advance_generation_workflow()

    def move_generation_workflow(
        self,
        stage: GenerationWorkflowStage,
        detail: str,
    ) -> None:
        """자동 진행 위치와 활성 8단계 수치를 GUI에 표시한다."""
        if self.workflow_context is None:
            return
        self.workflow_context.move_to(stage)
        current_step, total_steps = self.workflow_context.progress
        self.status_label.setText(
            f"상태: 전체 진행 {current_step}/{total_steps} - {detail}"
        )

    def pause_generation_workflow(
        self,
        failed_stage: GenerationWorkflowStage,
        reason: str,
    ) -> None:
        """성공한 이전 결과를 유지하고 실패·취소 위치에서 자동 진행을 멈춘다."""
        if self.workflow_context is None:
            return
        self.workflow_context.fail(failed_stage)
        current_step, total_steps = self.workflow_context.progress
        self.set_workflow_input_buttons_enabled(True)
        self.generate_button.setEnabled(self.style_path is not None)
        self.generate_button.setText("실패 단계 다시 시도")
        self.status_label.setText(
            f"상태: 전체 진행 {current_step}/{total_steps} 중지 - {reason}"
        )

    def set_workflow_input_buttons_enabled(self, enabled: bool) -> None:
        """자동 실행 중 입력 3개와 화면 범위 변경을 잠근다."""
        self.style_button.setEnabled(enabled)
        self.outfit_button.setEnabled(enabled)
        self.clear_outfit_button.setEnabled(enabled)
        self.pose_button.setEnabled(enabled)
        self.clear_pose_button.setEnabled(enabled)
        self.clothing_category_combo.setEnabled(enabled)
        self.framing_combo.setEnabled(enabled)

    def _start_model_generation(self):
        if self.pending_character_candidate is not None:
            QMessageBox.information(
                self,
                "안내",
                "현재 후보의 승인 또는 거절을 먼저 선택하세요.",
            )
            return

        if not self.style_path:
            QMessageBox.warning(self, "경고", "캐릭터 기준 이미지를 선택하세요.")
            return
        if self.approved_reference_image is None:
            QMessageBox.warning(
                self,
                "경고",
                "참조 이미지 화질 확인과 보정 이미지 승인을 먼저 완료하세요.",
            )
            return
        if (
            self.approved_pose_reference is not None
            and self.approved_pose_estimation is None
        ):
            QMessageBox.warning(
                self,
                "자세 승인 필요",
                "자세 이미지를 선택했습니다. 관절 추출과 뼈대 지도 승인을 먼저 완료하세요.",
            )
            return
        if (
            self.outfit_path is not None
            and self.confirmed_character_body_comparison is not None
        ):
            selected_clothing_category = ClothingCategory(
                self.clothing_category_combo.currentData()
            )
            if self.body_comparison_clothing_category is not selected_clothing_category:
                self.invalidate_character_body_comparison()
                QMessageBox.information(
                    self,
                    "의상 종류 변경",
                    "의상 종류가 바뀌어 이전 신체 비교 승인을 취소했습니다. 다시 비교하세요.",
                )
                return

        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.information(self, "안내", "이미지를 생성하고 있습니다.")
            return

        run_log: GenerationRunLog | None = None
        try:
            current_dir = Path(__file__).resolve().parent
            run_log = create_generation_run_log(current_dir)
            run_log.write_stage(
                "사용자 입력",
                f"참조 이미지={Path(self.style_path).name}",
            )
            config_path = current_dir / "configs" / "animagine.yaml"
            self.config = load_yaml(config_path)
            self.config["style"]["enabled"] = True
            self.config["style"]["reference_image"] = self.style_path
            validate_config(self.config)

            model_config = self.config["model"]
            generation_config = self.config["generation"]
            style_config = self.config["style"]
            generation_settings = CharacterGenerationSettings(
                model_id=str(model_config["id"]),
                reference_adapter_id="/".join(
                    (
                        str(style_config["adapter_repository"]),
                        str(style_config["adapter_subfolder"]),
                        str(style_config["adapter_weight"]),
                    )
                ),
                inference_steps=int(generation_config["steps"]),
                guidance_scale=float(generation_config["guidance_scale"]),
                original_image_change_strength=float(
                    generation_config["original_image_change_strength"]
                ),
                reference_image_strength=float(style_config["scale"]),
                default_negative_prompt=str(
                    generation_config.get("default_negative_prompt", "")
                ),
            )

            # CharacterGenerationInput(캐릭터 생성 입력)
            # - 포함: 사용자가 고른 참조 이미지와 화면 범위.
            # - 생성: GUI 선택값을 읽어 만든다.
            # - 처리: 규칙 검사만 수행하며 AI 모델은 실행하지 않는다.
            # - 저장: 저장하지 않고 생성 요청을 만드는 데만 사용한다.
            # - 다음 사용처: request.py에서 CharacterGenerationRequest로 변환한다.
            selected_framing_type = CharacterFramingType(
                self.framing_combo.currentData()
            )
            generation_input = CharacterGenerationInput(
                reference_image_path=Path(self.style_path),
                framing_type=selected_framing_type,
                approved_reference_image=self.approved_reference_image.image,
                reference_enhancement_applied=(
                    self.approved_reference_image.enhancement_applied
                ),
                reference_enhancement_model_id=(
                    self.approved_reference_image.enhancement_model_id
                ),
                reference_quality_status=self.approved_reference_image.quality_status.value,
            )
            generation_request = prepare_character_generation_request(
                generation_input,
                generation_settings,
                candidate_number=1,
            )
            clothing_reference_input = None
            catvton_settings = None
            approved_agnostic_input = None
            if (
                self.outfit_path
                and self.confirmed_character_body_comparison is not None
            ):
                if (
                    not self.clothing_region_candidates
                    or not self.clothing_region_measurements
                ):
                    raise ValueError(
                        "의상 영역 자동 탐지 또는 수동 선택 승인을 먼저 완료하세요."
                    )
                clothing_config = self.config.get("clothing_try_on", {})
                if not clothing_config.get("enabled", False):
                    raise ValueError(
                        "의상 참조 기능이 설정에서 꺼져 있습니다."
                    )
                clothing_category = ClothingCategory(
                    self.clothing_category_combo.currentData()
                )
                clothing_reference_input = ClothingReferenceInput(
                    image_path=Path(self.outfit_path),
                    category=clothing_category,
                    region_box_xyxy=None,
                    approved_image=(
                        self.pending_clothing_extraction.extracted_image
                    ),
                )
                approved_agnostic_input = CharacterAgnosticApprovedInput(
                    human_agnostic_image=(
                        self.confirmed_character_body_comparison
                        .approved_human_agnostic_image.copy()
                    ),
                    approved_change_mask=(
                        self.confirmed_character_body_comparison
                        .approved_change_mask.copy()
                    ),
                    clothing_type=(
                        self.confirmed_character_body_comparison.clothing_type
                    ),
                    approved_mask_pixel_count=(
                        self.confirmed_character_body_comparison
                        .safe_change_pixel_count
                    ),
                    approved_model_mask=(
                        self.confirmed_character_body_comparison
                        .approved_model_mask.copy()
                    ),
                    preflight_person_sha256=(
                        self.confirmed_character_body_comparison
                        .preflight_person_sha256
                    ),
                    preflight_binary_mask_sha256=(
                        self.confirmed_character_body_comparison
                        .preflight_binary_mask_sha256
                    ),
                    preflight_model_mask_sha256=(
                        self.confirmed_character_body_comparison
                        .preflight_model_mask_sha256
                    ),
                    preflight_clothing_sha256=(
                        self.confirmed_character_body_comparison
                        .preflight_clothing_sha256
                    ),
                )
                runner_path = Path(str(clothing_config["runner_path"]))
                if not runner_path.is_absolute():
                    runner_path = current_dir / runner_path
                catvton_settings = CatVTONLocalSettings(
                    python_executable=Path(
                        str(clothing_config["python_executable"])
                    ),
                    repository_path=Path(
                        str(clothing_config["repository_path"])
                    ),
                    runner_path=runner_path,
                    temporary_root=Path(
                        str(clothing_config["temporary_root"])
                    ),
                    cache_dir=Path(str(clothing_config["cache_dir"])),
                    model_id=str(clothing_config["model_id"]),
                    base_model_id=str(clothing_config["base_model_id"]),
                    width=int(clothing_config["width"]),
                    height=int(clothing_config["height"]),
                    inference_steps=int(
                        clothing_config["inference_steps"]
                    ),
                    guidance_scale=float(
                        clothing_config["guidance_scale"]
                    ),
                    mixed_precision=str(
                        clothing_config["mixed_precision"]
                    ),
                    timeout_seconds=int(
                        clothing_config["timeout_seconds"]
                    ),
                    safety_check_enabled=bool(
                        clothing_config.get(
                            "safety_check_enabled",
                            False,
                        )
                    ),
                    mask_blur_factor=int(
                        clothing_config.get("mask_blur_factor", 9)
                    ),
                )
                run_log.write_stage(
                    "의상 참조 입력",
                    (
                        f"파일={Path(self.outfit_path).name}, "
                        f"종류={clothing_category.value}, "
                        f"영역 수={len(self.clothing_region_candidates)}개, "
                        f"영역 목록="
                        f"{tuple(region.box_xyxy for region in self.clothing_region_candidates)}, "
                        "승인된 투명 의상 추출본 사용, "
                        f"마스크 팽창={catvton_settings.minimum_mask_expansion_pixels}~"
                        f"{catvton_settings.maximum_mask_expansion_pixels}px, "
                        f"닫기 반경={catvton_settings.mask_closing_radius_pixels}px, "
                        "보호 영역과 겹쳐 차단한 픽셀="
                        f"{self.confirmed_character_body_comparison.attempted_protected_overlap_pixels:,}px, "
                        "CatVTON 별도 환경 사용, "
                        "마스크 출처=user_approved, AutoMasker 실행=0회, "
                        "model_mask 출처=user_approved_preflight, "
                        "약한 침범="
                        f"{self.confirmed_character_body_comparison.preflight_soft_overlap_pixel_count:,}px, "
                        "강한 침범="
                        f"{self.confirmed_character_body_comparison.preflight_hard_overlap_pixel_count:,}px, "
                        "금지 영역 제거="
                        f"{self.confirmed_character_body_comparison.preflight_removed_pixel_count:,}px, "
                        "안전 검사="
                        f"{'활성화' if catvton_settings.safety_check_enabled else '비활성화'}, "
                        "승인 마스크 픽셀="
                        f"{approved_agnostic_input.approved_mask_pixel_count:,}px"
                    ),
                )


            run_log.write_stage(
                "요청 준비",
                (
                    f"화면 범위={generation_request.framing_type.value}, "
                    f"원본 크기={generation_request.reference_image.width}x"
                    f"{generation_request.reference_image.height}, "
                    f"크기={generation_request.width}x{generation_request.height}, "
                    f"시드={generation_request.seed}, "
                    f"원본 이미지 변경 강도="
                    f"{generation_request.original_image_change_strength:.2f}, "
                    f"참조 이미지 반영 강도="
                    f"{generation_request.reference_image_strength:.2f}, "
                    f"모델={generation_request.model_id}, "
                    "자세 제어="
                    f"{'사용' if self.approved_pose_estimation is not None else '미사용'}"
                ),
            )

            requested_pose_pipeline = (
                self.pending_clothing_base_candidate is None
                and self.approved_pose_estimation is not None
            )
            if (
                self.pending_clothing_base_candidate is None
                and self.pipeline is not None
                and bool(
                getattr(
                    self.pipeline,
                    "_genai_lab_pose_control_enabled",
                    False,
                )
                ) != requested_pose_pipeline
            ):
                run_log.write_stage(
                    "모델 전환",
                    (
                        "기존 모델의 자세 제어 상태와 요청이 달라 재사용하지 않음, "
                        f"새 자세 제어={'사용' if requested_pose_pipeline else '미사용'}"
                    ),
                )
                if hasattr(self.pipeline, "maybe_free_model_hooks"):
                    self.pipeline.maybe_free_model_hooks()
                self.pipeline = None
                import torch

                torch.cuda.empty_cache()

            self.generate_button.setEnabled(False)
            self.status_label.setText("상태: 생성 작업 준비 중...")

            self.worker_thread = QThread(self)
            self.worker = GenerationWorker(
                self.config,
                generation_request,
                current_dir,
                run_log,
                clothing_reference_input,
                catvton_settings,
                approved_agnostic_input,
                self.approved_pose_estimation,
                self.pending_clothing_base_candidate,
                self.pipeline,
            )
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.run)
            self.worker.status_changed.connect(self.show_worker_status)
            self.worker.completed.connect(self.generation_completed)
            self.worker.failed.connect(self.generation_failed)
            self.worker.completed.connect(self.worker_thread.quit)
            self.worker.failed.connect(self.worker_thread.quit)
            self.worker_thread.finished.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            self.worker_thread.finished.connect(self.clear_worker)
            self.worker_thread.start()
            run_log = None
        except Exception as error:
            if (
                "approved_agnostic_input" in locals()
                and approved_agnostic_input is not None
            ):
                approved_agnostic_input.close()
            error_message = str(error)
            if run_log is not None:
                run_log.write_failure(
                    "요청 준비",
                    error,
                    find_recovery_action(error),
                )
                error_message += f"\n\n로그 파일: {run_log.file_path}"
                run_log.close()
            self.generate_button.setEnabled(True)
            self.status_label.setText(f"상태: 설정 오류 ({error})")
            failed_stage = (
                self.workflow_context.current_stage
                if self.workflow_context is not None
                and self.workflow_context.current_stage
                is not GenerationWorkflowStage.FAILED
                else GenerationWorkflowStage.BASE_GENERATING
            )
            self.pause_generation_workflow(
                failed_stage,
                f"생성 요청 준비 실패: {error}",
            )
            QMessageBox.critical(self, "설정 오류", error_message)

    @Slot(str)
    def show_worker_status(self, message):
        self.status_label.setText(f"상태: {message}")

    @Slot(object)
    def show_garment_inpaint_progress(
        self,
        progress: GarmentInpaintProgress,
    ) -> None:
        phase_labels = {
            "runner_started": "실행기 시작",
            "pipeline_loading": "Animagine XL 파이프라인 로딩",
            "ip_adapter_loading": "IP-Adapter 로딩",
            "diffusion_running": "Diffusion 추론",
            "output_saving": "결과 변환·저장",
            "completed": "Inpaint 실행 완료",
        }
        callback_text = (
            f"콜백={progress.current_step}회"
            if progress.current_step is not None
            else "콜백=0회"
        )
        configured_text = (
            f"설정={progress.configured_steps}단계"
            if progress.configured_steps is not None
            else "설정=28단계"
        )
        self.status_label.setText(
            "상태: 7/8 "
            f"{phase_labels.get(progress.phase, progress.phase)} | "
            f"{callback_text} | {configured_text} | "
            f"단계 경과={progress.phase_elapsed_seconds:.1f}초 | "
            f"전체 경과={progress.total_elapsed_seconds:.1f}초 | "
            "제한=1,800초"
        )

    @Slot(object, object)
    def generation_completed(
        self,
        character_candidate: CharacterGenerationCandidate,
        pipeline,
    ):
        self.pipeline = pipeline
        if character_candidate.original_generated_image is not None:
            comparison_dialog = DetailCorrectionComparisonDialog(
                character_candidate,
                self,
            )
            dialog_result = self.execute_approval_dialog(comparison_dialog)
            use_corrected_image = (
                dialog_result == QDialog.DialogCode.Accepted
                and comparison_dialog.use_corrected_image
            )
            if use_corrected_image:
                character_candidate.original_generated_image.close()
                character_candidate = replace(
                    character_candidate,
                    original_generated_image=None,
                )
            else:
                character_candidate.image.close()
                character_candidate = replace(
                    character_candidate,
                    image=character_candidate.original_generated_image,
                    original_generated_image=None,
                    detail_correction_status="rejected_by_user",
                    detail_verification_warning_ko=(
                        "사용자가 부분 보정 전 이미지를 선택했습니다."
                    ),
                )
        if (
            self.outfit_path is not None
            and self.confirmed_character_body_comparison is None
            and character_candidate.before_clothing_image is None
        ):
            self.pending_clothing_base_candidate = character_candidate
            self.show_character_candidate(character_candidate)
            self.body_comparison_button.setEnabled(True)
            self.generate_button.setEnabled(False)
            self.status_label.setText(
                "상태: 기준 후보 생성 완료 - 같은 후보의 신체 비교 시작"
            )
            self.move_generation_workflow(
                GenerationWorkflowStage.BODY_MASKING,
                "생성 후보 신체와 기존 의상 마스크 추출",
            )
            self.start_character_body_comparison()
            return

        if self.pending_clothing_base_candidate is not None:
            self.pending_clothing_base_candidate = None

        if character_candidate.before_clothing_image is not None:
            clothing_dialog = ClothingTryOnComparisonDialog(
                character_candidate,
                self,
            )
            dialog_result = self.execute_approval_dialog(clothing_dialog)
            use_clothing_image = (
                dialog_result == QDialog.DialogCode.Accepted
                and clothing_dialog.use_clothing_image
            )
            if use_clothing_image:
                character_candidate.before_clothing_image.close()
                if character_candidate.clothing_change_mask is not None:
                    character_candidate.clothing_change_mask.close()
                if character_candidate.raw_clothing_try_on_image is not None:
                    character_candidate.raw_clothing_try_on_image.close()
                if character_candidate.clothing_difference_image is not None:
                    character_candidate.clothing_difference_image.close()
                character_candidate = replace(
                    character_candidate,
                    before_clothing_image=None,
                    clothing_change_mask=None,
                    raw_clothing_try_on_image=None,
                    clothing_difference_image=None,
                )
            else:
                character_candidate.image.close()
                if character_candidate.clothing_change_mask is not None:
                    character_candidate.clothing_change_mask.close()
                if character_candidate.raw_clothing_try_on_image is not None:
                    character_candidate.raw_clothing_try_on_image.close()
                if character_candidate.clothing_difference_image is not None:
                    character_candidate.clothing_difference_image.close()
                character_candidate = replace(
                    character_candidate,
                    image=character_candidate.before_clothing_image,
                    before_clothing_image=None,
                    clothing_change_mask=None,
                    raw_clothing_try_on_image=None,
                    clothing_difference_image=None,
                    clothing_try_on_status="rejected_by_user",
                    clothing_verification_warning_ko=(
                        "사용자가 의상 적용 전 후보를 선택했습니다."
                    ),
                )
        elif character_candidate.clothing_try_on_status == "failed":
            QMessageBox.warning(
                self,
                "의상 참조 적용 실패",
                (
                    character_candidate.clothing_verification_warning_ko
                    or "의상 합성에 실패해 기본 생성 후보를 유지합니다."
                ),
            )

        self.save_candidate_button.setEnabled(False)
        self.discard_candidate_button.setEnabled(False)
        self.open_original_size_button.setEnabled(True)
        self.pending_character_candidate = character_candidate
        self.candidate_is_approved = False
        self.show_character_candidate(character_candidate)
        self.approve_candidate_button.setEnabled(True)
        self.reject_candidate_button.setEnabled(True)
        self.move_generation_workflow(
            GenerationWorkflowStage.FINAL_REVIEW,
            "최종 후보 승인 대기 - 자동 저장 0개",
        )

    def show_character_candidate(
        self,
        character_candidate: CharacterGenerationCandidate,
    ) -> None:
        """메모리에 있는 생성 후보를 GUI 미리보기로 표시한다."""
        candidate_pixmap = self.create_character_candidate_pixmap(
            character_candidate
        )
        scaled_pixmap = candidate_pixmap.scaled(
            self.candidate_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.candidate_preview.setPixmap(scaled_pixmap)

    def create_character_candidate_pixmap(
        self,
        character_candidate: CharacterGenerationCandidate,
    ) -> QPixmap:
        """메모리 후보를 파일 저장 없이 Qt 화면 이미지로 변환한다."""
        rgb_image = character_candidate.image.convert("RGB")
        image_bytes = rgb_image.tobytes("raw", "RGB")
        qt_image = QImage(
            image_bytes,
            rgb_image.width,
            rgb_image.height,
            rgb_image.width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qt_image)

    @Slot()
    def show_candidate_original_size(self) -> None:
        """저장하지 않은 후보를 100% 크기의 스크롤 화면으로 표시한다."""
        if self.pending_character_candidate is None:
            return

        original_size_dialog = QDialog(self)
        original_size_dialog.setWindowTitle(
            "생성 후보 원본 크기 "
            f"({self.pending_character_candidate.image.width}x"
            f"{self.pending_character_candidate.image.height})"
        )
        original_size_dialog.resize(820, 900)
        dialog_layout = QVBoxLayout(original_size_dialog)
        scroll_area = QScrollArea(original_size_dialog)
        original_size_image_label = QLabel()
        original_size_image_label.setPixmap(
            self.create_character_candidate_pixmap(
                self.pending_character_candidate
            )
        )
        scroll_area.setWidget(original_size_image_label)
        dialog_layout.addWidget(scroll_area)
        original_size_dialog.exec()

    @Slot()
    def approve_candidate(self) -> None:
        """현재 후보를 사용자 승인 결과로 바꾸고 저장 결정을 기다린다."""
        if self.pending_character_candidate is None:
            return
        self.candidate_is_approved = True
        self.approve_candidate_button.setEnabled(False)
        self.reject_candidate_button.setEnabled(False)
        self.save_candidate_button.setEnabled(True)
        self.discard_candidate_button.setEnabled(True)
        self.status_label.setText(
            "상태: 저장 대기 - 저장 또는 저장하지 않음을 선택하세요."
        )

    @Slot()
    def reject_candidate(self) -> None:
        """승인하지 않은 후보를 파일 저장 없이 메모리에서 제거한다."""
        if self.pending_character_candidate is None:
            return
        self.release_pending_candidate(
            "상태: 후보 거절 - 이미지 파일을 만들지 않았습니다."
        )

    @Slot()
    def save_approved_candidate(self) -> None:
        """사용자가 저장까지 승인한 후보만 PNG와 JSON으로 기록한다."""
        if (
            self.pending_character_candidate is None
            or not self.candidate_is_approved
        ):
            return

        current_dir = Path(__file__).resolve().parent
        output_name = self.config.get("paths", {}).get("output_dir", "outputs")
        output_root = current_dir / output_name
        try:
            save_result = save_approved_character_candidate(
                self.pending_character_candidate,
                output_root,
            )
        except OSError as error:
            self.status_label.setText(f"상태: 저장 실패 ({error})")
            QMessageBox.critical(self, "저장 실패", str(error))
            return

        saved_image_path = save_result.image_path
        self.release_pending_candidate(
            f"상태: 저장 완료 - {saved_image_path}"
        )
        QMessageBox.information(
            self,
            "저장 완료",
            f"승인한 이미지만 저장했습니다.\n{saved_image_path}",
        )

    @Slot()
    def discard_approved_candidate(self) -> None:
        """승인했지만 저장하지 않기로 한 후보를 메모리에서 제거한다."""
        if self.pending_character_candidate is None:
            return
        self.release_pending_candidate(
            "상태: 저장 거절 - 이미지 파일을 만들지 않았습니다."
        )

    def release_pending_candidate(self, status_message: str) -> None:
        """현재 후보의 이미지 자원을 해제하고 다음 생성을 준비한다."""
        if self.pending_character_candidate is not None:
            self.pending_character_candidate.image.close()
            if (
                self.pending_character_candidate.original_generated_image
                is not None
            ):
                self.pending_character_candidate.original_generated_image.close()
            if self.pending_character_candidate.before_clothing_image is not None:
                self.pending_character_candidate.before_clothing_image.close()
            if self.pending_character_candidate.clothing_change_mask is not None:
                self.pending_character_candidate.clothing_change_mask.close()
            if (
                self.pending_character_candidate.raw_clothing_try_on_image
                is not None
            ):
                self.pending_character_candidate.raw_clothing_try_on_image.close()
            if (
                self.pending_character_candidate.clothing_difference_image
                is not None
            ):
                self.pending_character_candidate.clothing_difference_image.close()
        self.pending_character_candidate = None
        self.candidate_is_approved = False
        self.candidate_preview.clear()
        self.candidate_preview.setText("생성 후보가 여기에 표시됩니다.")
        self.approve_candidate_button.setEnabled(False)
        self.reject_candidate_button.setEnabled(False)
        self.save_candidate_button.setEnabled(False)
        self.discard_candidate_button.setEnabled(False)
        self.open_original_size_button.setEnabled(False)
        if self.workflow_context is not None:
            self.workflow_context.move_to(GenerationWorkflowStage.COMPLETED)
        self.workflow_context = None
        self.set_workflow_input_buttons_enabled(True)
        self.generate_button.setEnabled(True)
        self.generate_button.setText("전체 이미지 생성 시작")
        self.status_label.setText(status_message)

    @Slot(str, str, object)
    def generation_failed(self, message, details, pipeline):
        self.pipeline = pipeline
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"상태: 이미지 생성 실패 ({message})")
        failed_stage = (
            self.workflow_context.current_stage
            if self.workflow_context is not None
            and self.workflow_context.current_stage
            is not GenerationWorkflowStage.FAILED
            else GenerationWorkflowStage.BASE_GENERATING
        )
        self.pause_generation_workflow(
            failed_stage,
            f"이미지 생성 실패: {message}",
        )
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("이미지 생성 실패")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    def closeEvent(self, event) -> None:
        """실행 중 작업을 보호하고 소유한 이미지와 모델 참조를 해제한다."""
        active_threads = tuple(
            thread
            for thread in (
                self.reference_worker_thread,
                self.outfit_worker_thread,
                self.mask_worker_thread,
                self.design_worker_thread,
                self.body_comparison_worker_thread,
                self.pose_estimation_worker_thread,
                self.garment_geometry_worker_thread,
                self.garment_inpaint_worker_thread,
                self.worker_thread,
            )
            if thread is not None and thread.isRunning()
        )
        if active_threads:
            QMessageBox.information(
                self,
                "작업 실행 중",
                f"실행 중인 작업 {len(active_threads)}개가 끝난 뒤 종료하세요.",
            )
            event.ignore()
            return

        if self.pending_character_candidate is not None:
            self.release_pending_candidate(
                "상태: 앱 종료 - 저장하지 않은 후보를 메모리에서 해제했습니다."
            )
        self.release_pending_clothing_base_candidate()
        self.release_approved_garment_inputs()
        self.release_clothing_mask_state()
        self.release_approved_reference_image()
        self.release_approved_pose_reference()
        self.release_approved_pose_estimation()
        self.pipeline = None
        event.accept()

    @Slot()
    def clear_worker(self):
        self.worker = None
        self.worker_thread = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenAILabWindow()
    window.show()
    sys.exit(app.exec())
