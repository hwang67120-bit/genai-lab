import sys
import os
import traceback
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox,
    QDialog, QScrollArea,
)

from run import (
    check_environment,
    configure_system_certificates,
    load_yaml,
    validate_config,
)
from genai_lab.model import prepare_pipeline
from genai_lab.generator import generate_character_candidate
from genai_lab.clothing import (
    CatVTONLocalSettings,
    ClothingCategory,
    ClothingReferenceInput,
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

from genai_lab.result import (
    CharacterGenerationCandidate,
    save_approved_character_candidate,
)
from genai_lab.run_log import (
    GenerationRunLog,
    create_generation_run_log,
    find_recovery_action,
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
    image_bytes = rgb_image.tobytes("raw", "RGB")
    qt_image = QImage(
        image_bytes,
        rgb_image.width,
        rgb_image.height,
        rgb_image.width * 3,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qt_image)


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
        self.resize(1100, 720)
        layout = QVBoxLayout(self)
        summary = QLabel(
            f"의상 참조={character_candidate.clothing_reference_name}, "
            f"종류={character_candidate.clothing_category}\n"
            "마스크의 흰색 영역 안에서만 의상 합성 결과를 사용했습니다."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        if character_candidate.clothing_verification_warning_ko:
            warning = QLabel(
                character_candidate.clothing_verification_warning_ko
            )
            warning.setWordWrap(True)
            layout.addWidget(warning)

        comparison_layout = QHBoxLayout()
        comparison_images = (
            ("의상 적용 전", character_candidate.before_clothing_image),
            ("의상 변경 허용 영역", character_candidate.clothing_change_mask),
            ("의상 적용 후", character_candidate.image),
        )
        for title, image in comparison_images:
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setPixmap(
                create_pil_image_pixmap(image).scaled(
                    340,
                    520,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            column.addWidget(image_label)
            comparison_layout.addLayout(column)
        layout.addLayout(comparison_layout)

        button_layout = QHBoxLayout()
        clothing_button = QPushButton("의상 적용본 사용")
        original_button = QPushButton("의상 적용 전 후보 사용")
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
            if self.pipeline is None:
                model_started_at = perf_counter()
                self.status_changed.emit("모델과 참조 그림 장치 준비 중...")
                self.run_log.write_stage("모델 준비", "모델 불러오기 시작")
                self.pipeline = prepare_pipeline(self.config)
                self.run_log.write_stage(
                    "모델 준비",
                    f"완료, 소요 시간={perf_counter() - model_started_at:.1f}초",
                )
            else:
                self.run_log.write_stage("모델 준비", "메모리에 있는 모델 재사용")

            generation_started_at = perf_counter()
            self.status_changed.emit("이미지 생성 중...")
            self.run_log.write_stage("이미지 생성", "후보 1번 생성 시작")
            character_candidate = generate_character_candidate(
                self.pipeline,
                self.config,
                self.generation_request,
                self.current_dir,
                self.run_log,
                clothing_reference_input=self.clothing_reference_input,
                catvton_settings=self.catvton_settings,
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
        self.reference_worker = None
        self.reference_worker_thread = None
        self.approved_reference_image: ApprovedReferenceImage | None = None
        self.pending_character_candidate: CharacterGenerationCandidate | None = None
        self.candidate_is_approved = False

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
        style_button = QPushButton("이미지 선택")
        style_button.clicked.connect(lambda: self.select_image("style"))
        style_layout.addWidget(self.style_label)
        style_layout.addWidget(style_button)
        layout.addLayout(style_layout)

        # 의상 참조는 CatVTON 별도 환경으로 전달하고 보호 마스크로 제한한다.
        outfit_layout = QHBoxLayout()
        self.outfit_label = QLabel("2. 의상 참조: 선택하지 않음")
        outfit_button = QPushButton("의상 이미지 선택")
        outfit_button.clicked.connect(lambda: self.select_image("outfit"))
        clear_outfit_button = QPushButton("의상 선택 해제")
        clear_outfit_button.clicked.connect(self.clear_outfit_reference)
        outfit_layout.addWidget(clear_outfit_button)
        outfit_layout.addWidget(self.outfit_label)
        outfit_layout.addWidget(outfit_button)
        layout.addLayout(outfit_layout)
        clothing_type_layout = QHBoxLayout()
        clothing_type_layout.addWidget(QLabel("의상 종류:"))
        self.clothing_category_combo = QComboBox()
        for clothing_category, label in CLOTHING_OPTIONS:
            self.clothing_category_combo.addItem(label, clothing_category.value)
        clothing_type_layout.addWidget(self.clothing_category_combo)
        layout.addLayout(clothing_type_layout)

        pose_layout = QHBoxLayout()
        self.pose_label = QLabel("3. 자세 참조(ControlNet): 추후 지원")
        pose_button = QPushButton("아직 사용할 수 없음")
        pose_button.setEnabled(False)
        pose_layout.addWidget(self.pose_label)
        pose_layout.addWidget(pose_button)
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
        self.generate_button = QPushButton("후보 이미지 생성 시작")
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
                self.style_label.setText(f"1. 원본 캐릭터 기준 이미지: {file_name}")
                self.start_reference_preparation(Path(file_path))

            elif target_type == "outfit":
                self.outfit_path = file_path
                self.outfit_label.setText(f"2. 의상 참조: {file_name}")
                self.status_label.setText("상태: 의상 참조 선택 완료")

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
        self.reference_worker_thread.start()

    @Slot()
    def clear_outfit_reference(self) -> None:
        """선택 의상 입력만 해제하고 캐릭터 기준 이미지는 유지한다."""
        if self.pending_character_candidate is not None:
            QMessageBox.information(self, "안내", "현재 후보를 먼저 판단하세요.")
            return
        self.outfit_path = None
        self.outfit_label.setText("2. 의상 참조: 선택하지 않음")
        self.status_label.setText("상태: 의상 참조 선택 해제")

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
            self.generate_button.setEnabled(True)
            self.status_label.setText(
                "상태: 원본 참조 이미지 사용 가능 - 생성 준비 완료"
            )
            return

        comparison_dialog = ReferenceEnhancementDialog(
            enhancement_candidate,
            self,
        )
        comparison_dialog.exec()
        if comparison_dialog.use_enhanced_image:
            self.approved_reference_image = approve_enhanced_reference_image(
                enhancement_candidate,
                source_name,
            )
            self.generate_button.setEnabled(True)
            self.status_label.setText(
                "상태: 보정 참조 이미지 승인 - 생성 준비 완료"
            )
        else:
            self.approved_reference_image = None
            self.generate_button.setEnabled(False)
            self.status_label.setText(
                "상태: 보정 이미지 거절 - 다른 참조 이미지를 선택하세요"
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

    @Slot()
    def clear_reference_worker(self) -> None:
        """종료된 참조 이미지 준비 작업 객체를 해제한다."""
        self.reference_worker = None
        self.reference_worker_thread = None

    def start_generation(self):
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
            if self.outfit_path:
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
                )
                run_log.write_stage(
                    "의상 참조 입력",
                    (
                        f"파일={Path(self.outfit_path).name}, "
                        f"종류={clothing_category.value}, "
                        "CatVTON 별도 환경 사용"
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
                    f"모델={generation_request.model_id}"
                ),
            )

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
            QMessageBox.critical(self, "설정 오류", error_message)

    @Slot(str)
    def show_worker_status(self, message):
        self.status_label.setText(f"상태: {message}")

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
            dialog_result = comparison_dialog.exec()
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
        if character_candidate.before_clothing_image is not None:
            clothing_dialog = ClothingTryOnComparisonDialog(
                character_candidate,
                self,
            )
            dialog_result = clothing_dialog.exec()
            use_clothing_image = (
                dialog_result == QDialog.DialogCode.Accepted
                and clothing_dialog.use_clothing_image
            )
            if use_clothing_image:
                character_candidate.before_clothing_image.close()
                if character_candidate.clothing_change_mask is not None:
                    character_candidate.clothing_change_mask.close()
                character_candidate = replace(
                    character_candidate,
                    before_clothing_image=None,
                    clothing_change_mask=None,
                )
            else:
                character_candidate.image.close()
                if character_candidate.clothing_change_mask is not None:
                    character_candidate.clothing_change_mask.close()
                character_candidate = replace(
                    character_candidate,
                    image=character_candidate.before_clothing_image,
                    before_clothing_image=None,
                    clothing_change_mask=None,
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
        self.status_label.setText(
            "상태: 후보 검토 대기 - 아직 파일로 저장되지 않았습니다."
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
        self.pending_character_candidate = None
        self.candidate_is_approved = False
        self.candidate_preview.clear()
        self.candidate_preview.setText("생성 후보가 여기에 표시됩니다.")
        self.approve_candidate_button.setEnabled(False)
        self.reject_candidate_button.setEnabled(False)
        self.save_candidate_button.setEnabled(False)
        self.discard_candidate_button.setEnabled(False)
        self.open_original_size_button.setEnabled(False)
        self.generate_button.setEnabled(True)
        self.status_label.setText(status_message)

    @Slot(str, str, object)
    def generation_failed(self, message, details, pipeline):
        self.pipeline = pipeline
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"상태: 이미지 생성 실패 ({message})")
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("이미지 생성 실패")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec()

    @Slot()
    def clear_worker(self):
        self.worker = None
        self.worker_thread = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenAILabWindow()
    window.show()
    sys.exit(app.exec())