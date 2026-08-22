import sys
import os
import secrets
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox
)

from run import (
    PromptItem,
    check_environment,
    config_fingerprint,
    configure_system_certificates,
    load_yaml,
    validate_config,
)
from genai_lab.model import prepare_pipeline
from genai_lab.generator import generate_images
from genai_lab.result import create_new_run_directory, initial_result, write_json


# 화면 범위마다 캔버스 비율, 반드시 넣을 표현, 피할 표현을 한곳에서 관리한다.
FRAMING_PRESETS = {
    "full_body": {
        "label": "전신 (머리부터 발끝까지)",
        "width": 576,
        "height": 896,
        "prompt": (
            "full body, standing, head to toe, feet visible, entire character in frame, "
            "centered composition, long shot"
        ),
        "negative": (
            "cropped, out of frame, close-up, upper body, cowboy shot, "
            "feet out of frame, head out of frame"
        ),
    },
    "upper_body": {
        "label": "상반신 (허리 위)",
        "width": 768,
        "height": 768,
        "prompt": "upper body, waist up, centered composition, face visible",
        "negative": "full body, close-up, cropped head, out of frame",
    },
    "face": {
        "label": "얼굴 중심 (어깨 위)",
        "width": 768,
        "height": 768,
        "prompt": "portrait, close-up, face focus, head and shoulders, centered composition",
        "negative": "full body, upper body, wide shot, cropped face, out of frame",
    },
}


def apply_framing_preset(config, framing_key):
    """선택한 화면 범위를 생성 크기와 문장에 함께 반영한다."""
    if framing_key not in FRAMING_PRESETS:
        raise ValueError(f"지원하지 않는 화면 범위입니다: {framing_key}")

    preset = FRAMING_PRESETS[framing_key]
    config["generation"]["width"] = preset["width"]
    config["generation"]["height"] = preset["height"]
    base_negative = config["generation"].get("default_negative_prompt", "")
    negative_prompt = ", ".join(
        part for part in (base_negative, preset["negative"]) if part
    )
    return preset["prompt"], negative_prompt


class GenerationWorker(QObject):
    """모델 로딩과 이미지 생성을 화면 실행 흐름 밖에서 처리한다."""

    status_changed = Signal(str)
    completed = Signal(str, object)
    failed = Signal(str, str, object)

    def __init__(self, config, prompts, current_dir, pipeline=None):
        super().__init__()
        self.config = config
        self.prompts = prompts
        self.current_dir = current_dir
        self.pipeline = pipeline

    @Slot()
    def run(self):
        run_directory = None
        result = None
        try:
            configure_system_certificates()
            environment = check_environment()
            output_name = self.config.get("paths", {}).get("output_dir", "outputs")
            output_root = self.current_dir / output_name
            run_directory = create_new_run_directory(output_root)
            fingerprint = config_fingerprint(self.config, self.prompts)
            result = initial_result(
                self.config, self.prompts, environment, fingerprint
            )
            write_json(run_directory / "result.json", result)

            if self.pipeline is None:
                self.status_changed.emit("모델과 참조 그림 장치 준비 중...")
                self.pipeline = prepare_pipeline(self.config)

            self.status_changed.emit("이미지 생성 중...")
            generate_images(
                self.pipeline,
                self.config,
                self.prompts,
                run_directory,
                result,
                self.current_dir,
            )
            self.completed.emit(str(run_directory), self.pipeline)
        except Exception as error:
            details = traceback.format_exc()
            if result is not None and run_directory is not None:
                result["status"] = "failed"
                result["error"] = str(error)
                write_json(run_directory / "result.json", result)
            self.failed.emit(str(error), details, self.pipeline)


class GenAILabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenAI Lab - 캐릭터 후보 이미지 생성기")
        self.resize(800, 600)

        # 백엔드 파이프라인 및 설정 저장 변수
        self.pipeline = None
        self.config = None
        self.worker = None
        self.worker_thread = None

        # 현재 구현에서 실제 사용하는 입력은 캐릭터 기준 이미지 하나다.
        self.style_path = None

        user_profile = os.environ.get('USERPROFILE', '')
        self.default_get_dir = os.path.join(user_profile, "Downloads") if user_profile else ""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 1. 캐릭터 디자인 분위기 선택 UI
        style_layout = QHBoxLayout()
        self.style_label = QLabel("1. 캐릭터 디자인 분위기: 선택되지 않음")
        style_button = QPushButton("이미지 선택")
        style_button.clicked.connect(lambda: self.select_image("style"))
        style_layout.addWidget(self.style_label)
        style_layout.addWidget(style_button)
        layout.addLayout(style_layout)

        # 의상과 자세는 실제 처리 기능이 추가되기 전까지 선택할 수 없다.
        outfit_layout = QHBoxLayout()
        self.outfit_label = QLabel("2. 의상 참조: 추후 지원")
        outfit_button = QPushButton("아직 사용할 수 없음")
        outfit_button.setEnabled(False)
        outfit_layout.addWidget(self.outfit_label)
        outfit_layout.addWidget(outfit_button)
        layout.addLayout(outfit_layout)

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
        for key, preset in FRAMING_PRESETS.items():
            self.framing_combo.addItem(preset["label"], key)
        self.framing_combo.setCurrentIndex(0)
        framing_layout.addWidget(framing_label)
        framing_layout.addWidget(self.framing_combo)
        layout.addLayout(framing_layout)

        self.framing_help = QLabel(
            "전신은 세로 화면과 머리·발끝 포함 문구를 함께 사용합니다. "
            "생성형 모델 특성상 결과 확인은 필요합니다."
        )
        self.framing_help.setWordWrap(True)
        layout.addWidget(self.framing_help)

        self.generate_button = QPushButton("후보 이미지 생성 시작")
        self.generate_button.clicked.connect(self.start_generation)
        layout.addWidget(self.generate_button)

        self.status_label = QLabel("상태: 캐릭터 기준 이미지를 선택해 주세요")
        layout.addWidget(self.status_label)

    def select_image(self, target_type):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "참조 이미지 선택", self.default_get_dir, "이미지 파일 (*.png *.jpg *.jpeg)"
        )
        if file_path:
            file_name = os.path.basename(file_path)
            if target_type == "style":
                self.style_path = file_path
                self.style_label.setText(f"1. 캐릭터 디자인 분위기: {file_name}")
                self.status_label.setText("상태: 생성 준비 완료")

    def start_generation(self):
        if not self.style_path:
            QMessageBox.warning(self, "경고", "캐릭터 기준 이미지를 선택하세요.")
            return

        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.information(self, "안내", "이미지를 생성하고 있습니다.")
            return

        try:
            current_dir = Path(__file__).resolve().parent
            config_path = current_dir / "configs" / "animagine.yaml"
            self.config = load_yaml(config_path)
            self.config["style"]["enabled"] = True
            self.config["style"]["reference_image"] = self.style_path
            framing_key = self.framing_combo.currentData()
            framing_prompt, framing_negative = apply_framing_preset(
                self.config, framing_key
            )
            validate_config(self.config)

            prompts = [
                PromptItem(
                    request_id="gui_char_01",
                    description_ko="GUI 캐릭터 생성",
                    prompt=f"1girl, solo, masterpiece, best quality, {framing_prompt}, white background, simple background",
                    negative_prompt=framing_negative,
                    seed=secrets.randbelow(2**31),
                )
            ]

            self.generate_button.setEnabled(False)
            self.status_label.setText("상태: 생성 작업 준비 중...")

            self.worker_thread = QThread(self)
            self.worker = GenerationWorker(
                self.config, prompts, current_dir, self.pipeline
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
        except Exception as error:
            self.generate_button.setEnabled(True)
            self.status_label.setText(f"상태: 설정 오류 ({error})")
            QMessageBox.critical(self, "설정 오류", str(error))

    @Slot(str)
    def show_worker_status(self, message):
        self.status_label.setText(f"상태: {message}")

    @Slot(str, object)
    def generation_completed(self, run_directory, pipeline):
        self.pipeline = pipeline
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"상태: 생성 완료! 저장 경로: {run_directory}")
        os.startfile(str(Path(run_directory) / "images"))

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