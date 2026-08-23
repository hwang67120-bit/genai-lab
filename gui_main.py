import sys
import os
import traceback
from pathlib import Path
from time import perf_counter

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
from genai_lab.request import (
    CharacterFramingType,
    CharacterGenerationInput,
    CharacterGenerationSettings,
    prepare_character_generation_request,
)
from genai_lab.result import create_new_run_directory, initial_result, write_json
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


class GenerationWorker(QObject):
    """모델 로딩과 이미지 생성을 화면 실행 흐름 밖에서 처리한다."""

    status_changed = Signal(str)
    completed = Signal(str, object)
    failed = Signal(str, str, object)

    def __init__(
        self,
        config,
        prompts,
        current_dir,
        run_log: GenerationRunLog,
        pipeline=None,
    ):
        super().__init__()
        self.config = config
        self.prompts = prompts
        self.current_dir = current_dir
        self.run_log = run_log
        self.pipeline = pipeline

    @Slot()
    def run(self):
        execution_started_at = perf_counter()
        run_directory = None
        result = None
        try:
            self.run_log.write_stage("환경 검사", "GPU와 필수 도구 확인 시작")
            configure_system_certificates()
            environment = check_environment()
            gpu_memory_gb = environment["vram_bytes"] / (1024**3)
            self.run_log.write_stage(
                "환경 검사",
                f"GPU={environment['gpu']}, GPU 메모리={gpu_memory_gb:.1f}GB",
            )
            output_name = self.config.get("paths", {}).get("output_dir", "outputs")
            output_root = self.current_dir / output_name
            run_directory = create_new_run_directory(output_root)
            fingerprint = config_fingerprint(self.config, self.prompts)
            result = initial_result(
                self.config, self.prompts, environment, fingerprint
            )
            write_json(run_directory / "result.json", result)

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
            generate_images(
                self.pipeline,
                self.config,
                self.prompts,
                run_directory,
                result,
                self.current_dir,
                self.run_log,
            )
            self.run_log.write_stage(
                "이미지 생성",
                f"완료, 소요 시간={perf_counter() - generation_started_at:.1f}초",
            )
            self.run_log.write_stage(
                "실행 완료",
                (
                    f"전체 소요 시간={perf_counter() - execution_started_at:.1f}초, "
                    f"결과 폴더={run_directory}"
                ),
            )
            self.completed.emit(str(run_directory), self.pipeline)
        except Exception as error:
            details = traceback.format_exc()
            self.run_log.write_failure(
                "이미지 생성 실행",
                error,
                find_recovery_action(error),
            )
            if result is not None and run_directory is not None:
                result["status"] = "failed"
                result["error"] = str(error)
                write_json(run_directory / "result.json", result)
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

        # 1. 원본 캐릭터 기준 이미지 선택 UI
        style_layout = QHBoxLayout()
        self.style_label = QLabel("1. 원본 캐릭터 기준 이미지: 선택되지 않음")
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
        for framing_type, label in FRAMING_OPTIONS:
            self.framing_combo.addItem(label, framing_type.value)
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
                self.style_label.setText(f"1. 원본 캐릭터 기준 이미지: {file_name}")
                self.status_label.setText("상태: 생성 준비 완료")

    def start_generation(self):
        if not self.style_path:
            QMessageBox.warning(self, "경고", "캐릭터 기준 이미지를 선택하세요.")
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
            generation_input = CharacterGenerationInput(
                reference_image_path=Path(self.style_path),
                framing_type=CharacterFramingType(
                    self.framing_combo.currentData()
                ),
            )
            generation_request = prepare_character_generation_request(
                generation_input,
                generation_settings,
                candidate_number=1,
            )
            run_log.write_stage(
                "요청 준비",
                (
                    f"화면 범위={generation_request.framing_type.value}, "
                    f"원본 크기={generation_request.reference_image.width}x"
                    f"{generation_request.reference_image.height}, "
                    f"크기={generation_request.width}x{generation_request.height}, "
                    f"시드={generation_request.seed}, "
                    f"참조 이미지 반영 강도="
                    f"{generation_request.reference_image_strength:.2f}, "
                    f"모델={generation_request.model_id}"
                ),
            )

            # 기존 모델 실행 함수가 아직 config와 PromptItem을 받으므로
            # 확정 요청을 현재 실행 형식으로 옮긴다.
            generation_config["width"] = generation_request.width
            generation_config["height"] = generation_request.height
            validate_config(self.config)
            prompts = [
                PromptItem(
                    request_id=(
                        f"gui_char_{generation_request.candidate_number:02d}"
                    ),
                    description_ko="GUI 캐릭터 생성",
                    prompt=generation_request.prompt,
                    negative_prompt=generation_request.negative_prompt,
                    seed=generation_request.seed,
                )
            ]

            self.generate_button.setEnabled(False)
            self.status_label.setText("상태: 생성 작업 준비 중...")

            self.worker_thread = QThread(self)
            self.worker = GenerationWorker(
                self.config,
                prompts,
                current_dir,
                run_log,
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