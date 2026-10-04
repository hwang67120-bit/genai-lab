"""User-facing create/review/export flow, backed by production one-pass modules."""
from pathlib import Path
import threading
import traceback
import uuid
from PySide6.QtCore import QThread, Qt, Signal, QObject
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QComboBox, QPlainTextEdit, QPushButton, QMessageBox, QFileDialog, QScrollArea)
from genai_lab.studio_generation import (StudioRuntime, StudioResults, analyze_inputs,
    build_request, generate_onepass_request, record)
from genai_lab.onepass_generation import OnePassCancelled, validate_local_models
from genai_lab.character_preferences import load_character_gender


class StudioTask(QThread):
    progress = Signal(str)

    def __init__(self, action, parent):
        super().__init__(parent)
        self.action = action
        self.stop = threading.Event()
        self.result = None
        self.error = None
        self.detail = ""

    def run(self):
        try:
            self.result = self.action(self.stop.is_set, self.progress.emit)
            if self.stop.is_set():
                raise OnePassCancelled("요청을 취소했습니다.")
        except Exception as error:
            self.error = error
            self.detail = traceback.format_exc()


def picture(path, label, width=200, height=220):
    box = QVBoxLayout()
    box.addWidget(QLabel(label))
    image = QLabel()
    image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    image.setPixmap(QPixmap(str(path)).scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))
    box.addWidget(image)
    return box


# Display labels only; original approved tag strings are never translated for generation.
NAMES = {"shirt": "셔츠", "camisole": "캐미솔", "tank top": "민소매 상의", "crop top": "짧은 상의",
    "shorts": "반바지", "pants": "바지", "skirt": "치마", "dress": "원피스", "jacket": "재킷",
    "coat": "코트", "bodysuit": "바디슈트", "gloves": "장갑", "sweater": "스웨터",
    "necktie": "넥타이", "long sleeves": "긴 소매", "short sleeves": "짧은 소매",
    "thighhighs": "허벅지 길이 양말", "elbow gloves": "팔꿈치 길이 장갑",
    "jeans": "청바지", "leggings": "레깅스", "pantyhose": "스타킹", "long skirt": "긴 치마",
    "hoodie": "후드 상의", "blazer": "블레이저", "t-shirt": "티셔츠", "sweatshirt": "맨투맨",
    "cardigan": "카디건", "suspenders": "멜빵", "vest": "조끼", "shrug": "짧은 걸침옷"}


def confirm_inputs(window, analysis):
    from genai_lab.onepass_garment_vocabulary import garment_nouns
    dialog = QDialog(window)
    dialog.setWindowTitle("만들 이미지 확인")
    dialog.resize(780, 650)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("이 캐릭터에 선택한 옷을 입힌 이미지 4장을 만듭니다. 자세는 저장 후 따로 바꿀 수 있습니다."))
    pictures = QHBoxLayout()
    directory = Path(analysis["directory"])
    for filename, title in (("character.png", "캐릭터"), ("face.png", "생성에 사용할 얼굴"), ("garment.png", "입힐 옷")):
        pictures.addLayout(picture(directory / filename, title))
    layout.addLayout(pictures)
    nouns = garment_nouns(analysis["garment_tags"])
    summary = ", ".join(NAMES.get(n.name, n.name) for n in nouns)
    layout.addWidget(QLabel("읽어낸 옷: " + (summary or "아래 상세 설명을 확인해 주세요.")))
    gender = QComboBox()
    gender.addItem("캐릭터 성별을 선택해 주세요", None)
    for label, value in (("남성", "male"), ("여성", "female"), ("지정 안 함", "unspecified")):
        gender.addItem(label, value)
    saved = load_character_gender(analysis["source"])
    if saved is not None:
        gender.setCurrentIndex(gender.findData(saved))
    layout.addWidget(gender)
    details = QCheckBox("옷 설명 자세히 보기·수정 (선택)")
    layout.addWidget(details)
    tags = QPlainTextEdit(", ".join(analysis["garment_tags"]))
    tags.setMaximumHeight(85)
    tags.hide()
    details.toggled.connect(tags.setVisible)
    layout.addWidget(tags)
    checked = QCheckBox("얼굴에 팔·몸이 남지 않았고, 읽어낸 옷이 선택한 옷과 맞습니다.")
    layout.addWidget(checked)
    buttons = QHBoxLayout()
    back = QPushButton("다른 이미지 선택")
    back.clicked.connect(dialog.reject)
    create = QPushButton("이 조건으로 만들기")
    create.setEnabled(False)
    update = lambda: create.setEnabled(checked.isChecked() and gender.currentData() is not None and bool(tags.toPlainText().strip()))
    checked.toggled.connect(update)
    gender.currentIndexChanged.connect(update)
    tags.textChanged.connect(update)
    create.clicked.connect(dialog.accept)
    buttons.addWidget(back)
    buttons.addWidget(create)
    layout.addLayout(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return gender.currentData(), tuple(t.strip() for t in tags.toPlainText().split(",") if t.strip())


def confirm_result(window, candidate, analysis):
    dialog = QDialog(window)
    dialog.setWindowTitle("저장할 결과 확인")
    dialog.resize(800, 670)
    layout = QVBoxLayout(dialog)
    row = QHBoxLayout()
    for path, name in ((Path(analysis["directory"])/"character.png", "원래 캐릭터"),
                       (Path(analysis["directory"])/"garment.png", "선택한 옷"), (candidate.path, "저장할 결과")):
        row.addLayout(picture(path, name, 235, 380))
    layout.addLayout(row)
    checks = {}
    for key, text in (("character", "원하는 캐릭터로 보입니다."), ("garment", "옷의 모양과 색이 마음에 듭니다."),
                      ("exposure", "원치 않는 노출이나 잘못 그려진 부분이 없습니다.")):
        checks[key] = QCheckBox(text)
        layout.addWidget(checks[key])
    accept = QPushButton("이 결과 사용")
    accept.setEnabled(False)
    for check in checks.values():
        check.toggled.connect(lambda: accept.setEnabled(all(c.isChecked() for c in checks.values())))
    accept.clicked.connect(dialog.accept)
    back = QPushButton("다시 비교하기")
    back.clicked.connect(dialog.reject)
    layout.addWidget(accept)
    layout.addWidget(back)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return {key: check.isChecked() for key, check in checks.items()}
    return None


class StudioController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.task = None
        self.analysis = None
        self.results = None
        self.runtime = None
        self.run_directory = None
        self.confirming = False
        self.pending_action = None
        self.last_saved = None
        self.output_root = Path(__file__).resolve().parents[1] / "outputs" / "studio-runs"
        window.studio_candidate_combo.currentIndexChanged.connect(self.select)
        window.studio_cancel_button.clicked.connect(self.cancel)

    @property
    def occupied(self):
        return self.task is not None or self.confirming or self.results is not None

    def controls(self):
        w = self.window
        active = self.task is not None or self.confirming
        for control in (w.style_button, w.outfit_button, w.clear_outfit_button,
                        w.qwen_pose_button, w.external_candidate_button):
            control.setEnabled(not self.occupied)
        w.generate_button.setEnabled(not self.occupied and bool(w.style_path and w.selected_outfit_path))
        w.studio_cancel_button.setVisible(self.task is not None)
        w.studio_candidate_combo.setVisible(self.results is not None)
        w.studio_candidate_combo.setEnabled(not active)
        ready = self.results is not None and not active
        w.approve_candidate_button.setEnabled(ready)
        w.reject_candidate_button.setEnabled(ready)
        w.discard_candidate_button.setEnabled(ready)
        w.save_candidate_button.setEnabled(ready and self.results.status == "user_approved")
        w.open_original_size_button.setEnabled(ready)

    def launch(self, action, complete):
        if self.task is not None:
            raise RuntimeError("이미 실행 중입니다.")
        task = StudioTask(action, self.window)
        self.task = task
        # Existing application close/GPU-exclusion guards also see this task.
        self.window.worker_thread = task
        task.progress.connect(self.window.status_label.setText)
        self.pending_action = complete
        task.finished.connect(self.finished)
        self.controls()
        task.start()

    def finished(self):
        task = self.task
        complete = self.pending_action
        self.task = None
        self.window.worker_thread = None
        self.pending_action = None
        try:
            if task.error:
                self.fail(task.error, task.detail)
            else:
                complete(task.result)
        except Exception as error:
            self.fail(error, traceback.format_exc())
        finally:
            task.deleteLater()
            self.controls()

    def fail(self, error, details=""):
        cancelled = isinstance(error, OnePassCancelled)
        self.results = None  # Partial raw output stays on disk, never offered as a completed batch.
        self.window.candidate_preview.clear()
        self.window.candidate_preview.setText("취소되었습니다." if cancelled else "완성된 결과가 없습니다.")
        self.window.status_label.setText("상태: 취소됨" if cancelled else "상태: 생성하지 못했습니다. 입력과 실행 정보를 확인해 주세요.")
        if self.run_directory:
            self.run_directory.mkdir(parents=True, exist_ok=True)
            record(self.run_directory / "gui-status.json", {"status": "cancelled" if cancelled else "failed",
                   "error": str(error), "detail": details})
        if not cancelled:
            box = QMessageBox(self.window)
            box.setWindowTitle("작업을 마치지 못했습니다")
            box.setText(str(error))
            box.setDetailedText(details)
            box.exec()

    def start(self):
        w = self.window
        if self.occupied or not w.can_start_registered_generation():
            return
        try:
            # A failed new preflight must not overwrite a previous run's status.
            self.run_directory = self.output_root / uuid.uuid4().hex
            self.run_directory.mkdir(parents=True)
            self.runtime = StudioRuntime.from_environment()
            validate_local_models(self.runtime.generation)
            w.candidate_preview.clear()
            w.candidate_preview.setText("선택한 캐릭터와 옷을 준비하고 있습니다.")
            w.status_label.setText("상태: 캐릭터와 옷 확인 중")
            character, garment = Path(w.style_path), Path(w.selected_outfit_path)
            self.launch(lambda cancel, progress: analyze_inputs(character, garment,
                self.run_directory / "inputs", self.runtime, cancelled=cancel), self.inputs_ready)
        except Exception as error:
            self.fail(error, traceback.format_exc())
            self.controls()

    def inputs_ready(self, analysis):
        self.analysis = analysis
        self.confirming = True
        self.controls()
        try:
            decision = confirm_inputs(self.window, analysis)
        finally:
            self.confirming = False
        if decision is None:
            self.window.status_label.setText("상태: 입력 확인을 취소했습니다. 다른 이미지를 선택할 수 있습니다.")
            self.window.candidate_preview.clear()
            self.window.candidate_preview.setText("아직 생성하지 않았습니다.")
            return
        gender, tags = decision
        # Preferences access stays on GUI thread. Tokenizers only read local files.
        request = build_request(analysis, tags, gender, confirmed=True, runtime=self.runtime)
        self.window.status_label.setText("상태: 이미지 만드는 중 · 0/4")
        def generate(cancel, progress):
            count = [0]
            def on_image(candidate):
                count[0] += 1
                progress(f"상태: 이미지 만드는 중 · {count[0]}/4 · 아직 결과 확인 전")
            return generate_onepass_request(request, self.run_directory / "generation",
                                           cancelled=cancel, on_image=on_image)
        self.launch(generate, self.generated)

    def generated(self, batch):
        self.results = StudioResults(batch)
        combo = self.window.studio_candidate_combo
        combo.blockSignals(True)
        combo.clear()
        for i in range(len(batch.candidates)):
            combo.addItem(f"결과 {i+1}")
        combo.blockSignals(False)
        self.select(0)
        self.window.status_label.setText("상태: 결과 4장 준비됨 · 마음에 드는 결과를 골라 확인해 주세요.")

    def select(self, index):
        if self.results is None or index < 0:
            return
        self.results.select(index)
        self.window.candidate_preview.setPixmap(QPixmap(str(self.results.candidate.path)))
        self.window.status_label.setText(f"상태: 결과 {index+1} 확인 전 · 다른 결과와 비교할 수 있습니다.")
        self.controls()

    def approve(self):
        if self.results is None:
            return
        checks = confirm_result(self.window, self.results.candidate, self.analysis)
        if checks is not None:
            try:
                self.results.approve(checks)
                self.window.status_label.setText("상태: 결과 확인 완료 · 저장 위치를 선택해 주세요.")
            except Exception as error:
                QMessageBox.warning(self.window, "확인 필요", str(error))
        self.controls()

    def save(self):
        if self.results is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.window, "이미지 저장", "character.png", "PNG (*.png)")
        if not path:
            return
        try:
            self.last_saved = self.results.export(path)
            self.window.status_label.setText(f"상태: 저장 완료 · {path}")
            self.results = None
            self.controls()
        except Exception as error:
            QMessageBox.warning(self.window, "저장하지 못했습니다", str(error))

    def discard(self):
        if self.results is not None:
            self.results.status = "discarded"
            self.results.persist()
        self.results = None
        self.window.candidate_preview.clear()
        self.window.candidate_preview.setText("결과를 저장하지 않았습니다.")
        self.window.status_label.setText("상태: 새 이미지를 만들 수 있습니다.")
        self.controls()

    def reject(self):
        if self.results is not None:
            self.results.select(self.results.selected)
            self.window.status_label.setText("상태: 이 결과를 승인하지 않았습니다. 다른 결과를 선택하거나 다시 만드세요.")
        self.controls()

    def cancel(self):
        if self.task:
            self.task.stop.set()
            self.window.status_label.setText("상태: 취소 중 · 현재 단계를 안전하게 마칠 때까지 기다려 주세요.")

    def original(self):
        if self.results is None:
            return
        dialog = QDialog(self.window)
        dialog.setWindowTitle("결과 원본 크기")
        dialog.resize(850, 800)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        image = QLabel()
        image.setPixmap(QPixmap(str(self.results.candidate.path)))
        scroll.setWidget(image)
        layout.addWidget(scroll)
        dialog.exec()
