"""작업실 이미지 생성의 읽기 시작점.

gui_main.GenAILabWindow.start_generation → StudioController.start
주 흐름: start → inputs_ready → generated → select → approve → save.
분석·생성은 launch로 작업 스레드에 맡기고 finished가 GUI에서 다음 단계로 연결한다.
사용자 확인에서 취소하면 생성하지 않고, 실패하면 완성 후보를 제공하지 않는다.

이 파일은 주 흐름 → 실행 보조 → 확인창 순서로 읽는다.
모델·태그 계산의 세부 구현은 studio_generation, 전체 지도는 docs/CODE_FLOW.md.
"""

from pathlib import Path
import threading
import traceback
import uuid
from PySide6.QtCore import QThread, Qt, Signal, QObject
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QComboBox, QPlainTextEdit, QLineEdit, QWidget, QPushButton, QMessageBox, QFileDialog, QScrollArea)
from genai_lab.studio_generation import (StudioRuntime, StudioResults, analyze_inputs,
    build_request, generate_onepass_request, record)
from genai_lab.onepass_generation import OnePassCancelled, validate_local_models
from genai_lab.character_preferences import load_character_gender
from genai_lab.onepass_prompt import AppearanceOverrides, PartAppearance, appendage_tags

class StudioController(QObject):
    """한 요청의 입력 확인부터 승인·저장까지 연결한다. 자동 승인은 하지 않는다."""

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

    # 1. 입력 준비: CPU 분석 완료 후 inputs_ready로 이어진다.

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

    # 2. 사용자 확인: 승인한 조건만 생성 요청으로 만들고, 완료 후 generated로 이어진다.

    def inputs_ready(self, analysis):
        self.analysis = analysis
        self.confirming = True
        self.controls()
        try:
            decision = confirm_inputs(self.window, analysis,
                self.runtime.prompt.enable_ear_override if self.runtime else False)
        finally:
            self.confirming = False
        if decision is None:
            self.window.status_label.setText("상태: 입력 확인을 취소했습니다. 다른 이미지를 선택할 수 있습니다.")
            self.window.candidate_preview.clear()
            self.window.candidate_preview.setText("아직 생성하지 않았습니다.")
            return
        gender, tags, appearance = decision
        # Preferences access stays on GUI thread. Tokenizers only read local files.
        request = build_request(analysis, tags, gender, confirmed=True, runtime=self.runtime, appearance=appearance)
        self.window.status_label.setText("상태: 이미지 만드는 중 · 0/4")
        def generate(cancel, progress):
            count = [0]
            def on_image(candidate):
                count[0] += 1
                progress(f"상태: 이미지 만드는 중 · {count[0]}/4 · 아직 결과 확인 전")
            return generate_onepass_request(request, self.run_directory / "generation",
                                           cancelled=cancel, on_image=on_image)
        self.launch(generate, self.generated)

    # 3. 생성 완료: 원시 결과를 검토 대기로 표시한다. 아직 승인·저장 상태가 아니다.

    def generated(self, batch):
        self.results = StudioResults(batch, appendage_review_required=bool(
            self.analysis and any(appendage_tags(self.analysis["groups"]["fixed"]).values())))
        combo = self.window.studio_candidate_combo
        combo.blockSignals(True)
        combo.clear()
        for i in range(len(batch.candidates)):
            combo.addItem(f"결과 {i+1}")
        combo.blockSignals(False)
        self.select(0)
        self.window.status_label.setText("상태: 결과 4장 준비됨 · 마음에 드는 결과를 골라 확인해 주세요.")

    # 4. 결과 선택: 후보를 바꾸면 기존 승인은 해제된다.

    def select(self, index):
        if self.results is None or index < 0:
            return
        self.results.select(index)
        self.window.candidate_preview.setPixmap(QPixmap(str(self.results.candidate.path)))
        self.window.status_label.setText(f"상태: 결과 {index+1} 확인 전 · 다른 결과와 비교할 수 있습니다.")
        self.controls()

    # 5. 사용자 승인: 캐릭터·의상·노출 확인을 저장한다.

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

    # 6. 내보내기: 승인한 이미지와 동일한 파일만 지정 위치에 저장한다.

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

    # 다른 선택: 재검토·폐기·취소·원본 보기는 자동 재생성을 요청하지 않는다.

    def reject(self):
        if self.results is not None:
            self.results.select(self.results.selected)
            self.window.status_label.setText("상태: 이 결과를 승인하지 않았습니다. 다른 결과를 선택하거나 다시 만드세요.")
        self.controls()

    def discard(self):
        if self.results is not None:
            self.results.status = "discarded"
            self.results.persist()
        self.results = None
        self.window.candidate_preview.clear()
        self.window.candidate_preview.setText("결과를 저장하지 않았습니다.")
        self.window.status_label.setText("상태: 새 이미지를 만들 수 있습니다.")
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

    # 실행 보조: 아래 코드는 버튼 상태와 비동기 실행·실패 처리를 담당한다.

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

    # 작업 스레드 종료 신호가 도착하면 GUI 스레드에서 다음 단계 함수를 호출한다.

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

# 작업 실행기: 화면을 직접 바꾸지 않고 결과·오류·진행 신호를 돌려준다.

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

# 확인창의 배치 세부사항: 주 흐름에서는 confirm_inputs / confirm_result만 읽으면 된다.

def picture(path, label, width=200, height=220):
    box = QVBoxLayout()
    box.addWidget(QLabel(label))
    image = QLabel()
    image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    image.setPixmap(QPixmap(str(path)).scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))
    box.addWidget(image)
    return box

# 화면 표시용 번역. 생성에는 승인된 원문 태그를 그대로 전달한다.

NAMES = {"shirt": "셔츠", "camisole": "캐미솔", "tank top": "민소매 상의", "crop top": "짧은 상의",
    "shorts": "반바지", "pants": "바지", "skirt": "치마", "dress": "원피스", "jacket": "재킷",
    "coat": "코트", "bodysuit": "바디슈트", "gloves": "장갑", "sweater": "스웨터",
    "necktie": "넥타이", "long sleeves": "긴 소매", "short sleeves": "짧은 소매",
    "thighhighs": "허벅지 길이 양말", "elbow gloves": "팔꿈치 길이 장갑",
    "jeans": "청바지", "leggings": "레깅스", "pantyhose": "스타킹", "long skirt": "긴 치마",
    "hoodie": "후드 상의", "blazer": "블레이저", "t-shirt": "티셔츠", "sweatshirt": "맨투맨",
    "cardigan": "카디건", "suspenders": "멜빵", "vest": "조끼", "shrug": "짧은 걸침옷"}

def appearance_field(layout, part, detected, enabled):
    """One optional user description; changing text invalidates its confirmation."""
    title = "꼬리" if part == "tail" else "귀"
    label = QLabel(f"{title} · 자동으로 읽은 내용: {', '.join(detected)}")
    label.setWordWrap(True)
    layout.addWidget(label)
    text = QLineEdit()
    text.setObjectName(f"{part}_appearance_text")
    text.setPlaceholderText("예: light blue tail, striped tail" if part == "tail" else "영어 외형 문구 (현재 적용 안 함)")
    text.setEnabled(enabled)
    layout.addWidget(text)
    confirmed = QCheckBox(f"이 {title} 설명을 확인했고 생성에 적용합니다.")
    confirmed.setObjectName(f"{part}_appearance_confirmed")
    confirmed.setEnabled(False)
    layout.addWidget(confirmed)
    hint = QLabel("입력 후 확인한 문구만 적용합니다." if enabled else "귀 외형 변경은 현재 적용하지 않습니다.")
    hint.setWordWrap(True)
    layout.addWidget(hint)
    def changed():
        confirmed.setChecked(False)
        try:
            PartAppearance(text.text(), True)
        except (TypeError, ValueError) as error:
            confirmed.setEnabled(False)
            hint.setText(str(error) if text.text().strip() else "문구 없음 · 기존 태그를 그대로 사용합니다.")
        else:
            confirmed.setEnabled(enabled)
            hint.setText("아직 적용하지 않습니다. 내용을 확인한 뒤 체크해 주세요.")
    text.textChanged.connect(changed)
    confirmed.toggled.connect(lambda checked: hint.setText(
        "확인한 문구를 적용합니다." if checked else "문구 미확인 · 기존 태그를 그대로 사용합니다."))
    return text, confirmed


def confirm_inputs(window, analysis, enable_ear_override=False):
    from genai_lab.onepass_garment_vocabulary import garment_nouns
    dialog = QDialog(window)
    dialog.setWindowTitle("만들 이미지 확인")
    dialog.resize(780, 650)
    outer = QVBoxLayout(dialog)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    scroll.setWidget(content)
    outer.addWidget(scroll)
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
    fields = {}
    for part, detected in appendage_tags(analysis["groups"]["fixed"]).items():
        if detected:
            fields[part] = appearance_field(layout, part, detected,
                part == "tail" or enable_ear_override)
    if "tail" in fields:
        notice = QLabel("꼬리 색·모양을 적지 않으면 종 대표 색이나 머리색으로 그려질 수 있습니다.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
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
    outer.addLayout(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    appearance = AppearanceOverrides(**{
        part: PartAppearance(text.text(), confirmed.isChecked())
        for part, (text, confirmed) in fields.items()
    })
    return gender.currentData(), tuple(t.strip() for t in tags.toPlainText().split(",") if t.strip()), appearance

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
    if any(appendage_tags(analysis["groups"]["fixed"]).values()):
        checks["appendages"] = QCheckBox("꼬리·귀의 개수·종류·색이 원래 캐릭터와 같습니다.")
        layout.addWidget(checks["appendages"])
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
