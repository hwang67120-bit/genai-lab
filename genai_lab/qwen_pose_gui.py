"""Explicit Qwen B editor UI. Optional entry; existing generation remains unchanged."""
from dataclasses import replace
from pathlib import Path
from threading import Event
import uuid
import json

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QComboBox, QCheckBox, QMessageBox, QScrollArea, QWidget, QFileDialog)

from genai_lab.qwen_preservation import PreservationItem, PreservationSpec, draft_from_reports, file_sha
from genai_lab.qwen_pose_prompt import PoseEditInstructions
from genai_lab.qwen_pose_edit import PoseEditWorkflow, make_request, run_pose_edit, write_json
from genai_lab.qwen_pose_review import ITEM_STATES, GLOBAL_FIELDS, GLOBAL_STATES, LIMITATIONS, save_user_review
from genai_lab.visual_reference_review import preview


class TaskThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
    def run(self):
        try: self.succeeded.emit(self.operation())
        except Exception as error: self.failed.emit(str(error))
        finally: self.operation = None


class QwenPoseDialog(QDialog):
    def __init__(self, image_path, skeleton_path, settings, workflow, output_root, parent=None):
        super().__init__(parent)
        self.image_path, self.skeleton_path = Path(image_path), Path(skeleton_path)
        self.skeleton_sha256 = file_sha(skeleton_path)
        self.settings, self.workflow, self.output_root = settings, workflow, Path(output_root)
        self.spec, self.task = None, None
        self.saved_reports, self.generation_tags = [], []
        self.cancel_event = Event()
        self.setWindowTitle("승인 기준 이미지의 자세 편집")
        self.resize(1150, 850)
        layout = QVBoxLayout(self)
        notice = QLabel(LIMITATIONS); notice.setWordWrap(True); layout.addWidget(notice)
        from PIL import Image
        row = QHBoxLayout()
        for path in (self.image_path, self.skeleton_path):
            with Image.open(path) as image: row.addWidget(preview(image))
        layout.addLayout(row)
        self.approve_image = QCheckBox("이 완성 이미지를 보존 기준으로 승인합니다. 오른쪽은 반전 전 골격입니다.")
        layout.addWidget(self.approve_image)
        self.analyze_button = QPushButton("완성 이미지 분석 후 보존 항목 작성")
        self.analyze_button.clicked.connect(self.analyze); layout.addWidget(self.analyze_button)
        self.import_button = QPushButton("기존 분석 근거·생성 태그 JSON 불러오기 (선택)")
        self.import_button.clicked.connect(self.import_evidence); layout.addWidget(self.import_button)
        self.status = QLabel("기준 이미지 확인 대기"); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["ID/종류", "한국어 사실", "영어 지시문", "검토 위치", "근거·확인 필요", "상태", "성별 근거 확인", "충돌 해결 근거"])
        layout.addWidget(self.table)
        self.add_button = QPushButton("사용자 보존 항목 추가"); self.add_button.clicked.connect(self.add_manual); layout.addWidget(self.add_button)
        self.viewpoint = QLineEdit(); self.viewpoint.setPlaceholderText("선택 시점: 예 front-facing — 기본값 없음")
        self.hands = QLineEdit(); self.hands.setPlaceholderText("선택 손 동작 영어 문장 — 기본값 없음")
        layout.addWidget(self.viewpoint); layout.addWidget(self.hands)
        self.pose_confirmation = QCheckBox("입력한 시점·손 문장을 확인했습니다."); layout.addWidget(self.pose_confirmation)
        self.run_button = QPushButton("보존 문장 확인 후 Qwen 1회 편집")
        self.run_button.setEnabled(False); self.run_button.clicked.connect(self.run_edit); layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("취소/닫기"); self.cancel_button.clicked.connect(self.reject); layout.addWidget(self.cancel_button)

    def busy(self): return self.task is not None and self.task.isRunning()

    def start_task(self, operation, done):
        if self.busy(): return
        self.task = TaskThread(operation, self)
        self.task.succeeded.connect(done)
        self.task.failed.connect(self.failed)
        self.task.start()

    def failed(self, message):
        self.status.setText(message)
        QMessageBox.warning(self, "실행 중단", message)

    def analyze(self):
        if not self.approve_image.isChecked() or self.busy(): return
        self.analyze_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.approve_image.setEnabled(False)
        self.status.setText("앞 단계 해제 확인 → 완성 이미지 분석 → 분석 자원 해제")
        def operation():
            self.workflow.release_generation()
            def factory():
                if not self.settings.analysis_cache_dir:
                    raise ValueError("로컬 분석 캐시 미설정 — 사용자 직접 입력 필요")
                from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings
                from genai_lab.qwen_preservation_analysis import FinishedImageAnalyzer
                return FinishedImageAnalyzer(ClothingDesignAnalysisSettings(cache_dir=Path(self.settings.analysis_cache_dir)))
            return self.workflow.analyze(self.image_path, factory)
        self.start_task(operation, self.analysis_done)

    def analysis_done(self, result):
        reports, failure = result
        reports = [*reports, *self.saved_reports]
        preparation = self.output_root / ("preparation-" + uuid.uuid4().hex)
        preparation.mkdir(parents=True, exist_ok=False)
        write_json(preparation / "analysis.json", {"reports": reports, "failure": failure,
                   "image_sha256": self.workflow.analysis_image_sha256, "generation_tags": self.generation_tags,
                   "gpu_sequence": self.workflow.events})
        self.spec = draft_from_reports(self.image_path, "사용자가 기준 이미지 승인", reports, self.generation_tags)
        write_json(preparation / "draft.json", self.spec.record())
        for item in self.spec.items: self.append_item(item)
        self.run_button.setEnabled(True)
        self.status.setText("모든 후보는 미확인입니다. 근거와 실제 이미지를 비교하고 영어 문장까지 확인하세요." +
                            (" 분석 실패: " + failure["message"] if failure else ""))

    def import_evidence(self):
        path, _ = QFileDialog.getOpenFileName(self, "현재 완성 이미지의 저장 분석 근거", "", "JSON (*.json)")
        if not path: return
        try:
            bundle = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            from genai_lab.qwen_preservation_analysis import report_candidates
            reports = [report_candidates(x["source"], x["image_sha256"], x["report"])
                       for x in bundle.get("reports", [])]
            tags = bundle.get("generation_tags", [])
            if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
                raise ValueError("generation_tags는 문자열 목록이어야 합니다.")
            draft_from_reports(self.image_path, "근거 형식 확인", reports, tags)
            self.saved_reports, self.generation_tags = reports, tags
            self.status.setText("분석 근거를 불러왔습니다. 다른 이미지 SHA·생성 태그는 충돌 확인 대상으로 남습니다.")
        except Exception as error: self.failed(str(error))

    def append_item(self, item):
        i = self.table.rowCount(); self.table.insertRow(i)
        for col, text in enumerate((item.id + "/" + item.kind, item.fact_ko, item.english,
                                    item.check_at, item.source + ": " + ", ".join(item.issues))):
            cell = QTableWidgetItem(text)
            if col in (0, 4): cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, col, cell)
        self.table.item(i, 0).setData(Qt.ItemDataRole.UserRole, item)
        state = QComboBox(); state.addItems(["미확인", "사용자 확인", "제외"]); self.table.setCellWidget(i, 5, state)
        self.table.setCellWidget(i, 6, QCheckBox())
        self.table.setItem(i, 7, QTableWidgetItem(""))

    def add_manual(self):
        if self.spec is None: return
        self.append_item(PreservationItem("user-" + uuid.uuid4().hex[:8], "user", "", "", "user_input",
                                         self.spec.image_sha256, "사용자 지정"))

    def confirmed_spec(self):
        items = []
        for i in range(self.table.rowCount()):
            old = self.table.item(i, 0).data(Qt.ItemDataRole.UserRole)
            state = ("draft", "confirmed", "excluded")[self.table.cellWidget(i, 5).currentIndex()]
            items.append(replace(old, fact_ko=self.table.item(i, 1).text(), english=self.table.item(i, 2).text(),
                check_at=self.table.item(i, 3).text(), status=state,
                confirmation="사용자 화면 확인" if state == "confirmed" else "",
                resolution=self.table.item(i, 7).text(), approved_gender=self.table.cellWidget(i, 6).isChecked()))
        return replace(self.spec, items=tuple(items), revision=self.spec.revision + 1)

    def run_edit(self):
        if self.busy(): return
        try:
            spec = self.confirmed_spec()
            instructions = PoseEditInstructions(self.viewpoint.text(), self.hands.text(),
                "사용자 화면 확인" if self.pose_confirmation.isChecked() else "")
            request = make_request(spec, self.skeleton_path, self.skeleton_sha256,
                                   instructions=instructions, settings=self.settings)
            confirm = QMessageBox.question(self, "실제 전달 문장 확인", request["prompt"]["positive"])
            if confirm != QMessageBox.StandardButton.Yes: return
            self.workflow.confirm(spec)
            self.spec = spec
            self.directory = self.output_root / ("edit-" + uuid.uuid4().hex)
            self.run_button.setEnabled(False); self.add_button.setEnabled(False); self.table.setEnabled(False)
            self.status.setText("Qwen 별도 프로세스 실행 중 — 실패 시 자동 재시도하지 않습니다.")
            self.start_task(lambda: run_pose_edit(request, self.directory, self.workflow,
                            cancelled=self.cancel_event.is_set), self.edit_done)
        except Exception as error: self.failed(str(error))

    def edit_done(self, product):
        self.status.setText("완료 — 사용자 검토 전까지 최종 결과가 아닙니다.")
        ReviewDialog(self.directory, self.spec, self).exec()

    def closeEvent(self, event):
        if self.busy():
            self.cancel_event.set()
            self.status.setText("취소 요청됨. 실행 종료를 기다린 뒤 닫아 주세요.")
            event.ignore()
        else:
            event.accept()

    def reject(self):
        if self.busy():
            self.cancel_event.set()
            self.status.setText("취소 요청됨. 실행 종료를 기다린 뒤 닫아 주세요.")
            return
        super().reject()


class ReviewDialog(QDialog):
    def __init__(self, directory, spec, parent=None):
        super().__init__(parent)
        self.directory, self.spec = Path(directory), spec
        self.setWindowTitle("자세 편집 결과 — 사용자 검토"); self.resize(1000, 850)
        layout = QVBoxLayout(self)
        from PIL import Image
        row = QHBoxLayout()
        for path in (spec.image_path, self.directory / "raw.png", self.directory / "product.png"):
            with Image.open(path) as image: row.addWidget(preview(image))
        layout.addLayout(row)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); body = QWidget(); checks = QVBoxLayout(body)
        scroll.setWidget(body); layout.addWidget(scroll)
        self.items, self.globals = {}, {}
        for item in spec.items:
            if item.status != "confirmed": continue
            checks.addWidget(QLabel(item.fact_ko))
            combo = QComboBox(); combo.addItems(["선택 필요", *ITEM_STATES]); checks.addWidget(combo); self.items[item.id] = combo
        for name in GLOBAL_FIELDS:
            checks.addWidget(QLabel(name)); combo = QComboBox(); combo.addItems(["선택 필요", *GLOBAL_STATES])
            checks.addWidget(combo); self.globals[name] = combo
        for label, adopted in (("검토 기록 후 채택", True), ("검토 기록 후 미채택", False)):
            button = QPushButton(label); button.clicked.connect(lambda checked=False, a=adopted: self.save(a)); layout.addWidget(button)

    def save(self, adopted):
        try:
            save_user_review(self.directory, self.spec, {k:v.currentText() for k,v in self.items.items()},
                {k:v.currentText() for k,v in self.globals.items()}, adopted=adopted, reviewer="사용자 화면 판정")
            self.accept()
        except Exception as error: QMessageBox.warning(self, "검토 필요", str(error))
