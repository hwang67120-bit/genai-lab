"""최초 모델 준비 화면(D-173): 기존 로컬 모델 폴더 선택 → 공통 깊은 검사(별도 프로세스) → 결과 안내 → 재검사 → 작업실 열기.

화면은 검사 중에도 멈추지 않는다. 검사 중에는 폴더 선택·검사 버튼을 막고(중복 실행·폴더 바뀜 방지), 끝난 결과도
실행 번호와 폴더가 지금 것과 같을 때만 받는다. 취소·중단은 저장하지 않는다.
'모델 파일 검증 완료'와 'GPU 생성 검증'은 따로 보여 준다(후자는 이 화면에서 하지 않는다).
"""
import json
import sys
import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment
from shiboken6 import isValid
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout,
                               QWidget)

from genai_lab import model_setup
from genai_lab.child_env import child_environment

GPU_NOT_VERIFIED = "GPU 생성 검증: 아직 안 됨 — 이 화면은 파일만 확인합니다. 첫 '이미지 생성'에서 실제로 확인됩니다."


def default_command(home, out, run_id):
    return [sys.executable, "-m", "genai_lab.model_setup", "check", str(home), str(out), run_id]


class ModelSetupWindow(QWidget):
    def __init__(self, on_ready=None, *, command=default_command, record_file=None):
        super().__init__()
        self.setWindowTitle("모델 준비")
        self.resize(760, 560)
        self.on_ready, self.command, self.record_file = on_ready, command, record_file
        self.folder = None
        self.process = self.run_id = self.run_home = self.result_file = None
        self.cancelled_runs = set()
        self.last_record = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("모델 준비 — 이미 받아 둔 모델 폴더를 고르고 검사합니다(내려받기는 아직 없습니다)."))
        row = QHBoxLayout()
        self.folder_field = QLineEdit()
        self.folder_field.setReadOnly(True)
        self.folder_field.setPlaceholderText("모델 폴더를 고르세요")
        self.choose_button = QPushButton("폴더 선택")
        self.choose_button.clicked.connect(self.choose_folder)
        row.addWidget(self.folder_field)
        row.addWidget(self.choose_button)
        layout.addLayout(row)
        buttons = QHBoxLayout()
        self.check_button = QPushButton("검사")
        self.check_button.clicked.connect(self.start_check)
        self.cancel_button = QPushButton("검사 취소")
        self.cancel_button.clicked.connect(self.cancel)
        self.open_button = QPushButton("작업실 열기")
        self.open_button.clicked.connect(self.open_studio)
        for button in (self.check_button, self.cancel_button, self.open_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel("상태: 폴더를 고르세요.")
        self.files_status = QLabel("모델 파일 검증: 아직 안 됨")
        self.gpu_status = QLabel(GPU_NOT_VERIFIED)
        for label in (self.status, self.files_status, self.gpu_status):
            label.setWordWrap(True)
            layout.addWidget(label)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        saved = model_setup.load_record(record_file)
        if saved and Path(saved["model_home"]).is_dir():
            self.set_folder(saved["model_home"])
        self.refresh()

    # 상태
    def running(self):
        return self.process is not None

    def verified_for_current(self):
        record = self.last_record or model_setup.load_record(self.record_file)
        return bool(record and self.folder and record.get("status") == model_setup.FILES_VERIFIED
                    and Path(record["model_home"]).resolve() == self.folder.resolve()
                    and record.get("fingerprint") == model_setup.manifest_fingerprint())

    def refresh(self):
        busy = self.running()
        self.choose_button.setEnabled(not busy)
        self.check_button.setEnabled(not busy and self.folder is not None)
        self.check_button.setText("재검사" if self.verified_for_current() else "검사")
        self.cancel_button.setEnabled(busy)
        verified = self.verified_for_current() and not busy
        self.open_button.setEnabled(verified)
        self.files_status.setText("모델 파일 검증: 완료(있음·크기·SHA)" if verified else "모델 파일 검증: 아직 안 됨")

    # 폴더
    def choose_folder(self):
        if self.running():
            return
        path = QFileDialog.getExistingDirectory(self, "모델 폴더 선택", str(self.folder or ""))
        if path:
            self.set_folder(path)

    def set_folder(self, path):
        if self.running():                 # 검사 중에는 폴더를 바꾸지 않는다(결과 혼동 방지)
            return False
        self.folder = Path(path)
        self.folder_field.setText(str(self.folder))
        self.last_record = None
        record = model_setup.load_record(self.record_file)
        if record and Path(record["model_home"]).resolve() == self.folder.resolve():
            self.last_record = record
            self.show_record(record)
        else:
            self.status.setText("상태: 검사 전입니다.")
            self.details.setPlainText("")
        self.refresh()
        return True

    # 검사
    def start_check(self):
        if self.running() or self.folder is None:   # 중복 실행 방지
            return False
        self.run_id, self.run_home = uuid.uuid4().hex, self.folder.resolve()
        self.result_file = Path(tempfile.gettempdir()) / f"genai-model-setup-{self.run_id}.json"
        process = QProcess(self)
        environment = QProcessEnvironment()
        for key, value in child_environment(PYTHONIOENCODING="utf-8").items():   # 진행 문장(한국어)을 UTF-8로 받는다
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(model_setup.ROOT))
        command = self.command(self.run_home, self.result_file, self.run_id)
        run_id = self.run_id
        process.readyReadStandardOutput.connect(lambda p=process: self.on_output(p))
        process.finished.connect(lambda code, status, run_id=run_id: self.on_finished(run_id, code))
        self.process = process
        self.status.setText("상태: 검사 중 — 화면은 계속 쓸 수 있습니다. 이미지 생성 모델(약 28GB) 확인에 몇 분 걸립니다.")
        self.details.setPlainText("")
        self.refresh()
        process.start(command[0], command[1:])
        return True

    def on_output(self, process):
        if not isValid(self.status):
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("progress "):
                self.status.setText("상태: 검사 중 · " + line[len("progress "):])

    def cancel(self):
        if not self.running():
            return False
        self.cancelled_runs.add(self.run_id)
        self.process.kill()
        return True

    def on_finished(self, run_id, exit_code):
        if run_id != self.run_id or not isValid(self.status):   # 이전 실행·이미 닫힌 창의 늦은 신호는 무시
            return
        result_file, run_home = self.result_file, self.run_home
        self.process = None
        if run_id in self.cancelled_runs:
            self.status.setText("상태: 검사를 취소했습니다. 저장하지 않았습니다.")
        elif exit_code != 0 or not result_file.is_file():
            self.status.setText(f"상태: 검사가 중간에 끝났습니다(종료 코드 {exit_code}). 저장하지 않았습니다.")
        else:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if result.get("run_id") != run_id or Path(result["home"]).resolve() != run_home or self.folder is None \
                    or self.folder.resolve() != run_home:
                self.status.setText("상태: 다른 실행·다른 폴더의 결과라 버렸습니다. 다시 검사하세요.")
            else:
                self.last_record = model_setup.save_result(result, path=self.record_file)
                self.show_record(self.last_record)
        if result_file is not None and result_file.exists():
            result_file.unlink()
        self.refresh()

    def show_record(self, record):
        if record["status"] == model_setup.FILES_VERIFIED:
            self.status.setText(f"상태: 모델 파일 검증 완료({record['checked_at']}, {record['seconds']}초). '작업실 열기'를 누르세요.")
            self.details.setPlainText("이 폴더의 모델 파일이 목록과 같습니다(있음·크기·SHA).\n" + GPU_NOT_VERIFIED)
        else:
            self.status.setText("상태: 모델 파일에 문제가 있어 시작할 수 없습니다. 아래 안내를 확인한 뒤 '재검사'를 누르세요.")
            self.details.setPlainText(record.get("message", ""))

    def closeEvent(self, event):            # noqa: N802 — 창을 닫으면 돌던 검사도 끝낸다(저장하지 않음)
        self.cancel()
        if self.process is not None:
            self.process.waitForFinished(5000)
        super().closeEvent(event)

    def open_studio(self):
        if not self.verified_for_current() or self.running():
            return False
        import os
        os.environ[model_setup.ENV] = str(self.folder)
        if self.on_ready is not None:
            self.on_ready(self.folder)
        return True
