"""GUI-owned cancellable CPU process. No model loading on the GUI thread."""
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

from PySide6.QtCore import QObject, Signal, QProcess, QProcessEnvironment, QTimer
from genai_lab.qwen_preservation import json_sha
from genai_lab.qwen_record_io import write_json


class TailComplexityRunner(QObject):
    completed = Signal(dict)

    def __init__(self, parent=None, *, timeout_ms=120000, program=None, arguments=None):
        super().__init__(parent)
        self.timeout_ms = timeout_ms
        self.program = program or sys.executable
        self.arguments = arguments or ['-m', 'genai_lab.tail_complexity_worker']
        self.storage = tempfile.TemporaryDirectory(prefix='genai-tail-complexity-')
        self.process = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.timed_out)
        self.request = None
        self.directory = None

    def start(self, source, source_sha256, box):
        self.cancel()
        self.directory = Path(self.storage.name)/uuid.uuid4().hex
        self.directory.mkdir()
        self.request = dict(source=str(source), source_sha256=source_sha256, box=list(box))
        write_json(self.directory/'request.json', self.request)
        process = QProcess(self)
        self.process = process
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='',
                               PYTHONDONTWRITEBYTECODE='1').items():
            environment.insert(key, value)
        environment.remove('PYTHONPATH')
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setStandardOutputFile(str(self.directory/'worker.log'))
        process.finished.connect(lambda code, status: self.finished(process, code))
        process.errorOccurred.connect(lambda error: self.failed_start(process, error))
        process.start(self.program, self.arguments + [str(self.directory/'request.json')])
        self.timer.start(self.timeout_ms)

    def cancel(self):
        self.timer.stop()
        process, self.process = self.process, None
        if process is not None:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
                process.waitForFinished(1000)
            self.release_process(process)

    @staticmethod
    def release_process(process):
        process.finished.disconnect()
        process.errorOccurred.disconnect()
        process.deleteLater()

    def finished(self, process, code):
        if process is not self.process:
            return  # Superseded crop results cannot replace current selection.
        self.timer.stop()
        try:
            if code != 0:
                raise RuntimeError('CPU 분석 프로세스 실패')
            record = json.loads((self.directory/'result.json').read_text(encoding='utf-8'))
            if record.get('request_sha256') != json_sha(self.request):
                raise ValueError('다른 선택 영역의 분석 결과')
        except Exception as error:
            record = dict(status='failed', advisory_only=True, error=str(error))
        self.process = None
        self.release_process(process)
        self.completed.emit(record)

    def failed_start(self, process, error):
        if process is self.process and error == QProcess.ProcessError.FailedToStart:
            self.cancel()
            self.completed.emit(dict(status='failed', advisory_only=True, error='CPU 분석 실행 불가'))

    def timed_out(self):
        self.cancel()
        self.completed.emit(dict(status='timeout', advisory_only=True, error='분석 시간 초과'))

    def close(self):
        self.cancel()
        self.storage.cleanup()
