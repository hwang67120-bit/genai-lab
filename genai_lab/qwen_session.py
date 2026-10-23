"""Qwen 작업 프로세스 세션(D-180): 한 단계 안에서 seed만 다른 재시도에 작업 프로세스 하나를 다시 쓴다.

재시도 요청은 seed만 다르고 지시문·입력 그림·임베딩이 같다(요청 37 기록). 새 프로세스마다 하던 시작·모델 SHA 검사·라이브러리 가져오기·
문장 인코딩·파이프라인 불러오기를 첫 작업에서 한 번만 한다. 재시도 횟수·판정 규칙은 그대로다.

- 세션에 넣는 요청: 자유 생성, 가속·그림체 LoRA 없음, 첫 요청과 seed 밖의 내용이 같음(`qwen_pose_worker.reuse_key`). 아니면 세션을 쓰지 않는다.
- 요청마다 새로: 요청 폴더·request.json·launcher.json, seed·생성기, 입력 그림 SHA 확인, model_inputs, 진행·미리보기 기록, 메모리 감시.
- 한 작업이라도 완료되지 않으면(실패·취소·메모리 중단) 프로세스를 끝내고 세션을 닫는다 — 다음 요청은 새 프로세스.
- 실행기가 끝나면(표준 입력이 닫힘) 작업 프로세스도 끝난다. 다음 작업이 30분 안에 없으면 스스로 끝난다.
- 기본값은 꺼짐: GPU 확인(MEMORY_OPTIMIZATION_PLAN 11.4 M4) 전에는 `GENAI_QWEN_SESSION_REUSE=1`일 때만 쓴다.
"""
import os
import subprocess
from pathlib import Path

from genai_lab.child_env import child_environment
from genai_lab.qwen_preservation import json_sha
from genai_lab.qwen_pose_worker import reuse_key, session_reusable
from genai_lab.qwen_record_io import write_json

ENV = "GENAI_QWEN_SESSION_REUSE"


def session_enabled(environ=os.environ):
    return environ.get(ENV, "").strip() == "1"


class QwenSession:
    def __init__(self, settings, control, *, popen=subprocess.Popen):
        self.settings, self.control, self.popen = settings, Path(control), popen
        self.process = self.log = self.key = None
        self.command = [settings.python_executable, "-m", "genai_lab.qwen_pose_worker", "--session", str(self.control)]
        self.jobs = 0
        self.closed = False

    def accepts(self, request):
        """이 요청을 지금 세션에 보내도 되는가: 세션 대상이고, 첫 요청과 seed만 다르고, 프로세스가 살아 있다."""
        if self.closed or not session_reusable(request):
            return False
        if self.key is not None and self.key != reuse_key(request):
            return False
        return self.process is None or self.process.poll() is None

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def submit(self, request_path, request):
        if not self.accepts(request):
            raise RuntimeError("이 요청은 지금 세션에 보낼 수 없습니다.")
        if self.process is None:
            self.control.mkdir(parents=True, exist_ok=False)
            self.log = (self.control / "worker.log").open("w", encoding="utf-8")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            self.process = self.popen(self.command, cwd=str(Path(__file__).resolve().parents[1]), env=child_environment(),
                                      stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT, shell=False,
                                      creationflags=creationflags)
        self.key = reuse_key(request)
        self.jobs += 1
        write_json(self.control / f"job-{self.jobs:06d}.json", dict(request=str(request_path), request_sha256=json_sha(request)))
        return self.jobs

    def _stop(self, graceful):
        if self.process is not None and self.process.poll() is None:
            if graceful:
                (self.control / "close").touch()
                try:
                    self.process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        stdin = getattr(self.process, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except OSError:
                pass
        if self.log is not None:
            self.log.close()
            self.log = None
        self.closed = True

    def close(self):
        """정상 끝: close 파일로 알리고 기다린다."""
        self._stop(graceful=True)

    def kill(self):
        """실패·취소: 바로 끝낸다(넘길 상태가 깨끗하다고 보지 않는다)."""
        self._stop(graceful=False)

    def record(self):
        return dict(control=str(self.control), jobs=self.jobs, pid=getattr(self.process, "pid", None),
                    returncode=None if self.process is None else self.process.poll())

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        if kind is None:
            self.close()
        else:
            self.kill()
        return False
