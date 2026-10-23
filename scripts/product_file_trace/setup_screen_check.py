"""최초 모델 준비 화면 CPU 검증(D-173): 실제 검사 프로세스로 깊은 검사·취소·실패 안내·화면 멈춤 여부·시작 판단을 확인한다.

사용자 기록(%APPDATA%\\GenAILab\\model-setup.json)은 쓰지 않는다 — 기록 파일은 출력 폴더 안(GENAI_MODEL_SETUP_RECORD).
사용: python setup_screen_check.py <출력 폴더> <전체 모델 폴더> <WD 없는 모델 폴더>
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main(argv):
    out, full, no_wd = Path(argv[0]), Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    from genai_lab.child_env import isolate_hf_credentials
    isolate_hf_credentials()
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from genai_lab import model_setup
    from genai_lab.model_setup_ui import ModelSetupWindow
    report = {}

    def run(name, home, *, cancel_after=None, limit=900):
        record = out / f"record-{name}.json"
        os.environ["GENAI_MODEL_SETUP_RECORD"] = str(record)
        window = ModelSetupWindow(record_file=record)
        window.set_folder(home)
        ticks = [0]
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.__setitem__(0, ticks[0] + 1))
        timer.start(100)                           # 화면 이벤트가 도는지: 100ms마다 1씩
        statuses, start = [], time.time()
        assert window.start_check()
        duplicate = window.start_check()
        changed = window.set_folder(Path(str(home) + "-elsewhere"))
        while window.running() and time.time() - start < limit:
            app.processEvents()
            text = window.status.text()
            if not statuses or statuses[-1] != text:
                statuses.append(text)
            if cancel_after is not None and time.time() - start > cancel_after:
                window.cancel()
                cancel_after = None
            time.sleep(0.05)
        elapsed = time.time() - start
        timer.stop()
        app.processEvents()
        report[name] = dict(home=str(home), seconds=round(elapsed, 1), timer_ticks=ticks[0], expected_ticks=int(elapsed * 10),
                            duplicate_start_refused=duplicate is False, folder_change_refused=changed is False,
                            statuses=statuses[:12] + [window.status.text()], files_status=window.files_status.text(),
                            gpu_status=window.gpu_status.text(), open_enabled=window.open_button.isEnabled(),
                            record=json.loads(record.read_text(encoding="utf-8")) if record.exists() else None,
                            details=window.details.toPlainText()[:1500])
        window.close()
        print(name, report[name]["seconds"], "record:", (report[name]["record"] or {}).get("status"), "open:", report[name]["open_enabled"], flush=True)

    run("cancel-full", full, cancel_after=40)
    run("no-wd", no_wd)
    run("full-deep", full)
    os.environ["GENAI_MODEL_SETUP_RECORD"] = str(out / "record-none.json")
    report["startup_this_pc_without_record"] = list(map(str, model_setup.startup_decision(
        {k: v for k, v in os.environ.items() if k != model_setup.ENV}, quick_check=model_setup.current_paths_pass)))
    report["user_record_untouched"] = not (Path(os.environ.get("APPDATA", "")) / "GenAILab" / "model-setup.json").exists()
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("startup:", report["startup_this_pc_without_record"], "user record untouched:", report["user_record_untouched"])


if __name__ == "__main__":
    main(sys.argv[1:])
