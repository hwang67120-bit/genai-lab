"""사용자 화면 흐름 그대로 GPU 시험(D-156): 작업실을 사용자 화면(ProductFlow)으로 화면 없이 띄우고,
줄마다 캐릭터(·의상)를 고른 뒤 '이미지 생성'을 누르고, 끝나면 '저장'까지 한다. 장면 칸은 비워 자동 장면 조합을 쓴다.
확인 창이 없는 흐름이라 사람 대신 넘기는 창은 저장 위치 하나뿐이다.

사용: python scripts/product_flow_run.py <줄 목록 json> <출력 폴더>
  줄 목록: [{"case": "A1", "character": "...", "garment": null 또는 "..."}, ...]
출력: <출력 폴더>/runs/<run id>/(실행 기록), <출력 폴더>/<case>-saved.png(AI 생성 표시 포함), batch-status.json, summary.json
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("GENAI_STUDIO_DEVELOPER", None)          # 사용자 화면(D-151)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import genai_lab.studio_controller as controller  # noqa: E402

OUT = None
status = dict(state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), cases={})
errors = []
saved_path = {}


def save_status():
    temporary = OUT / "batch-status.tmp"
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, OUT / "batch-status.json")


QMessageBox.exec = lambda self: errors.append(self.text() + " | " + self.detailedText()[-1500:]) or 0
QMessageBox.warning = staticmethod(lambda parent, title, text, *a, **k: errors.append(f"{title}: {text}") or 0)
QMessageBox.information = staticmethod(lambda parent, title, text, *a, **k: errors.append(f"{title}: {text}") or 0)
controller.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(saved_path["next"]), "PNG (*.png)"))


def run_case(window, case, limit_minutes=240):
    app = QApplication.instance()
    errors.clear()
    record = status["cases"][case["case"]] = dict(state="running", character=case["character"], garment=case.get("garment"))
    save_status()
    window.style_path = case["character"]
    window.selected_outfit_path = Path(case["garment"]) if case.get("garment") else None
    window.studio.results = None
    window.update_input_ready_status()
    window.studio.product.scene_input.setText("")          # 장면 칸 비움 = 자동 장면 조합
    if not window.generate_button.isEnabled():
        record.update(state="failed", errors=["'이미지 생성'이 꺼져 있음: " + window.status_label.text()])
        save_status()
        return
    start = time.time()
    window.generate_button.click()
    while time.time() - start < limit_minutes * 60:
        app.processEvents()
        time.sleep(0.5)
        studio = window.studio
        if studio.task is None and not studio.confirming and (studio.results is not None or errors):
            break
        if int(time.time() - start) % 60 == 0:
            record.update(elapsed_seconds=round(time.time() - start), status_label=window.status_label.text())
            save_status()
    run_directory = window.studio.run_directory
    record.update(seconds=round(time.time() - start), run=str(run_directory), status_label=window.status_label.text(), errors=list(errors))
    scene = Path(run_directory) / "scene.json"
    if scene.is_file():
        record["scene"] = json.loads(scene.read_text(encoding="utf-8"))
    if window.studio.results is None:
        record["state"] = "failed"
        save_status()
        return
    record["result"] = str(window.studio.results.current_path)
    saved_path["next"] = OUT / f"{case['case']}-saved.png"
    window.save_candidate_button.click()                   # 사용자 화면 '저장' = 사용 결정 기록 + AI 생성 표시
    record["saved"] = str(saved_path["next"]) if saved_path["next"].exists() else None
    record["state"] = "ok" if record["saved"] and not errors else "failed"
    record["errors"] = list(errors)
    save_status()


def main(argv=None):
    global OUT
    argv = sys.argv[1:] if argv is None else argv
    cases, OUT = Path(argv[0]), Path(argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    status.update(state="running", cases={})
    app = QApplication.instance() or QApplication([])
    from gui_main import GenAILabWindow
    window = GenAILabWindow()
    if window.studio.product is None:
        raise SystemExit("사용자 화면이 아닙니다(GENAI_STUDIO_DEVELOPER가 켜져 있음).")
    window.studio.output_root = OUT / "runs"
    for case in json.loads(cases.read_text(encoding="utf-8")):
        try:
            run_case(window, case)
        except Exception:  # noqa: BLE001
            status["cases"][case["case"]] = dict(state="crashed", traceback=traceback.format_exc()[-3000:])
            save_status()
    status.update(state="finished", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_status()
    (OUT / "summary.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    window.close()
    print(json.dumps({k: v.get("state") for k, v in status["cases"].items()}, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main()
