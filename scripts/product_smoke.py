"""제품 점검(D-148): 작업실 화면 코드(GenAILabWindow → StudioController)로 실제 경로를 끝까지 돌린다. 화면은 띄우지 않는다(offscreen).

사람이 누르는 확인 창만 정해진 값으로 대신 넘긴다(입력 확인·결과 확인·저장 위치). 생성·분석·승인·저장 코드는 제품 그대로다.
여기서의 '승인'은 경로가 끝까지 도는지 보는 자동 점검이지 품질 판정이 아니다(판정 기록 source에 product_smoke로 남는다).
사용: python scripts/product_smoke.py <출력 폴더> <캐릭터 이미지> <의상 이미지> [edit|free|preview ...]
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GENAI_STUDIO_DEVELOPER", "1")   # 이 점검은 개발·평가 화면 경로(확인 창 대체)를 돈다. 사용자 화면은 D-151
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import genai_lab.studio_controller as controller  # noqa: E402
import genai_lab.studio_free_generation as free_ui  # noqa: E402

OUT, CHARACTER, GARMENT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FLOWS = sys.argv[4:] or ["edit", "free", "preview"]
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = dict(hair_length=["short_hair"], hair_style=None, hair_color=["white_hair"], eye_color=["blue_eyes"],
                head_accessories=[], skin=[])
status = dict(flows={}, started=time.strftime("%Y-%m-%d %H:%M:%S"))
errors = []


def save_status():
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


# 오류 상자는 띄우지 않고 기록한다(제품의 실패 안내 경로는 그대로 지난다)
QMessageBox.exec = lambda self: errors.append(self.text() + " | " + self.detailedText()[-1500:]) or 0
QMessageBox.warning = staticmethod(lambda parent, title, text, *a, **k: errors.append(f"{title}: {text}") or 0)
QMessageBox.information = staticmethod(lambda parent, title, text, *a, **k: errors.append(f"{title}: {text}") or 0)
controller.confirm_outfit_edit = lambda *a, **k: dict(gender="male", body_parts=(), features=FEATURES)
controller.confirm_result = lambda *a, **k: dict(character=True, garment=True, exposure=True, original_outfit_gone=True)
free_ui.confirm_free_result = lambda *a, **k: dict(character=True, garment=True, request=True, exposure=True, finish=True)
saved_path = {}
controller.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(saved_path["next"]), "PNG (*.png)"))


def wait(window, flow, limit_minutes=90):
    """작업이 끝날 때까지(결과가 생기거나 실패) 이벤트를 돌린다."""
    app = QApplication.instance()
    start = time.time()
    while time.time() - start < limit_minutes * 60:
        app.processEvents()
        time.sleep(0.5)
        studio = window.studio
        idle = studio.task is None and not studio.confirming     # occupied는 결과가 화면에 있어도 참이다
        if idle and (studio.results is not None or errors):
            break
        status["flows"][flow]["elapsed_seconds"] = round(time.time() - start)
        if int(time.time() - start) % 60 == 0:
            status["flows"][flow]["status_label"] = window.status_label.text()
            save_status()
    return round(time.time() - start)


def run_flow(window, flow):
    errors.clear()
    status["flows"][flow] = dict(state="running")
    save_status()
    window.style_path = str(CHARACTER)
    window.studio.results = None
    if flow == "edit":
        window.selected_outfit_path = GARMENT
        window.studio_method_combo.setCurrentIndex(window.studio_method_combo.findData("edit"))
        window.studio.start()
    else:
        window.selected_outfit_path = None
        decision = dict(base_kind="original", gender="male", body_parts=(), character_features=FEATURES,
                        request_text="sitting on a simple stool in a cozy cafe", confirmed=True, adult_confirmed=True)
        if flow == "preview":
            decision["experiment"] = dict(lightning8=True, single_attempt=True)
        free_ui.confirm_free_inputs = lambda *a, **k: dict(decision)
        window.studio.start_free_generation()
    seconds = wait(window, flow)
    record = status["flows"][flow]
    record.update(seconds=seconds, status_label=window.status_label.text(), errors=list(errors))
    results = window.studio.results
    if results is None:
        record["state"] = "failed"
        save_status()
        return
    record["result"] = str(results.current_path)
    record["head_notice"] = getattr(results, "head_notice", "")
    window.studio.approve()
    record["approved"] = results.status == "user_approved"
    saved_path["next"] = OUT / f"{flow}-saved.png"
    window.studio.save()
    record["saved"] = saved_path["next"].exists()
    record["state"] = "ok" if record["approved"] and record["saved"] and not errors else "failed"
    record["errors"] = list(errors)
    save_status()


def main():
    app = QApplication.instance() or QApplication([])
    from gui_main import GenAILabWindow
    window = GenAILabWindow()
    window.studio.output_root = OUT / "studio-runs"
    for flow in FLOWS:
        try:
            run_flow(window, flow)
        except Exception:  # noqa: BLE001
            status["flows"][flow] = dict(state="crashed", traceback=traceback.format_exc()[-3000:])
            save_status()
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_status()
    window.close()
    print(json.dumps({k: v.get("state") for k, v in status["flows"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
