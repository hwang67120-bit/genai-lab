"""배포 후보 흐름 확인(D-174, P2·P3) — 배포 후보의 runtime-main\\python.exe로 실행한다(실행기와 같은 환경 변수).

단계(프로세스마다 하나 = 앱 실행 한 번):
  first   첫 실행: 검증 기록 없음 → 준비 화면이어야 함 → 모델 폴더 선택 → 깊은 검사 → 작업실 열기 → 종료
  second  재실행: 기록으로 바로 작업실 → '이미지 생성' 전 빠른 검사 → 제품 CPU 단계(Qwen 생성·머리 분할은 기존 결과로 대신) → 종료
  moved   모델 폴더를 옮긴 뒤 실행: 준비 화면 + '찾을 수 없음' 안내
  missing 검증 기록은 있는데 파일이 빠진 폴더(WD 태거 없음): 작업실은 열리지만 '이미지 생성' 전 빠른 검사가 막음
사용: runtime-main\\python.exe app\\scripts\\product_file_trace\\package_flow_check.py <단계> <출력 폴더> <모델 폴더> [<참고 그림 폴더>]
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def wait(app, condition, seconds):
    end = time.time() + seconds
    while not condition() and time.time() < end:
        app.processEvents()
        time.sleep(0.05)
    return condition()


def main(argv):
    phase, out, home = argv[0], Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    report = dict(phase=phase, executable=sys.executable, base_prefix=sys.base_prefix, prefix=sys.prefix,
                  app_home=os.environ.get("GENAI_APP_HOME"), appdata=os.environ.get("APPDATA"), path=os.environ.get("PATH"),
                  sys_path=list(sys.path))
    try:
        from genai_lab.child_env import isolate_hf_credentials
        isolate_hf_credentials()
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from genai_lab.app_home import app_home
        from genai_lab import model_setup
        packaged = app_home() is not None
        decision, value = model_setup.startup_decision(quick_check=None, packaged=packaged)
        report.update(packaged=packaged, decision=decision, decision_value=str(value), record_path=str(model_setup.record_path()))
        if phase == "first":
            from genai_lab.model_setup_ui import ModelSetupWindow
            opened = []
            setup = ModelSetupWindow(on_ready=opened.append, reason=value if decision == "setup" else "")
            report["setup_status_at_start"] = setup.status.text()
            report["folder_set"] = setup.set_folder(home)
            report["check_started"] = setup.start_check()
            report["check_finished"] = wait(app, lambda: not setup.running(), 900)
            report.update(status=setup.status.text(), files_status=setup.files_status.text(), gpu_status=setup.gpu_status.text(),
                          details=setup.details.toPlainText()[:1500], open_enabled=setup.open_button.isEnabled())
            report["open_studio"] = setup.open_studio()
            if opened:
                import gui_main
                studio = gui_main.open_studio(opened[0])
                app.processEvents()
                report["studio_window"] = type(studio).__name__
                report["model_home_env"] = os.environ.get("GENAI_MODEL_HOME")
                studio.close()
            setup.close()
            report["record"] = model_setup.load_record()
        elif phase == "second":
            import gui_main
            studio = gui_main.open_studio(value if decision == "studio" else None)
            app.processEvents()
            report["studio_window"] = type(studio).__name__
            from genai_lab import model_check
            from genai_lab.qwen_pose_settings import QwenPoseSettings
            from genai_lab.studio_generation import StudioRuntime
            runtime = StudioRuntime.from_environment()
            report["runtime"] = {k: str(v) for k, v in vars(runtime).items() if isinstance(v, Path)}
            for garment in (False, True):
                result = model_check.require_product_models(runtime, QwenPoseSettings.load(runtime.qwen_settings), garment=garment)
                report[f"quick_check_{'outfit' if garment else 'staging'}"] = result["ok"]
            studio.close()
            # 제품 CPU 단계(장면 실행기와 같은 함수, Qwen 생성·머리 분할은 기존 결과로 대신)
            sys.argv = [sys.argv[0]]
            import scenarios
            scenarios_out = out / "scenario-base4"
            scenarios_out.mkdir(exist_ok=True)
            status = {}
            for stage, fn in scenarios.scenario_base4(scenarios_out):
                scenarios.mark(stage)
                start = time.time()
                try:
                    result = fn()
                    status[stage] = dict(state="ok", seconds=round(time.time() - start, 1),
                                         result=result.get("status") if isinstance(result, dict) else None)
                except Exception as error:  # noqa: BLE001
                    status[stage] = dict(state="failed", error=str(error)[:300], tb=traceback.format_exc()[-1200:])
            report["stages"] = status
        elif phase in ("moved", "missing"):
            report["record"] = model_setup.load_record()
            if decision == "setup":
                from genai_lab.model_setup_ui import ModelSetupWindow
                setup = ModelSetupWindow(reason=value)
                report["setup_status_at_start"] = setup.status.text()
                setup.close()
            else:
                import gui_main
                studio = gui_main.open_studio(value)
                app.processEvents()
                picture = Path(argv[3]) if len(argv) > 3 else None
                studio.style_path = str(picture) if picture else ""
                failures = []
                studio.studio.fail = lambda error, details="": failures.append(str(error))
                studio.studio.product.start()
                app.processEvents()
                report["start_blocked"] = bool(failures)
                report["start_message"] = failures[0][:1500] if failures else ""
                report["task_started"] = studio.studio.task is not None
                studio.close()
    except Exception as error:  # noqa: BLE001
        report.update(error=f"{type(error).__name__}: {error}", tb=traceback.format_exc()[-2000:])
    (out / f"report-{phase}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(phase, report.get("decision"), report.get("error", "ok"), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv[1:])
