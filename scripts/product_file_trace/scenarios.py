"""제품 의존 파일 관찰 — 사용자 화면이 부르는 CPU 단계를 조건별로 실제 실행하고 연 파일을 기록한다(D-166·D-168).

GPU는 쓰지 않는다(CUDA_VISIBLE_DEVICES=-1 + 시작 때 torch.cuda.is_available() 확인). 대신하는 단계:
- Qwen 생성(작업 프로세스) → 기존 결과 그림. 그 단계가 여는 파일은 정적 확인 목록(INSTALLER_PACKAGING_PREP 2-2)으로 따로 다룬다.
- 머리 분할(l2d_sam, CUDA 필요) → 기존 분할 결과(parts.npz). l2d_sam 체크포인트는 첫 관찰(GPU가 실제로 켜져 있었음)과 정적 확인으로 다룬다.
열린 파일 = 그 실행 경로에서 쓴 근거. 열리지 않은 파일 = 그 실행에서 쓰지 않았다는 뜻일 뿐, 제품 전체에서 불필요하다는 뜻이 아니다.

사용(Windows, 저장소 루트에서):
  set GENAI_OPEN_TRACE=<출력폴더>\\opens.jsonl  (출력 폴더는 비어 있는 새 폴더)
  set PYTHONPATH=scripts\\product_file_trace
  set CUDA_VISIBLE_DEVICES=-1
  python scripts\\product_file_trace\\scenarios.py <장면> <출력폴더>
장면: base4 · outfit_retry · hair_restore · hair_auto · memory_link · product_buttons
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DOWNLOADS = Path.home() / "Downloads"
B2_CHAR = DOWNLOADS / "참조 이미지" / "86a72742408d9b4d04f5fb23fb0b93bd.jpg"
B2_GARM = DOWNLOADS / "참조 의상" / "4b750036a001e45a39973c408855e970.jpg"
B2_RUN = ROOT / "outputs/product-run-20261022/runs/b5cb43fef5bf4277a15f16db4fd23644"
B2_FEATURES = {"hair_length": ["short_hair"], "hair_style": None, "hair_color": ["black_hair"], "eye_color": None,
               "head_accessories": None, "skin": None}
RED_RUN = ROOT / "outputs/studio-runs/4b0e5e2cbf3a4bad850d0f17b6cf94f7"          # 10/23 GUI: 두 시도 모두 머리색 위반 → 머리색 복원 후보
M6_TRIAL = ROOT / "outputs/style-trial-20261023"                                  # 요청 35 M6 없음: hair_auto auto_selected
MEMORY_CHARACTER = DOWNLOADS / "참조 이미지" / "26a8fe98706769af46e75e8663658909.jpg"  # 캐릭터 기억에 등록된 성인 캐릭터


def mark(stage):
    with open(os.environ["GENAI_OPEN_TRACE"], "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stage=stage, t=time.time()), ensure_ascii=False) + "\n")


def fake_qwen(*raws):
    """Qwen 작업 프로세스 대신: 호출 순서대로 주어진 결과 그림을 raw.png로 둔다(관찰 대상 아님)."""
    queue = list(raws)
    def run(request, directory, workflow, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True)
        source = queue.pop(0) if len(queue) > 1 else queue[0]
        if source == "white":
            from PIL import Image
            Image.new("RGB", (896, 1184), "white").save(directory / "raw.png")
        else:
            shutil.copy(source, directory / "raw.png")
        (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        (directory / "substituted.json").write_text(json.dumps(dict(qwen="substituted", source=str(source)), ensure_ascii=False),
                                                    encoding="utf-8")
        return directory / "raw.png"
    return run


def fake_segment(*runs):
    """머리 분할(l2d_sam)은 CUDA를 요구한다(head_paste_semantics: .to("cuda")) → GPU를 끈 관찰에서는 기존 실행의 분할 결과로 대신한다.
    폴더 이름(parts·base-parts·result-analysis[-N])이 같은 기존 parts.npz를 쓰고, 없으면 첫 실행의 같은 종류를 쓴다."""
    def run(prepared, directory, runtime, **kwargs):
        directory = Path(directory)
        result = directory.name == "parts" and directory.parent.name.startswith("result-analysis")
        key = directory.parent.name if result else directory.name
        candidates = []
        for r in runs:
            if result:
                candidates += [r / key / "parts" / "parts.npz", r / "result-analysis" / "parts" / "parts.npz"]
            elif key == "base-parts":
                candidates.append(r / "base-parts" / "parts.npz")
            else:
                candidates.append(r / "parts" / "parts.npz")
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"대신할 분할 결과가 없습니다: {key}")
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, directory / "parts.npz")
        (directory / "substituted.json").write_text(json.dumps(dict(segmentation="substituted", source=str(source)), ensure_ascii=False),
                                                    encoding="utf-8")
        return dict(parts=str(directory / "parts.npz"), substituted=True)
    return run


def patch_segmentation(*runs):
    import genai_lab.free_generation_pipeline as free
    import genai_lab.outfit_edit_pipeline as outfit
    fake = fake_segment(*runs)
    outfit.segment_canvas = fake
    free.segment_canvas = fake


def runtime_and_settings():
    from genai_lab import studio_generation
    from genai_lab.qwen_outfit_edit import load_qwen_settings
    runtime = studio_generation.StudioRuntime.from_environment()
    return runtime, load_qwen_settings(runtime.qwen_settings)


def trace_inputs(run):
    t = json.loads((run / "free-generation" / "trace.json").read_text(encoding="utf-8"))
    facts = t["input_facts"]
    return Path(facts["base"]["path"]), facts["confirmed_features"]["value"], facts["gender"]["value"], facts["body_parts"]["value"], facts["request"]["value"]


def scenario_base4(out):
    from genai_lab import studio_free_generation, studio_generation
    import genai_lab.free_generation_pipeline as free
    import genai_lab.qwen_outfit_edit as q
    runtime, settings = runtime_and_settings()
    q.run_outfit_edit = fake_qwen(B2_RUN / "outfit-edit/qwen/raw.png")
    free.run_free_generation_edit = fake_qwen(B2_RUN / "free-generation/qwen/raw.png")
    patch_segmentation(B2_RUN / "outfit-edit", B2_RUN / "free-generation")
    from genai_lab.outfit_edit_pipeline import run_outfit_pipeline
    return [
        ("A_analyze_inputs", lambda: studio_generation.analyze_inputs(B2_CHAR, B2_GARM, out / "A" / "inputs", runtime, require_face=False)),
        ("B_analyze_base", lambda: studio_free_generation.analyze_base(B2_CHAR, runtime)),
        ("C_outfit_pipeline", lambda: run_outfit_pipeline(B2_CHAR, B2_GARM, settings=settings, head_accessories="keep", runtime=runtime,
                                                          directory=out / "C", gender="female", body_parts=[], character_features=B2_FEATURES)),
        ("D_free_generation", lambda: free.run_free_generation(B2_CHAR, B2_FEATURES, "", base_kind="original", gender="female", body_parts=[],
                                                               confirmed=True, adult_confirmed=True, settings=settings, runtime=runtime,
                                                               directory=out / "D", selection_seed=42, confirmation_source="automatic_product")),
    ]


def scenario_outfit_retry(out):
    """첫 결과를 흰 그림으로 대신해 머리 붙이기가 실패하게 하고, 두 번째 seed 경로까지 지나가게 한다."""
    import genai_lab.qwen_outfit_edit as q
    runtime, settings = runtime_and_settings()
    q.run_outfit_edit = fake_qwen("white", B2_RUN / "outfit-edit/qwen/raw.png")
    patch_segmentation(B2_RUN / "outfit-edit")
    from genai_lab.outfit_edit_pipeline import run_outfit_pipeline
    return [("C_outfit_retry", lambda: run_outfit_pipeline(B2_CHAR, B2_GARM, settings=settings, head_accessories="keep", runtime=runtime,
                                                           directory=out / "C", gender="female", body_parts=[], character_features=B2_FEATURES))]


def scenario_hair_restore(out):
    """10/23 GUI 실행의 두 결과(둘 다 머리색 위반 → 복원 후보 restored)를 그대로 다시 지나가게 한다."""
    import genai_lab.free_generation_pipeline as free
    runtime, settings = runtime_and_settings()
    base, features, gender, parts, request = trace_inputs(RED_RUN)
    free.run_free_generation_edit = fake_qwen(RED_RUN / "free-generation/qwen/raw.png", RED_RUN / "free-generation/qwen-2/raw.png")
    patch_segmentation(RED_RUN / "free-generation")
    return [("D_hair_restore", lambda: free.run_free_generation(base, features, request, base_kind="original", gender=gender, body_parts=list(parts),
                                                                 confirmed=True, adult_confirmed=True, settings=settings, runtime=runtime,
                                                                 directory=out / "D", selection_seed=42, confirmation_source="automatic_product"))]


def scenario_hair_auto(out):
    """요청 35 M6 LoRA 없음 결과(hair_auto auto_selected)를 다시 지나가게 한다."""
    import genai_lab.free_generation_pipeline as free
    runtime, settings = runtime_and_settings()
    case = next(c for c in json.loads((M6_TRIAL / "cases.json").read_text(encoding="utf-8")) if c["case"] == "M6")
    from genai_lab.scene_composer import compose_scene
    free.run_free_generation_edit = fake_qwen(M6_TRIAL / "runs/M6-none/qwen/raw.png")
    patch_segmentation(M6_TRIAL / "runs/M6-none")
    return [("D_hair_auto", lambda: free.run_free_generation(ROOT / case["base"], case["features"], compose_scene(case["scene_seed"])["text"],
                                                              base_kind="original", gender=case["gender"], body_parts=list(case["body_parts"]),
                                                              confirmed=True, adult_confirmed=True, settings=settings, runtime=runtime,
                                                              directory=out / "D", selection_seed=42,
                                                              experiment=dict(single_attempt=True),
                                                              confirmation_source="automatic_product"))]


def scenario_memory_link(out):
    """캐릭터 기억 **사본**으로 사용자 화면 자동 정체성(auto_identity)을 지나가게 한다. 실제 사용자 기억은 읽기 전용으로 복사만."""
    source = Path(os.environ.get("APPDATA", "")) / "GenAILab" / "character-memory"
    copy = out / "character-memory-copy"
    shutil.copytree(source, copy)
    os.environ["GENAI_CHARACTER_MEMORY"] = str(copy)
    from genai_lab.character_memory import CharacterMemory
    from genai_lab.studio_product import auto_identity
    return [("M_auto_identity", lambda: auto_identity(MEMORY_CHARACTER, dict(character_scores={}), CharacterMemory()))]


def scenario_product_buttons(out):
    """사용자 화면에서 보이고 누를 수 있는 버튼·메뉴 목록(개발 화면 기능에 닿는지 판단 근거)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.pop("GENAI_STUDIO_DEVELOPER", None)
    def run():
        from PySide6.QtWidgets import QAbstractButton, QApplication, QLineEdit, QMenu, QComboBox
        import gui_main
        app = QApplication.instance() or QApplication(sys.argv)
        window = gui_main.GenAILabWindow()
        window.show()
        app.processEvents()
        controls = []
        for kind in (QAbstractButton, QComboBox, QLineEdit):
            for w in window.findChildren(kind):
                if w.isVisibleTo(window):
                    text = w.currentText() if isinstance(w, QComboBox) else w.text()
                    controls.append(dict(kind=type(w).__name__, name=w.objectName(), text=text, enabled=w.isEnabled()))
        actions = [dict(text=a.text(), enabled=a.isEnabled()) for m in window.findChildren(QMenu) for a in m.actions() if a.isVisible()]
        (out / "product-controls.json").write_text(json.dumps(dict(developer_mode=window.developer_mode, controls=controls, menu_actions=actions),
                                                              ensure_ascii=False, indent=2), encoding="utf-8")
        window.close()
        return dict(controls=len(controls), menu_actions=len(actions))
    return [("P_product_controls", run)]


SCENARIOS = dict(base4=scenario_base4, outfit_retry=scenario_outfit_retry, hair_restore=scenario_hair_restore,
                 hair_auto=scenario_hair_auto, memory_link=scenario_memory_link, product_buttons=scenario_product_buttons)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    name, out = argv[0], Path(argv[1])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":      # PowerShell에서 빈 값은 변수 삭제가 된다 → -1로 숨긴다
        raise SystemExit("GPU를 숨기고 실행하세요(CUDA_VISIBLE_DEVICES=-1).")
    import torch
    if torch.cuda.is_available():
        raise SystemExit("GPU가 보입니다 — 관찰을 시작하지 않습니다.")
    out.mkdir(parents=True, exist_ok=True)
    status = dict(scenario=name, qwen="substituted (기존 결과 그림)", segmentation="substituted (기존 분할 결과, CUDA 필요 단계)",
                  gpu="hidden (CUDA_VISIBLE_DEVICES=-1, torch.cuda.is_available()=False)")
    for stage, fn in SCENARIOS[name](out):
        mark(stage)
        start = time.time()
        try:
            result = fn()
            summary = {k: result.get(k) for k in ("status", "composited", "reason", "prefill", "gender", "memory_error", "controls", "menu_actions")
                       if isinstance(result, dict) and k in result}
            if isinstance(result, dict) and "attempts" in result:
                summary["attempts"] = [{k: a.get(k) for k in ("seed", "composited", "reason", "retry_reason") if k in a}
                                       | {"hair_auto": (a.get("hair_auto") or {}).get("decision"),
                                          "hair_restore": (a.get("hair_restore") or {}).get("status")}
                                       for a in result["attempts"]]
            status[stage] = dict(state="ok", seconds=round(time.time() - start, 1), result=summary)
        except Exception as error:  # noqa: BLE001 — 단계 실패도 관찰 결과
            status[stage] = dict(state="failed", seconds=round(time.time() - start, 1), error=str(error)[:300], tb=traceback.format_exc()[-1500:])
        print(stage, status[stage]["state"], status[stage]["seconds"], flush=True)
    mark("end")
    (out / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
