"""사용자 화면 GPU 시험 실행기(D-156, D-222 FLUX 한 경로): 생성만 가짜로 바꾸고 '이미지 생성' → 저장까지 화면 없이 도는지(CPU)."""
import importlib.util
import json
from pathlib import Path

from PIL import Image

from genai_lab import style_redraw_pipeline
from genai_lab import studio_free_generation as free_ui
from saved_image import saved_as
from test_studio_product import fake_flux

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "product_flow_run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("product_flow_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_clicks_generate_and_saves_each_case(tmp_path, monkeypatch):
    monkeypatch.delenv("GENAI_STUDIO_DEVELOPER", raising=False)
    character = tmp_path / "character.png"
    garment = tmp_path / "garment.png"
    Image.new("RGB", (8, 8), (30, 60, 90)).save(character)
    Image.new("RGB", (8, 8), (0, 0, 200)).save(garment)
    from genai_lab import model_check
    monkeypatch.setattr(model_check, "require_flux_models", lambda *a, **k: None)
    monkeypatch.setattr(free_ui, "analyze_base", lambda source, runtime: dict(source=str(source), character_scores={},
                                                                              groups=dict(fixed=[], appearance=[], body=[])))
    calls = []
    monkeypatch.setattr(style_redraw_pipeline, "run_style_redraw", fake_flux(calls))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([dict(case="A1", character=str(character), garment=None),
                                 dict(case="B1", character=str(character), garment=str(garment))]), encoding="utf-8")
    status = load_runner().main([str(cases), str(tmp_path / "out")])
    assert {k: v["state"] for k, v in status["cases"].items()} == dict(A1="ok", B1="ok")
    assert calls[0]["garment_path"] is None and calls[1]["garment_path"] == garment
    assert all(c["scene"] and "head to feet" in c["scene"] for c in calls)                 # 장면 자동 조합
    assert status["cases"]["A1"]["scene"]["source"] == "scene_composer"
    shown = Path(status["cases"]["B1"]["result"])                                           # 처음 보여 준 후보를 저장
    assert saved_as(tmp_path / "out" / "B1-saved.png", shown)
