"""사용자 화면 GPU 시험 실행기(D-156): 생성만 가짜로 바꾸고 '이미지 생성' → 저장까지 화면 없이 도는지(CPU)."""
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from genai_lab import free_generation_pipeline, free_generation_results, outfit_edit_pipeline, qwen_outfit_edit
from genai_lab import studio_free_generation as free_ui, studio_generation
from saved_image import saved_as
from test_studio_generation import analysis_at
from test_studio_outfit_edit import fake_outcome
from test_studio_product import FakeResults

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "product_flow_run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("product_flow_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_clicks_generate_and_saves_each_case(tmp_path, monkeypatch):
    monkeypatch.delenv("GENAI_STUDIO_DEVELOPER", raising=False)
    analysis = analysis_at(tmp_path / "inputs")
    character = Path(analysis["source"])
    garment = Path(analysis["directory"]) / "garment.png"
    monkeypatch.setattr(qwen_outfit_edit, "load_qwen_settings", lambda path: object())
    from genai_lab import model_check
    monkeypatch.setattr(model_check, "require_product_models", lambda *a, **k: None)
    monkeypatch.setattr(studio_generation, "analyze_inputs", lambda *a, **k: analysis)
    monkeypatch.setattr(free_ui, "analyze_base", lambda source, runtime: dict(source=str(source), character_scores={},
                                                                              groups=dict(fixed=[], appearance=[], body=[])))
    monkeypatch.setattr(outfit_edit_pipeline, "run_outfit_pipeline", lambda c, g, **kw: fake_outcome(kw["directory"]))
    final = tmp_path / "final.png"
    Image.new("RGB", (8, 8), (30, 60, 90)).save(final)
    requests = []
    monkeypatch.setattr(free_generation_pipeline, "run_free_generation",
                        lambda base, features, request_text="", **kw: requests.append((Path(base).name, request_text)) or
                        dict(status="awaiting_user_review", result=str(final)))

    class SavingResults(FakeResults):
        def export(self, destination):
            from genai_lab.ai_label import label_record, labeled_png
            Path(destination).write_bytes(labeled_png(Path(self.outcome["result"]).read_bytes(), label_record("generated")))
            return Path(destination)

    monkeypatch.setattr(free_generation_results, "FreeGenerationResults", SavingResults)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([dict(case="A1", character=str(character), garment=None),
                                 dict(case="B1", character=str(character), garment=str(garment))]), encoding="utf-8")
    status = load_runner().main([str(cases), str(tmp_path / "out")])
    assert {k: v["state"] for k, v in status["cases"].items()} == dict(A1="ok", B1="ok")
    assert requests[0][0] == character.name and requests[1][0] == "edit-result.png"
    assert all(text and "head to feet" in text for _, text in requests)                 # 장면 자동 조합
    assert status["cases"]["A1"]["scene"]["source"] == "scene_composer"
    assert saved_as(tmp_path / "out" / "B1-saved.png", final)
    assert json.loads((tmp_path / "out" / "batch-status.json").read_text(encoding="utf-8"))["state"] == "finished"
