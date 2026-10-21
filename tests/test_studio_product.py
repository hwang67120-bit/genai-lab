"""사용자 화면(D-151): 참조 이미지 + '이미지 생성' 한 번 + 생성 과정 미리보기. 확인 창 없이 ①→② 자동, 기록은 정직하게."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from genai_lab import free_generation_pipeline, free_generation_results, outfit_edit_pipeline, qwen_outfit_edit
from genai_lab import studio_controller as ui, studio_free_generation as free_ui, studio_generation
from test_studio_generation import analysis_at, wait_for
from test_studio_outfit_edit import fake_outcome


@pytest.fixture
def product_window(monkeypatch, tmp_path):
    monkeypatch.delenv("GENAI_STUDIO_DEVELOPER", raising=False)
    QApplication.instance() or QApplication([])
    from gui_main import GenAILabWindow
    window = GenAILabWindow()
    window.studio.output_root = tmp_path / "runs"
    monkeypatch.setattr(qwen_outfit_edit, "load_qwen_settings", lambda path: object())
    for name in ("confirm_outfit_edit", "confirm_inputs", "confirm_result"):
        monkeypatch.setattr(ui, name, lambda *a, **k: pytest.fail("사용자 화면은 확인 창을 띄우지 않는다"))
    monkeypatch.setattr(free_ui, "confirm_free_inputs", lambda *a, **k: pytest.fail("확인 창 없음"))
    yield window
    window.close()


class FakeResults(free_generation_results.FreeGenerationResults):
    """실제 결과 클래스의 화면용 속성은 그대로 물려받고, 생성 기록 검증만 건너뛴다."""

    def __init__(self, outcome):
        self.outcome, self.result_path, self.status = outcome, Path(outcome["result"]), "awaiting_user_review"
        self.candidates, self.selected = [dict(attempt=1, raw=outcome["result"], sha256="x")], 0
        self.directory = self.result_path.parent            # OutfitEditResults.__init__가 두는 속성들
        self.batch = SimpleNamespace(directory=self.directory, candidates=(None,))
        self.composited, self.raw_path, self.appendage_review_required = False, self.result_path, False
        self.expected_sha = self.approved_sha = None
        self.checks, self.quick_preview = {}, False

    def approve_saved_by_user(self):
        self.status = "user_approved"


def fake_free(calls, tmp_path):
    def run(base, features, request_text="", **kw):
        calls.append(dict(kw, base=Path(base), features=features, request_text=request_text))
        result = tmp_path / "final.png"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(result)
        return dict(status="awaiting_user_review", result=str(result))
    return run


def test_only_references_generate_and_preview_are_shown(product_window):
    w = product_window
    assert w.generate_button.text() == "이미지 생성"
    for widget in (w.approve_candidate_button, w.reject_candidate_button, w.discard_candidate_button, w.diagnostics_toggle,
                   w.free_generation_button.parentWidget(), w.studio_candidate_combo):
        assert widget.isHidden()
    assert not w.generate_button.isEnabled()
    w.style_path = Path("character.png")
    w.update_input_ready_status()
    assert w.generate_button.isEnabled() and "이미지 생성" in w.status_label.text()       # 의상은 선택


def test_with_outfit_runs_edit_then_staging_automatically(product_window, tmp_path, monkeypatch):
    w = product_window
    analysis = analysis_at(tmp_path / "inputs")
    w.style_path = Path(analysis["source"])
    w.selected_outfit_path = Path(analysis["directory"]) / "garment.png"
    w.update_input_ready_status()
    asked = []
    monkeypatch.setattr(studio_generation, "analyze_inputs", lambda *a, **k: asked.append(k.get("require_face")) or analysis)
    edits = []
    monkeypatch.setattr(outfit_edit_pipeline, "run_outfit_pipeline",
                        lambda c, g, **kw: edits.append(kw) or fake_outcome(kw["directory"]))
    calls = []
    monkeypatch.setattr(free_generation_pipeline, "run_free_generation", fake_free(calls, tmp_path))
    monkeypatch.setattr(free_generation_results, "FreeGenerationResults", FakeResults)
    w.generate_button.click()
    wait_for(QApplication.instance(), lambda: w.studio.task is None and w.studio.results is not None)
    assert asked == [False] and edits[0]["provenance"]["gender"]["source"] == "automatic_product"
    call = calls[0]
    assert call["base"].name == "edit-result.png" and call["base_kind"] == "edited" and call["request_text"] == ""
    assert call["automatic_edit_base"] is True and call["confirmation_source"] == "automatic_product"
    record = json.loads(call["base"].with_suffix(".review.json").read_text(encoding="utf-8"))
    assert record["reviewer"] == "automatic_chain" and record["sha256"] == hashlib.sha256(call["base"].read_bytes()).hexdigest()
    assert w.save_candidate_button.isEnabled() and "완성" in w.status_label.text()
    saved = []
    monkeypatch.setattr(w.studio, "save", lambda: saved.append(w.studio.results.status))
    w.save_candidate_button.click()
    assert saved == ["user_approved"]


def test_without_outfit_keeps_the_outfit_and_makes_pose_and_scene(product_window, tmp_path, monkeypatch):
    w = product_window
    character = tmp_path / "character.png"
    Image.new("RGB", (8, 8)).save(character)
    w.style_path = character
    w.update_input_ready_status()
    monkeypatch.setattr(free_ui, "analyze_base", lambda source, runtime: dict(source=str(source), character_scores={"1boy": 0.9},
                                                                              groups=dict(fixed=[], appearance=[], body=[])))
    calls = []
    monkeypatch.setattr(free_generation_pipeline, "run_free_generation", fake_free(calls, tmp_path))
    monkeypatch.setattr(free_generation_results, "FreeGenerationResults", FakeResults)
    w.generate_button.click()
    wait_for(QApplication.instance(), lambda: w.studio.task is None and w.studio.results is not None)
    call = calls[0]
    assert call["base"] == character and call["base_kind"] == "original" and call["automatic_edit_base"] is False
    assert call["gender"] == "male" and set(call["features"]) >= {"hair_color", "distinctive"}


def test_preview_shows_the_newest_step_image(product_window, tmp_path):
    w = product_window
    w.studio.run_directory = tmp_path / "run"
    qwen = w.studio.run_directory / "free-generation" / "qwen"
    qwen.mkdir(parents=True)
    Image.new("RGB", (16, 20), (200, 0, 0)).save(qwen / "preview.png")
    w.studio.product.poll_preview()
    assert not w.candidate_preview._original.isNull()


def test_saved_by_user_is_not_a_quality_judgement(tmp_path):
    results = free_generation_results.FreeGenerationResults.__new__(free_generation_results.FreeGenerationResults)
    recorded = []
    results.verify_current = lambda: "sha"
    results.persist = lambda **k: None
    results.record_human = lambda values, source: recorded.append((values, source))
    results.approve_saved_by_user()
    assert results.status == "user_approved" and results.checks == dict(saved_by_user=True)
    assert recorded and recorded[0][1] == "product_save" and "quality_pass" not in recorded[0][0]
