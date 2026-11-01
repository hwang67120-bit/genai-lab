"""사용자 화면(D-151, D-222 FLUX 한 경로): 참조 이미지 + '이미지 생성' 한 번 + 생성 과정 미리보기 + 후보 3장 고르기 + 저장."""
import json
import os
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QCheckBox, QLineEdit

from genai_lab import style_redraw_pipeline
from genai_lab import studio_controller as ui, studio_free_generation as free_ui
from genai_lab.style_redraw_results import StyleRedrawResults
from test_studio_generation import wait_for


@pytest.fixture
def product_window(monkeypatch, tmp_path):
    monkeypatch.delenv("GENAI_STUDIO_DEVELOPER", raising=False)
    QApplication.instance() or QApplication([])
    from gui_main import GenAILabWindow
    window = GenAILabWindow()
    window.studio.output_root = tmp_path / "runs"
    from genai_lab import model_check
    monkeypatch.setattr(model_check, "require_flux_models", lambda *a, **k: None)   # 모델 검사는 test_model_check에서
    for name in ("confirm_outfit_edit", "confirm_inputs", "confirm_result"):
        monkeypatch.setattr(ui, name, lambda *a, **k: pytest.fail("사용자 화면은 확인 창을 띄우지 않는다"))
    monkeypatch.setattr(free_ui, "confirm_free_inputs", lambda *a, **k: pytest.fail("확인 창 없음"))
    monkeypatch.setattr(free_ui, "analyze_base", lambda source, runtime: dict(source=str(source), character_scores={"1boy": 0.9},
                                                                              groups=dict(fixed=[], appearance=[], body=[])))
    yield window
    window.close()


def fake_flux(calls, flags=None):
    """run_style_redraw 대신: 후보 3장 + generation.json을 만들고 검토 대기를 돌려준다."""
    def run(source, directory, runtime=None, **kw):
        calls.append(dict(kw, source=Path(source)))
        directory = Path(directory)
        directory.mkdir(parents=True)
        files, rows = [], []
        for i, seed in enumerate((42, 43, 44)):
            path = directory / f"result-seed{seed}.png"
            Image.new("RGB", (8, 8), (seed, i, 0)).save(path)
            files.append(str(path))
            rows.append(dict(seed=seed, file=path.name, extra_hands=(flags or {}).get(seed, False), missing_body_parts=[]))
        (directory / "generation.json").write_text(json.dumps(dict(candidates=rows)), encoding="utf-8")
        return dict(state="review_pending", candidates=files)
    return run


def start(w, tmp_path, monkeypatch, calls, flags=None, garment=None):
    character = tmp_path / "character.png"
    Image.new("RGB", (8, 8)).save(character)
    w.style_path = character
    w.selected_outfit_path = garment
    w.update_input_ready_status()
    w.findChild(QCheckBox, "product_adult").setChecked(True)           # 성인 확인(D-227)
    monkeypatch.setattr(style_redraw_pipeline, "run_style_redraw", fake_flux(calls, flags))
    failed = []
    monkeypatch.setattr(w.studio, "fail", lambda error, details="": failed.append(f"{error}\n{details}"))   # 오류 창 대신 기록
    w.generate_button.click()
    wait_for(QApplication.instance(), lambda: w.studio.task is None and (w.studio.results is not None or bool(failed)))
    assert not failed, failed[0]
    return character


def test_only_references_generate_and_preview_are_shown(product_window):
    w = product_window
    assert w.generate_button.text() == "이미지 생성"
    for widget in (w.approve_candidate_button, w.reject_candidate_button, w.discard_candidate_button, w.diagnostics_toggle,
                   w.free_generation_button.parentWidget(), w.studio_candidate_combo):
        assert widget.isHidden()
    assert not w.generate_button.isEnabled()
    w.style_path = Path("character.png")
    w.update_input_ready_status()
    assert not w.generate_button.isEnabled() and "성인" in w.status_label.text()          # 성인 확인 전(D-227)
    adult = w.findChild(QCheckBox, "product_adult")
    adult.setChecked(True)
    assert w.generate_button.isEnabled() and "이미지 생성" in w.status_label.text()       # 의상은 선택
    w.style_path = Path("other.png")                                                    # 다른 캐릭터 → 확인이 풀림
    w.update_input_ready_status()
    assert not adult.isChecked() and not w.generate_button.isEnabled()


def test_one_flux_step_with_scene_and_three_candidates_to_choose(product_window, tmp_path, monkeypatch):
    w = product_window
    calls = []
    character = start(w, tmp_path, monkeypatch, calls)
    call = calls[0]
    assert call["source"] == character and call["garment_path"] is None
    scene = json.loads((w.studio.run_directory / "scene.json").read_text(encoding="utf-8"))      # 장면 자동 조합(D-153)
    assert scene["source"] == "scene_composer" and call["scene"] == scene["text"] and "head to feet" in scene["text"]
    assert call["confirmed_gender"] is None and call["confirmed_body_parts"] is None   # 기억이 없으면 추정은 준비 단계가(D-198)
    assert call["adult_confirmed"] is True
    record = json.loads((w.studio.run_directory / "adult-confirmation.json").read_text(encoding="utf-8"))
    assert record["source"] == "user_checkbox" and record["character"] == str(character) and len(record["character_sha256"]) == 64
    assert isinstance(w.studio.results, StyleRedrawResults)
    combo = w.studio_candidate_combo
    assert not combo.isHidden() and combo.count() == 3 and "후보 3장" in w.status_label.text()
    combo.setCurrentIndex(2)
    assert w.studio.results.selected == 2 and w.studio.results.current_path.name == "result-seed44.png"
    assert w.save_candidate_button.isEnabled()
    saved = []
    monkeypatch.setattr(w.studio, "save", lambda: saved.append(w.studio.results.status))
    w.save_candidate_button.click()
    assert saved == ["user_approved"]


def test_garment_goes_to_the_same_flux_step_and_flagged_candidate_is_not_shown_first(product_window, tmp_path, monkeypatch):
    w = product_window
    garment = tmp_path / "garment.png"
    Image.new("RGB", (8, 8), (0, 0, 200)).save(garment)
    calls = []
    start(w, tmp_path, monkeypatch, calls, flags={42: True}, garment=garment)
    assert calls[0]["garment_path"] == garment
    assert w.studio.results.selected == 1 and "손이 원본보다" not in w.studio.results.head_notice   # 손이 늘어난 42는 처음에 안 보임
    w.studio_candidate_combo.setCurrentIndex(0)
    assert "손이 원본보다" in w.studio.results.head_notice


def test_refused_outcome_is_reported_not_shown(product_window, tmp_path, monkeypatch):
    w = product_window
    character = tmp_path / "character.png"
    Image.new("RGB", (8, 8)).save(character)
    w.style_path = character
    w.update_input_ready_status()
    w.findChild(QCheckBox, "product_adult").setChecked(True)
    monkeypatch.setattr(style_redraw_pipeline, "run_style_redraw",
                        lambda *a, **k: dict(state="refused", code="safety", message="이 그림은 진행할 수 없습니다."))
    failed = []
    monkeypatch.setattr(w.studio, "fail", lambda error, details="": failed.append(str(error)))
    w.generate_button.click()
    wait_for(QApplication.instance(), lambda: w.studio.task is None and bool(failed))
    assert failed == ["이 그림은 진행할 수 없습니다."] and w.studio.results is None


def test_cancel_reaches_the_flux_step_and_discards_the_run(product_window, tmp_path, monkeypatch):
    """D-224: '현재 작업 취소'가 FLUX 단계에 전달되고, 결과는 버리고, 오류 창 없이 '취소됨'."""
    w = product_window
    character = tmp_path / "character.png"
    Image.new("RGB", (8, 8)).save(character)
    w.style_path = character
    w.update_input_ready_status()
    w.findChild(QCheckBox, "product_adult").setChecked(True)
    started, seen = [], []
    def slow(source, directory, runtime=None, *, cancelled=None, **kw):
        started.append(True)
        import time
        t0 = time.time()
        while not cancelled() and time.time() - t0 < 20:
            time.sleep(0.05)
        seen.append(cancelled())
        return dict(state="cancelled", stage="generate", message="취소했습니다.")
    monkeypatch.setattr(style_redraw_pipeline, "run_style_redraw", slow)
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "exec", lambda self: pytest.fail("취소는 오류 창을 띄우지 않는다: " + self.text()))
    w.generate_button.click()
    wait_for(QApplication.instance(), lambda: bool(started))
    assert not w.studio_cancel_button.isHidden()
    w.studio_cancel_button.click()
    wait_for(QApplication.instance(), lambda: w.studio.task is None)
    assert seen == [True] and w.studio.results is None
    assert "취소" in w.status_label.text() and w.generate_button.isEnabled()


def test_preview_shows_the_newest_candidate_and_progress(product_window, tmp_path):
    w = product_window
    w.studio.run_directory = tmp_path / "run"
    flux = w.studio.run_directory / "flux"
    flux.mkdir(parents=True)
    Image.new("RGB", (16, 20), (200, 0, 0)).save(flux / "result-raw-seed42.png")
    (flux / "progress.json").write_text(json.dumps(dict(candidate=2, candidates=3)), encoding="utf-8")
    w.studio.product.poll_preview()
    assert not w.candidate_preview._original.isNull() and "후보 2/3" in w.status_label.text()
    (flux / "progress.json").write_text(json.dumps(dict(candidate=1, candidates=3, phase="autofix")), encoding="utf-8")
    w.studio.product.poll_preview()
    assert "자동 보정" in w.status_label.text() and "후보 1/3" in w.status_label.text()      # D-241 빠진 특징 다시 그리기


def test_autofixed_candidate_says_what_was_asked_again(tmp_path):
    outcome = fake_flux([])(tmp_path / "c.png", tmp_path / "run")
    generation = tmp_path / "run" / "generation.json"
    record = json.loads(generation.read_text(encoding="utf-8"))
    for c in record["candidates"]:
        c["autofix"] = dict(round=1, asked=["furry", "animal_nose"])
    generation.write_text(json.dumps(record), encoding="utf-8")
    results = StyleRedrawResults(outcome)          # generation.json의 자동 보정 기록을 후보로 옮겨 온다(실제 화면 시험에서 빠졌음)
    assert "자동 보정: 빠진 특징(furry, animal nose)" in results.head_notice
    results.candidates[results.selected]["autofix"] = dict(round=0, asked=[])
    assert "자동 보정" not in results.head_notice


def test_saved_by_user_is_not_a_quality_judgement(tmp_path):
    directory = tmp_path / "run"
    calls = []
    outcome = fake_flux(calls)(tmp_path / "c.png", directory)
    results = StyleRedrawResults(outcome)
    results.approve_saved_by_user()
    review = json.loads((directory / "user-review.json").read_text(encoding="utf-8"))
    assert results.status == "user_approved" and review["checks"] == dict(saved_by_user=True)
    saved = results.export(tmp_path / "out.png")
    meta = json.loads(saved.with_suffix(".review.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "style_redraw" and meta["engine"] == "flux" and meta["ai_label"]
    with pytest.raises(FileExistsError):
        results.status = "user_approved"
        results.export(tmp_path / "out.png")


def test_scene_field_feeds_the_request_and_blocks_appearance_changes(product_window, tmp_path, monkeypatch):
    w = product_window
    character = tmp_path / "character.png"
    Image.new("RGB", (8, 8)).save(character)
    w.style_path = character
    w.update_input_ready_status()
    field = w.findChild(QLineEdit, "product_scene")
    field.setText("머리색을 금발로 바꿔")
    assert not w.generate_button.isEnabled() and "자세·배경·상황" in w.studio.product.scene_note.text()
    field.setText("jumping happily")
    w.findChild(QCheckBox, "product_adult").setChecked(True)
    assert w.generate_button.isEnabled()
    calls = []
    monkeypatch.setattr(style_redraw_pipeline, "run_style_redraw", fake_flux(calls))
    w.generate_button.click()
    wait_for(QApplication.instance(), lambda: w.studio.task is None and w.studio.results is not None)
    assert calls[0]["scene"].startswith("jumping happily. Setting: ")       # 배경 말이 없어 장소·빛을 덧붙임
