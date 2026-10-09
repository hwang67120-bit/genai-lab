"""CPU integration: same selected raw, mandatory IP, reversible review and tail coordinates."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
from genai_lab import studio_optional_finishing as optional
from genai_lab import studio_finishing as finishing
from genai_lab import studio_background as background
from genai_lab import studio_controller as ui
from genai_lab.studio_generation import StudioResults, StudioRuntime
from genai_lab.studio_finishing_gui import FinishingComparisonDialog, choose_finished
from genai_lab.onepass_generation import OnePassCancelled
from genai_lab.qwen_record_io import write_json
from genai_lab.proportion_inputs import sha, json_value
from gui_main import GenAILabWindow
from test_studio_finishing import batch_at, Backend, Foreground
from test_studio_generation import wait_for


def inputs(tmp_path, count=4):
    batch = batch_at(tmp_path / "generation", count)
    runtime = StudioRuntime()
    face = tmp_path / "face.png"
    Image.new("RGB", (64,64), (21,44,72)).save(face)
    for c in batch.candidates:
        data = json.loads(c.record_path.read_text(encoding="utf-8"))
        data.update(inputs={"face_file":str(face), "face_sha256":sha(face)},
                    settings=json_value(asdict(runtime.generation)), models=runtime.generation.model_record())
        write_json(c.record_path, data)
        background.prepare_candidate(c, Foreground(), background.BackgroundOptions(Path("unused")))
    return batch, runtime


def mock_models(monkeypatch):
    seen = []
    class FaceBackend(Backend):
        def set_face_reference(self, reference): self.reference = reference
        def refine(self, image, seed, cancelled):
            seen.append((seed, image.size))
            output, metrics = super().refine(image, seed, cancelled)
            output.putpixel((0,0), (7,8,9))
            return output, {**metrics, "ip_scale":.9}
    def finish(batch, prompt, runtime, **kwargs):
        assert len(batch.candidates) == 1
        assert runtime.finishing and runtime.finishing_face_reference
        def factory(settings, *, face_reference):
            assert face_reference is True
            return FaceBackend(settings)
        return finishing.finish_batch(batch, prompt, runtime, **kwargs, backend_factory=factory,
            detector=lambda *_:[([40,20,140,160], "head", .9)])
    def white(batch, model_cache, **kwargs):
        assert kwargs["finishing"] is True
        for c in batch.candidates:
            source, _ = finishing.finishing_source(c)
            background.prepare_candidate(c, Foreground(), background.BackgroundOptions(Path("unused")), source_file=source)
        return batch
    monkeypatch.setattr(optional, "finish_batch", finish)
    monkeypatch.setattr(optional, "prepare_backgrounds", white)
    return seen


def run_optional(tmp_path, monkeypatch, count=4):
    batch, runtime = inputs(tmp_path, count)
    result = StudioResults(batch)
    result.select(count-1)
    seen = mock_models(monkeypatch)
    info = optional.finish_selected(result.candidate, result.basis_path, result.verify_basis(),
        tmp_path / "optional", runtime)
    return result, info, seen


def checks(): return dict(character=True, garment=True, exposure=True)


def test_single_selected_raw_unchanged_prompt_and_face_reference(tmp_path, monkeypatch):
    result, info, seen = run_optional(tmp_path, monkeypatch)
    assert seen == [(4,(1104,1848)), (4,(1024,1024))]
    assert len(list((tmp_path / "optional").glob("seed-*"))) == 1
    copied = Path(info["candidate_file"])
    assert copied.read_bytes() == result.raw_path.read_bytes()
    assert Path(info["candidate_record"]).read_bytes() == result.candidate.record_path.read_bytes()
    assert not any(c.path.with_name("finishing.json").exists() for c in result.batch.candidates)
    record = info["finishing"]
    assert record["face_reference"] == "on" and record["ip_scale"] == .9
    assert record["contract"]["strength"] == .35 and record["contract"]["scale"] == 1.5
    assert record["contract"]["steps"] == 28 and record["contract"]["max_reserved_gib"] == 6.5
    assert record["prompt"] == json.loads(result.candidate.record_path.read_text())["prompt"]
    assert result.optional_finishing is None  # Completing an operation is not adoption.
    assert StudioRuntime().finishing is False and StudioRuntime().quality_tags is False


@pytest.mark.parametrize("adopt", [False, True])
def test_selection_export_sha_and_audit(tmp_path, monkeypatch, adopt):
    result, info, _ = run_optional(tmp_path, monkeypatch)
    original = result.current_path
    if adopt: result.adopt_finishing(info["directory"])
    result.approve(checks())
    target = result.export(tmp_path / "saved.png")
    expected = Path(info["product_file"]) if adopt else original
    assert target.read_bytes() == expected.read_bytes()
    audit = json.loads(target.with_suffix(".review.json").read_text())
    assert audit["finishing_applied"] is adopt and audit["sha256"] == sha(expected)
    if adopt:
        assert audit["finishing"]["face_reference"] == "on"
        assert audit["finishing"]["stages"]["hires"]["sha256"]
        assert audit["finishing"]["stages"]["face"]["max_reserved_bytes"] == 123
        assert audit["finishing"]["seconds"] > 0


@pytest.mark.parametrize("failure", ["cancel", "backend", "background"])
def test_incomplete_optional_does_not_replace_original(tmp_path, monkeypatch, failure):
    batch, runtime = inputs(tmp_path, 1)
    result = StudioResults(batch)
    result.approve(checks())
    previous = (result.current_path.read_bytes(), result.approved_sha, result.status)
    mock_models(monkeypatch)
    if failure == "backend":
        monkeypatch.setattr(optional, "finish_batch", lambda *a,**k: (_ for _ in ()).throw(RuntimeError("backend failed")))
    if failure == "background":
        monkeypatch.setattr(optional, "prepare_backgrounds", lambda *a,**k: None)
    with pytest.raises((OnePassCancelled, RuntimeError, ValueError)):
        optional.finish_selected(result.candidate, result.basis_path, result.verify_basis(), tmp_path/"failed", runtime,
                                 cancelled=lambda: failure == "cancel")
    state = json.loads((tmp_path/"failed/optional-finishing.json").read_text())
    assert state["status"] == ("cancelled" if failure == "cancel" else "failed")
    with pytest.raises((KeyError, ValueError)):
        result.adopt_finishing(tmp_path/"failed")
    assert (result.current_path.read_bytes(), result.approved_sha, result.status) == previous


@pytest.mark.parametrize("changed", ["product", "record", "source"])
def test_changed_artifact_rejected_before_adoption(tmp_path, monkeypatch, changed):
    result, info, _ = run_optional(tmp_path, monkeypatch, 1)
    if changed == "product": Path(info["product_file"]).write_bytes(b"changed")
    elif changed == "source": result.raw_path.write_bytes(b"changed")
    else:
        path = Path(info["candidate_file"]).with_name("finishing.json")
        data = json.loads(path.read_text()); data["ip_scale"] = 0
        write_json(path,data)
    with pytest.raises(ValueError): result.adopt_finishing(info["directory"])
    assert result.optional_finishing is None


def test_adoption_resets_approval_and_failed_save_rolls_back(tmp_path, monkeypatch):
    result, info, _ = run_optional(tmp_path, monkeypatch, 1)
    result.approve(checks())
    previous = result.approved_sha
    with monkeypatch.context() as patch:
        patch.setattr(result,"persist",lambda **k: (_ for _ in ()).throw(OSError("record unavailable")))
        with pytest.raises(OSError): result.adopt_finishing(info["directory"])
    assert result.optional_finishing is None and result.approved_sha == previous
    result.adopt_finishing(info["directory"])
    assert result.approved_sha is None and result.status == "awaiting_user_review"
    with pytest.raises(ValueError): result.export(tmp_path/"unapproved.png")
    result.select(0)
    assert result.optional_finishing is None and result.basis_path != Path(info["product_file"])


def test_dialog_defaults_original_and_face_zoom_is_displayed(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    result, info, _ = run_optional(tmp_path, monkeypatch, 1)
    dialog = FinishingComparisonDialog(result.basis_path, info)
    assert dialog.original.isChecked() and not dialog.finished.isChecked()
    from PySide6.QtWidgets import QLabel
    assert len([w for w in dialog.findChildren(QLabel) if not w.pixmap().isNull()]) == 4
    monkeypatch.setattr(QDialog,"exec",lambda self:QDialog.DialogCode.Accepted)
    assert choose_finished(None,result.basis_path,info) is False
    def choose(self):
        self.finished.setChecked(True)
        assert not self.original.isChecked()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog,"exec",choose)
    assert choose_finished(None,result.basis_path,info) is True
    dialog.close()


def test_tail_edit_uses_finished_dimensions_and_disables_reverse(tmp_path, monkeypatch):
    from test_qwen_tail import assets, finished_edit
    result, info, _ = run_optional(tmp_path, monkeypatch, 1)
    result.adopt_finishing(info["directory"])
    with Image.open(result.basis_path) as img: assert img.size == (1104,1848)
    source, _ = assets(tmp_path/"source")
    directory, product, request = finished_edit(tmp_path/"tail",source=source,basis=result.basis_path)
    result.adopt_tail_edit(directory)
    assert request["tail_spec"]["image_sha256"] == info["product_sha256"]
    with Image.open(result.current_path) as img: assert img.size == (1104,1848)
    assert "꼬리를 고친 뒤" in result.finishing_unavailable_reason
    result.approve(checks())
    saved = result.export(tmp_path/"tail-saved.png")
    assert saved.read_bytes() == product.read_bytes()
    assert json.loads(saved.with_suffix(".review.json").read_text())["finishing_applied"] is True


@pytest.mark.parametrize("outcome", ["original", "adopt", "cancel", "failure"])
def test_real_button_worker_selection_and_save(tmp_path, monkeypatch, outcome):
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    batch, runtime = inputs(tmp_path)
    controller = window.studio
    controller.runtime = runtime
    seen = mock_models(monkeypatch)
    controller.generated(batch)
    window.studio_candidate_combo.setCurrentIndex(2)
    original = controller.results.current_path
    controller.results.approve(checks())
    prior_sha = controller.results.approved_sha
    monkeypatch.setattr(ui,"choose_finished",lambda *a: outcome == "adopt")
    monkeypatch.setattr(QMessageBox,"warning",lambda *a: None)
    actual = optional.finish_selected
    if outcome in ("cancel", "failure"):
        def interrupted(*args, **kwargs):
            if outcome == "failure": raise RuntimeError("test failure")
            while not kwargs["cancelled"](): time.sleep(.01)
            raise OnePassCancelled("test cancel")
        monkeypatch.setattr(ui,"finish_selected",interrupted)
    else: monkeypatch.setattr(ui,"finish_selected",actual)
    try:
        assert window.studio_finish_button.isEnabled()
        window.studio_finish_button.click()
        assert controller.task is not None and not window.studio_finish_button.isEnabled()
        assert not window.studio_candidate_combo.isEnabled()
        if outcome == "cancel": window.studio_cancel_button.click()
        wait_for(app,lambda:controller.task is None)
        assert controller.results is not None and controller.results.selected == 2
        assert controller.finishing_context is None
        if outcome == "adopt":
            assert seen == [(3,(1104,1848)),(3,(1024,1024))]
            assert controller.results.current_path != original and controller.results.approved_sha is None
            assert not window.studio_finish_button.isEnabled()
        else:
            assert controller.results.current_path == original
            assert controller.results.approved_sha == prior_sha
        controller.results.approve(checks())
        controller.controls()
        destination = tmp_path/"gui-saved.png"
        expected = controller.results.current_path.read_bytes()
        monkeypatch.setattr(QFileDialog,"getSaveFileName",lambda *a:(str(destination),""))
        window.save_candidate_button.click()
        assert destination.read_bytes() == expected
        assert not window.studio_finish_button.isEnabled()
    finally: window.close()


def test_two_pass_rejected_and_disabled(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    batch, _ = inputs(tmp_path,1)
    try:
        assert not window.studio_finish_button.isEnabled()
        window.studio.generated(batch)
        window.studio.results.two_pass = True
        window.studio.controls()
        assert not window.studio_finish_button.isEnabled()
        assert "원본 비율" in window.studio_finish_notice.text()
        window.studio.finish_selected_image()
        assert window.studio.task is None
    finally: window.close()


@pytest.mark.parametrize("changed", ["model", "settings", "face"])
def test_mismatched_generation_conditions_stop_before_model_load(tmp_path, monkeypatch, changed):
    batch, runtime = inputs(tmp_path,1)
    result = StudioResults(batch)
    c = result.candidate
    data = json.loads(c.record_path.read_text())
    if changed == "model": data["models"]["base"]["revision"] = "different"
    elif changed == "settings": data["settings"]["guidance_scale"] = 6.0
    else: (tmp_path/"face.png").write_bytes(b"changed")
    write_json(c.record_path,data)
    monkeypatch.setattr(optional,"finish_batch",lambda *a,**k:pytest.fail("model load forbidden"))
    with pytest.raises(ValueError):
        optional.finish_selected(c,result.basis_path,result.verify_basis(),tmp_path/"blocked",runtime)
    assert json.loads((tmp_path/"blocked/optional-finishing.json").read_text())["status"] == "failed"


def test_cancellation_after_partial_output_keeps_original(tmp_path, monkeypatch):
    batch, runtime = inputs(tmp_path,1)
    result = StudioResults(batch)
    prior = result.current_path.read_bytes()
    cancelled = [False]
    mock_models(monkeypatch)
    actual = optional.finish_batch
    def interrupt(*a,**k):
        output = actual(*a,**k)
        cancelled[0] = True
        return output
    monkeypatch.setattr(optional,"finish_batch",interrupt)
    with pytest.raises(OnePassCancelled):
        optional.finish_selected(result.candidate,result.basis_path,result.verify_basis(),tmp_path/"interrupted",
                                 runtime,cancelled=lambda:cancelled[0])
    assert (tmp_path/"interrupted/seed-1/finished.png").is_file()
    assert json.loads((tmp_path/"interrupted/optional-finishing.json").read_text())["status"] == "cancelled"
    assert result.current_path.read_bytes() == prior and result.optional_finishing is None
