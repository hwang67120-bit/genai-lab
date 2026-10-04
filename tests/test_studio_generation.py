"""Actual GUI signals -> prompt/request -> raw batch -> explicit review -> export, CPU only."""
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from genai_lab import studio_controller as ui
from genai_lab import studio_generation as service
from genai_lab import onepass_generation as engine
from genai_lab.studio_analysis import classify_character
from gui_main import GenAILabWindow


class Tokens:
    bos_token_id = 1000
    eos_token_id = 1001
    pad_token_id = 1001
    def __call__(self, text, **kwargs):
        return {"input_ids": [sum(map(ord, w)) for w in text.split()]}


def analysis_at(path):
    path.mkdir(parents=True, exist_ok=True)
    for name, color in (("character", "red"), ("garment", "blue"), ("face", "white")):
        Image.new("RGB", (16, 24), color).save(path / f"{name}.png")
    return {"directory": str(path), "source": str(path / "character.png"),
            "groups": {"appearance": ["blue_hair"], "body": ["medium_breasts"], "fixed": ["raccoon_ears"]},
            "slim": False, "garment_tags": ["white camisole", "black shorts"],
            "face_sha256": service.digest(path / "face.png"), "references": {}}


def request_at(tmp_path, analysis=None, **kwargs):
    analysis = analysis or analysis_at(tmp_path / "inputs")
    store = QSettings(str(tmp_path / "test-only.ini"), QSettings.Format.IniFormat)
    runtime = service.StudioRuntime(generation=replace(service.OnePassGenerationSettings(), width=16, height=24))
    return service.build_request(analysis, analysis["garment_tags"], "male", confirmed=True,
        runtime=runtime, preferences=store, tokenizers=(Tokens(), Tokens()), seeds=(1,2,3,4), **kwargs)


class Backend:
    callback_mode = "legacy_callback"
    pipe = SimpleNamespace(model_cpu_offload_seq="test")
    def __init__(self, settings):
        self.settings = settings
        self.closed = False
    def generate(self, inputs, images, seed, observation, cancelled):
        assert inputs.pose_mode == "without_pose"
        assert inputs.control_file is None and inputs.control_sha256 is None
        for i in range(28):
            module = SimpleNamespace(attn_processors={"ip": SimpleNamespace(scale=[0.0 if i < 11 else .9])})
            observation(module, (), {"down_intrablock_additional_residuals": None}, None)
        return Image.new("RGB", (16,24), (seed,20,30)), {"max_memory_reserved_bytes": 1, "generation_seconds": .1}
    def close(self):
        self.closed = True


def batch_at(tmp_path):
    request = request_at(tmp_path)
    return engine.generate_onepass_request(request, tmp_path / "generated", backend_factory=Backend)


def wait_for(app, condition):
    until = time.monotonic()+10
    while not condition():
        app.processEvents()
        if time.monotonic() > until:
            pytest.fail("GUI task did not finish")
        time.sleep(.01)
    app.processEvents()


def test_create_button_to_review_and_export_without_legacy_fallback(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = GenAILabWindow()
    w.studio.output_root = tmp_path / "runs"
    initial = analysis_at(tmp_path / "selected")
    w.style_path = Path(initial["source"])
    w.selected_outfit_path = Path(initial["directory"])/"garment.png"
    w.update_input_ready_status()
    monkeypatch.setattr(ui, "validate_local_models", lambda _: None)
    monkeypatch.setattr(ui, "analyze_inputs", lambda c,g,d,r,**kw: analysis_at(d))
    monkeypatch.setattr(ui, "confirm_inputs", lambda *a: ("male", ("white camisole", "black shorts")))
    monkeypatch.setattr(ui, "build_request", lambda a,*args,**kw: request_at(tmp_path, a))
    calls = []
    def generate(req, folder, **kw):
        calls.append(req)
        return engine.generate_onepass_request(req, folder, backend_factory=Backend, **kw)
    monkeypatch.setattr(ui, "generate_onepass_request", generate)
    monkeypatch.setattr(w, "start_legacy_generation", lambda: pytest.fail("legacy must not run"))
    monkeypatch.setattr(w, "start_native_refinement", lambda *a: pytest.fail("refinement must not run"))
    monkeypatch.setattr(ui, "confirm_result", lambda *a: {"character":True,"garment":True,"exposure":True})
    destination = tmp_path/"chosen.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
    try:
        w.generate_button.click()
        assert not w.style_button.isEnabled()
        wait_for(app, lambda: w.studio.task is None and w.studio.results is not None)
        assert len(calls)==1 and calls[0].inputs.pose_mode == "without_pose"
        assert "white camisole" in calls[0].inputs.prompt.positive
        assert "raccoon ears" in calls[0].inputs.prompt.positive
        assert len(w.studio.results.batch.candidates)==4
        assert not w.save_candidate_button.isEnabled()
        w.studio_candidate_combo.setCurrentIndex(2)
        selected = w.studio.results.candidate
        w.approve_candidate_button.click()
        assert w.save_candidate_button.isEnabled()
        w.studio_candidate_combo.setCurrentIndex(1)
        assert not w.save_candidate_button.isEnabled()  # Choice change invalidates approval.
        w.studio_candidate_combo.setCurrentIndex(2)
        w.approve_candidate_button.click()
        w.save_candidate_button.click()
        assert destination.read_bytes() == selected.path.read_bytes()
        assert "저장 완료" in w.status_label.text()
        raw = json.loads(selected.record_path.read_text())
        assert raw["gates_executed"] is False and raw["final_return_eligible"] is False
        assert raw["adapter_applied_calls"]==[]
        assert len(raw["unet_calls"])==28
        ledger = json.loads((selected.path.parents[1]/"user-review.json").read_text())
        assert ledger["status"]=="saved" and all(ledger["checks"].values())
        assert w.generate_button.isEnabled()
    finally:
        w.close()


def test_export_requires_explicit_review_and_unchanged_bytes(tmp_path):
    results = service.StudioResults(batch_at(tmp_path))
    with pytest.raises(ValueError):
        results.export(tmp_path/"not-approved.png")
    with pytest.raises(ValueError):
        results.approve({"character": True})
    results.approve(dict(character=True,garment=True,exposure=True))
    results.candidate.path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        results.export(tmp_path/"changed.png")
    assert not (tmp_path/"changed.png").exists()


def test_export_does_not_overwrite_existing_file(tmp_path):
    results = service.StudioResults(batch_at(tmp_path))
    results.approve(dict(character=True,garment=True,exposure=True))
    path = tmp_path/"existing.png"
    path.write_bytes(b"user file")
    with pytest.raises(FileExistsError):
        results.export(path)
    assert path.read_bytes()==b"user file"


@pytest.mark.parametrize("error", [RuntimeError("generation failed"), engine.OnePassCancelled("cancel")])
def test_failure_and_cancel_never_offer_base_as_completed(tmp_path, monkeypatch, error):
    app = QApplication.instance() or QApplication([])
    w = GenAILabWindow()
    monkeypatch.setattr(QMessageBox, "exec", lambda *a: 0)
    w.studio.run_directory = tmp_path
    def fail(cancel, progress):
        raise error
    try:
        w.studio.launch(fail, lambda result: pytest.fail("No completed result expected"))
        wait_for(app, lambda: w.studio.task is None)
        assert not w.save_candidate_button.isEnabled()
        assert not w.approve_candidate_button.isEnabled()
        assert w.studio.results is None
        state = json.loads((tmp_path/"gui-status.json").read_text())
        assert state["status"] in ("failed", "cancelled")
        assert "5/5" not in w.status_label.text()
    finally:
        w.close()


def test_gender_is_explicit_and_character_classification_excludes_clothing(tmp_path):
    groups = classify_character((("blue_hair", .9),("small_breasts", .8),("raccoon_ears", .7)))
    assert groups == {"appearance":["blue_hair"], "body":["small_breasts"], "fixed":["raccoon_ears"]}
    a = analysis_at(tmp_path/"inputs")
    with pytest.raises(ValueError):
        service.build_request(a,a["garment_tags"],"male",confirmed=False,runtime=service.StudioRuntime())


def test_cpu_analysis_uses_explicit_offline_processes(tmp_path, monkeypatch):
    a = analysis_at(tmp_path/"source")
    exe = tmp_path/"python.exe"
    exe.touch()
    runtime = service.StudioRuntime(pose_python=exe, analysis_python=exe)
    commands=[]
    def run(cmd, log, cancelled, timeout):
        commands.append(cmd)
        if cmd[2]=="features":
            target=Path(cmd[3])
            service.record(target/"analysis.json", {"face_sha256":"abc", "groups":{}})
    monkeypatch.setattr(service,"cpu_process",run)
    result=service.analyze_inputs(a["source"],Path(a["directory"])/"garment.png",tmp_path/"analysis",runtime)
    assert [c[2] for c in commands]==["pose","features"]
    assert result["references"]["character"]["sha256"]==service.digest(a["source"])
    assert (tmp_path/"analysis/character.png").is_file()


def test_legacy_refinement_error_is_not_final_review(tmp_path, monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    monkeypatch.setattr(QMessageBox,"exec",lambda *a:0)
    w.pending_native_base_candidate=object()
    try:
        w.native_refinement_failed("failed", "test")
        assert w.pending_character_candidate is None
        assert not w.approve_candidate_button.isEnabled()
        assert not w.save_candidate_button.isEnabled()
        assert "실패" in w.status_label.text()
    finally:
        w.close()


def test_user_cancels_input_confirmation_no_generation(tmp_path, monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    monkeypatch.setattr(ui,"confirm_inputs",lambda *a:None)
    monkeypatch.setattr(ui,"generate_onepass_request",lambda *a,**kw:pytest.fail("No approval"))
    try:
        w.studio.inputs_ready(analysis_at(tmp_path/"inputs"))
        assert w.studio.task is None and w.studio.results is None
        assert "취소" in w.status_label.text()
    finally:
        w.close()


def test_cancel_button_ends_running_task_without_result(tmp_path):
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    w.studio.run_directory=tmp_path
    def wait(cancel,progress):
        while not cancel():
            time.sleep(.005)
        raise engine.OnePassCancelled("cancelled")
    try:
        w.studio.launch(wait,lambda result:pytest.fail("cancelled result"))
        w.studio_cancel_button.click()
        wait_for(app,lambda:w.studio.task is None)
        assert not w.approve_candidate_button.isEnabled()
        assert not w.save_candidate_button.isEnabled()
        assert json.loads((tmp_path/"gui-status.json").read_text())["status"]=="cancelled"
    finally:
        w.close()


def test_invalid_generated_record_cannot_be_user_approved(tmp_path):
    results=service.StudioResults(batch_at(tmp_path))
    path=results.candidate.record_path
    data=json.loads(path.read_text())
    data["valid"]=False
    service.record(path,data)
    with pytest.raises(ValueError):
        results.approve(dict(character=True,garment=True,exposure=True))


def test_cpu_process_forces_offline_cpu_and_no_console(tmp_path,monkeypatch):
    captured={}
    class Process:
        returncode=0
        def __init__(self,cmd,**kwargs):
            captured.update(kwargs)
        def poll(self):
            return 0
    monkeypatch.setattr(service.subprocess,"Popen",Process)
    service.cpu_process(["python"],tmp_path/"cpu.log",lambda:False,10)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"]==""
    assert captured["env"]["ONNX_MODE"]=="cpu"
    assert captured["env"]["HF_HUB_OFFLINE"]=="1"
    assert captured["env"]["TRANSFORMERS_OFFLINE"]=="1"


def test_new_preflight_failure_does_not_rewrite_previous_run(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    previous=tmp_path/"previous";previous.mkdir()
    service.record(previous/"gui-status.json",{"status":"saved"})
    original=(previous/"gui-status.json").read_bytes()
    w.studio.run_directory=previous
    w.studio.output_root=tmp_path/"runs"
    w.style_path=tmp_path/"character.png";w.selected_outfit_path=tmp_path/"garment.png"
    monkeypatch.setattr(QMessageBox,"exec",lambda *a:0)
    def fail(_):
        raise FileNotFoundError("missing model")
    monkeypatch.setattr(ui,"validate_local_models",fail)
    try:
        w.studio.start()
        assert (previous/"gui-status.json").read_bytes()==original
        assert w.studio.run_directory!=previous
        assert json.loads((w.studio.run_directory/"gui-status.json").read_text())["status"]=="failed"
    finally:
        w.close()
