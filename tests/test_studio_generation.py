"""실제 화면 신호 → 요청·문구 → 원본 묶음 → 명시적 검토 → 내보내기를 CPU로 검사한다."""
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


@pytest.mark.parametrize("use_description", [False, True])
def test_create_button_to_review_and_export_without_legacy_fallback(tmp_path, monkeypatch, use_description):
    app = QApplication.instance() or QApplication([])
    w = GenAILabWindow()
    w.studio.output_root = tmp_path / "runs"
    initial = analysis_at(tmp_path / "selected")
    w.style_path = Path(initial["source"])
    w.selected_outfit_path = Path(initial["directory"])/"garment.png"
    w.update_input_ready_status()
    monkeypatch.setattr(ui, "validate_local_models", lambda _: None)
    def analyze(c,g,d,r,**kw):
        analysis = analysis_at(d)
        analysis["groups"]["fixed"] += ["tail", "raccoon_tail"]
        return analysis
    monkeypatch.setattr(ui, "analyze_inputs", analyze)
    appearance = ui.AppearanceOverrides(
        tail=ui.PartAppearance("light blue tail, striped tail", use_description))
    monkeypatch.setattr(ui, "confirm_inputs", lambda *a: ("male", ("white camisole", "black shorts"), appearance))
    monkeypatch.setattr(ui, "build_request", lambda a,*args,**kw: request_at(tmp_path, a, appearance=kw["appearance"]))
    monkeypatch.setattr(service.StudioRuntime, "from_environment",
        lambda: service.StudioRuntime(model_cache=tmp_path/"missing-model-cache"))
    calls = []
    def generate(req, folder, **kw):
        calls.append(req)
        return engine.generate_onepass_request(req, folder, backend_factory=Backend, **kw)
    monkeypatch.setattr(ui, "generate_onepass_request", generate)
    monkeypatch.setattr(w, "start_legacy_generation", lambda: pytest.fail("legacy must not run"))
    monkeypatch.setattr(w, "start_native_refinement", lambda *a: pytest.fail("refinement must not run"))
    monkeypatch.setattr(ui, "confirm_result", lambda *a: {"character":True,"garment":True,"exposure":True,"appendages":True})
    destination = tmp_path/"chosen.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
    try:
        w.generate_button.click()
        assert not w.style_button.isEnabled()
        wait_for(app, lambda: w.studio.task is None and w.studio.results is not None)
        assert len(calls)==1 and calls[0].inputs.pose_mode == "without_pose"
        assert "white camisole" in calls[0].inputs.prompt.positive
        assert "raccoon ears" in calls[0].inputs.prompt.positive
        assert ("light blue tail" in calls[0].inputs.prompt.positive) == use_description
        assert ("raccoon tail" in calls[0].inputs.prompt.positive) != use_description
        assert len(w.studio.results.batch.candidates)==4
        assert not w.save_candidate_button.isEnabled()
        w.studio_candidate_combo.setCurrentIndex(2)
        selected = w.studio.results.candidate
        w.approve_candidate_button.click()
        assert w.save_candidate_button.isEnabled()
        w.studio_candidate_combo.setCurrentIndex(1)
        assert not w.save_candidate_button.isEnabled()  # 선택값을 바꾸면 승인을 무효화한다.
        w.studio_candidate_combo.setCurrentIndex(2)
        w.approve_candidate_button.click()
        w.save_candidate_button.click()
        assert destination.read_bytes() == selected.path.read_bytes()
        assert "저장 완료" in w.status_label.text()
        raw = json.loads(selected.record_path.read_text(encoding="utf-8"))
        assert raw["gates_executed"] is False and raw["final_return_eligible"] is False
        assert raw["adapter_applied_calls"]==[]
        assert len(raw["unet_calls"])==28
        ledger = json.loads((selected.path.parents[1]/"user-review.json").read_text(encoding="utf-8"))
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
    results.approve(dict(character=True,garment=True,exposure=True,appendages=True))
    results.candidate.path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        results.export(tmp_path/"changed.png")
    assert not (tmp_path/"changed.png").exists()


def test_export_does_not_overwrite_existing_file(tmp_path):
    results = service.StudioResults(batch_at(tmp_path))
    results.approve(dict(character=True,garment=True,exposure=True,appendages=True))
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
        state = json.loads((tmp_path/"gui-status.json").read_text(encoding="utf-8"))
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
        assert json.loads((tmp_path/"gui-status.json").read_text(encoding="utf-8"))["status"]=="cancelled"
    finally:
        w.close()


def test_invalid_generated_record_cannot_be_user_approved(tmp_path):
    results=service.StudioResults(batch_at(tmp_path))
    path=results.candidate.record_path
    data=json.loads(path.read_text(encoding="utf-8"))
    data["valid"]=False
    service.record(path,data)
    with pytest.raises(ValueError):
        results.approve(dict(character=True,garment=True,exposure=True,appendages=True))


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
        assert json.loads((w.studio.run_directory/"gui-status.json").read_text(encoding="utf-8"))["status"]=="failed"
    finally:
        w.close()


def test_confirmed_tail_reaches_request_approval_and_raw_record(tmp_path):
    a = analysis_at(tmp_path/"source")
    a["groups"]["fixed"] = ["tail", "raccoon_tail", "raccoon_ears"]
    override = ui.AppearanceOverrides(tail=ui.PartAppearance("light blue tail, striped tail", True))
    request = request_at(tmp_path, a, appearance=override)
    batch = engine.generate_onepass_request(request, tmp_path/"batch", backend_factory=Backend)
    approval = json.loads((Path(a["directory"])/"approval.json").read_text(encoding="utf-8"))
    raw = json.loads(batch.candidates[0].record_path.read_text(encoding="utf-8"))
    assert "light blue tail, striped tail" in raw["prompt"]["positive"]
    assert "raccoon tail" not in raw["prompt"]["positive"]
    assert raw["prompt"]["rules"]["appendage_appearance"]["tail"]["applied"]
    assert approval["appendage_appearance"]["tail"]["confirmed"]
    results = service.StudioResults(batch)
    with pytest.raises(ValueError):
        results.approve(dict(character=True, garment=True, exposure=True))
    results.approve(dict(character=True, garment=True, exposure=True, appendages=True))
    results.select(1)
    assert results.checks == {} and results.status == "awaiting_user_review"
    with pytest.raises(ValueError):
        results.export(tmp_path/"unreviewed.png")


def test_no_appendage_keeps_three_review_checks(tmp_path):
    a = analysis_at(tmp_path/"source")
    a["groups"]["fixed"] = []
    req = request_at(tmp_path, a)
    batch = engine.generate_onepass_request(req, tmp_path/"batch", backend_factory=Backend)
    results = service.StudioResults(batch)
    assert not results.appendage_review_required
    results.approve(dict(character=True, garment=True, exposure=True))
    assert results.status == "user_approved"


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
def test_opaque_analysis_snapshot_is_byte_identical(tmp_path, mode):
    image = Image.new(mode, (3, 3), (40, 90, 180, 255) if mode == "RGBA" else
                      (40, 90, 180) if mode == "RGB" else 1)
    if mode == "P":
        image.putpalette([0,0,0,40,90,180]+[0]*762)
    original = tmp_path/"old.png"
    image.convert("RGB").save(original)
    output = tmp_path/"new.png"
    record = service.save_analysis_reference(image, output)
    assert output.read_bytes() == original.read_bytes()
    assert not record["alpha_composited"]


def test_transparent_and_partial_alpha_are_composited_on_white(tmp_path):
    image = Image.new("RGBA", (3, 1))
    image.putdata([(12,34,56,0), (200,0,0,128), (40,90,180,255)])
    before = image.tobytes()
    path = tmp_path/"snapshot.png"
    info = service.save_analysis_reference(image, path)
    with Image.open(path) as out:
        assert [out.getpixel((x, 0)) for x in range(out.width)] == [(255,255,255), (227,127,127), (40,90,180)]
    assert image.tobytes() == before
    assert info == {"alpha_composited":True, "background_rgb":[255,255,255],
                    "analysis_sha256":service.digest(path)}


def test_palette_transparency_uses_white(tmp_path):
    image = Image.new("P", (2,1))
    image.putpalette([0,0,0,20,40,60]+[0]*762)
    image.putdata([0,1])
    image.info["transparency"] = 0
    path = tmp_path/"p.png"
    info = service.save_analysis_reference(image, path)
    with Image.open(path) as out:
        assert [out.getpixel((x, 0)) for x in range(out.width)] == [(255,255,255), (20,40,60)]
    assert info["alpha_composited"]


def test_input_dialog_preserves_draft_and_resets_confirmation(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QLineEdit, QCheckBox, QPushButton
    app = QApplication.instance() or QApplication([])
    a = analysis_at(tmp_path/"source")
    a["groups"]["fixed"] = ["tail", "snake_tail", "animal_ears"]
    monkeypatch.setattr(ui, "load_character_gender", lambda _: "male")
    def inspect(dialog):
        tail = dialog.findChild(QLineEdit, "tail_appearance_text")
        checked = dialog.findChild(QCheckBox, "tail_appearance_confirmed")
        ears = dialog.findChild(QLineEdit, "ears_appearance_text")
        assert tail.text() == "" and not checked.isEnabled()
        assert not ears.isEnabled()
        for invalid in ("grey tail, large breasts", "scaly tail, NSFW", "blue tail, bare_legs"):
            tail.setText(invalid)
            assert not checked.isEnabled() and not checked.isChecked()
        tail.setText("grey tail")
        assert checked.isEnabled()
        checked.setChecked(True)
        tail.setText("grey tail, scaly tail")
        assert not checked.isChecked()
        checked.setChecked(True)
        general = next(c for c in dialog.findChildren(QCheckBox) if c.text().startswith("얼굴에 팔"))
        general.setChecked(True)
        create = next(b for b in dialog.findChildren(QPushButton) if b.text()=="이 조건으로 만들기")
        assert create.isEnabled()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", inspect)
    gender, tags, appearance = ui.confirm_inputs(None, a)
    assert appearance.tail == ui.PartAppearance("grey tail, scaly tail", True)
    assert not appearance.ears.confirmed


@pytest.mark.parametrize("has_tail", [True, False])
def test_result_dialog_requires_appendage_check_only_when_present(tmp_path, monkeypatch, has_tail):
    from PySide6.QtWidgets import QDialog, QCheckBox, QPushButton
    app = QApplication.instance() or QApplication([])
    a = analysis_at(tmp_path/"source")
    a["groups"]["fixed"] = ["tail"] if has_tail else []
    def inspect(dialog):
        checks = dialog.findChildren(QCheckBox)
        assert len(checks) == (4 if has_tail else 3)
        accept = next(b for b in dialog.findChildren(QPushButton) if b.text()=="이 결과 사용")
        for c in checks[:3]:
            c.setChecked(True)
        assert accept.isEnabled() == (not has_tail)
        for c in checks:
            c.setChecked(True)
        assert accept.isEnabled()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", inspect)
    checks = ui.confirm_result(None, SimpleNamespace(path=Path(a["source"])), a)
    assert ("appendages" in checks) == has_tail
