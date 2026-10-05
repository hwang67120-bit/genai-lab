"""CPU contract and GUI lifecycle tests; synthetic images, never models or GPU."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QFileDialog
from genai_lab.qwen_preservation import file_sha, json_sha
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.qwen_record_io import write_json
from genai_lab.qwen_tail_edit import (assemble_tail_prompt, prepare_tail_spec, validate_tail_request,
    make_tail_request, crop_bytes, TailEditWorkflow, run_tail_edit, tail_product_info)
from genai_lab.qwen_tail_gui import TailInputDialog, screen_box_to_source
from genai_lab.qwen_pose_worker import validate_request
from genai_lab.studio_generation import StudioResults
from genai_lab import studio_controller as controller
from gui_main import GenAILabWindow

FIXTURES = json.loads((Path(__file__).parent / "fixtures/qwen_tail_prompts.json").read_text(encoding="utf-8"))
CHECKS = dict(character=True, garment=True, exposure=True, appendages=True)


def assets(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source, basis = tmp_path/"source.png", tmp_path/"basis.png"
    Image.new("RGBA", (936,2048), (200,0,0,128)).save(source)
    Image.new("RGB", (736,1232), "white").save(basis)
    return source, basis


def runtime(tmp_path):
    return QwenPoseSettings(sys.executable, str(tmp_path), str(tmp_path/"model.gguf"),
                            str(tmp_path/"models.json"), "a"*64)


def spec_at(tmp_path, source=None, basis=None):
    if source is None:
        source, basis = assets(tmp_path)
    return prepare_tail_spec(basis, source, (560,1100,936,1700), pattern=True, confirmed=True,
                            directory=tmp_path/"inputs", source_sha256=file_sha(source), image_sha256=file_sha(basis))


def finished_edit(tmp_path, *, source=None, basis=None):
    spec = spec_at(tmp_path, source, basis)
    request = make_tail_request(spec, settings=runtime(tmp_path))
    out = tmp_path/"edit"
    class Process:
        pid=17; returncode=0
        def __init__(self, command, **kwargs):
            assert command[1:3] == ["-m", "genai_lab.qwen_pose_worker"]
            assert kwargs["shell"] is False and kwargs["env"]["HF_HUB_OFFLINE"] == "1"
            Image.new("RGB", tuple(request["output_size"]), "blue").save(out/"raw.png")
            write_json(out/"run.json", {"status":"completed", "raw_sha256":file_sha(out/"raw.png"),
                                      "request_sha256":json_sha(request)})
        def poll(self): return 0
    result = run_tail_edit(request, out, TailEditWorkflow(spec, gpu_probe=lambda:{}), popen=Process)
    return out, result, request


def simple_batch(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidates=[]
    for i in range(4):
        path=tmp_path/f"raw-{i}.png"
        Image.new("RGB", (736,1232), (i,20,30)).save(path)
        record=tmp_path/f"run-{i}.json"
        data={"valid":True,"completed":True,"raw_sha256":file_sha(path)}
        write_json(record,data)
        candidates.append(SimpleNamespace(path=path,record_path=record,record=data))
    return SimpleNamespace(directory=tmp_path,candidates=candidates)


@pytest.mark.parametrize("case", FIXTURES)
def test_exact_experiment_tail_prompt(case):
    actual=assemble_tail_prompt(case["pattern"],case["tip"])
    assert actual["positive"].encode()==case["positive"].encode()
    assert actual["negative"]==case["negative"]==" "
    assert actual["positive_sha256"]==hashlib.sha256(case["positive"].encode()).hexdigest()


@pytest.mark.parametrize("tip", ["Make the tail a raccoon tail.","Use a chameleon tail.",
    "Make large breasts.","Make the character female.","Add a white camisole.","Add NSFW details.",
    "Use underwear.","Use skinny hips.","Add a dragon tail.","두꺼운 꼬리", "One. Two.","Tail\nchange", "A"*301])
def test_tip_rejects_sentence_embedded_forbidden_terms(tip):
    with pytest.raises(ValueError): assemble_tail_prompt(True,tip)


@pytest.mark.parametrize("pattern", [None,0,1,"yes"])
def test_pattern_requires_explicit_boolean(pattern):
    with pytest.raises(ValueError): assemble_tail_prompt(pattern)


def test_original_pixel_crop_and_alpha_white(tmp_path):
    spec=spec_at(tmp_path)
    with Image.open(spec.crop_path) as crop:
        assert crop.size==(376,600) and crop.mode=="RGB"
        assert crop.getpixel((0,0))==(227,127,127)
    assert spec.box==(560,1100,936,1700)
    assert Path(spec.source_path).read_bytes()==(tmp_path/"source.png").read_bytes()
    spec.verify_image()
    request=make_tail_request(spec,settings=runtime(tmp_path))
    assert request["output_size"]==[800,1312]
    assert validate_request(request).crop_sha256==spec.crop_sha256
    assert "skeleton_path" not in request


@pytest.mark.parametrize("box", [(0,0,936,2048),(10,10,10,20),(-1,2,5,6),(0,0,937,2048),(1.1,2,5,6)])
def test_bad_or_full_image_box_rejected(tmp_path,box):
    source,_=assets(tmp_path)
    with pytest.raises(ValueError): crop_bytes(source,box,file_sha(source))


def test_crop_cannot_be_substituted_with_full_reference(tmp_path):
    spec=spec_at(tmp_path)
    Path(spec.crop_path).write_bytes(Path(spec.source_path).read_bytes())
    changed=replace(spec,crop_sha256=file_sha(spec.crop_path))
    with pytest.raises(ValueError,match="크롭"): changed.verify_image()


def test_changed_input_or_prompt_cannot_launch(tmp_path):
    spec=spec_at(tmp_path)
    request=make_tail_request(spec,settings=runtime(tmp_path))
    request["prompt"]["positive"]+=" extra"
    with pytest.raises(ValueError,match="지시문"): validate_tail_request(request)
    Path(spec.image_path).write_bytes(b"changed")
    with pytest.raises(ValueError,match="변경"): spec.verify_image()


def test_crop_coordinates_account_for_letterbox_and_reverse_drag():
    rect=(100,0,468,1024)
    assert screen_box_to_source((380,550),(568,850),rect,(936,2048))==(560,1100,936,1700)
    assert screen_box_to_source((568,850),(380,550),rect,(936,2048))==(560,1100,936,1700)


def test_tail_workflow_checks_actual_gpu_release_and_confirmation(tmp_path):
    spec=spec_at(tmp_path)
    def residue(): raise RuntimeError("GPU retained")
    flow=TailEditWorkflow(spec,gpu_probe=residue)
    with pytest.raises(RuntimeError,match="retained"): flow.before_launch(spec)
    with pytest.raises(ValueError): TailEditWorkflow(replace(spec,confirmed=False))
    flow=TailEditWorkflow(spec,gpu_probe=lambda:{})
    with pytest.raises(ValueError): flow.before_launch(replace(spec,pattern=False))


def test_tail_process_reuses_preview_and_raw_is_immutable(tmp_path):
    out,product,request=finished_edit(tmp_path)
    raw_before=file_sha(out/"raw.png")
    info=tail_product_info(out,request["tail_spec"]["image_sha256"])
    assert info["product_sha256"]==file_sha(product)
    assert file_sha(out/"raw.png")==raw_before
    with Image.open(product) as im: assert im.size==(736,1232)


@pytest.mark.parametrize("failure", ["failure","cancel"])
def test_tail_process_terminal_record_on_failure_or_cancel(tmp_path,failure):
    spec=spec_at(tmp_path)
    request=make_tail_request(spec,settings=runtime(tmp_path))
    calls=[]
    class Process:
        pid=9; returncode=None
        def __init__(self,*a,**kw):
            calls.append("start")
            if failure=="failure": self.returncode=1
        def poll(self): return self.returncode
        def terminate(self): self.returncode=-1; calls.append("terminate")
        def wait(self,timeout=None): return self.returncode
    with pytest.raises(RuntimeError):
        run_tail_edit(request,tmp_path/"edit",TailEditWorkflow(spec,gpu_probe=lambda:{}),popen=Process,
                      cancelled=lambda:failure=="cancel" and bool(calls))
    assert calls.count("start")==1
    state=json.loads((tmp_path/"edit/run.json").read_text(encoding="utf-8"))
    assert state["status"]==("cancelled" if failure=="cancel" else "failed")
    assert (tmp_path/"edit/run.final.json").exists()


def test_adopt_review_export_and_candidate_switch(tmp_path):
    results=StudioResults(simple_batch(tmp_path/"batch"),appendage_review_required=True)
    source,_=assets(tmp_path/"source")
    original=[c.path.read_bytes() for c in results.batch.candidates]
    records=[c.record_path.read_bytes() for c in results.batch.candidates]
    out,product,_=finished_edit(tmp_path/"tail",source=source,basis=results.candidate.path)
    results.adopt_tail_edit(out)
    with pytest.raises(ValueError): results.export(tmp_path/"no.png")
    results.approve(CHECKS)
    saved=results.export(tmp_path/"saved.png")
    assert saved.read_bytes()==product.read_bytes() and saved.read_bytes()!=original[0]
    assert json.loads(saved.with_suffix(".review.json").read_text(encoding="utf-8"))["tail_edit"]["directory"]==str(out.resolve())
    assert [c.path.read_bytes() for c in results.batch.candidates]==original
    assert [c.record_path.read_bytes() for c in results.batch.candidates]==records
    results.select(1)
    assert results.tail_edit is None and results.approved_sha is None
    with pytest.raises(ValueError): results.adopt_tail_edit(out)


def test_tampered_product_rejected_at_export(tmp_path):
    results=StudioResults(simple_batch(tmp_path/"batch"),appendage_review_required=True)
    source,_=assets(tmp_path/"source")
    out,product,_=finished_edit(tmp_path/"tail",source=source,basis=results.candidate.path)
    results.adopt_tail_edit(out); results.approve(CHECKS)
    Image.new("RGB",(736,1232),"red").save(product)
    with pytest.raises(ValueError): results.export(tmp_path/"bad.png")


def test_tail_dialog_requires_pattern_confirmation_and_invalidates_changes(tmp_path):
    app=QApplication.instance() or QApplication([])
    source,_=assets(tmp_path)
    dialog=TailInputDialog(source,file_sha(source),settings_path="runtime.json")
    dialog.canvas.box=(560,1100,936,1700)
    dialog.confirm.setChecked(True)
    assert not dialog.run.isEnabled()
    dialog.pattern.setCurrentIndex(1)
    assert not dialog.confirm.isChecked() and not dialog.run.isEnabled()
    dialog.confirm.setChecked(True)
    assert dialog.run.isEnabled()
    dialog.tip.setText("The tip curls into a tight spiral.")
    assert not dialog.confirm.isChecked()
    dialog.confirm.setChecked(True)
    assert dialog.run.isEnabled()
    dialog.tip.setText("Add a bikini.")
    dialog.confirm.setChecked(True)
    assert not dialog.run.isEnabled()
    dialog.close()


@pytest.mark.parametrize("tags,visible", [(["raccoon_tail"],True),(["tail"],True),(["twintails"],False),(["animal_ears"],False),([],False)])
def test_tail_button_visibility_before_approval(tmp_path,tags,visible):
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    w.studio.analysis={"groups":{"fixed":tags}}
    try:
        w.studio.generated(simple_batch(tmp_path))
        assert (not w.tail_edit_button.isHidden())==visible
        assert w.tail_edit_button.isEnabled()==visible
        assert not w.save_candidate_button.isEnabled()
    finally:w.close()


def test_optional_failure_keeps_original_batch_and_approval(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    monkeypatch.setattr(QMessageBox,"warning",lambda *a:None)
    w.studio.analysis={"groups":{"fixed":["tail"]}}
    w.studio.generated(simple_batch(tmp_path))
    results=w.studio.results
    results.approve(CHECKS)
    try:
        w.studio.tail_failed(RuntimeError("failure"))
        w.studio.controls()
        assert w.studio.results is results
        assert results.status=="user_approved" and len(results.batch.candidates)==4
        assert w.save_candidate_button.isEnabled()
    finally:w.close()


def test_gui_tail_adoption_and_save_selected_product(tmp_path,monkeypatch):
    import genai_lab.qwen_tail_gui as gui
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    source,_=assets(tmp_path/"source")
    w.studio.analysis={"groups":{"fixed":["tail"]}}
    w.studio.generated(simple_batch(tmp_path/"batch"))
    out,product,request=finished_edit(tmp_path/"tail",source=source,basis=w.studio.results.candidate.path)
    w.studio.tail_context={"directory":out,"source":source,"before":w.studio.results.candidate.path,"selected":0}
    monkeypatch.setattr(gui,"review_tail_result",lambda *a:dict(CHECKS))
    destination=tmp_path/"chosen.png"
    monkeypatch.setattr(QFileDialog,"getSaveFileName",lambda *a,**k:(str(destination),""))
    try:
        w.studio.tail_edited(product)
        assert w.save_candidate_button.isEnabled()
        w.save_candidate_button.click()
        assert destination.read_bytes()==product.read_bytes()
        assert w.studio.results is None
    finally:w.close()


def test_gui_discard_edit_keeps_current_candidate(tmp_path,monkeypatch):
    import genai_lab.qwen_tail_gui as gui
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    source,_=assets(tmp_path/"source")
    w.studio.analysis={"groups":{"fixed":["tail"]}}
    w.studio.generated(simple_batch(tmp_path/"batch"))
    results=w.studio.results
    out,product,_=finished_edit(tmp_path/"tail",source=source,basis=results.candidate.path)
    w.studio.tail_context={"directory":out,"source":source,"before":results.candidate.path,"selected":0}
    monkeypatch.setattr(gui,"review_tail_result",lambda *a:None)
    try:
        w.studio.tail_edited(product)
        assert w.studio.results is results and results.tail_edit is None
        assert json.loads((out/"tail-review.json").read_text(encoding="utf-8"))["adopted"] is False
    finally:w.close()


def test_tail_seed_cannot_drift_from_trial(tmp_path):
    spec=spec_at(tmp_path)
    with pytest.raises(ValueError,match="seed"):
        make_tail_request(spec,settings=replace(runtime(tmp_path),default_seed=42))


def test_mouse_crop_keeps_original_box_after_resize(tmp_path):
    from genai_lab.qwen_tail_gui import TailCropCanvas
    app=QApplication.instance() or QApplication([])
    source,_=assets(tmp_path)
    canvas=TailCropCanvas(source)
    canvas.resize(668,1024)
    canvas.show();app.processEvents()
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=QPoint(380,550))
    QTest.mouseMove(canvas,QPoint(568,850))
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=QPoint(568,850))
    assert canvas.box==(560,1100,936,1700)
    canvas.resize(700,700);app.processEvents()
    assert canvas.box==(560,1100,936,1700)
    assert not canvas.dragging
    canvas.close()


@pytest.mark.parametrize("outcome", ["adopt", "failure", "cancel"])
def test_actual_tail_button_request_process_review_and_export(tmp_path,monkeypatch,outcome):
    import time
    import genai_lab.qwen_tail_edit as edit
    import genai_lab.qwen_tail_gui as gui
    app=QApplication.instance() or QApplication([])
    w=GenAILabWindow()
    source,_=assets(tmp_path/"source")
    w.studio.analysis={"source":str(source),"references":{"character":{"sha256":file_sha(source)}},
                       "groups":{"fixed":["tail"]}}
    w.studio.generated(simple_batch(tmp_path/"batch"))
    results=w.studio.results
    basis_sha=file_sha(results.candidate.path)
    original=[c.path.read_bytes() for c in results.batch.candidates]
    class Input:
        settings_path="test-runtime.json"
        settings=runtime(tmp_path)
        canvas=SimpleNamespace(box=(560,1100,936,1700))
        pattern=SimpleNamespace(currentData=lambda:True)
        tip=SimpleNamespace(text=lambda:"")
        confirm=SimpleNamespace(isChecked=lambda:True)
        def __init__(self,*a,**kw):pass
        def exec(self):return QDialog.DialogCode.Accepted
    monkeypatch.setattr(gui,"TailInputDialog",Input)
    monkeypatch.setattr(gui,"review_tail_result",lambda *a:dict(CHECKS))
    monkeypatch.setattr(QMessageBox,"warning",lambda *a:None)
    import weakref
    released=[]
    class IdlePipeline:
        def remove_all_hooks(self): released.append("hooks_removed")
    w.pipeline=IdlePipeline()
    prior_pipeline=weakref.ref(w.pipeline)
    actual_run=edit.run_tail_edit
    seen=[]
    def fake_worker_run(request,directory,flow,**kwargs):
        assert prior_pipeline() is None and released==["hooks_removed"]
        assert request["tail_spec"]["image_sha256"]==basis_sha
        assert request["seed"]==209212001
        assert "skeleton_path" not in request
        flow.gpu_probe=lambda:{}
        class Process:
            pid=18;returncode=None
            def __init__(self,command,**options):
                seen.append(request)
                if outcome=="failure":self.returncode=1
                elif outcome=="adopt":
                    Image.new("RGB",tuple(request["output_size"]),"blue").save(directory/"raw.png")
                    write_json(directory/"run.json",{"status":"completed","raw_sha256":file_sha(directory/"raw.png"),
                                                     "request_sha256":json_sha(request)})
                    self.returncode=0
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-1
            def wait(self,timeout=None):return self.returncode
        return actual_run(request,directory,flow,popen=Process,**kwargs)
    monkeypatch.setattr(edit,"run_tail_edit",fake_worker_run)
    destination=tmp_path/"user-saved.png"
    monkeypatch.setattr(QFileDialog,"getSaveFileName",lambda *a,**k:(str(destination),""))
    try:
        w.tail_edit_button.click()
        assert w.studio.task is not None
        assert not w.tail_edit_button.isEnabled() and not w.studio_candidate_combo.isEnabled()
        until=time.monotonic()+15
        if outcome=="cancel":
            while not seen:
                app.processEvents();time.sleep(.01)
                assert time.monotonic()<until
            w.studio_cancel_button.click()
        while w.studio.task is not None:
            app.processEvents();time.sleep(.01)
            assert time.monotonic()<until
        app.processEvents()
        assert len(seen)==1
        assert w.studio.results is results
        assert [c.path.read_bytes() for c in results.batch.candidates]==original
        if outcome=="adopt":
            assert results.status=="user_approved"
            expected=results.current_path.read_bytes()
            w.save_candidate_button.click()
            assert destination.read_bytes()==expected and expected!=original[0]
        else:
            assert results.tail_edit is None and results.status=="awaiting_user_review"
            assert not w.save_candidate_button.isEnabled()
            run=list((tmp_path/"batch/tail-edits").glob("*/edit/run.json"))[0]
            assert json.loads(run.read_text(encoding="utf-8"))["status"]==("failed" if outcome=="failure" else "cancelled")
    finally:w.close()
