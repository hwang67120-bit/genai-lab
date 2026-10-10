"""영역 확인·기억·실패·새 후보 보존 흐름을 CPU 대역으로 검사한다."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication,QDialog,QMessageBox
from genai_lab import studio_head_paste as service
from genai_lab import studio_head_paste_gui as gui
from genai_lab.studio_generation import StudioRuntime,StudioResults
from genai_lab.onepass_generation import OnePassCancelled
from genai_lab.proportion_inputs import sha
from genai_lab.qwen_record_io import write_json
from gui_main import GenAILabWindow
from test_studio_proportion import two_pass_batch
from test_studio_generation import wait_for, batch_at


def preview_at(folder,identity=None):
    folder.mkdir(parents=True)
    for name in ("original","automatic-preview","corrected-preview","H"):
        Image.new("RGB",(12,12),"white").save(folder/(name+".png"))
    info=dict(directory=str(folder),identity=identity or {"source_sha256":"one"},
        original_file=str(folder/"original.png"),automatic_preview=str(folder/"automatic-preview.png"),
        corrected_preview=str(folder/"corrected-preview.png"),H_sha256=sha(folder/"H.png"),
        manifest={str(folder/(name+".png")):sha(folder/(name+".png")) for name in
            ("original","automatic-preview","corrected-preview","H")})
    write_json(folder/"result.json",info)
    return info


@pytest.mark.parametrize("accepted",[False,True])
def test_confirmation_records_user_only_and_dialog_has_two_choices(tmp_path,monkeypatch,accepted):
    app=QApplication.instance() or QApplication([])
    info=preview_at(tmp_path/"preview")
    def decide(dialog):
        assert isinstance(dialog,gui.HairConfirmationDialog)
        assert "붙이기" in dialog.yes.text() and "원래" in dialog.no.text()
        return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
    monkeypatch.setattr(QDialog,"exec",decide)
    decision=gui.review_hair(None,info)
    service.confirm_hair(info,decision)
    saved=service.read(tmp_path/"preview/confirmation.json")
    assert saved["confirmed"] is accepted and saved["reviewer"]=="user"
    assert saved["preview_sha256"]==sha(tmp_path/"preview/result.json")


def test_cache_reused_only_same_source_and_unchanged_confirmed_pixels(tmp_path,monkeypatch):
    identity={"source_sha256":"one"}
    info=preview_at(tmp_path/"cache/key/item",identity)
    service.confirm_hair(info,True)
    req=dict(cache_key="key",identity=identity,manifest={})
    monkeypatch.setattr(service,"run_worker",lambda *a,**k:pytest.fail("승인한 영역은 재분할하지 않음"))
    assert service.prepare_hair(req,tmp_path/"cache",StudioRuntime())["remembered"]
    calls=[]
    monkeypatch.setattr(service,"run_worker",lambda *a,**k:calls.append(a) or {"fresh":True})
    assert service.prepare_hair({**req,"identity":{"source_sha256":"two"}},tmp_path/"cache",StudioRuntime())=={"fresh":True}
    (Path(info["directory"])/"H.png").write_bytes(b"changed")
    assert service.prepare_hair(req,tmp_path/"cache",StudioRuntime())=={"fresh":True}
    assert len(calls)==2


def test_rejected_preview_is_not_reused_or_executed(tmp_path,monkeypatch):
    info=preview_at(tmp_path/"cache/key/item")
    service.confirm_hair(info,False)
    req=dict(cache_key="key",identity=info["identity"],manifest={})
    with pytest.raises(ValueError,match="확인"):
        service.apply_head(req,info,tmp_path/"execute",StudioRuntime())
    monkeypatch.setattr(service,"run_worker",lambda *a,**k:{"fresh":True})
    assert service.prepare_hair(req,tmp_path/"cache",StudioRuntime())=={"fresh":True}
    assert not (tmp_path/"execute").exists()


def test_cancel_before_worker_records_final_state_without_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(service.subprocess,"Popen",lambda *a,**k:pytest.fail("취소 후 실행 금지"))
    with pytest.raises(OnePassCancelled):
        service.run_worker("apply",{"manifest":{}},tmp_path/"cancelled",StudioRuntime(),lambda:True,lambda _:None)
    assert service.read(tmp_path/"cancelled/operation.json")["status"]=="cancelled"


def test_worker_failure_has_reason_and_final_record(tmp_path,monkeypatch):
    class Failed:
        returncode=1
        def poll(self): return 1
    monkeypatch.setattr(service.subprocess,"Popen",lambda *a,**k:Failed())
    with pytest.raises(ValueError,match="원래 후보"):
        service.run_worker("apply",{"manifest":{}},tmp_path/"failed",StudioRuntime(),lambda:False,lambda _:None)
    assert service.read(tmp_path/"failed/operation.json")["status"]=="failed"


def product_at(folder,result):
    folder.mkdir()
    product=folder/"product.png";Image.new("RGB",(16,24),(4,5,6)).save(product)
    record=folder/"run.json"
    write_json(record,dict(status="review_pending",unet_calls=[{"batch":2,"adapter":True}]*28,callback=[{"index":i,"mode":"overwrite" if i<24 else "blend"} for i in range(28)],
                           adapter_calls=1,max_reserved_gib=4.67))
    info=dict(directory=str(folder),status="awaiting_user_review",raw_sha256=result.verify_original(),
              before_sha256=result.verify_current(),product_file=str(product),product_sha256=sha(product),
              checks=dict(outside_equal=True,face_equal=True),redraw_record=str(record),manifest={str(record):sha(record)})
    write_json(folder/"result.json",info)
    return info


def test_new_candidate_keeps_four_originals_reapproval_and_export_provenance(tmp_path):
    result=StudioResults(two_pass_batch(tmp_path/"generation"))
    originals=tuple((c.path.read_bytes(),c.raw.path.read_bytes()) for c in result.batch.candidates)
    info=product_at(tmp_path/"head",result)
    result.approve(dict(character=True,garment=True,exposure=True))
    result.add_head_candidate(info["directory"])
    assert len(result.batch.candidates)==5 and result.selected==4
    assert result.status=="awaiting_user_review" and result.approved_sha is None
    assert result.head_paste_unavailable_reason
    for candidate,(product,raw) in zip(result.batch.candidates,originals):
        assert candidate.path.read_bytes()==product and candidate.raw.path.read_bytes()==raw
    result.approve(dict(character=True,garment=True,exposure=True))
    result.export(tmp_path/"saved.png")
    assert (tmp_path/"saved.png").read_bytes()==Path(info["product_file"]).read_bytes()
    assert service.read(tmp_path/"saved.review.json")["head_paste"]==info
    result.select(0)
    assert not result.head_paste_unavailable_reason


def test_changed_head_execution_record_blocks_saving(tmp_path):
    result=StudioResults(two_pass_batch(tmp_path/"generation"))
    info=product_at(tmp_path/"head",result)
    result.add_head_candidate(info["directory"])
    Path(info["redraw_record"]).write_bytes(b"changed")
    with pytest.raises(ValueError): result.verify_current()


@pytest.mark.parametrize("accept",[False,True])
def test_actual_gui_button_confirmation_to_new_candidate(tmp_path,monkeypatch,accept):
    app=QApplication.instance() or QApplication([])
    window=GenAILabWindow()
    window.studio.output_root=tmp_path/"runs"
    info=preview_at(tmp_path/"preview")
    monkeypatch.setattr(QMessageBox,"warning",lambda *a:pytest.fail(str(a)))
    try:
        assert not window.studio_head_paste_button.isEnabled()
        window.studio.generated(two_pass_batch(tmp_path/"generation"))
        window.studio.runtime=StudioRuntime()
        original=window.studio.results.current_path
        req=dict(identity=info["identity"],manifest={})
        monkeypatch.setattr(gui,"selected_request",lambda *a:req)
        monkeypatch.setattr(gui,"prepare_hair",lambda *a,**k:info)
        monkeypatch.setattr(gui,"review_hair",lambda *a:accept)
        seen=[]
        def apply(request,preview,directory,*a,**k):
            seen.append(directory)
            directory.parent.mkdir(parents=True,exist_ok=True)
            return product_at(directory,window.studio.results)
        monkeypatch.setattr(gui,"apply_head",apply)
        assert window.studio_head_paste_button.isEnabled()
        window.studio_head_paste_button.click()
        wait_for(app,lambda:window.studio.task is None)
        # 두 번째 작업이 확인 콜백에서 시작됐을 때도 기다린다.
        wait_for(app,lambda:window.studio.head_paste_controller.context is None)
        assert bool(seen)==accept
        assert len(window.studio.results.batch.candidates)==(5 if accept else 4)
        if accept:
            assert window.studio.results.current_path!=original
            window.studio_candidate_combo.setCurrentIndex(0)
        assert window.studio.results.current_path==original
    finally:
        window.close()


def test_gui_error_and_cancel_keep_results(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    window=GenAILabWindow()
    monkeypatch.setattr(QMessageBox,"warning",lambda *a:None)
    try:
        window.studio.generated(two_pass_batch(tmp_path/"generation"))
        result=window.studio.results; before=result.current_path
        for error in (RuntimeError("분할 실패"),OnePassCancelled("취소")):
            window.studio.head_paste_controller.failed(error)
            assert window.studio.results is result and result.current_path==before
            assert len(result.batch.candidates)==4
    finally: window.close()


def test_normal_generation_does_not_expose_head_feature(tmp_path):
    app=QApplication.instance() or QApplication([])
    window=GenAILabWindow()
    try:
        window.studio.generated(batch_at(tmp_path))
        assert window.studio_head_paste_button.isHidden()
    finally: window.close()


@pytest.mark.parametrize("layout",["common/modules","modules"])
def test_semantic_tool_matches_existing_cache_layout(tmp_path,layout):
    from genai_lab.head_paste_semantics import semantic_source_file
    source=tmp_path/layout/"semanticsam.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"")
    assert semantic_source_file(tmp_path)==source


def test_missing_semantic_tool_fails_without_download(tmp_path):
    from genai_lab.head_paste_semantics import semantic_source_file
    with pytest.raises(ValueError,match="로컬"):
        semantic_source_file(tmp_path)
