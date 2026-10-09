"""비율 화면의 CPU 규칙 검사다. 모델 추론이나 GPU 생성은 없다."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QSettings, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QCheckBox, QFileDialog
from genai_lab import studio_controller as ui
from genai_lab import studio_generation as service
from genai_lab import studio_proportion as prep
from genai_lab import studio_proportion_worker as worker
from genai_lab import studio_proportion_gui as gui
from genai_lab.proportion_inputs import prepare_head_outline, sha, json_value
from genai_lab.onepass_generation import OnePassCandidate
from genai_lab.proportion_generation import ProportionCandidate, ProportionBatch
from genai_lab.qwen_record_io import write_json
from gui_main import GenAILabWindow
from test_studio_generation import analysis_at, Tokens, wait_for


def prepared(tmp_path):
    analysis = analysis_at(tmp_path/"inputs")
    analysis["character_scores"] = {"profile":.9}
    folder = tmp_path/"pose"
    folder.mkdir()
    for name in ("normalized","control","overlay"):
        Image.new("RGB",(736,1232),"white").save(folder/(name+".png"))
    pose = {"source_file":str(Path(analysis["directory"])/"character.png"),
            "source_sha256":sha(Path(analysis["directory"])/"character.png"),
            "normalized_file":str(folder/"normalized.png"), "normalized_sha256":sha(folder/"normalized.png"),
            "control_file":str(folder/"control.png"), "control_sha256":sha(folder/"control.png"),
            "overlay_file":str(folder/"overlay.png"), "nose":[340.,170.],"neck":[340.,230.],"ip_early":.5,
            "decision":{"status":"pass","rejects":[],"warnings":[],
                "choices":["proceed_with_pose","proceed_without_pose","choose_other_image"]}}
    mask = np.zeros((1232,736),np.uint8)
    mask[40:220,220:460] = 255
    Image.fromarray(mask).save(folder/"mask.png")
    head = prepare_head_outline(folder/"normalized.png",folder/"control.png",folder/"mask.png",
        folder/"preview",nose=pose["nose"],neck=pose["neck"])
    draft = json_value({"head":asdict(head),"box":[220,40,460,220],"face_point":[340,170],
        "preview_file":str(folder/"preview/preview.png"),"geometry_error":None})
    return analysis,pose,draft


def two_pass_batch(folder):
    folder.mkdir(parents=True)
    candidates=[]
    for seed in (1,2,3,4):
        raw_folder=folder/f"CONTOUR_{seed}";raw_folder.mkdir()
        raw_file=raw_folder/"raw.png"
        Image.new("RGB",(16,24),(seed,30,40)).save(raw_file)
        raw_record={"valid":True,"completed":True,"raw_sha256":sha(raw_file)}
        write_json(raw_folder/"run.json",raw_record)
        raw=OnePassCandidate(seed,raw_file,raw_folder/"run.json",raw_record)
        product_folder=folder/f"seed-{seed}";product_folder.mkdir()
        product_file=product_folder/"product.png"
        Image.new("RGB",(16,24),(255,seed,255)).save(product_file)
        record={**raw_record,"product_sha256":sha(product_file),"raw_file":str(raw_file),
                "product_file":str(product_file),"proportion_mode":"two_pass"}
        write_json(product_folder/"run.json",record)
        candidates.append(ProportionCandidate(seed,raw,product_file,product_folder/"run.json",record))
    return ProportionBatch(folder,tuple(candidates))


def test_default_off_and_frozen_choice_during_work():
    app=QApplication.instance() or QApplication([])
    window=GenAILabWindow()
    try:
        assert not window.studio_proportion_checkbox.isChecked()
        window.studio_proportion_checkbox.setChecked(True)
        assert "2배" in window.studio_mode_note.text()
        window.studio.confirming=True
        window.studio.controls()
        assert not window.studio_proportion_checkbox.isEnabled()
        window.studio.confirming=False
    finally: window.close()


def test_head_box_and_point_are_explicit_and_draft_is_not_approved(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    _,pose,draft=prepared(tmp_path)
    def select(dialog):
        canvas=dialog.findChild(gui.HeadCanvas)
        canvas.box=(220,40,460,220)
        canvas.selectionChanged.emit()
        accept=next(b for b in dialog.findChildren(QPushButton) if b.text()=="이 범위로 윤곽 미리보기")
        assert accept.isEnabled()
        canvas.face_point=(100,100);canvas.selectionChanged.emit()
        assert not accept.isEnabled()
        canvas.face_point=(340,170);canvas.selectionChanged.emit()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog,"exec",select)
    assert gui.select_head_region(None,pose)=={"box":(220,40,460,220),"face_point":(340,170)}
    assert prep.restore_head(draft).confirmed is False


@pytest.mark.parametrize("invalid",[False,True])
def test_head_review_requires_checkbox_and_valid_geometry(tmp_path,monkeypatch,invalid):
    app=QApplication.instance() or QApplication([])
    _,pose,draft=prepared(tmp_path)
    if invalid: draft["geometry_error"]="코 위치가 머리 밖입니다."
    def review(dialog):
        accept=next(b for b in dialog.findChildren(QPushButton) if b.text()=="확인한 비율로 4장 만들기")
        assert not accept.isEnabled()
        dialog.findChild(QCheckBox,"head_outline_confirmed").setChecked(True)
        assert accept.isEnabled() != invalid
        return QDialog.DialogCode.Rejected
    monkeypatch.setattr(QDialog,"exec",review)
    assert gui.review_head_outline(None,pose,draft) is None
    assert prep.restore_head(draft).confirmed is False


def test_confirmed_head_binds_pose_and_preserves_prompt_and_gender(tmp_path):
    analysis,pose,draft=prepared(tmp_path)
    runtime=service.StudioRuntime()
    options=prep.confirmed_options(pose,draft,runtime)
    assert options.enabled and options.head.confirmed
    assert prep.restore_head(draft).confirmed is False
    store=QSettings(str(tmp_path/"prefs.ini"),QSettings.Format.IniFormat)
    kwargs=dict(confirmed=True,runtime=runtime,preferences=store,tokenizers=(Tokens(),Tokens()),seeds=(1,2,3,4))
    old=service.build_request(analysis,analysis["garment_tags"],"male",**kwargs)
    new=service.build_request(analysis,analysis["garment_tags"],"male",proportion_pose=pose,**kwargs)
    assert old.inputs.prompt == new.inputs.prompt
    assert old.inputs.pose_mode=="without_pose" and new.inputs.pose_mode=="with_pose"
    assert new.inputs.ip_early==.5 and new.inputs.control_sha256==pose["control_sha256"]
    assert new.inputs.face_sha256==old.inputs.face_sha256
    record=json.loads((Path(analysis["directory"])/"approval.json").read_text(encoding="utf-8"))
    assert record["proportion_mode"]=="two_pass" and record["gender"]=="male"


@pytest.mark.parametrize("change",["source","normalized","control","mask"])
def test_changed_input_cannot_be_confirmed(tmp_path,change):
    _,pose,draft=prepared(tmp_path)
    path=Path(draft["head"]["mask_file"]) if change=="mask" else Path(pose[change+"_file"])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError): prep.confirmed_options(pose,draft,service.StudioRuntime())


def test_rejected_pose_does_not_become_with_pose(tmp_path):
    _,pose,draft=prepared(tmp_path)
    pose["decision"]={"status":"reject","rejects":["K2"],"warnings":[],"choices":["proceed_without_pose","choose_other_image"]}
    with pytest.raises(ValueError): prep.confirmed_options(pose,draft,service.StudioRuntime())


def test_merge_face_mask_matches_trial_rule_and_does_not_change_inputs():
    hair=np.zeros((100,100),bool);hair[10:40,20:70]=True
    face=np.zeros_like(hair);face[30:90,30:60]=True
    original=face.copy()
    mask=worker.merge_head_parts(hair,face,(20,10,70,50))
    assert np.array_equal(face,original)
    assert mask[64,40]==255 and mask[65,40]==0
    assert mask[10,20]==255 and set(np.unique(mask))=={0,255}


def test_head_worker_preview_remains_unconfirmed(tmp_path):
    _,pose,_=prepared(tmp_path)
    folder=tmp_path/"worker";folder.mkdir()
    hair=np.zeros((1232,736),bool);hair[40:160,220:460]=True
    face=np.zeros_like(hair);face[140:220,240:440]=True
    segmenter=SimpleNamespace(segment=lambda im,box:(hair,.9),snapshot="local-only")
    result=worker.save_head({"pose":pose,"box":[220,40,460,220],"face_point":[340,170]},folder,
                           segmenter,point_segment=lambda *a:face)
    assert result["head"]["confirmed"] is False and result["geometry_error"] is None
    assert result["status"]=="needs_user_review"
    assert not (folder/"preview/user-confirmation.json").exists()


@pytest.mark.parametrize("cancel_stage",["box","review"])
def test_cancelling_head_setup_never_generates(tmp_path,monkeypatch,cancel_stage):
    app=QApplication.instance() or QApplication([])
    analysis,pose,draft=prepared(tmp_path)
    window=GenAILabWindow()
    window.studio.run_directory=tmp_path/"run";window.studio.run_directory.mkdir()
    window.studio.runtime=service.StudioRuntime()
    window.studio.analysis=analysis
    window.studio.proportion_pending=("male",tuple(analysis["garment_tags"]),ui.AppearanceOverrides())
    monkeypatch.setattr(ui,"generate_onepass_request",lambda *a,**k:pytest.fail("must not generate"))
    try:
        if cancel_stage=="box":
            monkeypatch.setattr(ui,"select_head_region",lambda *a:None)
            window.studio.proportion_pose_ready(pose)
        else:
            window.studio.prepared_proportion_pose=pose
            monkeypatch.setattr(ui,"review_head_outline",lambda *a:None)
            window.studio.proportion_head_ready(draft)
        assert window.studio.task is None and window.studio.results is None
        assert json.loads((window.studio.run_directory/"gui-status.json").read_text())["status"]=="cancelled"
    finally: window.close()


def test_gui_two_pass_request_approval_export_and_no_second_background(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    analysis,pose,draft=prepared(tmp_path)
    window=GenAILabWindow()
    window.studio.output_root=tmp_path/"runs"
    window.style_path=Path(analysis["source"])
    window.selected_outfit_path=Path(analysis["directory"])/"garment.png"
    window.studio_proportion_checkbox.setChecked(True)
    window.update_input_ready_status()
    monkeypatch.setattr(ui,"validate_local_models",lambda _:None)
    monkeypatch.setattr(ui,"analyze_inputs",lambda *a,**k:analysis)
    monkeypatch.setattr(ui,"confirm_inputs",lambda *a:("male",tuple(analysis["garment_tags"]),ui.AppearanceOverrides()))
    monkeypatch.setattr(ui,"prepare_studio_pose",lambda *a,**k:pose)
    monkeypatch.setattr(ui,"select_head_region",lambda *a:{"box":draft["box"],"face_point":draft["face_point"]})
    monkeypatch.setattr(ui,"prepare_studio_head",lambda *a,**k:draft)
    monkeypatch.setattr(ui,"review_head_outline",lambda *a:"confirm")
    monkeypatch.setattr(ui,"prepare_backgrounds",lambda *a,**k:pytest.fail("already white"))
    def build(a,tags,gender,**kw):
        return service.build_request(a,tags,gender,**kw,tokenizers=(Tokens(),Tokens()),seeds=(1,2,3,4),
            preferences=QSettings(str(tmp_path/"prefs.ini"),QSettings.Format.IniFormat))
    monkeypatch.setattr(ui,"build_request",build)
    calls=[]
    def generate(request,folder,**kw):
        calls.append((request,kw["proportion"]))
        assert kw["proportion"].head.confirmed and request.inputs.pose_mode=="with_pose"
        batch=two_pass_batch(folder)
        for candidate in batch.candidates: kw["on_image"](candidate)
        return batch
    monkeypatch.setattr(ui,"generate_onepass_request",generate)
    monkeypatch.setattr(ui,"confirm_result",lambda *a:dict(character=True,garment=True,exposure=True,appendages=True))
    destination=tmp_path/"saved.png"
    monkeypatch.setattr(QFileDialog,"getSaveFileName",lambda *a:(str(destination),""))
    try:
        window.generate_button.click()
        wait_for(app,lambda:window.studio.task is None and window.studio.results is not None)
        assert len(calls)==1 and window.studio.results.two_pass
        assert window.studio.results.raw_path.name=="raw.png"
        expected=window.studio.results.current_path.read_bytes()
        window.approve_candidate_button.click();window.save_candidate_button.click()
        assert destination.read_bytes()==expected
        record=json.loads(destination.with_suffix(".review.json").read_text(encoding="utf-8"))
        assert record["proportion_mode"]=="two_pass" and record["raw_sha256"]!=record["sha256"]
    finally: window.close()


@pytest.mark.parametrize("target",["raw","product","record"])
def test_two_pass_tampering_blocks_approval(tmp_path,target):
    result=service.StudioResults(two_pass_batch(tmp_path/"batch"))
    path=result.raw_path if target=="raw" else result.current_path if target=="product" else result.candidate.record_path
    path.write_bytes(b"{}" if target=="record" else b"changed")
    with pytest.raises(ValueError): result.approve(dict(character=True,garment=True,exposure=True))


def test_cpu_pose_uses_existing_normalization_and_control_pixels(tmp_path):
    from genai_lab.onepass_pose import PoseObservation, prepare_onepass_pose
    from test_onepass_inputs import standard_joints
    source=tmp_path/"source.png"
    Image.new("RGB",(100,240),(30,80,110)).save(source)
    request={"source_file":str(source),"source_sha256":sha(source),"tag_scores":{"profile":.9}}
    def estimate(image):
        from PIL import ImageDraw
        control=Image.new("RGB",image.size)
        ImageDraw.Draw(control).line((30,30,50,200),fill=(255,0,0),width=3)
        return PoseObservation(standard_joints(),1,image.convert("RGB"),control)
    folder=tmp_path/"pose";folder.mkdir()
    saved=worker.save_pose(request,folder,estimate)
    with Image.open(source) as image:
        expected=prepare_onepass_pose(image,estimate=estimate,tag=lambda _:request["tag_scores"])
    try:
        with Image.open(saved["normalized_file"]) as actual:
            assert actual.tobytes()==expected.normalized_image.tobytes()
        with Image.open(saved["control_file"]) as actual:
            assert actual.tobytes()==expected.control_image.tobytes()
        assert saved["crop_box"]==expected.crop_box and saved["confirmed"] is False
    finally: expected.close()


def test_cpu_preparation_uses_separate_offline_interpreters_and_retains_json(tmp_path,monkeypatch):
    runtime=service.StudioRuntime(pose_python=Path(__import__("sys").executable),analysis_python=Path(__import__("sys").executable))
    calls=[]
    def execute(command,log,cancelled,timeout):
        calls.append(command)
        request=json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        assert request=={"source_file":"example.png"}
        write_json(log.parent/"result.json",{"mode":command[-2]})
    monkeypatch.setattr(service,"cpu_process",execute)
    for mode in ("pose","head"):
        assert prep.run_cpu_preparation(mode,{"source_file":"example.png"},runtime,tmp_path/mode,
                                       cancelled=lambda:False)=={"mode":mode}
    assert calls[0][-2]=="pose" and calls[1][-2]=="head"
    assert calls[0][0]==str(runtime.pose_python) and calls[1][0]==str(runtime.analysis_python)


def test_head_retry_discards_approval_and_reopens_selection(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    _,pose,draft=prepared(tmp_path)
    window=GenAILabWindow()
    window.studio.prepared_proportion_pose=pose
    monkeypatch.setattr(ui,"review_head_outline",lambda *a:"retry")
    seen=[]
    monkeypatch.setattr(window.studio,"proportion_pose_ready",lambda value:seen.append(value))
    monkeypatch.setattr(ui,"confirmed_options",lambda *a:pytest.fail("retry is not approval"))
    try:
        window.studio.proportion_head_ready(draft)
        assert seen==[pose] and window.studio.task is None
    finally: window.close()


def test_head_point_uses_display_to_source_coordinates(tmp_path):
    app=QApplication.instance() or QApplication([])
    _,pose,_=prepared(tmp_path)
    canvas=gui.HeadCanvas(pose["normalized_file"],pose["nose"])
    canvas.resize(400,600);canvas.show();app.processEvents()
    rect=canvas.image_rect()
    pos=QPoint(round(rect.x()+rect.width()*.5),round(rect.y()+rect.height()*.2))
    QTest.mouseClick(canvas,Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier,pos)
    assert abs(canvas.face_point[0]-368)<=2 and abs(canvas.face_point[1]-246.4)<=2
    canvas.close()


def test_pose_contract_can_load_without_gui_package():
    """분리된 자세 실행 환경에는 의도적으로 Qt를 설치하지 않는다."""
    import subprocess
    import sys
    code = """
import builtins
original = builtins.__import__
def isolated_import(name, *args, **kwargs):
    if name == 'PySide6' or name.startswith('PySide6.'):
        raise ImportError('GUI packages must not enter pose preparation')
    return original(name, *args, **kwargs)
builtins.__import__ = isolated_import
from genai_lab.studio_proportion import verify_pose
assert callable(verify_pose)
"""
    result = subprocess.run([sys.executable, '-B', '-c', code],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def low_neck(tmp_path):
    """윤곽은 맞지만 검출 목이 너무 아래인 사례다. 2026-10-09 목 장식 사례를 재현한다."""
    analysis,pose,draft=prepared(tmp_path)
    from genai_lab.proportion_inputs import NECK_GAP_MESSAGE
    pose["neck"]=[340.,300.]
    draft["head"]["neck"]=[340.,300.]
    draft["geometry_error"]=NECK_GAP_MESSAGE
    return analysis,pose,draft


def test_neck_gap_blocks_without_override_and_passes_only_with_user_confirmation(tmp_path):
    from genai_lab.proportion_inputs import validate_head, NECK_GAP_MESSAGE
    _,pose,draft=low_neck(tmp_path)
    head=replace(prep.restore_head(draft),confirmed=True)
    inputs=SimpleNamespace(control_sha256=pose["control_sha256"])
    with pytest.raises(ValueError,match=NECK_GAP_MESSAGE):
        validate_head(head,inputs,(736,1232))
    validate_head(replace(head,geometry_override="user_confirmed"),inputs,(736,1232))
    with pytest.raises(ValueError):
        replace(head,geometry_override="auto")


def test_override_never_bypasses_other_geometry_checks(tmp_path):
    from genai_lab.proportion_inputs import validate_head
    _,pose,draft=prepared(tmp_path)
    head=replace(prep.restore_head(draft),confirmed=True,nose=(100.,600.),geometry_override="user_confirmed")
    with pytest.raises(ValueError,match="코 좌표가 머리 밖"):
        validate_head(head,SimpleNamespace(control_sha256=pose["control_sha256"]),(736,1232))


def test_confirmed_options_records_neck_override(tmp_path):
    _,pose,draft=low_neck(tmp_path)
    options=prep.confirmed_options(pose,draft,service.StudioRuntime())
    assert options.head.geometry_override=="user_confirmed"
    record=json.loads((Path(options.head.contour_file).parent/"user-confirmation.json").read_text(encoding="utf-8"))
    assert record["geometry_override"]=="user_confirmed" and record["automatic_approval"] is False
    assert record["geometry_error_confirmed_past"]==draft["geometry_error"]


def test_confirmed_options_without_neck_error_sets_no_override(tmp_path):
    _,pose,draft=prepared(tmp_path)
    options=prep.confirmed_options(pose,draft,service.StudioRuntime())
    assert options.head.geometry_override is None


def test_review_shows_neck_warning_and_allows_confirmation(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    _,pose,draft=low_neck(tmp_path)
    from PySide6.QtWidgets import QLabel
    def review(dialog):
        assert dialog.findChild(QLabel,"head_geometry_override_notice") is not None
        accept=next(b for b in dialog.findChildren(QPushButton) if b.text()=="확인한 비율로 4장 만들기")
        assert not accept.isEnabled()
        dialog.findChild(QCheckBox,"head_outline_confirmed").setChecked(True)
        assert accept.isEnabled()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog,"exec",review)
    assert gui.review_head_outline(None,pose,draft)=="confirm"
