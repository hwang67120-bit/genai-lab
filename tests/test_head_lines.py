"""CPU contracts for locked Canny, face exclusion and preview-before-generation."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import cv2
import numpy as np
from PIL import Image
import pytest
from genai_lab import head_lines as lines
from genai_lab import proportion_inputs as contract
from genai_lab import proportion_generation as flow
from scripts import head_lines_trial as trial
from genai_lab.qwen_record_io import write_json
from test_proportion_generation import case,factories,Foreground
from test_outline_source_trial import studio_at


def face_stub(monkeypatch, options, *, ratio=None):
    mask=np.zeros((1232,736),dtype=bool)
    mask[135:205,290:390]=True
    if ratio=='large':mask[40:220,220:460]=True
    if ratio=='empty':mask[:]=False
    monkeypatch.setattr(lines,'sam_record',lambda _:{'id':'fake_cpu','revision':'test','files':{},'device':'cpu'})
    monkeypatch.setattr(lines,'segment_face',lambda *a:(mask,.9))
    return lambda:object()


def joint(head):return {'detected':True,'confidence_score':.9,'x':head.nose[0],'y':head.nose[1],'joint_name':'nose'}


def prepared(case,tmp_path,monkeypatch):
    inputs,settings,options=case
    pixels=np.asarray(Image.open(options.head.normalized_file)).copy()
    cv2.line(pixels,(260,55),(290,210),(255,255,255),2)
    cv2.line(pixels,(280,70),(340,180),(0,0,0),3)
    Image.fromarray(pixels).save(options.head.normalized_file)
    options=replace(options,head=replace(options.head,normalized_sha256=contract.sha(options.head.normalized_file)))
    record=lines.prepare_head_lines(options.head,joint(options.head),tmp_path/'lines',face_stub(monkeypatch,options))
    path=tmp_path/'lines/head-lines.json'
    return inputs,settings,replace(options,head_lines_file=path,head_lines_sha256=contract.sha(path)),record


def test_default_none_does_not_open_files_or_run_recognition(monkeypatch):
    monkeypatch.setattr(lines,'checked_image',lambda *a:pytest.fail('none must not inspect'))
    sketch=np.zeros((12,8,3),dtype=np.uint8)
    assert lines.compose_head_lines(sketch,contract.ProportionOptions())[0] is sketch
    with pytest.raises(ValueError):contract.ProportionOptions(head_lines='auto')


def test_fixed_lines_are_deterministic_clipped_and_small_noise_removed():
    rgb=np.zeros((120,120,3),dtype=np.uint8);mask=np.zeros((120,120),np.uint8);mask[10:110,10:110]=255
    rgb[20:90,30:34]=255;rgb[100,100]=255;rgb[:,1:4]=255
    first=lines.extract_lines(rgb,mask);second=lines.extract_lines(rgb,mask)
    assert np.array_equal(first,second) and first.any() and not first[mask==0].any()
    assert not first[95:105,95:105].any()
    assert set(np.unique(first))=={0,255}


def test_hair_excludes_dilated_face_and_locks_parameters(case,tmp_path,monkeypatch):
    _,_,options,record=prepared(case,tmp_path,monkeypatch)
    assert record['rules']==lines.RULES and record['sam_input']['label']==1
    assert record['sam_input']['box']==(220,40,460,220)
    assert record['status']=='needs_user_review'
    with pytest.raises(ValueError,match='사용자 확인'):lines.checked_lines(replace(options,head_lines='hair'),(736,1232))
    hair,_=lines.checked_lines(replace(options,head_lines='hair',head_lines_confirmed=True),(736,1232))
    all_lines,_=lines.checked_lines(replace(options,head_lines='all',head_lines_confirmed=True),(736,1232))
    face=np.asarray(Image.open(record['files']['face']['path']))
    excluded=cv2.dilate(face,np.ones((5,5),np.uint8))>0
    assert not hair[excluded].any() and np.count_nonzero(all_lines)>np.count_nonzero(hair)>0
    assert record['line_pixels']['hair']==int(np.count_nonzero(hair))


@pytest.mark.parametrize('failure',['nose','empty','large'])
def test_rejected_masks_keep_reason_and_never_guess(case,tmp_path,monkeypatch,failure):
    options=case[2];nose=joint(options.head)
    if failure=='nose':nose['confidence_score']=.29
    with pytest.raises(ValueError):
        lines.prepare_head_lines(options.head,nose,tmp_path/'bad',face_stub(monkeypatch,options,ratio=failure))
    record=trial.read(tmp_path/'bad/head-lines.json')
    assert record['status']=='failed' and record['gpu_generation'] is False
    assert {'nose':'nose_unavailable','empty':'face_empty','large':'face_too_large'}[failure] in record['error']
    if failure!='nose':assert (tmp_path/'bad/mask-preview.png').is_file()


@pytest.mark.parametrize('change',['line','preview','head','record'])
def test_changed_preparation_is_rejected(case,tmp_path,monkeypatch,change):
    _,_,options,record=prepared(case,tmp_path,monkeypatch)
    options=replace(options,head_lines='all',head_lines_confirmed=True)
    path={'line':record['files']['all']['path'],'preview':record['files']['all-preview']['path'],
          'head':options.head.mask_file,'record':options.head_lines_file}[change]
    Path(path).write_bytes(b'changed')
    with pytest.raises(ValueError):lines.checked_lines(options,(736,1232))


def test_none_keeps_existing_sketch_and_generation_bytes(case,tmp_path):
    inputs,settings,options=case
    results=[]
    for name,opt in (('default',options),('none',replace(options,head_lines='none'))):
        result=flow.generate_proportion_batch(inputs,(1,2),tmp_path/name,settings=settings,options=opt,**factories(case,[]))
        results.append([(c.raw.path.read_bytes(),c.path.read_bytes(),(tmp_path/name/f'sketch_{c.seed}.png').read_bytes()) for c in result.candidates])
    assert results[0]==results[1]


def test_product_uses_reviewed_lines_and_records_their_source(case,tmp_path,monkeypatch):
    inputs,settings,options,record=prepared(case,tmp_path,monkeypatch)
    active=replace(options,head_lines='all',head_lines_confirmed=True)
    batch=flow.generate_proportion_batch(inputs,(1,),tmp_path/'run',settings=settings,options=active,**factories((inputs,settings,active),[]))
    metadata=batch.candidates[0].record['sketch']['head_lines']
    assert metadata['mode']=='all' and metadata['rules']==lines.RULES
    assert metadata['record_sha256']==active.head_lines_sha256
    sketch=np.asarray(Image.open(tmp_path/'run/sketch_1.png'))[...,0]
    inner=np.asarray(Image.open(record['files']['all']['path']))
    assert np.all(sketch[inner>0]==255)


def studio_case(case,tmp_path):
    folder=tmp_path/'studio';inputs,settings,options=studio_at(case,folder)
    confirmation=options.head.contour_file.parent/'user-confirmation.json'
    data=trial.read(confirmation);data['pose']['joints']=[joint(options.head)]
    write_json(confirmation,data)
    return trial.load_case(folder,None)


def test_trial_reuses_first_two_c0_seeds_and_generates_only_contour(case,tmp_path,monkeypatch):
    source=studio_case(case,tmp_path)
    monkeypatch.setattr(trial,'run_prepared',trial.run_prepared)
    folder=tmp_path/'prepared'
    lock,digest=trial.prepare_case(source,folder,face_stub(monkeypatch,source[2]),lambda opt:Foreground(opt,[]))
    assert lock['seeds']==[1,2] and lock['new_generation_count']==4 and set(lock['modes'])=={'hair','all'}
    assert trial.read(folder/'status.json')['status']=='needs_user_review'
    with pytest.raises(ValueError,match='미리보기'):trial.run_prepared(folder,None,tmp_path/'unapproved')
    events=[];providers=factories(case,events)
    result=trial.run_prepared(folder,digest,tmp_path/'gpu-mock',contour_factory=providers['contour_factory'],foreground_factory=providers['foreground_factory'])
    assert result['status']=='review_pending' and 'base-open' not in events and events.count('contour-open')==2
    assert all(row['CONTOUR']==[1,2] and row['BASE']==[] for row in result['conditions'].values())
    with pytest.raises(FileExistsError):trial.run_prepared(folder,digest,tmp_path/'gpu-mock')


def test_trial_rejects_modified_preview_before_any_gpu(case,tmp_path,monkeypatch):
    source=studio_case(case,tmp_path);folder=tmp_path/'prepare'
    lock,digest=trial.prepare_case(source,folder,face_stub(monkeypatch,source[2]),lambda opt:Foreground(opt,[]))
    Path(lock['modes']['hair']['maps']['1']['preview_file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='변경'):
        trial.run_prepared(folder,digest,tmp_path/'gpu',contour_factory=lambda *a:pytest.fail('GPU'))
    assert not (tmp_path/'gpu').exists()


def test_cli_cannot_prepare_and_generate_without_preview_confirmation(tmp_path):
    with pytest.raises(ValueError,match='CPU 미리보기'):
        trial.main(['--r6','--run','--output',str(tmp_path/'out')])


def test_trial_generation_failure_records_status_and_stops_other_condition(case,tmp_path,monkeypatch):
    source=studio_case(case,tmp_path);folder=tmp_path/'prepare'
    _,digest=trial.prepare_case(source,folder,face_stub(monkeypatch,source[2]),lambda opt:Foreground(opt,[]))
    calls=[]
    def fail(*args):calls.append(args);raise RuntimeError('load failed')
    with pytest.raises(RuntimeError,match='load failed'):
        trial.run_prepared(folder,digest,tmp_path/'gpu',contour_factory=fail)
    assert len(calls)==1 and not (tmp_path/'gpu/all').exists()
    assert trial.read(tmp_path/'gpu/hair/status.json')['status']=='failed'
    assert trial.read(tmp_path/'gpu/status.json')['status']=='failed'
