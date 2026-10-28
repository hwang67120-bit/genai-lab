"""어깨 미리보기는 파생 입력이며 머리 기준의 원본을 재작성하지 않는다."""
from dataclasses import replace
import numpy as np
import pytest
from PIL import Image
from genai_lab import shoulder_control as shoulder
from genai_lab.proportion_inputs import ProportionOptions, sha
from test_proportion_generation import case, factories
from genai_lab.proportion_generation import generate_proportion_batch


def skeleton():
    points=[None]*18
    for i,p in {0:(340,170),1:(340,230),2:(240,280),3:(200,430),4:(190,550),5:(440,280),6:(480,430),7:(490,550),8:(300,600),11:(380,600)}.items():points[i]=p
    joints=[{'joint_name':name,'x':0 if p is None else p[0],'y':0 if p is None else p[1],
        'detected':p is not None,'confidence_score':0 if p is None else 1} for name,p in zip(shoulder.NAMES,points)]
    return np.ascontiguousarray(shoulder.draw_body(points,1232,736)[...,::-1]),joints


def prepared(case,tmp_path):
    inputs,settings,options=case;pixels,joints=skeleton();Image.fromarray(pixels).save(inputs.control_file)
    digest=sha(inputs.control_file);inputs=replace(inputs,control_sha256=digest)
    options=replace(options,head=replace(options.head,control_sha256=digest))
    record=shoulder.prepare_shoulder_control(inputs.control_file,digest,joints,options.head,tmp_path/'shoulder')
    path=tmp_path/'shoulder/shoulder.json'
    options=replace(options,shoulder_pull='0.85',shoulder_record_file=path,shoulder_record_sha256=sha(path))
    return inputs,settings,options,record


def test_default_off_and_invalid_combinations(case):
    inputs,_,options=case
    assert shoulder.resolve_shoulder_inputs(inputs,options)==(inputs,None)
    with pytest.raises(ValueError):ProportionOptions(shoulder_pull='0.85')
    with pytest.raises(ValueError):replace(options,shoulder_pull='0.85',body_widths='match_original')
    with pytest.raises(ValueError):replace(options,shoulder_pull='0.70')


def test_real_reconstruction_detects_corrupted_body_and_keeps_parent(case,tmp_path):
    inputs,settings,options,record=prepared(case,tmp_path)
    assert record['status']=='applied'
    child,details=shoulder.resolve_shoulder_inputs(inputs,options)
    assert options.head.control_sha256==inputs.control_sha256!=child.control_sha256
    before=np.asarray(Image.open(inputs.control_file));after=np.asarray(Image.open(child.control_file))
    assert np.array_equal(before[600:],after[600:])
    damaged=before.copy();damaged[850,50]=(255,0,0)
    with pytest.raises(ValueError,match='영역 밖'):shoulder.correct_pixels(damaged,record['joints'],np.asarray(Image.open(options.head.mask_file)))
    Image.fromarray(damaged).save(inputs.control_file)
    fallback=shoulder.prepare_shoulder_control(inputs.control_file,sha(inputs.control_file),record['joints'],options.head,tmp_path/'failed')
    assert fallback['status']=='not_applied' and '적용하지 못함' in fallback['warning']
    assert fallback['control_sha256']==sha(inputs.control_file)


def test_two_stages_share_child_and_keep_parent_in_record(case,tmp_path):
    inputs,settings,options,record=prepared(case,tmp_path)
    batch=generate_proportion_batch(inputs,(1,),tmp_path/'run',settings=settings,options=options,**factories((inputs,settings,options),[]))
    import json
    for name in ('BASE_1','CONTOUR_1'):
        data=json.loads((tmp_path/'run'/name/'run.json').read_text(encoding='utf-8'))
        assert data['shoulder_correction']['control_sha256']==record['control_sha256']
        assert data['shoulder_correction']['parent_sha256']==inputs.control_sha256
    assert batch.candidates[0].record['shoulder_correction']['status']=='applied'
