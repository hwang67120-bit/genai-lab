"""보정 전후의 검출 순서·구도·미확인 상태가 자동 채택 경계를 넘지 않는지 검사한다."""
import copy
import json
import numpy as np
import pytest
from genai_lab.style_redraw_arm_evidence import assess_arm_evidence
from genai_lab.style_redraw_anatomy import assess_hands, attach_arm_evidence, candidate_anatomy
from genai_lab.style_redraw_correction_evidence import guard_correction_anatomy


def record():
    p = dict(state="observed", points=[[[50,10],[50,30],[30,30],[20,50],[10,70],
                                       [70,30],[80,50],[90,70]]], scores=[[.9]*8])
    boxes = [[5,65,15,75],[85,65,95,75]]
    a = attach_arm_evidence(assess_hands(2), assess_arm_evidence(p,boxes,100,100))
    a.update(image_size=[100,100], detections=[dict(box=b, confidence=.9) for b in boxes])
    return a


def compare(before, after):
    return guard_correction_anatomy(before, after, (100,100))


def issues(a):
    return a['arm_evidence']['correction_comparison']['issues']


def test_same_geometry_different_detection_order_remains_eligible_not_approved():
    a=record(); b=copy.deepcopy(a);b['detections'].reverse()
    result=compare(a,b)
    assert result['automatic_correction_eligible'] and not result['user_quality_approved']
    assert not issues(result)
    assert 'correction_comparison' not in a['arm_evidence']


def test_same_hand_count_does_not_hide_new_hand_location():
    a=record();b=copy.deepcopy(a);b['detections'][0]['box']=[35,65,45,75]
    result=compare(a,b)
    assert 'hand_position_changed' in issues(result)
    assert not result['automatic_correction_eligible']
    assert not candidate_anatomy(dict(hands=2,anatomy=result))['automatic_correction_eligible']


def test_prior_uncertainty_cannot_be_cleared_by_better_clothing_score():
    a=record();a['automatic_correction_eligible']=False
    a['arm_evidence']['issues']=['two_wrists_share_one_hand_detection']
    result=compare(a,record())
    assert 'prior_anatomy_not_cleared' in issues(result)
    assert result['eligible_for_review'] and not result['automatic_correction_eligible']


def test_extra_hand_appearance_held_without_calling_occlusion_a_defect():
    a=record();a['hands']=1;a['detections']=a['detections'][:1]
    result=compare(a,record())
    assert 'hand_visibility_changed' in issues(result)
    assert result['arm_evidence']['correction_comparison']['reference_role']=='comparison_only_not_ground_truth'


def test_elbow_motion_held_even_when_hands_stay_put():
    a=record();b=copy.deepcopy(a);b['arm_evidence']['arms'][0]['joints']['elbow']['x']+=20
    assert 'arm_position_changed' in issues(compare(a,b))


@pytest.mark.parametrize('problem',['missing','nan','dimensions','unreliable','empty'])
def test_missing_evidence_cannot_pass_by_equal_hand_count(problem):
    a=record();b=copy.deepcopy(a)
    if problem=='missing':b.pop('detections')
    elif problem=='nan':b['detections'][0]['box'][0]=float('nan')
    elif problem=='dimensions':b['image_size']=[101,100]
    elif problem=='unreliable':b['arm_evidence']['arms'][0]['joints']['wrist']['reliable']=False
    else:b['arm_evidence']['arms']=[]
    result=compare(a,b)
    assert 'correction_geometry_unassessed' in issues(result)
    assert not result['automatic_correction_eligible']
    json.dumps(result['arm_evidence']['correction_comparison'],allow_nan=False)


def test_scale_independent_geometry_and_small_detection_jitter():
    a=record();b=copy.deepcopy(a)
    b['arm_evidence']['arms'][0]['joints']['elbow']['x']+=1
    b['detections'][0]['box'][0]+=1
    assert compare(a,b)['automatic_correction_eligible']
    for v in (a,b):
        v['image_size']=[1000,1000]
        for arm in v['arm_evidence']['arms']:
            for j in arm['joints'].values():j['x']*=10;j['y']*=10
        for d in v['detections']:d['box']=[x*10 for x in d['box']]
    assert guard_correction_anatomy(a,b,(1000,1000))['automatic_correction_eligible']


def test_elbow_boundary_detected_without_visible_hand_boundary():
    from genai_lab.style_redraw_arm_evidence import assess_arm_framing
    a=record();a['arm_evidence']['arms'][0]['joints']['elbow']['x']=1
    f=assess_arm_framing(a['arm_evidence']['arms'],[d['box'] for d in a['detections']],100,100)
    assert f['state']=='boundary_risk' and not f['crop_confirmed']
    assert f['boundary'][0]['joints']==['elbow']


@pytest.mark.parametrize('phase',['autofix','face-seed42','colour-seed42','furry-seed42'])
def test_product_inspector_persists_pairwise_evidence(tmp_path,monkeypatch,phase):
    from genai_lab import style_redraw_worker as w
    a=record()
    p=dict(state='observed',points=[[[50,10],[50,30],[30,30],[20,50],[10,70],
                                  [70,30],[80,50],[90,70]]],scores=[[.9]*8])
    monkeypatch.setattr(w,'observe_arm_poses',lambda *args:[p])
    monkeypatch.setattr(w,'hand_detections',lambda *args:[[(d['box'],'hand',.9) for d in a['detections']]])
    a['automatic_correction_eligible']=False
    result=w.inspect_anatomy(tmp_path,[np.zeros((100,100,3),np.uint8)],{},phase,previous=[a])[0]
    saved=json.loads((tmp_path/f'anatomy-{phase}.json').read_text(encoding='utf-8'))[0]
    assert result==saved
    assert 'prior_anatomy_not_cleared' in issues(saved)
    assert not saved['automatic_correction_eligible']


def test_wrong_candidate_pairing_fails_before_detector(tmp_path):
    from genai_lab.style_redraw_worker import inspect_anatomy
    with pytest.raises(ValueError,match='기록 수'):
        inspect_anatomy(tmp_path,[np.zeros((2,2,3),np.uint8)],{},'autofix',previous=[])
