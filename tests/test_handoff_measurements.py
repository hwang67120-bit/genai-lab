"""팔 겹침 측정과 볼에 연결된 머리 마스크의 회귀 검사다."""
import numpy as np
import pytest
from genai_lab.identity_measurement import arm_bands, touching_arms, measure_body
from genai_lab.identity_report import compare_measurements
from genai_lab.head_lines import cheek_skin_candidates, connected_cheeks
from test_skin_tone import face_points


def test_arm_unknown_and_frozen_fallback_width():
    bands,notes=arm_bands(np.zeros((100,100),bool),{'right_shoulder':(10,10),'right_elbow':(10,50)},20)
    assert 'left_arm_unknown' in notes and 'right_width_fallback' in notes
    assert bands[0]['r']==pytest.approx(.55*.35*20)
    assert touching_arms(np.array([[10,30]]),bands)==['right']


def test_torso_touch_is_not_a_comparison_value_and_merge_precedes_touch():
    mask=np.zeros((100,100),bool);mask[5:95,20:80]=True
    joints={'neck':(50,10),'right_hip':(40,80),'left_hip':(60,80),
        'right_shoulder':(25,10),'left_shoulder':(75,10),'right_elbow':(25,60),'right_wrist':(25,80)}
    result={'notes':[]};draw={'lines':[]}
    measure_body(mask,mask,joints,40,result,draw)
    assert result['B3']['invalid']=='arm_touch'
    assert result['B3']['touch']==['right']
    assert not compare_measurements(result,result)['items']['B3']['measurable']
    joints['right_shoulder']=(40,10);joints['left_shoulder']=(60,10)
    result={'notes':[]};draw={'lines':[]};measure_body(mask,mask,joints,40,result,draw)
    assert result['B3']['invalid']=='arm_merge'


def test_cheek_mask_ignores_highlight_and_rejects_missing_landmarks():
    rgb=np.full((40,40,3),(30,60,100),np.uint8)
    rgb[10:30,8:33]=(222,151,110);rgb[17:24,17:24]=255
    rgb[2:5,10:15]=(222,151,110)  # 관련 없는 떨어진 성분이다.
    mask=np.full((40,40),255,np.uint8)
    candidates,record=cheek_skin_candidates(rgb,mask,*face_points())
    face=connected_cheeks(candidates,record['cheek_pixels'])
    assert face[20,20] and not face[3,12]  # 밝은 빈 구멍은 채우고 떨어진 성분은 제외한다.
    with pytest.raises(ValueError):cheek_skin_candidates(rgb,mask,[],[])
