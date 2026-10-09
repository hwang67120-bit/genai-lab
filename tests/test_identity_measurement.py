import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from genai_lab.identity_measurement import measure_pixels
from genai_lab.identity_report import compare_measurements, summary


def synthetic():
    image=np.full((200,120,3),255,np.uint8);alpha=np.zeros((200,120),np.uint8)
    alpha[10:190,40:80]=255;image[10:190,40:80]=[160,110,70]
    image[10:23,40:80]=[20,30,120]
    coordinates={'nose':(60,32),'neck':(60,50),'right_shoulder':(45,55),'left_shoulder':(75,55),
        'right_hip':(50,110),'left_hip':(70,110),'right_knee':(50,145),'left_knee':(70,145),
        'right_ankle':(50,185),'left_ankle':(70,185),'right_eye':(52,30),'left_eye':(68,30)}
    pose={'joints':[dict(joint_name=n,x=x,y=y,confidence_score=1) for n,(x,y) in coordinates.items()]}
    return Image.fromarray(image),pose,alpha,(40,10,80,50)


def test_body_formulas_and_invalid_merged_limbs():
    image,pose,alpha,head=synthetic();measurement,art=measure_pixels(image,pose,alpha,head)
    assert measurement['B1']==4.475
    assert measurement['B2']==.75
    assert measurement['B3']['w']==.988
    assert measurement['B3']['bare'] is True
    assert measurement['B6']['invalid']=='no_valid_side'
    assert not set(measurement)&{'H1','H2','H3','H4'}
    assert art['hair_quantiles'].shape==(101,3)


@pytest.mark.parametrize('missing',['no_head','no_nose','empty_foreground'])
def test_missing_inputs_are_not_perfect_matches(missing):
    image,pose,alpha,head=synthetic()
    if missing=='no_head':head=None
    if missing=='no_nose':pose={'joints':[]}
    if missing=='empty_foreground':alpha[:]=0
    m,_=measure_pixels(image,pose,alpha,head)
    report=compare_measurements(m,m)
    assert missing in m['notes']
    assert report['body_difference'] is None and report['unavailable_count']==9


def test_comparison_exposure_mismatch_and_hair_quantiles():
    a={'B1':4,'B2':1,'B3':{'w':1,'bare':True}}
    b={'B1':5,'B2':1.2,'B3':{'w':2,'bare':False}}
    report=compare_measurements(a,b,np.zeros((101,3)).tolist(),np.ones((101,3)).tolist())
    assert report['items']['B3']['reason']=='exposure_mismatch'
    assert report['items']['B1']['difference']==.25
    assert report['body_difference']==.2
    assert report['hair_distance']==1.73
    assert '20.0%' in summary(report)
    json.dumps(report,allow_nan=False)
