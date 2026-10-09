from dataclasses import replace
import json
import numpy as np
import pytest
from genai_lab.body_widths import match_widths, width_scale
from genai_lab.proportion_inputs import ProportionOptions
from test_identity_measurement import synthetic
from test_proportion_generation import case, factories


@pytest.mark.parametrize('original,expected',[(.5,'scale_out_of_range'),(1.5,'scale_out_of_range')])
def test_out_of_range_is_skipped(original,expected):
    item=width_scale({'B2':original},{'B2':1},'B2')
    assert not item['applied'] and item['reason']==expected


def test_exposure_mismatch_is_skipped():
    a={'B3':{'w':1,'bare':True}};b={'B3':{'w':1,'bare':False}}
    assert width_scale(a,b,'B3')['reason']=='exposure_mismatch'


def test_capsules_and_protected_head_remain_unchanged():
    _,pose,alpha,_=synthetic();base={'H':40,'B2':.75,'B3':{'w':1,'bare':False}}
    source={'H':40,'B2':.6,'B3':{'w':.8,'bare':False}}
    protected=np.zeros_like(alpha,bool);protected[:51]=True
    changed,record,support=match_widths(alpha,source,base,pose,protected=protected)
    assert np.array_equal(changed[~support],alpha[~support])
    assert np.array_equal(changed[protected],alpha[protected])
    assert np.count_nonzero(changed>127)<np.count_nonzero(alpha>127)
    assert any(x['applied'] for x in record['items'])


def test_options_reject_other_paths():
    assert ProportionOptions().body_widths=='off'
    with pytest.raises(ValueError):ProportionOptions(body_widths='match_original')
    with pytest.raises(ValueError):ProportionOptions(enabled=True,body_widths='match_original',head_lines='hair')


def test_trial_shares_first_stage_and_off_never_calls_matching(case,tmp_path,monkeypatch):
    from scripts.body_width_trial import generate_comparison
    import genai_lab.body_widths as module
    calls=[]
    def adjust(path,alpha,options,directory,**kw):
        calls.append(options.body_widths)
        return alpha,{'mode':'match_original','path':'test','sha256':'fake'}
    monkeypatch.setattr(module,'adjust_base_widths',adjust)
    inputs,settings,options=case;events=[]
    output=tmp_path/'trial'
    result=generate_comparison(inputs,settings,options,(1,2),output,**factories(case,events))
    assert result['status']=='review_pending'
    assert events.count('base-open')==1 and events.count('contour-open')==2
    assert calls==['match_original','match_original']
    for seed in (1,2):
        assert (output/f'W0/CONTOUR_{seed}/raw.png').read_bytes()==(output/f'W1/CONTOUR_{seed}/raw.png').read_bytes()
    assert 'body_widths' not in json.loads((output/'W0/maps.json').read_text())['1']
    assert json.loads((output/'W1/maps.json').read_text())['1']['body_widths']['mode']=='match_original'
