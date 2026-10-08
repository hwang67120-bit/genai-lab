"""CPU-only geometry, prompt delivery, real generation recording and failure contracts."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from PIL import Image
import pytest

from genai_lab.body_proportions import (
    BodyProportionMeasurement, BodyProportionReference, compare_body_proportions,
    image_digest, measure_visible_proportions, suggest_proportions_from_pose)
from genai_lab.body_proportion_trial import (
    generate_body_proportion_trial, prepare_body_proportion_inputs,
    review_body_proportion_candidate)
from genai_lab.onepass_generation import OnePassInputs, OnePassCancelled, OnePassGenerationError
from genai_lab.onepass_generation_settings import OnePassGenerationSettings
from genai_lab.onepass_prompt import OnePassPrompt, plan_prompt_chunks


class Tokenizer:
    bos_token_id=1
    eos_token_id=2
    pad_token_id=2
    def __call__(self,text,**kwargs):
        return {'input_ids':[3+sum(map(ord,w)) for w in text.split()]}

TOKS=(Tokenizer(),Tokenizer())


def measured(path, heads=4, reviewed=True, shoulders=True):
    return measure_visible_proportions(path,head_box=(50,0,150,400/heads),sole_y=400,
        shoulders=((50,160),(150,160)) if shoulders else None,geometry_reviewed=reviewed)


@pytest.fixture
def case(tmp_path):
    source=tmp_path/'source.png';face=tmp_path/'face.png';control=tmp_path/'control.png'
    Image.new('RGB',(200,400),'white').save(source)
    Image.new('RGB',(100,100),'pink').save(face)
    Image.new('RGB',(200,400),'black').save(control)
    positive='1boy, male focus, masculine silhouette, white shirt, white pants, blue hair'
    negative='1girl, bad anatomy'
    prompt=OnePassPrompt(positive,negative,plan_prompt_chunks(positive,negative,TOKS),{'prefix':('1boy','male focus','masculine silhouette')})
    inputs=OnePassInputs(prompt,control,face,image_digest(control),image_digest(face),.5,pose_mode='with_pose')
    reference=BodyProportionReference(measured(source),inputs.face_sha256,confirmed=True)
    return source,inputs,reference,OnePassGenerationSettings(width=200,height=400)


class Backend:
    callback_mode='cpu_test'
    pipe=SimpleNamespace(model_cpu_offload_seq='unchanged')
    def __init__(self,settings,fail=False):self.settings=settings;self.fail=fail;self.received=None
    def generate(self,inputs,images,seed,observation,cancelled):
        self.received=inputs
        if self.fail:raise RuntimeError('generation failed')
        for i in range(self.settings.steps):
            observation.calls.append(dict(index=i,ip_scales=[inputs.ip_early if i<self.settings.ip_start else self.settings.ip_scale],adapter_applied=i<self.settings.adapter_steps))
        return Image.new('RGB',(200,400),'blue'),dict(generation_seconds=0,max_memory_reserved_bytes=0)


def run(case,tmp_path,measure=measured,**kwargs):
    source,inputs,reference,settings=case
    return generate_body_proportion_trial(Backend(settings),inputs,reference,source_image=source,
        tokenizers=TOKS,seed=7,directory=tmp_path/'run',measure_output=measure,settings=settings,**kwargs)


def test_disabled_returns_identical_request(case):
    assert prepare_body_proportion_inputs(case[1]) is case[1]


def test_prompt_delivery_preserves_every_other_input(case):
    _,inputs,ref,_=case
    actual=prepare_body_proportion_inputs(inputs,ref,tokenizers=TOKS)
    assert actual.prompt.positive == inputs.prompt.positive+', body proportions of 4 head lengths tall, shoulders 1 head widths wide'
    assert actual.prompt.negative == inputs.prompt.negative
    before,after=asdict(inputs),asdict(actual)
    before.pop('prompt');after.pop('prompt')
    assert before==after
    assert all(a.negative==b.negative for a,b in zip(inputs.prompt.encoders,actual.prompt.encoders))
    assert actual.prompt.rules['body_proportion_condition']['spatial_constraint'] is False
    with pytest.raises(ValueError):prepare_body_proportion_inputs(actual,ref,tokenizers=TOKS)


@pytest.mark.parametrize('count',[72,80,160])
def test_long_positive_rechunks_without_losing_negative(case,count):
    _,inputs,ref,_=case
    positive=' '.join(['tag']*count)
    prompt=replace(inputs.prompt,positive=positive,encoders=plan_prompt_chunks(positive,inputs.prompt.negative,TOKS))
    actual=prepare_body_proportion_inputs(replace(inputs,prompt=prompt),ref,tokenizers=TOKS)
    assert actual.prompt.encoders==plan_prompt_chunks(actual.prompt.positive,prompt.negative,TOKS)
    for plan in actual.prompt.encoders:
        assert len(plan.positive)==len(plan.negative)
        assert plan.negative[0].text==prompt.negative
        assert all(len(c.token_ids)==77 for c in (*plan.positive,*plan.negative))


@pytest.mark.parametrize('change',[{'confirmed':False},{'face_sha256':'0'*64}])
def test_unconfirmed_or_wrong_face_rejected(case,change):
    with pytest.raises(ValueError):prepare_body_proportion_inputs(case[1],replace(case[2],**change),tokenizers=TOKS)


def test_missing_sole_not_invented(case):
    source,inputs,_,_=case
    observation=measure_visible_proportions(source,head_box=(50,0,150,100),geometry_reviewed=True)
    assert observation.heads_tall is None
    with pytest.raises(ValueError):
        prepare_body_proportion_inputs(inputs,BodyProportionReference(observation,inputs.face_sha256,True),tokenizers=TOKS)


def test_scaling_geometry_keeps_ratios(case,tmp_path):
    small=tmp_path/'small.png';Image.new('RGB',(100,200)).save(small)
    obs=measure_visible_proportions(small,head_box=(25,0,75,50),sole_y=200,shoulders=((25,80),(75,80)),geometry_reviewed=True)
    assert (obs.heads_tall,obs.shoulder_to_head_width)==(4,1)


@pytest.mark.parametrize('bad',[(0,0,0,10),(-1,0,10,20),(0,0,300,500),(0,0,float('nan'),20)])
def test_invalid_head_boxes_rejected(case,bad):
    with pytest.raises(ValueError):measure_visible_proportions(case[0],head_box=bad)


@pytest.mark.parametrize('ratio',[0,-1,float('nan'),float('inf'),True])
def test_invalid_ratios_rejected(ratio):
    with pytest.raises(ValueError):BodyProportionMeasurement('a'*64,ratio,1,True)


def test_automatic_ankles_never_become_confirmed_heads():
    joints=[SimpleNamespace(joint_name=n,x=x,y=y,confidence_score=.9) for n,x,y in (
        ('left_ankle',80,380),('right_ankle',120,380),('left_shoulder',50,160),('right_shoulder',150,160))]
    out=suggest_proportions_from_pose(joints,(50,0,150,100),(200,400))
    assert out['heads_tall'] is None
    assert out['head_top_to_ankles_per_head_height']==3.8
    assert out['shoulder_to_head_box_width']==1
    assert out['status']=='needs_review'


def test_unreviewed_output_never_passes(case):
    out=compare_body_proportions(case[2],measured(case[0],reviewed=False))
    assert out['status']=='unmeasurable_or_unreviewed'
    assert out['product_approved'] is False


def test_generated_six_heads_fails_four_head_reference(case,tmp_path):
    result=run(case,tmp_path,measure=lambda p:measured(p,heads=6))
    assert result.comparison['status']=='outside_tolerance'
    assert result.comparison['fields']['heads_tall']['relative_error']==.5
    saved=json.loads(result.candidate.record_path.read_text())
    assert saved['valid'] is True
    assert '4 head lengths tall' in saved['prompt']['positive']
    assert saved['inputs']['face_sha256']==case[1].face_sha256
    assert result.comparison['product_approved'] is False


def test_matching_geometry_is_not_product_approval(case,tmp_path):
    result=run(case,tmp_path)
    assert result.comparison['status']=='within_tolerance'
    assert result.comparison['user_review_required']
    assert not result.comparison['product_approved']


def test_measurement_failure_preserves_successful_raw(case,tmp_path):
    def fail(path):raise RuntimeError('detector unavailable')
    result=run(case,tmp_path,measure=fail)
    assert result.comparison['status']=='measurement_failed'
    assert result.candidate.path.is_file()
    assert image_digest(result.candidate.path)==result.candidate.record['raw_sha256']
    assert result.candidate.record['valid']


def test_wrong_measurement_image_rejected(case,tmp_path):
    result=run(case,tmp_path,measure=lambda p:measured(case[0]))
    assert result.comparison['status']=='measurement_failed'


def test_review_record_is_not_overwritten(case,tmp_path):
    result=run(case,tmp_path)
    with pytest.raises(FileExistsError):review_body_proportion_candidate(result.candidate,case[2],measured)


def test_changed_source_rejected_before_backend(case,tmp_path):
    source,inputs,ref,settings=case
    source.write_bytes(b'changed')
    backend=Backend(settings)
    with pytest.raises(ValueError):
        generate_body_proportion_trial(backend,inputs,ref,source_image=source,tokenizers=TOKS,
            seed=7,directory=tmp_path/'run',measure_output=measured,settings=settings)
    assert backend.received is None


def test_generation_failure_propagates_and_keeps_run_record(case,tmp_path):
    source,inputs,ref,settings=case
    with pytest.raises(RuntimeError,match='generation failed'):
        generate_body_proportion_trial(Backend(settings,fail=True),inputs,ref,source_image=source,
            tokenizers=TOKS,seed=7,directory=tmp_path/'run',measure_output=measured,settings=settings)
    record=json.loads((tmp_path/'run/run.json').read_text())
    assert not record['valid']
    assert not (tmp_path/'run/body-proportion-review.json').exists()


def test_review_cancellation_keeps_raw_and_cancelled_record(case,tmp_path):
    def cancelled(path):raise OnePassCancelled('cancelled')
    with pytest.raises(OnePassCancelled):run(case,tmp_path,measure=cancelled)
    record=json.loads((tmp_path/'run/body-proportion-review.json').read_text())
    assert record['status']=='measurement_cancelled'
    assert (tmp_path/'run/raw.png').is_file()


@pytest.mark.parametrize('tolerance',[-1,1,float('nan'),True])
def test_invalid_tolerance_rejected_before_generation(case,tmp_path,tolerance):
    with pytest.raises(ValueError):run(case,tmp_path,relative_tolerance=tolerance)
    assert not (tmp_path/'run').exists()


def test_other_target_cannot_silently_review_generation(case,tmp_path):
    result=run(case,tmp_path)
    # A new review destination isolates the target-binding check from overwrite guard.
    result.comparison_path.unlink()
    changed=replace(case[2],measurement=replace(case[2].measurement,heads_tall=5))
    reviewed=review_body_proportion_candidate(result.candidate,changed,measured)
    assert reviewed.comparison['status']=='measurement_failed'


def test_low_confidence_and_missing_ankles_are_not_invented():
    joints=[SimpleNamespace(joint_name='left_ankle',x=80,y=380,confidence_score=.1)]
    proposal=suggest_proportions_from_pose(joints,(50,0,150,100),(200,400))
    assert proposal['head_top_to_ankles_per_head_height'] is None
    assert proposal['shoulder_to_head_box_width'] is None



def test_measurement_keeps_coordinate_evidence(case):
    observation=case[2].measurement
    assert observation.head_box==(50,0,150,100)
    assert observation.sole_y==400
    assert observation.shoulders==((50,160),(150,160))



def test_user_target_is_explicitly_not_a_reviewed_measurement(case):
    _,inputs,reference,_=case
    target=replace(reference,measurement=replace(reference.measurement,geometry_reviewed=False),basis='user_declared_target')
    prepared=prepare_body_proportion_inputs(inputs,target,tokenizers=TOKS)
    record=prepared.prompt.rules['body_proportion_condition']['reference']
    assert record['basis']=='user_declared_target'
    assert record['measurement']['geometry_reviewed'] is False
    assert '4 head lengths tall' in prepared.prompt.positive
    with pytest.raises(ValueError):replace(target,confirmed=False).require_confirmed()
    with pytest.raises(ValueError):replace(reference,basis='user_declared_target')
    with pytest.raises(ValueError):replace(reference,basis='inferred')
