"""CPU 원인 대조가 누락·충돌·토큰 손실·중복 기록을 구분하는지 검사한다."""
from dataclasses import asdict
import json
from types import SimpleNamespace
import pytest
from scripts.audit_garment_delivery import check_prompt_delivery, generation_records, audit_run
from genai_lab.onepass_prompt import plan_prompt_chunks


class Tokenizer:
    bos_token_id=1000
    eos_token_id=1001
    pad_token_id=0
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(c) for c in text]}


def prompt(text):
    tokens=(Tokenizer(), Tokenizer())
    return dict(positive=text, negative='nsfw, underwear', rules={},
                encoders=json.loads(json.dumps([asdict(p) for p in plan_prompt_chunks(text, 'nsfw, underwear', tokens)])))


def test_missing_approved_tag_is_distinct_from_model_behavior():
    result=check_prompt_delivery(['pants', 'zipper'], prompt('pants, long sleeves'))
    assert result['missing_approved_tags']==['zipper']
    assert result['encoders_reproduced'] is None


def test_present_conflicting_tags_are_not_reported_as_delivery_failure():
    result=check_prompt_delivery(['black pants', 'skirt', 'zipper'], prompt('black pants, skirt, zipper'))
    assert not result['missing_approved_tags']
    assert [w['rule'] for w in result['warnings']]==['bottom_types']
    assert result['detail_terms']['zipper']==['zipper']
    assert not result['detail_terms']['zipper_state']
    assert not result['detail_terms']['zipper_parts']


def test_ribbon_and_necktie_are_distinct_and_coat_does_not_match_raincoat():
    result=check_prompt_delivery(['neck ribbon', 'coat'], prompt('neck ribbon, raincoat'))
    assert result['missing_approved_tags']==['coat']
    assert 'necktie' not in result['positive_terms']


def test_layered_clothes_are_not_automatically_declared_contradictory():
    result=check_prompt_delivery(['shirt', 'sweater', 'open jacket'], prompt('shirt, sweater, open jacket'))
    assert not result['warnings']
    assert not result['detail_terms']['zipper_state']


def test_long_prompt_chunks_reproduce_and_tampered_last_chunk_is_detected():
    data=prompt('white shirt, pants, zipper, '+('long sleeves, '*8)+'wrinkles')
    assert len(data['encoders'][0]['positive'])>1
    assert check_prompt_delivery(['wrinkles'],data,(Tokenizer(),Tokenizer()))['encoders_reproduced']
    data['encoders'][0]['positive'][-1]['token_ids'][1]=999
    assert not check_prompt_delivery(['wrinkles'],data,(Tokenizer(),Tokenizer()))['encoders_reproduced']


def test_declared_crop_top_is_recorded_without_claiming_unexpected_exposure():
    result=check_prompt_delivery(['crop top', 'shirt'],prompt('crop top, shirt'))
    assert result['automatic_uncovered']==('bare legs',)
    assert result['detail_terms']['waist_coverage']==['crop top']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_two_pass_product_duplicate_is_not_counted_twice(tmp_path):
    for folder in ['BASE_1','CONTOUR_1','seed-1']:
        write(tmp_path/folder/'run.json',dict(seed=1,prompt=prompt('pants')))
    rows=generation_records(tmp_path)
    assert sorted(stage for _,_,stage in rows)==['BASE','CONTOUR']
    write(tmp_path/'seed-2/run.json',dict(seed=2,prompt=prompt('pants')))
    assert len(generation_records(tmp_path))==3


def test_audit_preserves_all_inputs_and_reports_removed_approval_tags(tmp_path):
    write(tmp_path/'inputs/analysis.json',dict(garment_tags=['pants','skirt']))
    write(tmp_path/'inputs/approval.json',dict(garment_tags=['pants'],prompt='pants'))
    write(tmp_path/'inputs/references.json',dict(garment={'source':'clothing.jpg','sha256':'reference'}))
    write(tmp_path/'generation/seed-1/run.json',dict(seed=1,completed=True,prompt=prompt('pants')))
    files={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    result=audit_run(tmp_path,(Tokenizer(),Tokenizer()))
    assert result['analysis_tags_removed']==['skirt']
    assert result['files_unchanged']
    assert result['records'][0]['checks']['encoders_reproduced']
    assert all(p.read_bytes()==data for p,data in files.items())
