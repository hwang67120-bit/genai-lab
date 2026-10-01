from dataclasses import replace
from pathlib import Path
import json

import pytest
from PySide6.QtCore import QSettings

from genai_lab.character_preferences import save_character_gender
from genai_lab.onepass_gender import prepare_onepass_gender
from genai_lab.onepass_prompt import CharacterTagGroups, assemble_onepass_prompt, plan_prompt_chunks
from genai_lab.onepass_prompt_settings import OnePassPromptSettings
from genai_lab.onepass_prompt_tokenizers import load_onepass_tokenizers
from genai_lab.onepass_garment_vocabulary import garment_nouns, uncovered_parts, filter_pocket_tags
from genai_lab.onepass_character import select_pose_tags
from genai_lab.onepass_pose import Finding, PoseAssessment, apply_input_policy, findings_for_display

RECORDS=json.loads((Path(__file__).parent/'fixtures/onepass_prompt_records.json').read_text(encoding='utf-8'))

class Tokenizer:
    bos_token_id=1000
    eos_token_id=1001
    pad_token_id=1001
    def __call__(self,text,**kwargs):
        return {'input_ids':[sum(map(ord,w)) for w in text.split()]}

TOKS=(Tokenizer(),Tokenizer())

@pytest.fixture
def store(tmp_path):
    # Always pass this explicit INI store; never access actual user preferences.
    return QSettings(str(tmp_path/'gender.ini'),QSettings.Format.IniFormat)


def build(tmp_path,store,*,gender='unspecified',groups=CharacterTagGroups(),garments=('shorts',),
          poses=(),slim=False,policy='ps'):
    cfg=OnePassPromptSettings()
    cfg=replace(cfg,character=replace(cfg.character,slim_policy=policy))
    source=tmp_path/'key-only.png'
    save_character_gender(source,gender,store)
    condition=prepare_onepass_gender(source,(*groups.appearance,*groups.body,*groups.fixed),cfg.negative_template,settings=store)
    return assemble_onepass_prompt(condition,groups,garments,poses,slim=slim,settings=cfg,tokenizers=TOKS)


def strip_prefix(s):
    ts=s.split(', ')
    while ts and ts[0] in ('1girl','1boy','male focus','female focus','masculine silhouette','feminine silhouette'):
        ts.pop(0)
    return ', '.join(ts)

@pytest.mark.parametrize('row',RECORDS['main'],ids=lambda x:x['id'])
def test_recorded_whole_strings(tmp_path,store,row):
    out=build(tmp_path,store,gender=row['gender'],groups=CharacterTagGroups(**{k:tuple(v) for k,v in row['groups'].items()}),
              garments=row['garments'],poses=row['poses'],slim=row['slim'],policy=row['policy'])
    transform=(lambda x:x) if row['comparison']=='full' else strip_prefix
    assert transform(out.positive)==transform(row['positive'])
    if not row['suite'].startswith('integration'):
        assert transform(out.negative)==transform(row['negative'])

@pytest.mark.parametrize('gender,prefix,guard',[
    ('male','1boy, male focus, masculine silhouette, ','1girl, '),
    ('female','1girl, female focus, feminine silhouette, ','1boy, '),
    ('unspecified','',''),
])
def test_full_gender_strings_in_temporary_settings(tmp_path,store,gender,prefix,guard):
    out=build(tmp_path,store,gender=gender,groups=CharacterTagGroups(('blue hair',),('medium breasts',),('animal ears',)),
              garments=('white camisole','black shorts'),poses=('holding phone',))
    assert out.positive==prefix+'white camisole, black shorts, bare legs, bare shoulders, bare arms, blue hair, medium breasts, animal ears, holding phone, solo, full body, white background, simple background, coherent anatomy, best quality'
    assert out.negative==guard+OnePassPromptSettings().negative_template

@pytest.mark.parametrize('policy,expected',[
    ('ps',('slender','skinny','breasts','small breasts')),
    ('ns',('slender','breasts','small breasts')),
    ('off',('breasts','medium breasts')),
])
def test_slim_policy_order_and_no_young_body_additions(tmp_path,store,policy,expected):
    out=build(tmp_path,store,groups=CharacterTagGroups(body=('breasts','medium breasts')),slim=True,policy=policy)
    assert out.rules['body_after']==expected
    assert not {'petite','flat chest','loli'} & set(out.positive.split(', '))

@pytest.mark.parametrize('slim',[False,None])
def test_no_slim_change_without_positive_measurement(tmp_path,store,slim):
    out=build(tmp_path,store,groups=CharacterTagGroups(body=('medium breasts',)),slim=slim)
    assert out.rules['body_after']==('medium breasts',)

@pytest.mark.parametrize('garments,retained',[(('blazer',),True),(('white blazer',),True),(('white dress',),False)])
def test_one_pocket_table_for_stage2_and_stage3(tmp_path,store,garments,retained):
    poses=('hands in pockets','holding phone','holding gun')
    out=build(tmp_path,store,garments=garments,poses=poses)
    assert ('hands in pockets' in out.positive)==retained
    assert 'holding phone' in out.positive and 'holding gun' in out.positive
    selected=select_pose_tags({'hands_in_pockets':.9,'holding_phone':.8},garments)
    assert ('hands_in_pockets' in dict(selected))==retained


def test_suffix_parentheses_and_longest_noun_match():
    nouns=garment_nouns(('white camisole','cropped sweater','white shrug (clothing)','white long skirt'))
    assert {n.name for n in nouns}=={'camisole','sweater','shrug','long skirt'}
    assert uncovered_parts(('crop top','white shirt'))==('bare legs',)
    assert uncovered_parts(('crop top','tank top'))==('bare legs','bare shoulders','bare arms')


def test_approved_tags_kept_and_uncovered_not_duplicated(tmp_path,store):
    tags=('white crop top','midriff','bare_legs','white shrug (clothing)','white thighhighs')
    out=build(tmp_path,store,garments=tags)
    assert out.positive.startswith('white crop top, midriff, bare legs, white shrug (clothing), white thighhighs, ')
    assert out.positive.split(', ').count('midriff')==1
    assert out.rules['uncovered_added']==()
    # Even if thighhighs was a mannequin misclassification, approved tag remains.
    assert 'white thighhighs' in out.positive

@pytest.mark.parametrize('term',['1boy','1girl','male_focus','feminine silhouette','gender_swap'])
def test_settings_share_stage1_gender_template_rejection(term):
    with pytest.raises(ValueError,match='템플릿에는 성별'):
        OnePassPromptSettings(negative_template='nsfw, '+term)


def test_classification_cannot_reintroduce_candidate_gender(tmp_path,store):
    source=tmp_path/'key.png';save_character_gender(source,'unspecified',store)
    cfg=OnePassPromptSettings();gender=prepare_onepass_gender(source,('1boy','blue hair'),cfg.negative_template,settings=store)
    with pytest.raises(ValueError,match='정확히 대응'):
        assemble_onepass_prompt(gender,CharacterTagGroups(('1boy','blue hair')),(),(),slim=False,tokenizers=TOKS)
    out=assemble_onepass_prompt(gender,CharacterTagGroups(('blue hair',)),(),(),slim=False,tokenizers=TOKS)
    assert '1boy' not in out.positive

@pytest.mark.parametrize('n,chunks',[(0,1),(75,1),(76,2),(150,2),(151,3)])
def test_chunks_keep_all_content_ids_and_pad_both_encoders(n,chunks):
    t1,t2=Tokenizer(),Tokenizer();t2.pad_token_id=0
    text=' '.join('x'+str(i) for i in range(n))
    plan=plan_prompt_chunks(text,'bad',(t1,t2))
    for i,e in enumerate(plan):
        assert len(e.positive)==len(e.negative)==chunks
        assert all(len(c.token_ids)==77 for c in (*e.positive,*e.negative))
        flat=[]
        for k,c in enumerate(e.positive):
            take=min(75,max(0,n-75*k));flat.extend(c.token_ids[1:1+take])
        assert flat==t1(text)['input_ids']
        assert e.positive[0].text==(text if n<=75 else None)
        if chunks>1:
            assert e.negative[1].text==''
            assert e.negative[1].token_ids[2:]==((1001 if i==0 else 0),)*75


def test_longer_negative_expands_positive_without_truncation():
    plan=plan_prompt_chunks('good',' '.join(['bad']*151),TOKS)
    assert len(plan[0].positive)==len(plan[0].negative)==3
    assert plan[0].positive[1].text==''


def test_long_dual_tokenizer_id_divergence_is_explicit_error():
    class Other(Tokenizer):
        def __call__(self,text,**kwargs):
            return {'input_ids':[x+1 for x in super().__call__(text,**kwargs)['input_ids']]}
    with pytest.raises(ValueError,match='ID가 달라'):
        plan_prompt_chunks(' '.join(['x']*76),'bad',(Tokenizer(),Other()))


def test_no_remote_tokenizer_fallback(tmp_path):
    with pytest.raises(FileNotFoundError,match='자동 다운로드하지 않습니다'):
        load_onepass_tokenizers(replace(OnePassPromptSettings(),tokenizer_root=tmp_path))


def test_k1_display_dedup_preserves_both_observations_and_policy():
    findings=(Finding('K1',{'count':2,'stage':'original'}),Finding('K1',{'count':0,'stage':'normalized'}),Finding('K2',{'missing':['nose']}))
    assessment=PoseAssessment(findings,None,(),{},False,False)
    before=apply_input_policy(assessment)
    display=findings_for_display(assessment)
    assert [x.code for x in display]==['K1','K2']
    assert len(display[0].observations)==2
    assert assessment.findings==findings and apply_input_policy(assessment)==before
    display[0].observations[0]['count']=99
    assert assessment.findings[0].values['count']==2
