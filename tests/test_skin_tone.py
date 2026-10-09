"""사용자 선택과 고정 T1/T2 문구 재현 검사다. GPU나 외부 시험 자료 의존성은 없다."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel
from genai_lab.skin_tone import (SkinToneChoice, replace_skin_tags, default_skin_choice, measure_cheeks,
    SkinRecommendationSettings, observe_skin_tone)
from genai_lab.onepass_prompt import AppearanceOverrides
from genai_lab.studio_skin_tone import skin_tone_field
from test_studio_generation import analysis_at, request_at


def test_no_tags_and_default_preserve_exact_prompt(tmp_path):
    a = analysis_at(tmp_path/'inputs')
    base = request_at(tmp_path, a)
    assert request_at(tmp_path, a, appearance=AppearanceOverrides()).inputs.prompt == base.inputs.prompt
    a['groups']['appearance'] += ['dark_skin','dark-skinned_male']
    base = request_at(tmp_path, a)
    chosen = request_at(tmp_path, a, appearance=AppearanceOverrides(skin_tone=SkinToneChoice('dark')))
    assert base.inputs.prompt.positive == chosen.inputs.prompt.positive
    assert chosen.inputs.prompt.rules['skin_tone']['was_default'] is True
    approval = json.loads((tmp_path/'inputs/approval.json').read_text(encoding='utf-8'))
    assert approval['skin_tone']['removed_tags'] == []


@pytest.mark.parametrize('value,trial', [(None,'T0'),('tan','T2'),('unspecified','T1')])
def test_product_assembler_matches_frozen_trial(tmp_path, value, trial):
    fixture = json.loads((Path(__file__).parent/'fixtures/skin_tone_trial.json').read_text(encoding='utf-8'))
    a = analysis_at(tmp_path/'inputs')
    a.update({k:fixture[k] for k in ('groups','slim','garment_tags')})
    request = request_at(tmp_path,a,appearance=AppearanceOverrides(skin_tone=SkinToneChoice(value)))
    assert request.inputs.prompt.positive == fixture['expected'][trial]
    assert request.inputs.prompt.rules['skin_tone']['automatic_replacement'] is False


def face_points():
    points = np.full((68,2),(20.,20.));scores=np.ones(68)
    points[[2,31]] = (12,20);points[[14,35]] = (28,20)
    return points,scores


def test_both_cheeks_ignore_white_nose_and_do_not_change_choice():
    rgb=np.full((40,40,3),(222,151,110),np.uint8);rgb[17:24,17:24]=255
    points,scores=face_points()
    result=measure_cheeks(rgb,points,scores)
    assert result['status']=='measured' and result['lab'][0]<90
    assert SkinToneChoice().value is None
    scores[2]=.1
    with pytest.raises(ValueError):measure_cheeks(rgb,points,scores)
    assert observe_skin_tone(Path('/missing'))['status']=='unavailable'
    changed=measure_cheeks(rgb,*face_points(),SkinRecommendationSettings(lightness_min=100, lightness_max=100))
    assert changed['recommend_tan'] is False


def test_tag_replacement_removes_all_skin_tags_at_first_position():
    tags=('blue hair','dark skin','short hair','dark-skinned male','tan')
    changed,record=replace_skin_tags(tags,SkinToneChoice('tan'))
    assert changed==('blue hair','tan','short hair')
    assert len(record['removed_tags'])==3
    assert default_skin_choice(('tan',))=='tan'
    assert replace_skin_tags(('tan',),SkinToneChoice())[0]==('tan',)
    with pytest.raises(ValueError):replace_skin_tags(('blue hair',),SkinToneChoice('tan'))


def test_gui_hidden_without_tags_and_default_dark_despite_recommendation(tmp_path, monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=QWidget();layout=QVBoxLayout(w);a=analysis_at(tmp_path/'inputs')
    getter=skin_tone_field(layout,a,SkinRecommendationSettings())
    assert not w.findChildren(QComboBox) and getter().value is None
    a['groups']['appearance']+=['dark_skin']
    monkeypatch.setattr('genai_lab.studio_skin_tone.observe_skin_tone',lambda *args:{'status':'measured','rgb':[220,150,110],'recommend_tan':True})
    getter=skin_tone_field(layout,a,SkinRecommendationSettings())
    combo=w.findChild(QComboBox,'skin_tone_choice')
    assert combo.currentData()=='dark' and getter().value=='dark'
    assert 'tan 추천' in w.findChild(QLabel,'skin_tone_recommendation').text()
    combo.setCurrentIndex(combo.findData('tan'));assert getter().value=='tan'
    w.close()


def test_confirmed_choice_is_saved_per_character_and_preselected(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from genai_lab.character_preferences import load_character_skin_tone, save_character_skin_tone
    a = analysis_at(tmp_path/'inputs'); a['groups']['appearance'] += ['dark_skin','dark-skinned_male']
    store = QSettings(str(tmp_path/'test-only.ini'), QSettings.Format.IniFormat)
    assert load_character_skin_tone(a['source'], store) is None
    request_at(tmp_path, a)  # 명시적 선택이 없으면 저장하지 않는다.
    assert load_character_skin_tone(a['source'], QSettings(str(tmp_path/'test-only.ini'), QSettings.Format.IniFormat)) is None
    request_at(tmp_path, a, appearance=AppearanceOverrides(skin_tone=SkinToneChoice('tan')))
    store = QSettings(str(tmp_path/'test-only.ini'), QSettings.Format.IniFormat)
    assert load_character_skin_tone(a['source'], store) == 'tan'
    other = analysis_at(tmp_path/'other'); other['groups']['appearance'] += ['dark_skin']
    assert load_character_skin_tone(other['source'], store) is None
    with pytest.raises(ValueError): save_character_skin_tone(a['source'], 'pale', store)
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr('genai_lab.studio_skin_tone.observe_skin_tone',lambda *args:{'status':'measured','rgb':[220,150,110],'recommend_tan':False})
    w=QWidget();layout=QVBoxLayout(w)
    getter=skin_tone_field(layout,a,SkinRecommendationSettings(),preferences=store)
    combo=w.findChild(QComboBox,'skin_tone_choice')
    assert combo.currentData()=='tan' and getter().value=='tan' and getter().measurement['saved_choice']=='tan'
    assert '저장한 선택' in w.findChild(QLabel,'skin_tone_recommendation').text()
    w.close()
    w=QWidget();layout=QVBoxLayout(w)
    getter=skin_tone_field(layout,other,SkinRecommendationSettings(),preferences=store)
    assert w.findChild(QComboBox,'skin_tone_choice').currentData()=='dark' and getter().measurement['saved_choice'] is None
    w.close()


def test_saved_skin_tone_does_not_affect_characters_without_skin_tags(tmp_path):
    from PySide6.QtCore import QSettings
    from genai_lab.character_preferences import save_character_skin_tone
    a = analysis_at(tmp_path/'inputs')
    save_character_skin_tone(a['source'], 'tan', QSettings(str(tmp_path/'test-only.ini'), QSettings.Format.IniFormat))
    base = request_at(tmp_path, a)
    assert 'tan' not in base.inputs.prompt.positive.split(', ')


def test_light_skin_is_not_recommended_tan():
    # 2026-10-09의 21장 검사에서 상한이 없으면 밝은 피부(L 89~94, 색상각 40~60)까지 tan으로 잘못 추천했다.
    points, scores = face_points()
    for color, expected in (((236,160,128), True), ((245,226,214), False)):
        rgb = np.full((40,40,3), color, np.uint8)
        result = measure_cheeks(rgb, points, scores)
        assert result['recommend_tan'] is expected, (color, result['lab'], result['hue'])
    with pytest.raises(ValueError): SkinRecommendationSettings(lightness_min=85, lightness_max=80)


def female_request_at(tmp_path, analysis, **kwargs):
    from PySide6.QtCore import QSettings
    import test_studio_generation as base
    store = QSettings(str(tmp_path/'test-only.ini'), QSettings.Format.IniFormat)
    runtime = base.service.StudioRuntime(generation=replace(base.service.OnePassGenerationSettings(), width=16, height=24))
    return base.service.build_request(analysis, analysis["garment_tags"], "female", confirmed=True, runtime=runtime,
        preferences=store, tokenizers=(base.Tokens(), base.Tokens()), seeds=(1,2,3,4), **kwargs)


MEASURED = {'status':'measured','rgb':[205,160,140],'recommend_tan':True,'lab':[71.8,19.0,20.5],'hue':47.0}


@pytest.mark.parametrize('value,trial', [(None,'N0'),('unspecified','N0'),('tan','N1')])
def test_measured_tan_without_tags_matches_gpu_trial(tmp_path, value, trial):
    # 2026-10-09의 9번에서 태거가 tan을 놓쳐 생성 볼이 밝아졌고(L72 → 86), tan 선택으로 복원됐다.
    fixture = json.loads((Path(__file__).parent/'fixtures/skin_tone_measured_tan.json').read_text(encoding='utf-8'))
    a = analysis_at(tmp_path/'inputs'); a.update({k:fixture[k] for k in ('groups','slim','garment_tags')})
    choice = SkinToneChoice() if value is None else SkinToneChoice(value, {**MEASURED, 'trigger':'measured_tan'})
    request = female_request_at(tmp_path, a, appearance=AppearanceOverrides(skin_tone=choice))
    assert request.inputs.prompt.positive == fixture['expected'][trial]
    if value is not None:
        record = request.inputs.prompt.rules['skin_tone']
        assert record['trigger'] == 'measured_tan' and record['default'] == 'unspecified'
        assert record['was_default'] is (value == 'unspecified') and record['automatic_replacement'] is False


def test_measured_tan_dark_choice_and_untriggered_choice(tmp_path):
    fixture = json.loads((Path(__file__).parent/'fixtures/skin_tone_measured_tan.json').read_text(encoding='utf-8'))
    a = analysis_at(tmp_path/'inputs'); a.update({k:fixture[k] for k in ('groups','slim','garment_tags')})
    dark = female_request_at(tmp_path, a, appearance=AppearanceOverrides(skin_tone=SkinToneChoice('dark', {**MEASURED, 'trigger':'measured_tan'})))
    assert dark.inputs.prompt.positive == fixture['expected']['N1'].replace(', tan,', ', dark skin,')
    with pytest.raises(ValueError):  # 측정 입력란에서 나온 선택이 아니면 태그를 추가할 수 없다.
        female_request_at(tmp_path, a, appearance=AppearanceOverrides(skin_tone=SkinToneChoice('tan')))


def test_gui_shows_field_without_tags_only_for_measured_tan(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    app=QApplication.instance() or QApplication([])
    a=analysis_at(tmp_path/'inputs'); store=QSettings(str(tmp_path/'gui.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr('genai_lab.studio_skin_tone.observe_skin_tone',lambda *args:{**MEASURED,'recommend_tan':False})
    w=QWidget();layout=QVBoxLayout(w)
    getter=skin_tone_field(layout,a,SkinRecommendationSettings(),preferences=store)
    assert not w.findChildren(QComboBox) and getter().value is None
    monkeypatch.setattr('genai_lab.studio_skin_tone.observe_skin_tone',lambda *args:dict(MEASURED))
    getter=skin_tone_field(layout,a,SkinRecommendationSettings(),preferences=store)
    combo=w.findChild(QComboBox,'skin_tone_choice')
    assert combo.currentData()=='unspecified'
    choice=getter(); assert choice.value=='unspecified' and choice.measurement['trigger']=='measured_tan'
    assert '황갈색으로 측정' in w.findChild(QLabel,'skin_tone_recommendation').text()
    combo.setCurrentIndex(combo.findData('tan')); assert getter().value=='tan'
    w.close()
