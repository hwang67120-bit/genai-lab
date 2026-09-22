"""Deterministic input/decision/wiring regressions; not model-accuracy tests."""
from dataclasses import replace
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest
from PIL import Image, ImageDraw

from genai_lab.clothing_reference import ClothingDesignAnalysisResult, ClothingDesignTagCandidate
from genai_lab.eye_color_analysis import (
    crop_square_head, decide_eye_color, analyze_with_eye_review, review_eye_candidates,
)


def result(**scores):
    return ClothingDesignAnalysisResult(
        model_id='fake-wd', execution_provider='CPUExecutionProvider',
        input_width=100, input_height=120, model_input_size=32,
        score_threshold=.35, total_label_count=len(scores), general_label_count=len(scores),
        excluded_rating_label_count=0, excluded_character_label_count=0,
        tag_candidates=(), elapsed_seconds=0., raw_general_scores=tuple(scores.items()))


@pytest.mark.parametrize('box', [(0, 0, 33, 45), (67, 75, 100, 120), (20, 35, 55, 80)])
@pytest.mark.parametrize('padding', [0., .15])
def test_square_crop_retains_rgb_and_odd_roi_at_edges(box, padding):
    with Image.new('RGB', (100, 120), (17, 39, 57)) as source, Image.new('L', (100, 120)) as mask:
        ImageDraw.Draw(mask).rectangle((box[0], box[1], box[2]-1, box[3]-1), fill=255)
        before = source.tobytes()
        crop, geometry = crop_square_head(source, mask, pad_ratio=padding)
        with crop:
            assert crop.width == crop.height
            x, y, right, bottom = geometry['requested_box']
            assert x <= box[0] and y <= box[1] and right >= box[2] and bottom >= box[3]
            array = np.asarray(crop)
            left, top, cr, cb = geometry['source_intersection']
            assert np.all(array[top-y:cb-y, left-x:cr-x] == (17, 39, 57))
            outside = np.ones(array.shape[:2], bool)
            outside[top-y:cb-y, left-x:cr-x] = False
            assert np.all(array[outside] == 255)
        assert source.tobytes() == before


@pytest.mark.parametrize('kind,reason', [
    ('missing', 'missing_head_mask'), ('empty', 'empty_head_mask'), ('small', 'head_roi_too_small')])
def test_missing_or_small_roi_is_not_guessed(kind, reason):
    with Image.new('RGB', (80, 80)) as source, Image.new('L', (80, 80)) as mask:
        if kind == 'small':
            ImageDraw.Draw(mask).rectangle((0, 0, 10, 10), fill=255)
        crop, geometry = crop_square_head(source, None if kind == 'missing' else mask, pad_ratio=.15)
        assert crop is None and geometry['reason'] == reason


@pytest.mark.parametrize('mode,size,value', [('RGB', (80, 80), 0), ('L', (40, 40), 255), ('L', (80, 80), 1)])
def test_invalid_mask_contract_rejected(mode, size, value):
    with Image.new('RGB', (80, 80)) as source, Image.new(mode, size, value) as mask:
        with pytest.raises(ValueError):
            crop_square_head(source, mask, pad_ratio=.15)


@pytest.mark.parametrize('first,second,status,reason', [
    ({'yellow_eyes': .7, 'purple_eyes': .1}, {'yellow_eyes': .65, 'purple_eyes': .2}, 'suggested', 'consistent'),
    ({'yellow_eyes': .12, 'purple_eyes': .1}, {'yellow_eyes': .13, 'purple_eyes': .1}, 'unresolved', 'low_score'),
    ({'blue_eyes': .45, 'red_eyes': .42}, {'blue_eyes': .46, 'red_eyes': .42}, 'unresolved', 'ambiguous'),
    ({'blue_eyes': .8, 'red_eyes': .1}, {'blue_eyes': .1, 'red_eyes': .8}, 'unresolved', 'disagreement'),
    ({'blue_eyes': .8, 'heterochromia': .6}, {'blue_eyes': .8}, 'unresolved', 'multicolor'),
    ({'blue_hair': .9}, {'blue_hair': .8}, 'unresolved', 'no_supported'),
])
def test_evidence_rules(first, second, status, reason):
    report = decide_eye_color([result(**first), result(**second)])
    assert report['status'] == status
    assert reason in ' '.join(report['reasons'])
    if status == 'unresolved':
        assert report['suggested_tag'] is None
    else:
        assert report['suggested_tag'] == 'yellow_eyes'
    assert all('relative_prob' not in view for view in report['views'])


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1., 1.01])
def test_invalid_raw_scores_rejected(score):
    with pytest.raises(ValueError):
        decide_eye_color([result(blue_eyes=score), result(blue_eyes=.9)])


def test_full_and_two_crops_saved_without_topn_loss(tmp_path):
    responses = iter([result(blue_hair=.9, purple_eyes=.2, yellow_eyes=.1),
                      result(yellow_eyes=.7, purple_eyes=.1), result(yellow_eyes=.8, purple_eyes=.2)])
    seen = []
    def analyze(image):
        seen.append(image.copy())
        return next(responses)
    with Image.new('RGB', (100, 120), 'orange') as source, Image.new('L', source.size) as mask:
        ImageDraw.Draw(mask).rectangle((20, 20, 59, 64), fill=255)
        reviewed = analyze_with_eye_review(SimpleNamespace(analyze=analyze), source, mask, debug_dir=tmp_path)
    report = json.loads((tmp_path/'eye_color_report.json').read_text(encoding='utf8'))
    assert reviewed.eye_color_report['suggested_tag'] == 'yellow_eyes'
    assert report['inference_count'] == 3 and not report['thresholds_calibrated']
    assert report['full_eye_scores'][0] == ['purple_eyes', .2]
    assert reviewed.tag_candidates == ()  # Crops cannot inject clothing/gender tags.
    for i in range(2):
        with Image.open(tmp_path/f'head_input_{i}.png') as saved:
            assert saved.tobytes() == seen[i+1].tobytes()
    assert seen[1].width != seen[2].width
    for image in seen:
        image.close()


def test_full_multicolor_signal_blocks_single_color(tmp_path):
    responses = iter([result(heterochromia=.8), result(yellow_eyes=.8), result(yellow_eyes=.8)])
    with Image.new('RGB', (64, 64)) as source, Image.new('L', source.size, 255) as mask:
        reviewed = analyze_with_eye_review(SimpleNamespace(analyze=lambda image: next(responses)),
                                          source, mask, debug_dir=tmp_path)
    assert reviewed.eye_color_report['status'] == 'unresolved'
    assert reviewed.eye_color_report['reasons'] == ['full_view_multicolor_signal']


def test_missing_mask_keeps_full_tags_but_no_eye_default(tmp_path):
    full = result(purple_eyes=.95)
    with Image.new('RGB', (64, 64)) as source:
        reviewed = analyze_with_eye_review(SimpleNamespace(analyze=lambda image: full),
                                          source, None, debug_dir=tmp_path)
    assert reviewed.eye_color_report['status'] == 'unresolved'
    assert reviewed.eye_color_report['inference_count'] == 1
    assert list(tmp_path.glob('*.png')) == []


def test_cancel_between_inferences_keeps_partial_evidence(tmp_path):
    calls = []
    def analyze(image):
        calls.append(image.size)
        return result(yellow_eyes=.8)
    with Image.new('RGB', (64, 64)) as source, Image.new('L', source.size, 255) as mask:
        with pytest.raises(InterruptedError):
            analyze_with_eye_review(SimpleNamespace(analyze=analyze), source, mask,
                debug_dir=tmp_path, cancelled=lambda: len(calls) >= 2)
    report = json.loads((tmp_path/'eye_color_report.json').read_text(encoding='utf8'))
    assert report['status'] == 'failed' and len(calls) == 2
    assert len(report['views']) == 1


def test_session_reuses_model_and_preserves_below_threshold_scores(monkeypatch, tmp_path):
    import genai_lab.clothing_analysis as module
    labels = tuple(module.ClothingTagLabel(tag, category) for tag, category in [
        ('safe', 9), ('some_character', 4), ('blue_hair', 0), ('purple_eyes', 0), ('yellow_eyes', 0)])
    monkeypatch.setattr(module, 'download_wd14_model_files', lambda s: (tmp_path/'m', tmp_path/'l'))
    monkeypatch.setattr(module, 'load_wd14_tag_labels', lambda p: labels)
    loads, calls = [], []
    class Session:
        def __init__(self, path, providers):
            loads.append(providers)
        def get_inputs(self):
            return [SimpleNamespace(name='image', shape=['batch', 32, 32, 3])]
        def run(self, outputs, feed):
            calls.append(feed['image'].shape)
            return [np.array([[.99, .99, .9, .8, .12]], np.float32)]
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        InferenceSession=Session, get_available_providers=lambda: ['CPUExecutionProvider']))
    with module.WdTagSession(module.ClothingDesignAnalysisSettings(maximum_tag_count=1)) as session:
        with Image.new('RGB', (64, 64)) as source, Image.new('L', source.size, 255) as mask:
            reviewed = analyze_with_eye_review(session, source, mask, debug_dir=tmp_path)
        assert [tag.tag_name for tag in reviewed.tag_candidates] == ['blue_hair']
        assert dict(reviewed.raw_general_scores)['yellow_eyes'] == pytest.approx(.12)
        assert len(reviewed.raw_general_scores) == 3
        disposition = {item['tag']: item['state']
                       for item in reviewed.eye_color_report['full_eye_display_filter']}
        assert disposition == {'purple_eyes': 'outside_display_top_n',
                               'yellow_eyes': 'below_display_threshold'}
    assert loads == [['CPUExecutionProvider']] and len(calls) == 3
    assert session.session is None
    with pytest.raises(module.ClothingDesignAnalysisError):
        session.analyze(None)


@pytest.mark.parametrize('status,suggestion', [('unresolved', None), ('suggested', 'yellow_eyes')])
def test_ui_eye_selection_separate_from_general_tags(status, suggestion):
    from PySide6.QtWidgets import QApplication
    from genai_lab.character_tag_review import CharacterTagReview
    app = QApplication.instance() or QApplication([])
    report = {'status': status, 'suggested_tag': suggestion, 'reasons': ['test'],
              'full_eye_scores': [('purple_eyes', .9), ('blue_eyes', .49), ('yellow_eyes', .1)], 'views': []}
    analyzed = replace(result(), eye_color_report=report,
        tag_candidates=tuple(ClothingDesignTagCandidate(tag, tag, .9) for tag in
                             ('1girl', 'blue_hair', 'purple_eyes', 'multicolored_eyes')))
    with Image.new('RGB', (64, 64)) as source:
        dialog = CharacterTagReview(source, analyzed, character_gender='male')
        assert dialog.approved_tags == (('blue_hair', 'yellow_eyes') if suggestion else ('blue_hair',))
        assert 'purple_eyes' not in dict(dialog.tag_checkboxes)
        assert not dict(dialog.tag_checkboxes)['1girl'].isEnabled()
        eyes = dict(dialog.eye_options)
        assert set(eyes) == {'purple_eyes', 'blue_eyes', 'yellow_eyes'}
        dialog.clear_eye_selection_button.click()
        eyes['purple_eyes'].click()
        eyes['blue_eyes'].click()
        assert dialog.approved_tags == ('blue_hair', 'purple_eyes', 'blue_eyes')
        assert 'heterochromia' not in dialog.approved_tags
        assert 'multicolored_eyes' not in dialog.approved_tags
        eyes['blue_eyes'].click()
        assert dialog.approved_tags == ('blue_hair', 'purple_eyes')
        dialog.clear_eye_selection_button.click()
        assert dialog.approved_tags == ('blue_hair',)
        assert not dialog.result()  # Clearing selections must not accept the dialog.
        dialog.close()


def test_character_analysis_wrapper_uses_cpu_and_releases_session(monkeypatch):
    import genai_lab.clothing_analysis as wd
    import genai_lab.eye_color_analysis as eye
    from genai_lab.character_tag_review import analyze_character_tags
    events = []
    class Session:
        def __init__(self, settings):
            assert settings.execution_provider == 'CPUExecutionProvider'
        def __enter__(self):
            events.append('enter')
            return self
        def __exit__(self, *args):
            events.append('close')
    def analyze(session, image, mask, **kwargs):
        assert image is source and mask is head
        assert kwargs['threshold'] == .4
        assert kwargs['minimum_margin'] == .15
        assert not kwargs['cancelled']()
        events.append('analyze')
        return 'result'
    monkeypatch.setattr(wd, 'WdTagSession', Session)
    monkeypatch.setattr(eye, 'analyze_with_eye_review', analyze)
    with Image.new('RGB', (64, 64)) as source, Image.new('L', source.size, 255) as head:
        reviewed = analyze_character_tags(source, {
            'clothing_design_analysis': {'execution_provider': 'CUDAExecutionProvider'},
            'character_eye_analysis': {'threshold': .4, 'minimum_margin': .15}},
            face_hair_mask=head)
        assert reviewed == 'result'
        assert events == ['enter', 'analyze', 'close']
        events.clear()
        with pytest.raises(InterruptedError):
            analyze_character_tags(source, {}, cancelled=lambda: True)
        assert events == []
