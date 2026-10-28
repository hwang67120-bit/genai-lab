from dataclasses import replace
from types import SimpleNamespace
import sys
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from genai_lab.reference_features import tag_scope, automatic_feature_selection
from genai_lab.reference_prompt_budget import build_reference_prompt, PromptBudgetError
from genai_lab.character_tag_review import CharacterTagReview
from genai_lab.clothing_reference_generation import prepare_design_reference_request
from genai_lab.request import CharacterFramingType


class Tokenizer:
    model_max_length = 77
    def __call__(self, text, **kwargs):
        assert kwargs['truncation'] is False and kwargs['add_special_tokens'] is True
        assert kwargs['verbose'] is False
        return {'input_ids': [0, *text.split(), 1]}


def candidates(*tags):
    return tuple(SimpleNamespace(tag_name=t, display_name=t.replace('_', ' '), score=.9) for t in tags)


@pytest.mark.parametrize('tag,scope', [
    ('blue_hair', 'character'), ('animal_ears', 'character'), ('raccoon_tail', 'character'),
    ('blue_jacket', 'garment'), ('long_sleeves', 'garment'), ('high_heels', 'garment'),
    ('1girl', 'context'), ('virtual_youtuber', 'context'), ('text', 'context'),
    ('white_background', 'scene'), ('full_body', 'scene'), ('novel_unknown_design', 'unknown'),
])
def test_roles_are_bounded(tag, scope):
    assert tag_scope(tag) == scope


def test_auto_selection_separates_sources_and_does_not_guess_unknown():
    tags = candidates('blue_hair', 'animal_ears', 'tail', 'jacket', 'white_shirt',
                      '1girl', 'white_background', 'unknown_feature')
    char = automatic_feature_selection(tags, 'character')
    outfit = automatic_feature_selection(tags, 'garment')
    assert char['selected'] == ('blue_hair', 'animal_ears', 'tail')
    assert outfit['selected'] == ('jacket', 'white_shirt')
    assert char['unresolved'] == outfit['unresolved'] == ('unknown_feature',)


def test_close_species_alternatives_are_not_auto_confirmed():
    selection = automatic_feature_selection(candidates('animal_ears', 'cat_ears', 'raccoon_ears'), 'character')
    assert selection['selected'] == ('animal_ears',)
    assert set(selection['unresolved']) == {'cat_ears', 'raccoon_ears'}


def test_budget_reserves_all_core_and_skips_only_optional():
    small = Tokenizer()
    small.model_max_length = 18
    prompt, report = build_reference_prompt(
        (Tokenizer(), small), core_character=('1boy', 'blue_eyes', 'purple_eyes', 'full_body'),
        core_outfit=('blue_jacket', 'white_shirt'), framing=('full_body',),
        optional=('too long ' * 50, 'quality', 'blue_eyes'))
    assert 'wearing blue jacket, white shirt' in prompt
    assert 'blue eyes' in prompt and 'purple eyes' in prompt
    assert 'same face' not in prompt
    assert prompt.count('full body') == 1
    assert prompt.endswith('quality')
    assert report['effective_token_counts'][1] <= 18
    assert len(report['omitted_optional']) == 1


def test_core_overflow_is_reported_not_truncated():
    t = Tokenizer()
    t.model_max_length = 6
    with pytest.raises(PromptBudgetError) as caught:
        build_reference_prompt((t,), core_character=('1boy', 'blue_hair'),
                               core_outfit=('blue_jacket', 'white_shirt'))
    assert 'white shirt' in caught.value.report['required_prompt']
    assert caught.value.report['omitted_optional'] == []


def test_missing_tokenizer_is_not_silently_ignored():
    with pytest.raises(ValueError):
        build_reference_prompt((Tokenizer(), None), core_character=(), core_outfit=('jacket',))


def test_character_summary_defaults_and_uncertain_eye_confirmation():
    app = QApplication.instance() or QApplication([])
    analyzed = SimpleNamespace(
        tag_candidates=candidates('blue_hair', 'tail', 'jacket', 'white_background', '1girl'),
        eye_color_report={'status': 'unresolved', 'suggested_tag': None,
                          'reasons': ['crop_disagreement'],
                          'full_eye_scores': [('blue_eyes', .5), ('purple_eyes', .49)], 'views': []})
    with Image.new('RGB', (64, 64)) as image:
        dialog = CharacterTagReview(image, analyzed, character_gender='male')
        assert dialog.details_panel.isHidden()
        assert dialog.approved_tags == ('blue_hair', 'tail')
        assert not dict(dialog.tag_checkboxes)['jacket'].isEnabled()
        assert not dialog.approve_button.isEnabled()
        dialog.eye_omit_confirmation.click()
        assert dialog.approve_button.isEnabled()
        dialog.details_button.click()
        assert not dialog.details_panel.isHidden()
        eyes = dict(dialog.eye_options)
        eyes['blue_eyes'].click()
        eyes['purple_eyes'].click()
        assert not dialog.eye_omit_confirmation.isChecked()
        assert dialog.approve_button.isEnabled()
        assert dialog.approved_tags[-2:] == ('blue_eyes', 'purple_eyes')
        dialog.eye_omit_confirmation.click()
        assert not any(box.isChecked() for _, box in dialog.eye_options)
        assert dialog.approve_button.isEnabled()
        assert dialog.result() == 0
        dialog.close()


def test_hair_detail_candidates_require_explicit_user_approval():
    app = QApplication.instance() or QApplication([])
    analyzed = SimpleNamespace(
        tag_candidates=candidates('blue_hair', 'long_hair', 'tail'),
        eye_color_report=None,
        hair_detail_report={
            'status': 'analyzed',
            'view_count': 4,
            'views': [],
            'evidence': [{
                'tag': 'long_hair', 'group': 'length',
                'status': 'model_candidate',
            }],
            'optional_detail_tags': ['long_hair'],
        },
    )
    with Image.new('RGB', (64, 64)) as image:
        dialog = CharacterTagReview(image, analyzed, character_gender='male')
        boxes = dict(dialog.tag_checkboxes)
        assert boxes['blue_hair'].isChecked()
        assert not boxes['long_hair'].isChecked()
        assert dialog.approved_tags == ('blue_hair', 'tail')
        boxes['long_hair'].click()
        assert dialog.approved_tags == ('blue_hair', 'long_hair', 'tail')
        dialog.close()


def test_live_prompt_preview_blocks_overflow_then_recovers():
    from dataclasses import dataclass
    @dataclass
    class Request:
        framing_type: object = CharacterFramingType.FULL_BODY
        prompt: str = ''
        negative_prompt: str = 'bad anatomy'
    app = QApplication.instance() or QApplication([])
    small = Tokenizer()
    def builder(tags):
        return prepare_design_reference_request(Request(), ('blue_jacket',), (small, small),
            character_tags=tags, character_gender='male')
    _, baseline_record = builder(())
    small.model_max_length = baseline_record[
        'positive']['required_token_counts'][0]
    with Image.new('RGB', (64, 64)) as image:
        dialog = CharacterTagReview(image, SimpleNamespace(tag_candidates=candidates(
            'very_long_blue_hair', 'animal_ears', 'tail')),
            character_gender='male', prompt_builder=builder)
        assert not dialog.approve_button.isEnabled()
        assert dialog.prepared_request is None
        assert '필수' in dialog.budget_label.text()
        for _, box in dialog.tag_checkboxes:
            box.setChecked(False)
        assert dialog.approve_button.isEnabled()
        assert dialog.prepared_request.prompt.startswith('1boy, ')
        assert 'wearing blue jacket' in dialog.prepared_request.prompt
        assert dialog.prompt_record['positive']['omitted_optional']
        assert '선택 표현' in dialog.budget_label.text()
        dialog.close()


def test_garment_summary_filters_people_and_hides_details(monkeypatch):
    import gui_main
    app = QApplication.instance() or QApplication([])
    analyzed = SimpleNamespace(
        tag_candidates=candidates('jacket', 'blue_shirt', 'blue_hair', '1girl', 'white_background', 'novel_detail'),
        model_id='mock', execution_provider='CPU', input_width=64, input_height=64,
        model_input_size=32, total_label_count=6, general_label_count=6,
        excluded_rating_label_count=0, excluded_character_label_count=0,
        score_threshold=.35, elapsed_seconds=.1)
    monkeypatch.setattr(gui_main, 'create_white_background_clothing_preview',
                        lambda candidate: Image.new('RGB', (64, 64), 'white'))
    dialog = gui_main.ClothingDesignAnalysisReviewDialog(None, analyzed)
    assert dialog.details_panel.isHidden()
    boxes = dict(dialog.tag_checkboxes)
    assert set(boxes) == {'jacket', 'blue_shirt', 'novel_detail'}
    assert boxes['jacket'].isChecked() and boxes['blue_shirt'].isChecked()
    assert not boxes['novel_detail'].isChecked()
    assert 'jacket' in dialog.summary_label.text()
    dialog.details_button.click()
    assert not dialog.details_panel.isHidden()
    boxes['jacket'].setChecked(False)
    boxes['blue_shirt'].setChecked(False)
    assert not dialog.approve_button.isEnabled()
    boxes['blue_shirt'].setChecked(True)
    dialog.approve_selected_tags()
    assert dialog.approved_tag_names == ('blue_shirt',)
    dialog.close()


def test_tokenizer_preflight_loads_only_two_tokenizers(monkeypatch):
    from genai_lab.reference_prompt_budget import load_reference_tokenizers
    calls = []
    class MockTokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append((model, kwargs))
            return Tokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(CLIPTokenizer=MockTokenizer))
    loaded = load_reference_tokenizers({'model': {'id': 'test-model', 'cache_dir': 'cache', 'revision': 'rev'}})
    assert len(loaded) == 2
    assert [kwargs['subfolder'] for _, kwargs in calls] == ['tokenizer', 'tokenizer_2']
    assert all(kwargs['revision'] == 'rev' for _, kwargs in calls)
