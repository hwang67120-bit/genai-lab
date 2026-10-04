from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from PIL import Image

from genai_lab.clothing_reference_generation import (
    prepare_design_reference_request,
    resolve_redundant_ensemble_tags,
)
from genai_lab.request import (CharacterGenerationInput, CharacterGenerationSettings,
                              CharacterFramingType, prepare_character_generation_request)
from genai_lab.generator import generate_character_candidate
from genai_lab.workflow import GenerationWorkflowContext, GenerationWorkflowStage


class Tokenizer:
    model_max_length = 77
    def __call__(self, text, **kwargs):
        return {'input_ids': [0, *range(len(text.split())), 1]}


def test_specific_garment_components_suppress_redundant_suit():
    tags, removed = resolve_redundant_ensemble_tags(
        ('blue jacket', 'white shirt', 'blue skirt', 'suit', 'formal'),
        {'redundant_ensemble_rules': ({
            'remove': 'suit',
            'when_groups_present': (
                ('jacket', 'blazer'),
                ('shirt', 'blouse'),
                ('skirt', 'pants', 'trousers'),
            ),
        },)},
    )
    assert tags == ('blue jacket', 'white shirt', 'blue skirt', 'formal')
    assert removed == ('suit',)


def test_ensemble_policy_does_not_infer_or_replace_lower_garment():
    tags, removed = resolve_redundant_ensemble_tags(
        ('jacket', 'shirt', 'skirt'),
        {'redundant_ensemble_rules': ({
            'remove': 'suit',
            'when_groups_present': (
                ('jacket',), ('shirt',), ('skirt', 'pants'),
            ),
        },)},
    )
    assert tags == ('jacket', 'shirt', 'skirt')
    assert removed == ()


@pytest.fixture
def request_data():
    source = Image.new('RGB', (32, 48), (60, 80, 100))
    request = prepare_character_generation_request(
        CharacterGenerationInput(Path('character.png'), CharacterFramingType.FULL_BODY,
                                 approved_reference_image=source),
        CharacterGenerationSettings('test-model', 'test-adapter', 28, 5.5, 0.25, 0.55, 'low quality'),
        candidate_number=1, seed=123,
    )
    yield replace(request, width=64, height=112, original_image_change_strength=0.65)
    request.reference_image.close()
    source.close()


def test_approved_multiple_eye_colors_survive_prompt_without_inferred_heterochromia(request_data):
    modified, record = prepare_design_reference_request(
        request_data, ('blue_jacket',), (Tokenizer(), Tokenizer()),
        character_tags=('blue_hair', 'purple_eyes', 'blue_eyes'), character_gender='male')
    assert 'purple eyes' in modified.prompt and 'blue eyes' in modified.prompt
    assert record['approved_character_tags'] == ('1boy', 'blue hair', 'purple eyes', 'blue eyes')
    assert 'heterochromia' not in modified.prompt
    assert 'multicolored eyes' not in modified.prompt
    assert modified.prompt.startswith('1boy, ')
    assert 'male focus' in modified.prompt
    assert 'masculine silhouette' in modified.prompt
    assert record['base_gender_condition_tags'] == (
        'male focus', 'masculine silhouette')


def test_design_tags_replace_outfit_preservation_without_mutation(request_data):
    modified, record = prepare_design_reference_request(request_data,
        ('blue_jacket', 'gold_buttons', 'blue_jacket'), (Tokenizer(), Tokenizer()))
    assert 'wearing blue jacket, gold buttons' in modified.prompt
    assert 'matching outfit and colors' not in modified.prompt
    assert 'matching outfit and colors' in request_data.prompt
    assert record['approved_tags'] == ('blue jacket', 'gold buttons')
    assert modified.reference_image is request_data.reference_image
    assert modified.seed == request_data.seed


def test_empty_tags_block_instead_of_generating_old_outfit(request_data):
    with pytest.raises(ValueError):
        prepare_design_reference_request(request_data, (), (Tokenizer(), Tokenizer()))


def test_outfit_conflicts_removed_before_token_limit_without_mutating_source(request_data):
    source = replace(request_data, negative_prompt=(
        'Different Outfit, mismatched colors, different hairstyle, bad anatomy'))
    prepared, record = prepare_design_reference_request(source, ('blue jacket',), (Tokenizer(), Tokenizer()))
    assert prepared.negative_prompt == 'different hairstyle, bad anatomy'
    assert record['removed_conflicting_negative_terms'] == ['Different Outfit', 'mismatched colors']
    assert record['negative']['original'] == 'different hairstyle, bad anatomy'
    assert record['negative']['source_before_conflict_removal'] == source.negative_prompt
    assert source.negative_prompt.startswith('Different Outfit')
    assert prepared.seed == source.seed
    assert prepared.original_image_change_strength == source.original_image_change_strength


def test_long_prompt_respects_both_tokenizers(request_data):
    from genai_lab.reference_prompt_budget import PromptBudgetError
    other = Tokenizer()
    other.model_max_length = 45
    with pytest.raises(PromptBudgetError) as caught:
        prepare_design_reference_request(request_data,
            tuple('blue jacket ' + str(i) for i in range(50)), (Tokenizer(), other))
    assert caught.value.report['tokenizer_limits'] == (77, 45)
    assert 'blue jacket 49' in caught.value.report['required_prompt']


def test_runtime_rejects_prompt_changed_after_approval(request_data):
    config = {'clothing_reference_generation': {
        'enabled': True, 'approved_tags': ('jacket',),
        'visual_inputs': SimpleNamespace(),
        'visual_condition': {'ip_adapter_image_embeds': ['cached']},
        'require_prompt_approval': True, 'approved_prompt_pair': ('old prompt', 'old negative')}}
    pipeline = SimpleNamespace(tokenizer=Tokenizer(), tokenizer_2=Tokenizer())
    with pytest.raises(ValueError, match='승인한 생성 조건'):
        generate_character_candidate(pipeline, config, request_data, Path('.'))


@pytest.mark.parametrize('diagnostic_colors_present', [False, True])
@pytest.mark.parametrize('detail_tags', [(), ('gold_buttons',)])
@pytest.mark.parametrize('garment_scale', [.7, .25])
@pytest.mark.parametrize('character_gender', ['unspecified', 'male', 'female'])
@pytest.mark.parametrize('refinement_enabled', [False, True])
def test_reference_mode_pipeline_stays_approved_without_legacy_stages(
        request_data, monkeypatch, tmp_path, garment_scale, character_gender,
        detail_tags, diagnostic_colors_present, refinement_enabled):
    calls = []
    checks = []
    import torch
    from genai_lab.visual_reference import VisualInputs
    from genai_lab.approved_reference_run import approve_reference_run, seal_encoded_condition
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    def forbidden(*args, **kwargs):
        pytest.fail('legacy generation stage was called')
    import genai_lab.generator as generator_module
    assert not hasattr(generator_module, 'execute_catvton_clothing_try_on')
    monkeypatch.setattr('genai_lab.generator.correct_character_candidate_details', forbidden)
    monkeypatch.setattr('genai_lab.generator.prepare_pose_control_input', forbidden)
    class Pipeline:
        tokenizer = Tokenizer()
        tokenizer_2 = Tokenizer()
        vae_scale_factor = 8
        def set_ip_adapter_scale(self, value):
            assert value == pytest.approx((.85 + garment_scale) / 2)
        def __call__(self, **kwargs):
            payload = {'latents': 'unchanged'}
            assert kwargs['callback_on_step_end'](self, 0, 0, payload) is payload
            assert 'control_image' not in kwargs and 'mask_image' not in kwargs
            assert 'ip_adapter_image' not in kwargs
            assert torch.equal(kwargs['ip_adapter_image_embeds'][0], torch.ones(2, 1, 4))
            assert 'cross_attention_kwargs' not in kwargs
            assert 'blue jacket' in kwargs['prompt']
            for detail in detail_tags:
                assert detail.replace('_', ' ') in kwargs['prompt']
            assert 'yellow eyes' in kwargs['prompt']
            assert 'blue tail' not in kwargs['prompt']
            if character_gender != 'unspecified':
                expected = '1boy' if character_gender == 'male' else '1girl'
                opposite = '1girl' if character_gender == 'male' else '1boy'
                assert kwargs['prompt'].startswith(expected + ', ')
                assert opposite not in kwargs['prompt']
            if isinstance(kwargs['image'], Image.Image):
                assert kwargs['image'].mode == 'RGB'
                assert kwargs['image'].size == (64, 112)
                assert kwargs['strength'] == request_data.original_image_change_strength
                assert (kwargs['width'], kwargs['height']) == (64, 112)
                assert kwargs.get('output_type', 'pil') == (
                    'latent' if refinement_enabled else 'pil')
            else:
                assert refinement_enabled
                assert kwargs['image'].shape == (1, 4, 14, 8)
                assert kwargs['strength'] == .25
                assert kwargs['num_inference_steps'] == 10
                assert 'width' not in kwargs and 'height' not in kwargs
            calls.append(kwargs.get('strength'))
            if refinement_enabled and isinstance(kwargs['image'], Image.Image):
                # 실제 Diffusers output_type="latent" 반환과 같이 images 자체가 텐서다.
                return SimpleNamespace(images=torch.zeros(1, 4, 14, 8))
            return SimpleNamespace(images=[Image.new('RGB', (64, 112), 'blue')])
    head = Image.new('L', (64, 112))
    head.paste(255, (0, 0, 64, 32))
    body = Image.new('L', (64, 112))
    body.paste(255, (0, 32, 64, 112))
    inputs = VisualInputs(Image.new('RGB', (64, 112)), Image.new('RGB', (32, 32), 'red'),
                          Image.new('RGB', (32, 32), 'blue'), head, body)
    from genai_lab.part_color_descriptions import PartColorDescription
    inputs.part_color_descriptions = ((PartColorDescription(
        "tail", ("blue",), "proposed", (), b"{}", "test"),) if diagnostic_colors_present else ())
    visual_config = {'identity_reference_scale': .85, 'garment_reference_scale': garment_scale,
        'part_color_descriptions': inputs.part_color_descriptions,
        'check_running': lambda: checks.append(True),
        'character_gender': character_gender,
        'approved_detail_tags': detail_tags,
        'garment_detail_report': {'version': 'test', 'semantic_accuracy_verified': False},
        'approved_character_tags': ('1girl', '1boy', 'blue_hair', 'yellow_eyes'),
        'eye_color_report': {'status': 'suggested', 'suggested_tag': 'yellow_eyes'},
        'visual_inputs': inputs, 'candidate_count': 2,
        'visual_condition': {'ip_adapter_image_embeds': [torch.ones(2, 1, 4)]}}
    reviewed_request, _ = prepare_design_reference_request(
        request_data, ('blue_jacket',), (Tokenizer(), Tokenizer()),
        character_tags=visual_config['approved_character_tags'], character_gender=character_gender,
        approved_detail_tags=detail_tags, part_color_descriptions=inputs.part_color_descriptions)
    visual_config.update(require_prompt_approval=True, approved_prompt_pair=(
        reviewed_request.prompt, reviewed_request.negative_prompt))
    config = {
        'generation': {'mode': 'image_to_image'},
        'reference_analysis': {'latent_refinement': {
            'enabled': refinement_enabled, 'inference_steps': 10,
            'strength': .25, 'guidance_scale': 5.5,
            'latent_scale_factor': 1.,
        }},
        'detail_correction': {'enabled': True},
        'clothing_reference_generation': {'enabled': True, 'approved_tags': ('blue_jacket',),
                                          'source_name': 'outfit.png', **visual_config},
    }
    approve_reference_run(inputs, config, request_data)
    config['clothing_reference_generation']['encoded_reference_receipt'] = seal_encoded_condition(
        inputs, visual_config['visual_condition'])
    pipeline = Pipeline()
    monkeypatch.setattr(
        'genai_lab.latent_refinement.refinement_pipeline_from',
        lambda base: pipeline,
    )
    candidate = generate_character_candidate(pipeline, config, request_data, tmp_path)
    try:
        assert calls == ([request_data.original_image_change_strength, .25] if refinement_enabled else [request_data.original_image_change_strength])
        assert checks == ([True, True, True, True] if refinement_enabled else [True, True])
        assert candidate.before_clothing_image is None
        assert candidate.detail_correction_status == 'disabled'
        assert candidate.design_reference_record['mode'] == (
            'visual_reference_regeneration')
        assert candidate.clothing_reference_name == 'outfit.png'
        assert candidate.design_reference_record['initial_image_used'] is True
        assert candidate.design_reference_record['generation_mode'] == 'image_to_image'
        assert candidate.design_reference_record['effective_strength'] == request_data.original_image_change_strength
        assert candidate.design_reference_record['garment_scale'] == garment_scale
        assert candidate.design_reference_record['identity_scale'] == .85
        assert candidate.design_reference_record['character_gender'] == character_gender
        assert candidate.design_reference_record['prompt_preflight_approved'] is True
        assert candidate.design_reference_record['eye_color_analysis'] == visual_config['eye_color_report']
        assert candidate.design_reference_record['garment_detail_analysis'] == visual_config['garment_detail_report']
        assert candidate.design_reference_record['approved_detail_tags'] == tuple(tag.replace('_', ' ') for tag in detail_tags)
        assert candidate.design_reference_record['garment_image_adapter'] is (garment_scale > 0)
        assert candidate.design_reference_record['latent_refinement']['status'] == (
            'completed' if refinement_enabled else 'disabled')
        assert candidate.pose_control_status == 'not_requested'
        assert not list(tmp_path.iterdir())  # approval before image persistence
    finally:
        candidate.image.close()
        inputs.close()


def test_removed_synthesis_has_no_runner_or_model_configuration():
    from run import load_yaml
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / 'configs' / 'animagine.yaml')
    assert 'clothing_try_on' not in config
    assert not (root / 'scripts' / 'catvton_runner.py').exists()
    assert config['character_body_comparison']['repository_path'].endswith('CatVTON')
    assert config['model']['id'] == 'cagliostrolab/animagine-xl-3.1'


def test_removed_synthesis_api_fails_before_model_or_files(request_data, tmp_path):
    from genai_lab.clothing import execute_catvton_clothing_try_on, CharacterClothingProtectionError
    from genai_lab.generator import apply_clothing_to_generated_candidate
    with pytest.raises(CharacterClothingProtectionError, match='제거'):
        execute_catvton_clothing_try_on(None, None, None, None, 1)
    with pytest.raises(ValueError, match='제거'):
        apply_clothing_to_generated_candidate(None, None, None, None)
    with pytest.raises(ValueError, match='제거'):
        generate_character_candidate(None, {}, request_data, tmp_path, catvton_settings=object())
    assert not list(tmp_path.iterdir())


def test_garment_gender_removed_but_user_character_gender_remains(request_data):
    modified, record = prepare_design_reference_request(request_data,
        ('1girl', 'male_focus', '2boys', 'blue_jacket', 'high_heels'),
        (Tokenizer(), Tokenizer()), character_tags=('1boy', 'silver_hair'))
    assert record['approved_tags'] == ('blue jacket', 'high heels')
    assert record['approved_character_tags'] == ('1boy', 'silver hair')
    assert set(record['excluded_non_design_tags']) == {'1girl', 'male focus', '2boys'}
    assert '1girl' not in modified.prompt and 'male focus' not in modified.prompt
    assert '1boy' in modified.prompt and 'silver hair' in modified.prompt
    assert modified.prompt.index('1boy') < modified.prompt.index('wearing blue jacket')



def test_native_base_prioritizes_full_body_and_deduplicates_animal_features(
        request_data):
    from genai_lab.native_pipeline_contract import (
        prepare_character_only_base_request,
    )

    modified, record = prepare_character_only_base_request(
        request_data,
        (Tokenizer(), Tokenizer()),
        character_tags=(
            'animal_ears', 'tail', 'blue_hair', 'raccoon_tail',
            'raccoon_ears', 'multicolored_hair',
        ),
        character_gender='male',
    )
    terms = modified.prompt.split(', ')

    assert terms[:7] == [
        '1boy', 'male focus', 'masculine silhouette', 'full body',
        'head to toe', 'feet visible', 'entire character inside frame',
    ]
    assert 'animal ears' not in terms
    assert 'tail' not in terms
    assert terms.index('blue hair') < terms.index('raccoon ears')
    assert terms.index('raccoon ears') < terms.index('raccoon tail')
    assert record['removed_redundant_character_tags'] == [
        'animal ears', 'tail']
    assert record['approved_character_tags'] == [
        '1boy', 'animal ears', 'tail', 'blue hair', 'raccoon tail',
        'raccoon ears', 'multicolored hair']


def test_native_base_unspecified_gender_does_not_emit_none_prompt(request_data):
    from genai_lab.native_pipeline_contract import (
        prepare_character_only_base_request,
    )

    modified, record = prepare_character_only_base_request(
        request_data,
        (Tokenizer(), Tokenizer()),
        character_tags=("blue_hair", "raccoon_ears", "raccoon_tail"),
        character_gender="unspecified",
    )
    terms = modified.prompt.split(", ")

    assert "none" not in terms
    assert terms[0] == "full body"
    assert record["character_gender"] == "unspecified"
    assert record["base_gender_condition_tags"] == []


def test_reference_mode_refuses_legacy_inputs(request_data, tmp_path):
    with pytest.raises(ValueError, match='외부 자세'):
        generate_character_candidate(None, {'clothing_reference_generation': {'enabled': True}},
            request_data, tmp_path, approved_pose_estimation=object())


def test_reference_progress_has_no_body_restoration_step():
    workflow = GenerationWorkflowContext(Path('character.png'), Path('outfit.png'), None,
                                         reference_generation=True)
    workflow.move_to(GenerationWorkflowStage.BASE_GENERATING)
    assert workflow.progress == (4, 5)
    workflow.move_to(GenerationWorkflowStage.FINAL_REVIEW)
    assert workflow.progress == (5, 5)


def test_optional_garment_details_use_remaining_budget_and_reject_components(request_data):
    modified, record = prepare_design_reference_request(request_data, ('skirt',),
        (Tokenizer(), Tokenizer()), character_gender='male',
        approved_detail_tags=('gold_buttons', '1girl', 'pants', 'blue_hair', 'suit'))
    assert modified.prompt.startswith('1boy, ')
    assert 'gold buttons' in modified.prompt
    assert 'pants' not in modified.prompt
    assert record['approved_detail_tags'] == ('gold buttons',)
    assert set(record['excluded_non_detail_tags']) == {'1girl', 'pants', 'blue hair', 'suit'}
    assert record['approved_tags'] == ('skirt',)


def test_optional_detail_overflow_keeps_core_and_records_omission(request_data):
    tokenizer = Tokenizer()
    baseline, _ = prepare_design_reference_request(request_data, ('blue_jacket',),
        (tokenizer, tokenizer), character_gender='male')
    # Use the same token counter as production; no assumed token lengths.
    from scripts.generation_inputs import _token_count
    from genai_lab.reference_prompt_budget import build_reference_prompt
    _, required = build_reference_prompt(
        (tokenizer,),
        core_character=('1boy', 'male focus', 'masculine silhouette'),
        core_outfit=('blue jacket',), framing=('full body', 'head to toe', 'feet visible'))
    tokenizer.model_max_length = required['required_token_counts'][0]
    modified, record = prepare_design_reference_request(request_data, ('blue_jacket',),
        (tokenizer, tokenizer), character_gender='male', approved_detail_tags=('gold_buttons',))
    assert 'wearing blue jacket' in modified.prompt
    assert 'gold buttons' not in modified.prompt
    assert record['omitted_detail_tags'] == ('gold buttons',)


@pytest.mark.parametrize('character_gender', ['unspecified', 'male', 'female'])
def test_color_diagnostics_do_not_change_generation_prompt(request_data, character_gender):
    from genai_lab.part_color_descriptions import PartColorDescription
    colors = (
        PartColorDescription('ears', ('light purple', 'blue', 'dark gray'), 'proposed', (), b'{}', 'test'),
        PartColorDescription('tail', ('dark blue', 'light blue'), 'proposed', (), b'{}', 'test'),
    )
    kwargs = dict(
        approved_tags=('skirt', 'jacket', 'blue_skirt', 'shirt', 'buttons', 'long_sleeves',
                       'formal', 'uniform', 'white_shirt', 'suit', 'blue_jacket'),
        tokenizers=(Tokenizer(), Tokenizer()),
        character_tags=('animal_ears', 'tail', 'short_hair', 'blue_hair', 'raccoon_tail',
                        'raccoon_ears', 'multicolored_hair', 'blue_eyes'),
        character_gender=character_gender,
    )
    baseline, baseline_record = prepare_design_reference_request(request_data, **kwargs)
    actual, record = prepare_design_reference_request(
        request_data, **kwargs, part_color_descriptions=colors,
        approved_part_color_descriptions=colors)
    assert actual.prompt == baseline.prompt
    assert actual.negative_prompt == baseline.negative_prompt
    assert record['positive'] == baseline_record['positive']
    assert record['approved_character_tags'] == baseline_record['approved_character_tags']
    assert record['part_color_descriptions']
    assert record['applied_part_color_tags'] == ()
    assert record['generic_part_tags_replaced_by_color_description'] == ()
    terms = actual.prompt.split(', ')
    assert 'animal ears' in terms and 'tail' in terms
    assert actual.prompt.endswith('natural fabric folds, coherent anatomy, best quality')
    assert all(description.prompt_text not in actual.prompt for description in colors)
    assert actual.seed == baseline.seed == request_data.seed
