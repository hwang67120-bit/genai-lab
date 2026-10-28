from types import SimpleNamespace
import pytest
from PIL import Image, ImageDraw
from genai_lab.reference_contract import validate_reference_mode, validate_visual_inputs, validate_visual_condition


def valid_inputs():
    head = Image.new('L', (16, 24))
    garment = Image.new('L', head.size)
    ImageDraw.Draw(head).rectangle((4, 0, 10, 5), fill=255)
    ImageDraw.Draw(garment).rectangle((2, 6, 12, 23), fill=255)
    return SimpleNamespace(source=Image.new('RGB', head.size), identity_mask=head, garment_mask=garment,
        identity=Image.new('RGB', (8, 8), 'blue'), garment=Image.new('RGB', (8, 8), 'white'))


def test_valid_disjoint_inputs_are_not_modified():
    inputs = valid_inputs()
    before = inputs.garment_mask.tobytes()
    validate_visual_inputs(inputs)
    assert inputs.garment_mask.tobytes() == before


@pytest.mark.parametrize('case', ['overlap', 'empty', 'size', 'soft', 'old_initial'])
def test_invalid_reference_regions_fail_closed(case):
    inputs = valid_inputs()
    if case == 'overlap': inputs.garment_mask.putpixel((4, 0), 255)
    if case == 'empty': inputs.garment_mask.paste(0, (0, 0, 16, 24))
    if case == 'size': inputs.identity_mask = Image.new('L', (8, 8), 255)
    if case == 'soft': inputs.identity_mask.putpixel((4, 0), 1)
    if case == 'old_initial': inputs.initial = inputs.source
    with pytest.raises(ValueError): validate_visual_inputs(inputs)


@pytest.mark.parametrize('extra', ['image', 'strength', 'control_image', 'mask_image', 'prompt'])
def test_reference_payload_cannot_override_generation_arguments(extra):
    condition = {'ip_adapter_image_embeds': ['cached'], extra: 'injected'}
    with pytest.raises(ValueError): validate_visual_condition(condition)


def test_only_cropped_visual_embedding_payload_allowed():
    validate_visual_condition({'ip_adapter_image_embeds': ['cached']})
    with pytest.raises(ValueError):
        validate_visual_condition({'ip_adapter_image_embeds': None})
    with pytest.raises(ValueError, match='좌표 마스크'):
        validate_visual_condition({
            'ip_adapter_image_embeds': ['cached'],
            'cross_attention_kwargs': {'ip_adapter_masks': [1]},
        })


def test_edit_based_reference_mode_is_required_and_pose_mode_is_blocked():
    with pytest.raises(ValueError):
        validate_reference_mode({'clothing_reference_generation': {'enabled': True, 'without_initial_image': True}})
    with pytest.raises(ValueError):
        validate_reference_mode({'clothing_reference_generation': {'enabled': True}}, True)
    validate_reference_mode({'clothing_reference_generation': {'enabled': True, 'without_initial_image': False}})
