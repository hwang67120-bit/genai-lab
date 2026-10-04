"""IP conditioning changes must never replace edit authorization or latent masking."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from genai_lab.output_coordinate_control import run_isolated_inpaint
from genai_lab.garment_edit_plan import project_target_garment_coverage

@pytest.fixture
def pipeline(monkeypatch):
    import diffusers
    class Pipeline:
        vae_scale_factor = 8
        unet = SimpleNamespace(config=SimpleNamespace(in_channels=4))
        mask_processor = object()
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            for i in range(2):
                kwargs['callback_on_step_end'](self, i, 2-i, {})
            return 'ok'
    monkeypatch.setattr(diffusers, 'StableDiffusionXLInpaintPipeline', Pipeline)
    return Pipeline()

@pytest.mark.parametrize('condition', ['default', 'restricted', 'zero'])
def test_condition_changes_attention_only_and_preserves_soft_weights(pipeline, condition):
    import torch
    edit = np.zeros((32,32), np.uint8); edit[8:24,8:24] = 127
    hard = (edit > 0).astype(np.uint8) * 255
    expected = edit.copy()
    if condition == 'restricted': expected[:,16:] = 0
    if condition == 'zero': expected[:] = 0
    kwargs = {} if condition == 'default' else {'ip_adapter_mask':Image.fromarray(expected)}
    edit_image = Image.fromarray(edit); image = Image.new('RGB',(32,32))
    processor = pipeline.mask_processor; state = torch.random.get_rng_state().clone()
    report = {}
    result = run_isolated_inpaint(pipeline, control_report=report,
        image=image, mask_image=edit_image, hard_authorized_mask=Image.fromarray(hard), **kwargs)
    assert result == 'ok'
    assert pipeline.kwargs['mask_image'] is edit_image
    assert pipeline.kwargs['image'] is image
    assert np.array_equal(np.asarray(edit_image),edit)
    tensor = pipeline.kwargs['cross_attention_kwargs']['ip_adapter_masks'][0]
    np.testing.assert_allclose(tensor[0,0].numpy(),expected.astype(np.float32)/255,rtol=0,atol=0)
    assert pipeline.mask_processor is processor
    assert report['output_coordinate_control']['latent_restore_steps'] == 2
    assert torch.equal(state,torch.random.get_rng_state())
    assert 'ip_adapter_mask' not in pipeline.kwargs

@pytest.mark.parametrize('invalid', ['size','rgb','object'])
def test_invalid_condition_coordinates_fail_before_inference(pipeline, invalid):
    condition = {'size':Image.new('L',(16,16)), 'rgb':Image.new('RGB',(32,32)), 'object':object()}[invalid]
    with pytest.raises(ValueError,match='IP-Adapter mask'):
        run_isolated_inpaint(pipeline,control_report={},image=Image.new('RGB',(32,32)),
            mask_image=Image.new('L',(32,32),255),ip_adapter_mask=condition)
    assert not hasattr(pipeline,'kwargs')

def test_condition_mask_cannot_make_empty_latent_edit_valid(pipeline):
    edit=np.zeros((32,32),np.uint8);edit[1:3,1:3]=255
    with pytest.raises(ValueError,match='empty at latent resolution'):
        run_isolated_inpaint(pipeline,control_report={},image=Image.new('RGB',(32,32)),
            mask_image=Image.fromarray(edit),ip_adapter_mask=Image.new('L',(32,32),255))
    assert not hasattr(pipeline,'kwargs')

def test_preunion_observation_preserves_union_and_exposes_source_only_region():
    foreground=np.zeros((160,100),np.uint8);foreground[10:150,30:70]=255
    source=foreground.copy()
    result=project_target_garment_coverage(Image.fromarray(source),Image.fromarray(foreground),('shorts',),growth_pixels=0)
    try:
        before=np.asarray(result.pre_union_mask).copy()==255
        after=np.asarray(result.mask)==255
        assert np.any((source==255)&~before)
        assert np.array_equal(after,before|(source==255))
        assert result.record['target_pixels']==int(after.sum())
        result.mask.paste(0,(0,0,100,160))
        assert np.array_equal(np.asarray(result.pre_union_mask)==255,before)
    finally:result.close()

def test_unresolved_target_does_not_invent_preunion_capture():
    result=project_target_garment_coverage(Image.new('L',(32,32),255),Image.new('L',(32,32),255),('unknown garment',))
    try:
        assert result.record['status']=='UNRESOLVED'
        assert result.pre_union_mask is None
    finally:result.close()
