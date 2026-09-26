"""Explicit diagnostics must not mutate Base, W, or the independent channel."""
import numpy as np
import pytest
from PIL import Image
from genai_lab.selected_garment_correction import _diagnostic_inputs


def assets():
    base = Image.fromarray(np.arange(4*5*3, dtype=np.uint8).reshape(4,5,3))
    w = Image.fromarray(np.full((4,5), 254, np.uint8))
    roi = np.zeros((4,5), np.uint8)
    roi[1:3, 2:4] = 255
    return base, w, Image.fromarray(roi)


def test_exclusion_only_changes_attention_not_edit_mask_or_base():
    base, w, roi = assets()
    before = base.tobytes(), w.tobytes()
    target = np.array(w)
    target[0] = 0
    condition = Image.fromarray(target)
    attention, initial, record = _diagnostic_inputs(base,w,condition,exclusion_mask=roi)
    expected = target.copy()
    expected[np.array(roi)==255] = 0
    assert np.array_equal(np.array(attention), expected)
    assert initial is None
    assert (base.tobytes(),w.tobytes()) == before
    assert np.array_equal(np.array(condition), target)
    assert record['ip_exclusion_pixels'] == 4


def test_prefill_changes_only_initial_roi_and_keeps_attention_independent():
    base,w,roi = assets()
    original = np.array(base)
    attention, initial, record = _diagnostic_inputs(base,w,None,prefill_mask=roi,prefill_rgb=(111,122,133))
    expected = original.copy()
    expected[np.array(roi)==255] = (111,122,133)
    assert np.array_equal(np.array(initial),expected)
    assert np.array_equal(np.array(base),original)
    assert attention is None
    assert record['initial_prefill_rgb'] == [111,122,133]
    assert record['initial_prefill_pixels'] == 4


def test_disabled_diagnostics_produce_no_replacement():
    base,w,_ = assets()
    attention,initial,record = _diagnostic_inputs(base,w,None)
    assert attention is None and initial is None
    assert record['initial_prefill_pixels'] == record['ip_exclusion_pixels'] == 0


@pytest.mark.parametrize('invalid', ['wrong_size','gray','empty'])
def test_reject_nonbinary_or_misaligned_diagnostic_mask(invalid):
    base,w,roi = assets()
    bad = Image.new('L',(1,1),255) if invalid=='wrong_size' else Image.new('L',base.size,127 if invalid=='gray' else 0)
    with pytest.raises(ValueError):
        _diagnostic_inputs(base,w,None,exclusion_mask=bad)


@pytest.mark.parametrize('missing', ['mask','color'])
def test_prefill_requires_explicit_mask_and_color(missing):
    base,w,roi = assets()
    with pytest.raises(ValueError):
        _diagnostic_inputs(base,w,None,prefill_mask=None if missing=='mask' else roi,prefill_rgb=None if missing=='color' else (1,2,3))
