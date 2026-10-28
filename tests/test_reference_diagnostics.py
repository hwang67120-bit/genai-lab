from types import SimpleNamespace
from PIL import Image
import pytest
from genai_lab.reference_diagnostics import timed_stage
from genai_lab.reference_diagnostics import verify_and_save_ip_adapter_input
import json
import numpy as np


@pytest.mark.parametrize('mode,color', [('RGB', 'black'), ('RGBA', (80, 100, 120, 0))])
def test_invalid_adapter_input_is_saved_before_rejection(tmp_path, mode, color):
    image = Image.new(mode, (8, 6), color)
    with pytest.raises(ValueError, match='저장 위치='):
        verify_and_save_ip_adapter_input(image, tmp_path)
    with Image.open(tmp_path / 'debug_garment_ip_adapter.png') as saved:
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(image))
    record = json.loads((tmp_path / 'debug_garment_ip_adapter.json').read_text(encoding='utf-8'))
    assert record['status'] == 'rejected'
    assert record['errors']


@pytest.mark.parametrize('color', [(0, 0, 255), (255, 255, 255), (100, 100, 100)])
def test_uniform_adapter_input_warns_even_when_rgb_std_is_high(tmp_path, color):
    record = verify_and_save_ip_adapter_input(Image.new('RGB', (8, 6), color), tmp_path)
    assert record['spatial_std'] == 0
    assert record['status'] == 'warning'
    assert not record['errors']
    if color == (0, 0, 255):
        assert record['std'] > 5


def test_contrast_is_not_called_valid_texture_and_input_is_unchanged(tmp_path):
    image = Image.new('RGB', (8, 6), 'white')
    image.paste((0, 0, 180), (2, 1, 6, 5))
    original = np.array(image)
    messages = []
    record = verify_and_save_ip_adapter_input(image, tmp_path,
        SimpleNamespace(write_stage=lambda *args: messages.append(args)))
    assert record['status'] == 'measured'
    assert record['spatial_std'] > 5
    assert 'passed' not in record
    np.testing.assert_array_equal(np.asarray(image), original)
    with Image.open(record['image_path']) as saved:
        np.testing.assert_array_equal(np.asarray(saved), original)
    assert any('정상 판정은 하지 않았습니다' in message for _, message in messages)










@pytest.mark.parametrize('fails', [False, True])
def test_timing_captures_completion_or_failure_without_swallowing(monkeypatch, fails):
    clock = iter([10., 12.5])
    monkeypatch.setattr('genai_lab.reference_diagnostics.perf_counter', lambda: next(clock))
    messages, timings = [], {}
    log = SimpleNamespace(write_stage=lambda *args: messages.append(args))
    def run():
        with timed_stage(log, 'generation', 2, timings):
            if fails:
                raise RuntimeError('test failure')
    if fails:
        with pytest.raises(RuntimeError):
            run()
    else:
        run()
    assert timings['generation'] == {'elapsed_seconds': 2.5, 'status': 'failed' if fails else 'completed'}
    assert len(messages) == 2
