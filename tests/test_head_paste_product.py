"""회색 테두리 제거·원본 보호·CPU 배경 모델 해제와 기록을 검사한다."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image
from genai_lab.head_paste_rules import blend_white_product, protected_support
from genai_lab.head_paste_worker import prepare_white_product
from genai_lab.proportion_inputs import sha


def materials():
    before = np.full((100, 100, 3), 255, np.uint8)
    redrawn = np.full_like(before, (208, 208, 212))
    mask = np.zeros((80, 80), np.uint8)
    mask[4:76, 4:76] = 255
    features = np.zeros_like(mask)
    features[1:3, 20:25] = 255
    before[11:13, 30:35] = (30, 40, 50)
    alpha = np.zeros((100, 100), np.uint8)
    alpha[40:50, 40:50] = 255
    return before, redrawn, alpha, mask, features, (10, 10, 90, 90)


def test_5361_gray_background_pixels_stay_white_with_face_and_outside_protected():
    before, redrawn, alpha, mask, features, box = materials()
    support, _, _ = protected_support(mask, features, box)
    allowed = np.zeros(alpha.shape, bool)
    allowed[10:90, 10:90] = support
    white_scope = allowed & np.all(before >= 250, axis=2)
    positions = np.flatnonzero(white_scope)
    assert len(positions) > 5361
    alpha[allowed] = 255
    regression = np.zeros_like(alpha, bool)
    regression.flat[positions[:5361]] = True
    alpha[regression] = 0
    inputs = [a.copy() for a in (before, redrawn, alpha, mask, features)]

    result, checks = blend_white_product(before, redrawn, alpha, mask, features, box)

    assert checks['white_background_checked_pixels'] == 5361
    assert np.all(result[regression] == 255)
    assert np.array_equal(result[~allowed], before[~allowed])
    assert np.array_equal(result[11:13, 30:35], before[11:13, 30:35])
    assert np.array_equal(result[allowed & ~regression], redrawn[allowed & ~regression])
    assert checks['outside_equal'] and checks['face_equal'] and checks['white_background_equal']
    for actual, frozen in zip((before, redrawn, alpha, mask, features), inputs):
        assert np.array_equal(actual, frozen)


def test_soft_foreground_alpha_uses_existing_white_composition():
    before, redrawn, alpha, mask, features, box = materials()
    alpha[40:50, 40:50] = 128
    result, _ = blend_white_product(before, redrawn, alpha, mask, features, box)
    assert tuple(result[45, 45]) == (231, 231, 233)


@pytest.mark.parametrize('fault', ['empty', 'shape', 'face'])
def test_invalid_alpha_or_face_overlap_stops_without_success(fault):
    before, redrawn, alpha, mask, features, box = materials()
    if fault == 'empty': alpha[:] = 0
    elif fault == 'shape': alpha = alpha[:-1]
    else: features[20:25, 20:25] = 255
    with pytest.raises(ValueError):
        blend_white_product(before, redrawn, alpha, mask, features, box)


def test_worker_reads_raw_redraw_closes_cpu_model_and_records_model_sha(tmp_path):
    before, redrawn, alpha, mask, features, box = materials()
    source = tmp_path / 'raw_redraw.png'
    Image.fromarray(redrawn).save(source)
    model = tmp_path / 'model.onnx'; model.write_bytes(b'local CPU model')
    options = SimpleNamespace(foreground_model=model, foreground_sha256=sha(model))
    original_sha = sha(source)
    calls = []
    class Foreground:
        def __init__(self, opts): assert opts is options
        def alpha(self, image):
            calls.append('alpha')
            assert np.array_equal(np.asarray(image), redrawn)
            return alpha
        def close(self): calls.append('close')
    result, checks, record = prepare_white_product(before, source, mask, features, box,
                                                   options, tmp_path, Foreground)
    assert calls == ['alpha', 'close']
    assert sha(source) == original_sha
    assert record['model']['sha256'] == sha(model)
    assert record['model']['provider'] == 'CPUExecutionProvider'
    assert record['source_sha256'] == original_sha
    assert sha(record['alpha_file']) == record['alpha_sha256']
    assert checks['white_background_equal'] and np.all(result[20, 20] == 255)
    assert json.loads(json.dumps(record)) == record


def test_worker_alpha_failure_releases_model_and_does_not_write_alpha(tmp_path):
    model = tmp_path / 'model.onnx'; model.write_bytes(b'local CPU model')
    options = SimpleNamespace(foreground_model=model, foreground_sha256=sha(model))
    source = tmp_path / 'raw_redraw.png'; Image.new('RGB', (4, 4)).save(source)
    closed = []
    class Foreground:
        def __init__(self, options): pass
        def alpha(self, image): raise RuntimeError('분리 실패')
        def close(self): closed.append(True)
    with pytest.raises(RuntimeError, match='분리 실패'):
        prepare_white_product(np.zeros((4, 4, 3), np.uint8), source,
            np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8),
            (0, 0, 4, 4), options, tmp_path, Foreground)
    assert closed == [True] and not (tmp_path / 'product-alpha.png').exists()


def test_changed_foreground_model_is_rejected_before_loading(tmp_path):
    model = tmp_path / 'model.onnx'; model.write_bytes(b'changed')
    options = SimpleNamespace(foreground_model=model, foreground_sha256='old')
    with pytest.raises(ValueError, match='변경'):
        prepare_white_product(None, None, None, None, None, options, tmp_path,
                             lambda _: pytest.fail('변경된 모델 로드 금지'))
