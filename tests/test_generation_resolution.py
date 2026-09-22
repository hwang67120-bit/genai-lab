import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from scripts.garment_inpaint_runner import resolve_inference_size, resize_inference_inputs
from genai_lab.generation_resolution_review import GenerationResolutionDialog


def test_sizes_preserve_original_default_and_ratio():
    assert resolve_inference_size((768, 1344)) == (768, 1344)
    assert resolve_inference_size((768, 1344), 768) == (768, 1344)
    assert resolve_inference_size((768, 1344), 512) == (512, 896)
    assert resolve_inference_size((768, 1344), 384) == (384, 672)


@pytest.mark.parametrize('width', [0, 255, 513, 2056])
def test_invalid_size_rejected(width):
    with pytest.raises(ValueError):
        resolve_inference_size((768, 1344), width)


def test_inputs_same_canvas_binary_mask_source_unchanged():
    image = Image.new('RGB', (768, 1344), 'white')
    mask = Image.new('L', image.size, 0)
    mask.paste(255, (192, 336, 576, 1008))
    pose = image.copy()
    before = image.tobytes(), mask.tobytes(), pose.tobytes()
    resized = resize_inference_inputs(image, mask, pose, (512, 896))
    assert all(im.size == (512, 896) for im in resized)
    assert set(np.unique(np.asarray(resized[1]))) == {0, 255}
    assert resized[1].getbbox() == (128, 224, 384, 672)
    assert before == (image.tobytes(), mask.tobytes(), pose.tobytes())
    for im in (*resized, image, mask, pose):
        im.close()


def test_dialog_preview_and_invalid_width():
    app = QApplication.instance() or QApplication([])
    dialog = GenerationResolutionDialog((768, 1344))
    dialog.width_input.setValue(512)
    assert '512×896' in dialog.preview.text()
    assert dialog.inference_width == 512
    dialog.width_input.setValue(513)
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.close()
