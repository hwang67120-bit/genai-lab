import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from scripts.generation_inputs import resolve_inference_size
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




def test_dialog_preview_and_invalid_width():
    app = QApplication.instance() or QApplication([])
    dialog = GenerationResolutionDialog((768, 1344))
    dialog.width_input.setValue(512)
    assert '512×896' in dialog.preview.text()
    assert dialog.inference_width == 512
    dialog.width_input.setValue(513)
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.close()
