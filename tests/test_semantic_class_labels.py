import logging
import numpy as np
import pytest
from PIL import Image
from scripts.body_comparison_runner import (
    read_class_labels, create_layered_semantic_masks,
    create_parser_foreground_mask,
)


def test_palette_indices_survive_even_when_palette_colors_are_white(caplog):
    labels = np.array([[0, 2, 13, 5]], dtype=np.uint8)
    lip = Image.fromarray(labels).convert('P')
    lip.putpalette([255] * 768)
    atr = Image.new('L', lip.size)
    auto = Image.new('L', lip.size)
    assert np.all(np.asarray(lip.convert('L')) == 255)  # Old path loses IDs.
    with caplog.at_level(logging.INFO):
        identity, clothes = create_layered_semantic_masks(auto, lip, atr,
            {'Face': 13, 'Hair': 2, 'Upper-clothes': 5}, {'Face': 11, 'Hair': 2})
    assert np.array_equal(read_class_labels(lip, 'LIP'), labels)
    assert list(identity.getdata()) == [0, 255, 255, 0]
    assert list(clothes.getdata()) == [0, 0, 0, 255]
    assert 'mode=P' in caplog.text and 'ID=13' in caplog.text
    identity.close()
    clothes.close()


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'F'])
def test_visualization_or_float_is_not_a_class_id_array(mode):
    with pytest.raises(ValueError, match='원시 클래스 ID'):
        read_class_labels(Image.new(mode, (4, 4)), 'test')


def test_parser_foreground_uses_non_background_union():
    lip = Image.fromarray(
        np.array([[0, 2, 0, 0]], dtype=np.uint8)).convert("P")
    lip.putpalette([255] * 768)
    atr = Image.fromarray(
        np.array([[0, 0, 11, 0]], dtype=np.uint8)).convert("P")
    with create_parser_foreground_mask(
            lip, atr, {"Background": 0}, {"Background": 0}) as foreground:
        assert np.asarray(foreground).tolist() == [[0, 255, 255, 0]]


def test_parser_foreground_rejects_true_empty_output():
    lip = Image.new("P", (4, 4), 0)
    atr = Image.new("P", (4, 4), 0)
    with pytest.raises(RuntimeError, match="SCHP-LIP/ATR"):
        create_parser_foreground_mask(
            lip, atr, {"Background": 0}, {"Background": 0})


def test_true_empty_face_hair_is_not_faked():
    mask = Image.new('L', (4, 4))
    with pytest.raises(ValueError, match='0px'):
        create_layered_semantic_masks(mask, mask, mask,
            {'Face': 13, 'Hair': 2}, {'Face': 11, 'Hair': 2})


def test_face_reference_excludes_hair_and_cross_parser_conflicts():
    from scripts.body_comparison_runner import create_face_reference_mask
    lip = Image.fromarray(np.array([[2, 13, 13, 0]], dtype=np.uint8)).convert('P')
    lip.putpalette([255] * 768)
    atr = Image.fromarray(np.array([[2, 11, 2, 0]], dtype=np.uint8))
    with create_face_reference_mask(lip, atr, {'Face': 13, 'Hair': 2}, {'Face': 11, 'Hair': 2}) as face:
        assert np.asarray(face).tolist() == [[0, 255, 0, 0]]


def test_face_reference_does_not_fall_back_to_hair():
    from scripts.body_comparison_runner import create_face_reference_mask
    mask = Image.new('L', (4, 4), 2)
    with pytest.raises(ValueError, match='얼굴 단독'):
        create_face_reference_mask(mask, mask, {'Face': 13, 'Hair': 2}, {'Face': 11, 'Hair': 2})


def test_face_reference_output_is_loaded_by_parent(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from PIL import ImageDraw
    from genai_lab.body_comparison import CharacterBodyComparisonSettings, execute_character_body_comparison
    from genai_lab.target_masks import approve_target_masks
    source = Image.new('RGB', (256, 256), 'white')
    blank = Image.new('L', source.size)
    approved = approve_target_masks(source, blank, blank, use_automatic_base=True, layered_priority=True)
    expected = Image.new('L', source.size)
    ImageDraw.Draw(expected).rectangle((105, 30, 145, 55), fill=255)
    def run(command, **kwargs):
        def path(flag):
            return Path(command[command.index(flag)+1])
        assert '--output-face-reference-mask' in command
        expected.save(path('--output-face-reference-mask'))
        expected.save(path('--output-face-hair-mask'))
        expected.save(path('--output-protection-mask'))
        body = Image.new('L', source.size)
        ImageDraw.Draw(body).rectangle((70, 70, 180, 230), fill=255)
        body.save(path('--output-raw-mask'))
        Image.new('L', source.size, 255).save(path('--output-foreground-mask'))
        source.save(path('--output-densepose'))
        path('--output-metadata-json').write_text(json.dumps(dict(foreground_model_id='isnet-anime',
            foreground_pixel_count=256*256, foreground_percent=100., foreground_elapsed_seconds=.1,
            model_ids=['mock'], elapsed_seconds=.2)), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr('genai_lab.body_comparison.subprocess.run', run)
    settings = CharacterBodyComparisonSettings(Path(sys.executable), tmp_path, Path(__file__),
        tmp_path, tmp_path, 256, 256, 60, face_reference_enabled=True)
    result = execute_character_body_comparison(source, 'overall', settings, approved_target_masks=approved)
    try:
        assert np.array_equal(np.asarray(result.mask_layers.images['face_reference']), np.asarray(expected))
    finally:
        result.close()
        approved.close()
        expected.close()
        blank.close()
        source.close()


def test_missing_mapping_is_explicit_error():
    mask = Image.new('L', (4, 4))
    with pytest.raises(ValueError, match='매핑'):
        create_layered_semantic_masks(mask, mask, mask, {}, {})
