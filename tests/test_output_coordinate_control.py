from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from genai_lab.output_coordinate_control import (
    create_dual_masks,
    enforce_hard_paste,
    exact_pixel_composite,
    isolate_output_mask,
    run_isolated_inpaint,
)


def test_exact_pixel_restoration_with_arbitrary_decoder_changes():
    rng = np.random.default_rng(42)
    base = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    proposal = Image.new('RGB', base.size, 'red')
    region = np.zeros((32, 32), dtype=np.uint8)
    region[8:24, 8:24] = 255
    report = {}
    result = exact_pixel_composite(base, proposal, Image.fromarray(region), report)
    assert np.array_equal(np.asarray(result)[region == 0], np.asarray(base)[region == 0])
    assert np.all(np.asarray(result)[region == 255] == [255, 0, 0])
    assert report['outside_changed_pixels'] == 0


def test_face_animal_ears_accessory_and_hair_are_excluded_from_garment_mask(monkeypatch):
    image = Image.new('RGB', (32, 32))
    mask = Image.new('L', image.size, 255)
    parts = []
    for index, name in enumerate(['face', 'human_ears', 'animal_ears', 'hair_accessory', 'hair']):
        a = np.zeros((32, 32), dtype=np.uint8)
        a[index * 6:index * 6 + 2, 12:16] = 255
        parts.append(SimpleNamespace(name='output_' + name,
                     source_mask=Image.fromarray(a), close=lambda: None))
    analyzer = SimpleNamespace(analyze=lambda *a, **kw: parts, close=lambda: None)
    monkeypatch.setattr('genai_lab.hair_error_correction.create_output_hair_analyzer', lambda c: analyzer)
    output = isolate_output_mask(image, mask, {}, target='garment', check=lambda: None)
    for part in parts:
        selected = np.asarray(output)[np.asarray(part.source_mask) > 0]
        if part.name == 'output_human_ears':
            assert selected.all()
        else:
            assert not selected.any()
    assert np.asarray(output)[:, 0].all()


def test_missing_face_protection_fails_closed(monkeypatch):
    analyzer = SimpleNamespace(analyze=lambda *a, **kw: [], close=lambda: None)
    monkeypatch.setattr('genai_lab.hair_error_correction.create_output_hair_analyzer', lambda c: analyzer)
    with pytest.raises(ValueError, match='face protection'):
        isolate_output_mask(Image.new('RGB', (32, 32)), Image.new('L', (32, 32), 255),
                            {}, target='garment', check=lambda: None)


def test_unknown_pipeline_cannot_silently_skip_latent_preservation():
    with pytest.raises(ValueError, match='unverified'):
        run_isolated_inpaint(object(), control_report={})


def test_attention_uses_output_mask_and_native_steps_are_recorded(monkeypatch):
    import diffusers
    class Pipeline:
        vae_scale_factor = 8
        unet = SimpleNamespace(config=SimpleNamespace(in_channels=4))
        mask_processor = object()
        def __call__(self, **kwargs):
            assert self.mask_processor.config.do_binarize is False
            self.kwargs = kwargs
            for i in range(3):
                kwargs['callback_on_step_end'](self, i, 3-i, {})
            return 'result'
    monkeypatch.setattr(diffusers, 'StableDiffusionXLInpaintPipeline', Pipeline)
    pipe = Pipeline()
    original_mask_processor = pipe.mask_processor
    a = np.zeros((32, 32), dtype=np.uint8); a[8:24, 8:24] = 255
    report = {}
    assert run_isolated_inpaint(pipe, control_report=report,
        image=Image.new('RGB', (32, 32)), mask_image=Image.fromarray(a)) == 'result'
    assert pipe.mask_processor is original_mask_processor
    tensor = pipe.kwargs['cross_attention_kwargs']['ip_adapter_masks'][0]
    assert tensor.shape == (1, 1, 32, 32)
    assert np.array_equal(tensor[0, 0].numpy() > 0, a > 0)
    assert report['output_coordinate_control']['latent_restore_steps'] == 3
    pipe.unet = SimpleNamespace(config=SimpleNamespace(in_channels=9))
    with pytest.raises(ValueError, match='four-channel'):
        run_isolated_inpaint(pipe, control_report={})


def test_nonempty_pixel_mask_that_vanishes_in_latent_space_is_blocked(monkeypatch):
    import diffusers
    class Pipeline:
        vae_scale_factor = 8
        unet = SimpleNamespace(config=SimpleNamespace(in_channels=4))
        def __call__(self, **kwargs):
            raise AssertionError('empty latent edit must not invoke GPU inference')
    monkeypatch.setattr(diffusers, 'StableDiffusionXLInpaintPipeline', Pipeline)
    pixels = np.zeros((32, 32), dtype=np.uint8); pixels[1:3, 1:3] = 255
    report = {}
    with pytest.raises(ValueError, match='empty at latent resolution'):
        run_isolated_inpaint(Pipeline(),control_report=report,
            image=Image.new('RGB', (32, 32)),mask_image=Image.fromarray(pixels))
    assert report['output_coordinate_control']['latent_restore_steps'] == 0



def test_dual_mask_feather_never_crosses_hard_boundary():
    detected = np.zeros((64, 64), dtype=np.uint8)
    detected[8:56, 8:56] = 255
    hard, soft = create_dual_masks(Image.fromarray(detected), feather_radius=6)
    try:
        hard_values = np.asarray(hard)
        soft_values = np.asarray(soft)
        assert set(np.unique(hard_values)) <= {0, 255}
        assert not np.any(soft_values[hard_values == 0])
        assert np.any((soft_values > 0) & (soft_values < 255))
        assert np.count_nonzero(soft_values) < np.count_nonzero(hard_values)
    finally:
        hard.close()
        soft.close()


def test_dual_mask_fails_closed_when_inward_erosion_removes_region():
    detected = np.zeros((32, 32), dtype=np.uint8)
    detected[15:17, 15:17] = 255
    with pytest.raises(ValueError, match="empty after inward erosion"):
        create_dual_masks(Image.fromarray(detected), feather_radius=10)


def test_enforce_hard_paste_restores_every_outside_rgb_pixel():
    rng = np.random.default_rng(7)
    base_values = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    proposed_values = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    hard_values = np.zeros((24, 24), dtype=np.uint8)
    hard_values[6:18, 6:18] = 255
    report = {}
    result = enforce_hard_paste(
        Image.fromarray(base_values),
        Image.fromarray(proposed_values),
        Image.fromarray(hard_values),
        report,
    )
    try:
        result_values = np.asarray(result)
        assert np.array_equal(
            result_values[hard_values == 0], base_values[hard_values == 0])
        assert np.array_equal(
            result_values[hard_values == 255], proposed_values[hard_values == 255])
        assert report["outside_changed_pixels"] == 0
        assert report["outside_changed_ratio"] == 0.0
    finally:
        result.close()


def test_animal_ear_mask_ignores_human_ear_detector_but_keeps_face_protection(
        monkeypatch):
    image = Image.new("RGB", (32, 32))
    target = np.zeros((32, 32), dtype=np.uint8)
    target[6:26, 6:26] = 255
    face = np.zeros((32, 32), dtype=np.uint8)
    face[6:10, 6:10] = 255
    human_ears = target.copy()
    parts = [
        SimpleNamespace(
            name="output_face", source_mask=Image.fromarray(face),
            close=lambda: None),
        SimpleNamespace(
            name="output_human_ears", source_mask=Image.fromarray(human_ears),
            close=lambda: None),
    ]
    analyzer = SimpleNamespace(
        analyze=lambda *args, **kwargs: parts, close=lambda: None)
    monkeypatch.setattr(
        "genai_lab.hair_error_correction.create_output_hair_analyzer",
        lambda config: analyzer,
    )
    output = isolate_output_mask(
        image, Image.fromarray(target), {}, target="animal_ears",
        check=lambda: None)
    try:
        values = np.asarray(output)
        assert values[16, 16] == 255
        assert values[7, 7] == 0
        assert np.count_nonzero(values) > 0
    finally:
        output.close()
        image.close()
        for part in parts:
            part.source_mask.close()


def test_soft_mask_outside_hard_boundary_is_rejected(monkeypatch):
    import diffusers

    class Pipeline:
        vae_scale_factor = 8
        unet = SimpleNamespace(config=SimpleNamespace(in_channels=4))

    monkeypatch.setattr(diffusers, "StableDiffusionXLInpaintPipeline", Pipeline)
    hard = np.zeros((32, 32), dtype=np.uint8)
    hard[8:24, 8:24] = 255
    soft = hard.copy()
    soft[0, 0] = 1
    with pytest.raises(ValueError, match="exceeds hard authorization"):
        run_isolated_inpaint(
            Pipeline(),
            control_report={},
            hard_authorized_mask=Image.fromarray(hard),
            image=Image.new("RGB", (32, 32)),
            mask_image=Image.fromarray(soft),
        )


def test_dual_mask_adapts_feather_radius_for_thin_valid_region():
    detected = np.zeros((48, 48), dtype=np.uint8)
    detected[18:30, 8:40] = 255
    hard, soft = create_dual_masks(Image.fromarray(detected), feather_radius=10)
    try:
        hard_values = np.asarray(hard)
        soft_values = np.asarray(soft)
        assert soft.info["requested_feather_radius"] == 10
        assert 0 < soft.info["effective_feather_radius"] < 10
        assert np.count_nonzero(soft_values) > 0
        assert not np.any(soft_values[hard_values == 0])
    finally:
        hard.close()
        soft.close()


def test_garment_mask_ignores_independent_human_ear_false_positive(monkeypatch):
    image = Image.new("RGB", (32, 32))
    target = np.zeros((32, 32), dtype=np.uint8)
    target[4:29, 4:29] = 255
    face = np.zeros((32, 32), dtype=np.uint8)
    face[4:9, 4:9] = 255
    animal_ears = np.zeros((32, 32), dtype=np.uint8)
    animal_ears[4:9, 23:28] = 255
    human_ears = target.copy()
    parts = [
        SimpleNamespace(name="output_face", source_mask=Image.fromarray(face), close=lambda: None),
        SimpleNamespace(name="output_animal_ears", source_mask=Image.fromarray(animal_ears), close=lambda: None),
        SimpleNamespace(name="output_human_ears", source_mask=Image.fromarray(human_ears), close=lambda: None),
    ]
    analyzer = SimpleNamespace(analyze=lambda *args, **kwargs: parts, close=lambda: None)
    monkeypatch.setattr(
        "genai_lab.hair_error_correction.create_output_hair_analyzer",
        lambda config: analyzer,
    )
    output = isolate_output_mask(
        image, Image.fromarray(target), {}, target="garment", check=lambda: None)
    try:
        values = np.asarray(output)
        assert values[16, 16] == 255
        assert values[6, 6] == 0
        assert values[6, 25] == 0
    finally:
        output.close()


def test_output_protection_distinguishes_human_animal_ears_and_tail(monkeypatch):
    from types import SimpleNamespace
    from genai_lab.output_coordinate_control import analyze_output_protection_masks

    arrays = {}
    for name, box in {
        "output_face": (8, 8, 16, 16),
        "output_human_ears": (4, 10, 8, 14),
        "output_animal_ears": (10, 2, 14, 6),
        "output_hair": (6, 1, 18, 9),
        "output_hair_accessory": (16, 3, 19, 6),
        "output_tail": (20, 16, 28, 28),
    }.items():
        values = np.zeros((32, 32), dtype=np.uint8)
        values[box[1]:box[3], box[0]:box[2]] = 255
        arrays[name] = Image.fromarray(values, mode="L")
    parts = [
        SimpleNamespace(name=name, source_mask=image, close=lambda: None)
        for name, image in arrays.items()
    ]
    analyzer = SimpleNamespace(
        analyze=lambda *args, **kwargs: parts,
        close=lambda: None,
    )
    monkeypatch.setattr(
        "genai_lab.hair_error_correction.create_output_hair_analyzer",
        lambda config: analyzer,
    )
    analysis = analyze_output_protection_masks(
        Image.new("RGB", (32, 32)),
        {},
        target="garment",
        check=lambda: None,
    )
    try:
        assert "output_animal_ears" in analysis.record["hard_parts"]
        assert "output_human_ears" not in analysis.record["hard_parts"]
        assert analysis.record["diagnostic_only_parts"] == ["output_human_ears"]
        assert analysis.record["conditional_parts"] == ["output_tail"]
        assert np.asarray(analysis.conditional_protection)[20, 24] == 255
    finally:
        analysis.close()
        for image in arrays.values():
            image.close()

