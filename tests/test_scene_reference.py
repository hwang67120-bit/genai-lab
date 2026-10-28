from dataclasses import replace
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from PIL import Image
from genai_lab.scene_reference import (
    AnalysisRejected, PartReference, SceneAnalysis, SceneCondition, as_mask,
    validate_scene, validate_points, analyze_with_retry, warp_garment, compose_scene,
)
from genai_lab.scene_generation import (
    scene_settings, validate_scene_condition, scene_arguments, prepare_scene_inputs,
)
from genai_lab.visual_reference import VisualInputs, configure_visual_condition


@pytest.fixture
def scene():
    shape = (32, 24)
    head = np.zeros(shape, bool)
    head[2:8, 8:16] = True
    tail = np.zeros(shape, bool)
    tail[22:28, 18:22] = True
    old = np.zeros(shape, bool)
    old[12:20, 8:16] = True
    placement = np.zeros(shape, bool)
    placement[10:22, 6:18] = True
    value = SceneAnalysis(
        Image.new("RGB", (24, 32), "white"),
        Image.new("RGBA", (8, 8), (20, 80, 160, 255)),
        as_mask(head | tail), as_mask(old), as_mask(np.zeros(shape, bool)), as_mask(placement),
        np.array([[0, 0], [7, 0], [0, 7], [7, 7]], float),
        np.array([[8, 12], [15, 12], [8, 19], [15, 19]], float),
        [PartReference("identity", Image.new("RGB", (8, 6), "red"), as_mask(head), .7),
         PartReference("tail", Image.new("RGB", (4, 6), "blue"), as_mask(tail), .5)],
        point_labels=("left_shoulder", "right_shoulder", "left_hem", "right_hem"),
        evidence={"input_type": "garment_only", "coordinates": "pixel", "occlusion": "fixture"},
    )
    yield value
    value.close()


class Lines:
    def extract(self, image):
        result = np.zeros((image.height, image.width), np.uint8)
        result[:, ::2] = 255
        return result


def config():
    return {
        "model": {"family": "sdxl", "id": "mock", "cache_dir": "unused"},
        "style": {"enabled": True},
        "clothing_reference_generation": {"enabled": True},
        "scene_lineart": {"enabled": True, "model_id": "TheMistoAI/MistoLine",
                         "extractor_output_white_lines": False},
    }


def composed(scene, tmp_path):
    with warp_garment(scene.garment_rgba, scene.src_points,
                      scene.dst_points, scene.source.size) as warped:
        return compose_scene(scene, warped, Lines(), tmp_path)


def test_scene_accepts_disjoint_semantic_masks(scene):
    validate_scene(scene)


@pytest.mark.parametrize("change", ["overlap", "missing_identity", "unresolved", "point_labels",
                                    "evidence", "soft", "transparent", "bad_scale"])
def test_scene_rejects_unproven_or_invalid_analysis(scene, change):
    if change == "overlap":
        scene.keep_mask.putpixel((8, 12), 255)
    elif change == "missing_identity":
        scene.parts[0].name = "face_unknown"
    elif change == "unresolved":
        scene.unresolved = ("tail front/back unknown",)
    elif change == "point_labels":
        scene.point_labels = ()
    elif change == "evidence":
        scene.evidence = {}
    elif change == "soft":
        scene.front_mask.putpixel((0, 0), 100)
    elif change == "transparent":
        scene.garment_rgba.putalpha(0)
    else:
        scene.parts[0].scale = float("nan")
    with pytest.raises(AnalysisRejected):
        validate_scene(scene)


@pytest.mark.parametrize("points", [
    [[0, 0], [0, 0], [2, 2]],
    [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [7, 0], [float("nan"), 2]],
    [[0, 0], [8, 0], [0, 7]],
])
def test_bad_landmarks(points):
    with pytest.raises(AnalysisRejected):
        validate_points(points, (8, 8), "test")


def test_identity_tps_preserves_color_and_alpha(scene):
    pixels = np.array(scene.garment_rgba)
    pixels[2:6, 2:6, 3] = 128
    garment = Image.fromarray(pixels)
    with warp_garment(garment, scene.src_points, scene.src_points, garment.size) as result:
        np.testing.assert_array_equal(np.asarray(result), pixels)


def test_tps_rejects_reflection(scene):
    target = scene.dst_points.copy()
    target[:, 0] = 23 - target[:, 0]
    with pytest.raises(AnalysisRejected, match="fold"):
        warp_garment(scene.garment_rgba, scene.src_points, target, scene.source.size)


def test_tps_budget_and_cancel(scene):
    with pytest.raises(AnalysisRejected, match="budget"):
        warp_garment(scene.garment_rgba, scene.src_points, scene.dst_points,
                     scene.source.size, maximum_pixels=10)
    with pytest.raises(InterruptedError):
        warp_garment(scene.garment_rgba, scene.src_points, scene.dst_points,
                     scene.source.size, cancelled=lambda: True)


def test_composition_is_grayscale_and_keeps_separate_tail(scene, tmp_path):
    before = scene.source.tobytes()
    condition = composed(scene, tmp_path)
    try:
        validate_scene_condition(condition, scene.source.size)
        assert [p.name for p in condition.references] == ["identity", "tail", "garment"]
        assert scene.source.tobytes() == before
        assert (tmp_path / "analysis.json").exists()
        assert condition.references[-1].rgb.getpixel((2, 2)) == (20, 80, 160)
        assert condition.references[1].rgb.getpixel((0, 0)) == (0, 0, 255)
    finally:
        condition.close()


def test_blank_garment_hint_does_not_pass_on_face_lines(scene, tmp_path):
    class BlankGarment:
        calls = 0
        def extract(self, image):
            self.calls += 1
            return np.full((image.height, image.width), 255 if self.calls == 1 else 0, np.uint8)
    with warp_garment(scene.garment_rgba, scene.src_points, scene.dst_points, scene.source.size) as warped:
        with pytest.raises(AnalysisRejected, match="lineart is empty"):
            compose_scene(scene, warped, BlankGarment(), tmp_path)


def test_garment_cannot_cover_identity(scene, tmp_path):
    scene.placement_mask.paste(255, (0, 0, 24, 32))
    with Image.new("RGBA", scene.source.size, (20, 80, 160, 255)) as warped:
        with pytest.raises(AnalysisRejected, match="identity"):
            compose_scene(scene, warped, Lines(), tmp_path)


def test_alignment_cannot_silently_clip_placement(scene, tmp_path):
    scene.placement_mask.paste(0, (0, 0, 24, 32))
    scene.placement_mask.putpixel((8, 12), 255)
    with warp_garment(scene.garment_rgba, scene.src_points, scene.dst_points, scene.source.size) as warped:
        with pytest.raises(AnalysisRejected, match="placement"):
            compose_scene(scene, warped, Lines(), tmp_path)


def test_retry_is_bounded_without_repeated_generation(scene):
    calls = []
    class Analyzer:
        def analyze(self, *args, attempt):
            calls.append(attempt)
            raise AnalysisRejected("no reliable garment landmarks")
    with pytest.raises(AnalysisRejected):
        analyze_with_retry(Analyzer(), scene.source, scene.garment_rgba)
    assert calls == [0, 1]


def test_retry_success_stays_open_after_previous_failure(scene):
    class Analyzer:
        def analyze(self, *args, attempt):
            if attempt == 0:
                raise AnalysisRejected("uncertain")
            return scene
    assert analyze_with_retry(Analyzer(), scene.source, scene.garment_rgba).source.getpixel((0, 0)) == (255, 255, 255)


def test_no_backend_blocks_before_model_load(scene, monkeypatch):
    monkeypatch.setattr("genai_lab.scene_generation.LineartExtractor",
                        lambda *a, **k: pytest.fail("model loaded before validation"))
    with pytest.raises(ValueError, match="분석기가 아직"):
        prepare_scene_inputs(scene.source, scene.garment_rgba, config())


def test_rgb_init_cannot_be_passed_as_hint(scene, tmp_path):
    condition = composed(scene, tmp_path)
    condition.hint.paste("blue", (0, 0, 24, 32))
    with pytest.raises(ValueError, match="RGB"):
        validate_scene_condition(condition, scene.source.size)
    condition.close()


def test_grayscale_empty_hint_rejected(scene, tmp_path):
    condition = composed(scene, tmp_path)
    condition.hint.paste("white", (0, 0, 24, 32))
    with pytest.raises(ValueError, match="단색"):
        validate_scene_condition(condition, scene.source.size)
    condition.close()


def test_scene_arguments_use_separate_control_image(scene, tmp_path):
    condition = composed(scene, tmp_path)
    inputs = SimpleNamespace(scene_condition=condition)
    pipe = SimpleNamespace(_genai_lab_scene_lineart_model_id="TheMistoAI/MistoLine")
    args = scene_arguments(pipe, config(), inputs, scene.source.size)
    assert args["control_image"] is condition.hint
    assert not {"image", "mask_image", "strength"} & args.keys()
    assert args["callback_on_step_end"](pipe, 0, None, {"latents": 1}) == {"latents": 1}
    with pytest.raises(ValueError, match="다시 준비"):
        scene_arguments(object(), config(), inputs, scene.source.size)
    with pytest.raises(ValueError, match="꺼져"):
        scene_arguments(pipe, {}, inputs, scene.source.size)
    condition.close()


def test_callback_cancel_and_timeout(scene, tmp_path, monkeypatch):
    condition = composed(scene, tmp_path)
    inputs = SimpleNamespace(scene_condition=condition)
    pipe = SimpleNamespace(_genai_lab_scene_lineart_model_id="TheMistoAI/MistoLine")
    cfg = config()
    clock = [0.]
    monkeypatch.setattr("genai_lab.scene_generation.perf_counter", lambda: clock[0])
    args = scene_arguments(pipe, cfg, inputs, scene.source.size)
    clock[0] = 1801
    with pytest.raises(TimeoutError):
        args["callback_on_step_end"](pipe, 0, None, {})
    cfg["scene_lineart"]["cancelled"] = lambda: True
    with pytest.raises(InterruptedError):
        scene_arguments(pipe, cfg, inputs, scene.source.size)
    condition.close()


def test_existing_mode_ignores_disabled_scene():
    assert scene_settings({}) is None
    assert scene_arguments(None, {}, SimpleNamespace(), (24, 32)) == {}


def test_multi_reference_encoding_uses_exact_rgb_order(scene, tmp_path, monkeypatch):
    condition = composed(scene, tmp_path)
    identity, tail, garment = condition.references
    inputs = VisualInputs(scene.source.copy(), identity.rgb.copy(), garment.rgb.copy(),
                          identity.region.copy(), garment.region.copy(), scene_condition=condition)
    observed = {}
    pipe = SimpleNamespace(
        _execution_device="cpu",
        set_ip_adapter_scale=lambda scales: observed.update(scales=scales),
        prepare_ip_adapter_image_embeds=lambda **kwargs: observed.update(kwargs) or ["encoded"])
    output = configure_visual_condition(pipe, inputs)
    assert observed["ip_adapter_image"][0] == [identity.rgb, tail.rgb, garment.rgb]
    assert observed["scales"] == pytest.approx(.55)
    assert output == {"ip_adapter_image_embeds": ["encoded"]}
    assert "masks" not in observed
    inputs.close()


def test_scene_pipeline_loads_img2img_on_cpu_then_offloads(monkeypatch, tmp_path):
    from genai_lab.model import prepare_pipeline
    calls = []
    pipe = SimpleNamespace(
        load_ip_adapter=lambda *a, **k: calls.append("adapter"),
        set_ip_adapter_scale=lambda value: None,
        enable_model_cpu_offload=lambda: calls.append("offload"))
    scene_factory = SimpleNamespace(from_pretrained=lambda *a, **k: calls.append(k) or pipe)
    forbidden = SimpleNamespace(from_pretrained=lambda *a, **k: pytest.fail("wrong pipeline"))
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        AutoPipelineForImage2Image=SimpleNamespace(from_pipe=lambda p: pytest.fail("img2img")),
        ControlNetModel=SimpleNamespace(from_pretrained=lambda *a, **k: "controlnet"),
        StableDiffusionPipeline=forbidden, StableDiffusionXLPipeline=forbidden,
        StableDiffusionXLControlNetImg2ImgPipeline=scene_factory,
        StableDiffusionXLControlNetPipeline=forbidden))
    cfg = config()
    cfg["model"]["cache_dir"] = str(tmp_path)
    cfg["generation"] = {"mode": "image_to_image"}
    cfg["style"].update(adapter_repository="h94/IP-Adapter", adapter_subfolder="sdxl_models",
                        adapter_weight="ip-adapter_sdxl.bin", scale=.8)
    result = prepare_pipeline(cfg)
    assert result is pipe
    assert calls[0]["controlnet"] == "controlnet"
    assert calls[0]["local_files_only"] is True
    assert calls[-1] == "offload"
    assert result._genai_lab_scene_lineart_model_id == "TheMistoAI/MistoLine"


def test_preprocessing_releases_analyzer_before_lineart(scene, tmp_path, monkeypatch):
    original = scene.source.copy()
    garment = scene.garment_rgba.copy()
    state = []
    class Analyzer:
        def analyze(self, *args, **kwargs):
            return scene
        def close(self):
            state.append("analysis_closed")
    class Extractor(Lines):
        def __init__(self, *args, **kwargs):
            assert state == ["analysis_closed"]
        def close(self):
            state.append("lineart_closed")
    cfg = config()
    cfg["scene_lineart"]["analyzer"] = Analyzer()
    monkeypatch.setattr("genai_lab.scene_generation.LineartExtractor", Extractor)
    monkeypatch.setattr("genai_lab.scene_generation.tempfile.mkdtemp", lambda **k: str(tmp_path))
    inputs = prepare_scene_inputs(original, garment, cfg)
    assert state == ["analysis_closed", "lineart_closed"]
    assert inputs.source.tobytes() == original.tobytes()
    assert len(inputs.scene_condition.references) == 3
    assert (tmp_path / "timings.json").exists()
    inputs.close()
    original.close()
    garment.close()


def test_preview_and_execution_reference_mismatch_rejected(scene, tmp_path):
    from genai_lab.reference_contract import validate_visual_inputs
    condition = composed(scene, tmp_path)
    identity, tail, garment = condition.references
    inputs = VisualInputs(scene.source.copy(), identity.rgb.copy(), garment.rgb.copy(),
                          identity.region.copy(), garment.region.copy(), scene_condition=condition)
    inputs.garment.putpixel((0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="실제 생성 참조"):
        validate_visual_inputs(inputs)
    inputs.close()
