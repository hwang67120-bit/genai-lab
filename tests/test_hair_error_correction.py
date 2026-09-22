from types import SimpleNamespace
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from genai_lab.hair_error_correction import (
    HairCorrectionContractError,
    _isolated_hair_mask,
    _isolated_hair_masks,
    correct_selected_hair,
    resolve_hair_error_correction,
)
from genai_lab.regional_reference import DetectedPart


def mask(box):
    result = Image.new("L", (32, 32))
    ImageDraw.Draw(result).rectangle(box, fill=255)
    return result


def settings_config():
    return {
        "reference_analysis": {
            "hair_error_correction": {
                "enabled": True,
                "minimum_similarity": .68,
                "minimum_improvement": .01,
                "hair_reference_scale": .55,
                "inpaint_strength": .45,
                "inference_steps": 10,
                "guidance_scale": 5.5,
                "mask_padding_pixels": 1,
                "protection_padding_pixels": 1,
                "timeout_seconds": 300,
            },
            "hair_view_gate": {
                "enabled": True,
                "score_threshold": .35,
                "minimum_margin": .08,
            },
        },
        "clothing_reference_generation": {
            "identity_reference_scale": .7,
            "garment_reference_scale": .45,
        },
    }


def test_output_hair_mask_excludes_face_ears_and_accessory():
    detected = [
        DetectedPart("output_hair", mask((4, 2, 27, 24)), None, .5),
        DetectedPart("output_face", mask((10, 10, 21, 23)), None, .5),
        DetectedPart("output_ears", mask((5, 2, 10, 8)), None, .5),
        DetectedPart("output_hair_accessory", mask((23, 5, 27, 9)), None, .5),
    ]
    try:
        isolated, reason = _isolated_hair_mask(
            detected, (32, 32),
            resolve_hair_error_correction(settings_config()),
            require_ears=True,
        )
        assert reason is None
        pixels = np.asarray(isolated) >= 128
        assert pixels.any()
        assert not pixels[15, 15]
        assert not pixels[5, 7]
        assert not pixels[7, 25]
        isolated.close()
    finally:
        for part in detected:
            part.close()


def test_implausibly_broad_hair_accessory_detection_is_diagnostic_only():
    detected = [
        DetectedPart("output_hair", mask((4, 2, 27, 24)), None, .5),
        DetectedPart("output_face", mask((10, 10, 21, 23)), None, .5),
        DetectedPart(
            "output_hair_accessory", mask((4, 2, 27, 24)), None, .5),
    ]
    diagnostics = {}
    try:
        isolated, protected, reason = _isolated_hair_masks(
            detected, (32, 32),
            resolve_hair_error_correction(settings_config()),
            require_ears=False, diagnostics=diagnostics,
        )
        assert reason is None
        assert np.asarray(isolated).any()
        assert diagnostics["hair_accessory"]["status"] == (
            "diagnostic_only_implausibly_broad")
        assert diagnostics["hair_accessory"]["hair_overlap_ratio"] > .5
        isolated.close()
        protected.close()
    finally:
        for part in detected:
            part.close()


def test_selected_hair_inpaint_receives_only_hair_reference(monkeypatch):
    config = settings_config()
    settings = resolve_hair_error_correction(config)
    generated = Image.new("RGB", (32, 32), "blue")
    source = Image.new("RGB", (32, 32), "green")
    source_hair = mask((4, 2, 27, 24))
    face_hair = mask((4, 2, 27, 24))
    garment = mask((2, 25, 29, 31))
    visual_inputs = SimpleNamespace(
        source=source,
        hair_mask=source_hair,
        extra_references=(),
        identity=source,
        identity_mask=face_hair,
        garment=source,
        garment_mask=garment,
        scene_condition=None,
    )

    class Analyzer:
        def analyze(self, *args, **kwargs):
            return [
                DetectedPart("output_hair", mask((4, 2, 27, 24)), None, .5),
                DetectedPart("output_face", mask((10, 10, 21, 23)), None, .5),
            ]

    calls = []

    class Inpaint:
        def set_ip_adapter_scale(self, value):
            calls.append(("scale", value))

        def __call__(self, **kwargs):
            calls.append(("images", kwargs["ip_adapter_image"]))
            return SimpleNamespace(
                images=[Image.new("RGB", (32, 32), "red")])

    generation_scales = []
    pipeline = SimpleNamespace(
        set_ip_adapter_scale=generation_scales.append)
    scores = iter((.4, .7))
    monkeypatch.setattr(
        "genai_lab.visual_reference.image_feature",
        lambda *args: np.array([1.0]))
    monkeypatch.setattr(
        "genai_lab.part_error_correction._similarity",
        lambda *args: next(scores))
    request = SimpleNamespace(
        prompt="approved hair prompt",
        negative_prompt="negative",
        seed=42,
    )
    approval = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        # ApprovedReferenceRun.record() returns a JSON round trip: tuples become
        # lists even though the semantic contract is unchanged.
        "hair_error_correction": json.loads(json.dumps(settings.record())),
    }
    result, report = correct_selected_hair(
        pipeline,
        generated,
        visual_inputs,
        config,
        request,
        approved_run_record=approval,
        analyzer=Analyzer(),
        inpaint_pipeline=Inpaint(),
        view_analyzer=SimpleNamespace(
            analyze_pair=lambda *args, **kwargs: {
                "status": "compatible",
                "reason": None,
                "reference": {"view": "front"},
                "candidate": {"view": "front"},
                "allowed_candidate_views": ("front", "front_three_quarter"),
                "inpaint_allowed": True,
            }),
    )
    try:
        assert report["status"] == "corrected_and_accepted"
        assert calls[0] == ("scale", .55)
        assert calls[1][0] == "images"
        assert len(calls[1][1]) == 1
        assert calls[1][1][0].size[0] == calls[1][1][0].size[1]
        assert generation_scales[-1] == [[.7, .45]]
        assert result.getpixel((5, 5)) == (255, 0, 0)
        assert result.getpixel((15, 15)) == (0, 0, 255)
    finally:
        result.close()
        generated.close()
        source.close()
        source_hair.close()
        face_hair.close()
        garment.close()


def test_incompatible_view_skips_inpaint(monkeypatch):
    config = settings_config()
    settings = resolve_hair_error_correction(config)
    generated = Image.new("RGB", (32, 32), "blue")
    source = Image.new("RGB", (32, 32), "green")
    hair = mask((4, 2, 27, 24))

    class Analyzer:
        def analyze(self, *args, **kwargs):
            return [
                DetectedPart("output_hair", mask((4, 2, 27, 24)), None, .5),
                DetectedPart("output_face", mask((10, 10, 21, 23)), None, .5),
            ]

    class Inpaint:
        def __call__(self, **kwargs):
            raise AssertionError("시점 불일치 후보에 Inpaint가 실행됐습니다.")

    monkeypatch.setattr(
        "genai_lab.visual_reference.image_feature", lambda *args: np.array([1.0]))
    monkeypatch.setattr(
        "genai_lab.part_error_correction._similarity", lambda *args: .4)
    request = SimpleNamespace(prompt="approved", negative_prompt="negative", seed=1)
    result, report = correct_selected_hair(
        SimpleNamespace(), generated,
        SimpleNamespace(source=source, hair_mask=hair, extra_references=()),
        config, request,
        approved_run_record={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "hair_error_correction": settings.record(),
        },
        analyzer=Analyzer(),
        inpaint_pipeline=Inpaint(),
        view_analyzer=SimpleNamespace(
            analyze_pair=lambda *args, **kwargs: {
                "status": "incompatible",
                "reason": "unobserved_hair_geometry_required",
                "reference": {"view": "front"},
                "candidate": {"view": "side"},
                "allowed_candidate_views": ("front", "front_three_quarter"),
                "inpaint_allowed": False,
            }),
    )
    try:
        assert result is generated
        assert report["status"] == "skipped_view_incompatible"
    finally:
        generated.close()
        source.close()
        hair.close()


def test_hair_correction_rejects_unapproved_parameter_change():
    config = settings_config()
    generated = Image.new("RGB", (32, 32))
    try:
        with pytest.raises(HairCorrectionContractError):
            correct_selected_hair(
                SimpleNamespace(),
                generated,
                SimpleNamespace(hair_mask=None),
                config,
                SimpleNamespace(
                    prompt="changed", negative_prompt="negative", seed=42),
                approved_run_record={
                    "prompt": "approved",
                    "negative_prompt": "negative",
                    "hair_error_correction":
                        resolve_hair_error_correction(config).record(),
                },
            )
    finally:
        generated.close()


@pytest.fixture(autouse=True)
def isolate_model_dependencies(monkeypatch):
    # These tests exercise promotion decisions using synthetic proposals. The
    # real control contract is exercised separately in test_output_coordinate_control.
    monkeypatch.setattr('genai_lab.hair_error_correction.run_isolated_inpaint',
                        lambda pipe, control_report, **kwargs: pipe(**kwargs))


def test_high_clip_similarity_cannot_skip_failed_hair_color_gate(monkeypatch):
    config = settings_config()
    config['reference_analysis']['hair_error_correction']['promotion_gate'] = {'enabled': True}
    settings = resolve_hair_error_correction(config)
    source = Image.new('RGB', (32, 32), 'red')
    generated = Image.new('RGB', (32, 32), 'blue')
    parts = [DetectedPart('output_hair', mask((4, 2, 27, 24)), None, .5),
             DetectedPart('output_face', mask((10, 10, 21, 23)), None, .5)]
    analyzer = SimpleNamespace(analyze=lambda *args, **kwargs: parts)
    monkeypatch.setattr('genai_lab.visual_reference.image_feature', lambda *args: np.array([1., 0.]))
    request = SimpleNamespace(prompt='blue hair', negative_prompt='', seed=42)
    inputs = SimpleNamespace(source=source,hair_mask=mask((4, 2, 27, 24)),extra_references=())
    image, report = correct_selected_hair(SimpleNamespace(),generated,inputs,config,request,
        approved_run_record={'prompt':request.prompt,'negative_prompt':'',
            'hair_error_correction':json.loads(json.dumps(settings.record()))},
        analyzer=analyzer,view_analyzer=SimpleNamespace(analyze_pair=lambda *args, **kwargs: {
            'status':'incompatible','reason':'test_stop_before_model','inpaint_allowed':False}))
    assert report['before_promotion_gate']['checks']['color'] is False
    assert report['status'] != 'accepted_without_correction'
    assert image is generated


def test_human_ear_detector_is_diagnostic_only_for_hair_protection():
    detected = [
        DetectedPart("output_hair", mask((4, 2, 27, 24)), None, .5),
        DetectedPart("output_face", mask((10, 10, 21, 23)), None, .5),
        DetectedPart("output_human_ears", mask((4, 2, 27, 24)), None, .5),
    ]
    try:
        isolated, reason = _isolated_hair_mask(
            detected, (32, 32),
            resolve_hair_error_correction(settings_config()),
            require_ears=False,
        )
        assert reason is None
        pixels = np.asarray(isolated) >= 128
        assert pixels.any()
        assert not pixels[15, 15]
        isolated.close()
    finally:
        for part in detected:
            part.close()


def test_hair_refinement_prompt_uses_confirmed_parts_only():
    from genai_lab.hair_error_correction import build_hair_refinement_prompt
    from genai_lab.hair_structure import resolve_hair_structure

    contract = resolve_hair_structure(("short_hair", "blunt_bangs"))
    prompt = build_hair_refinement_prompt("1girl, blue eyes", contract)

    assert "overall length (short hair)" in prompt
    assert "front hair (blunt bangs)" in prompt
    assert "side hair" not in prompt
