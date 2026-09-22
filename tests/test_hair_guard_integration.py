from types import SimpleNamespace
import json

import numpy as np
from PIL import Image, ImageDraw

from genai_lab.hair_error_correction import (
    correct_selected_hair,
    resolve_hair_error_correction,
)
from genai_lab.regional_reference import DetectedPart


def mask(box):
    result = Image.new("L", (32, 32))
    ImageDraw.Draw(result).rectangle(box, fill=255)
    return result


def test_guard_and_feature_gate_are_connected_to_hair_correction(monkeypatch):
    monkeypatch.setattr('genai_lab.hair_error_correction.run_isolated_inpaint',
                        lambda pipe, control_report, **kwargs: pipe(**kwargs))
    config = {
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
                "boundary_guard": {
                    "enabled": True,
                    "boundary_width_pixels": 3,
                    "feather_radius_pixels": 2,
                    "maximum_raw_outside_changed_ratio": 1.0,
                    "seam_mean_difference_max": 255,
                    "seam_p95_difference_max": 255,
                    "seam_changed_ratio_max": 1.0,
                },
                "promotion_gate": {
                    "enabled": True,
                    "minimum_color_similarity": 0.0,
                    "minimum_shape_iou": .65,
                    "maximum_length_change_ratio": .2,
                    "maximum_area_change_ratio": .25,
                },
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
    settings = resolve_hair_error_correction(config)
    generated = Image.new("RGB", (32, 32), "blue")
    source = generated.copy()
    source_hair = mask((4, 2, 27, 24))
    identity = source_hair.copy()
    garment = mask((2, 25, 29, 31))

    class Analyzer:
        def analyze(self, *args, **kwargs):
            return [
                DetectedPart("output_hair", mask((4, 2, 27, 24)), None, .5),
                DetectedPart("output_face", mask((10, 10, 21, 23)), None, .5),
            ]

    class Inpaint:
        def set_ip_adapter_scale(self, value):
            self.scale = value

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[generated.copy()])

    pipeline = SimpleNamespace(set_ip_adapter_scale=lambda value: None)
    scores = iter((.4, .7))
    monkeypatch.setattr(
        "genai_lab.visual_reference.image_feature",
        lambda *args: np.array([1.0]))
    monkeypatch.setattr(
        "genai_lab.part_error_correction._similarity",
        lambda *args: next(scores))
    request = SimpleNamespace(
        prompt="approved hair prompt", negative_prompt="negative", seed=42)
    result, report = correct_selected_hair(
        pipeline,
        generated,
        SimpleNamespace(
            source=source,
            hair_mask=source_hair,
            extra_references=(),
            identity=source,
            identity_mask=identity,
            garment=source,
            garment_mask=garment,
            scene_condition=None,
        ),
        config,
        request,
        approved_run_record={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "hair_error_correction": json.loads(json.dumps(settings.record())),
        },
        analyzer=Analyzer(),
        inpaint_pipeline=Inpaint(),
        view_analyzer=SimpleNamespace(
            analyze_pair=lambda *args, **kwargs: {
                "status": "compatible",
                "reason": None,
                "reference": {"view": "front"},
                "candidate": {"view": "front"},
                "allowed_candidate_views": ("front",),
                "inpaint_allowed": True,
            }),
    )
    try:
        assert report["status"] == "corrected_and_accepted"
        assert report["boundary_guard_result"]["status"] == "passed"
        assert report["boundary_guard_result"][
            "final_protected_changed_pixels"] == 0
        assert report["promotion_gate_result"]["passed"] is True
    finally:
        result.close()
        generated.close()
        source.close()
        source_hair.close()
        identity.close()
        garment.close()
