from genai_lab.candidate_semantic_gate import (
    CandidateSemanticGateSettings,
    evaluate_candidate_semantics,
)
from genai_lab.selected_garment_correction import (
    resolve_selected_garment_correction,
)


def test_base_semantic_gate_defers_lower_body_garment_conflict():
    approved = {
        'gender': 'male',
        'approved_character_tags': ['raccoon ears', 'raccoon tail'],
        'approved_tags': ['skirt', 'jacket'],
    }
    result = evaluate_candidate_semantics(
        approved,
        {'1boy': .9, 'pants': .95},
        CandidateSemanticGateSettings(enabled=True),
        stage='base',
    )
    assert result['action'] == 'pass'
    check = result['checks']['unapproved_lower_body_garment']
    assert check['status'] == 'DIAGNOSTIC'
    assert check['detected_conflicts'] == {'pants': .95}
    assert result['diagnostics'] == [
        'base_unapproved_lower_body_garment:pants'
    ]


def test_final_semantic_gate_rejects_lower_body_garment_conflict():
    approved = {
        'gender': 'male',
        'approved_character_tags': ['raccoon ears', 'raccoon tail'],
        'approved_tags': ['skirt', 'jacket'],
    }
    result = evaluate_candidate_semantics(
        approved,
        {'1boy': .9, 'pants': .95},
        CandidateSemanticGateSettings(enabled=True),
        stage='final',
    )
    assert result['action'] == 'retry'
    assert result['checks']['unapproved_lower_body_garment']['status'] == 'FAIL'


def test_staged_garment_settings_are_resolved():
    config = {
        'staged_reference_generation': {
            'enabled': True,
            'garment': {
                'enabled': True,
                'reference_scale': .45,
                'inpaint_strength': .4,
            },
        },
    }
    settings = resolve_selected_garment_correction(config)
    assert settings.enabled is True
    assert settings.reference_scale == .45
    assert settings.inpaint_strength == .4


def test_garment_similarity_imports_the_actual_crop_function(monkeypatch):
    import numpy as np
    from PIL import Image
    from genai_lab.selected_garment_correction import _similarity
    monkeypatch.setattr('genai_lab.visual_reference.image_feature', lambda *args: np.array([1., 0.]))
    with Image.new('RGB', (16, 16), 'blue') as image, Image.new('L', (16, 16), 255) as mask:
        assert _similarity(None, image, mask, np.array([1., 0.])) == 1.0


def test_local_garment_prompt_adds_only_confirmed_topology_guidance():
    from genai_lab.garment_topology import resolve_garment_topology
    from genai_lab.selected_garment_correction import (
        build_garment_refinement_prompt,
    )

    prompt, applied = build_garment_refinement_prompt(
        "approved character",
        resolve_garment_topology(("bikini",)),
    )
    assert applied is True
    assert "two visually separate components" in prompt
    assert "abdomen between them visibly uncovered" in prompt

    unresolved, applied = build_garment_refinement_prompt(
        "approved character",
        resolve_garment_topology(("swimsuit",)),
    )
    assert applied is False
    assert unresolved == "approved character"


def test_selected_garment_correction_uses_soft_mask_without_hard_paste(
    monkeypatch, tmp_path
):
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from genai_lab.output_coordinate_control import OutputProtectionAnalysis
    from genai_lab.reference_regions import ReferenceRegions
    from genai_lab.selected_garment_correction import correct_selected_garment

    size = (64, 64)
    generated = Image.new("RGB", size, "white")
    garment_reference = Image.new("RGB", size, "blue")
    foreground = Image.new("L", size, 0)
    foreground.paste(255, (8, 4, 56, 60))
    garment = Image.new("L", size, 0)
    garment.paste(255, (20, 20, 44, 48))
    identity = Image.new("L", size, 0)
    identity.paste(255, (20, 4, 44, 20))
    hair = identity.copy()
    face = Image.new("L", size, 0)
    face.paste(255, (24, 8, 40, 18))

    def load_regions(directory, expected_size, **kwargs):
        assert expected_size == size
        assert kwargs["required_regions"] == ("identity", "garment")
        assert kwargs["analysis_scope"] == "garment_edit_input"
        return ReferenceRegions(
            {
                "identity": identity.copy(),
                "hair": hair.copy(),
                "face": face.copy(),
                "garment": garment.copy(),
                "foreground": foreground.copy(),
            },
            {"mode": "reference_regions_v1", "size": list(size)},
        )

    hard = face.copy()
    soft_boundary = Image.new("L", size, 0)
    conditional = Image.new("L", size, 0)

    def protection(*args, **kwargs):
        return OutputProtectionAnalysis(
            hard.copy(),
            soft_boundary.copy(),
            conditional.copy(),
            {},
            {
                "version": "output_protection_analysis_v1",
                "hard_parts": ["output_face"],
                "conditional_parts": [],
                "diagnostic_only_parts": [],
            },
        )

    captured = {}

    def isolated(pipe, *, control_report, hard_authorized_mask, **kwargs):
        captured["hard"] = hard_authorized_mask.copy()
        captured["soft"] = kwargs["mask_image"].copy()
        return SimpleNamespace(images=[Image.new("RGB", size, "black")])

    fake_inpaint = SimpleNamespace(
        set_ip_adapter_scale=lambda scale: captured.setdefault("scale", scale)
    )
    monkeypatch.setattr(
        "genai_lab.reference_regions.load_reference_regions", load_regions)
    monkeypatch.setattr(
        "genai_lab.output_coordinate_control.analyze_output_protection_masks",
        protection,
    )
    monkeypatch.setattr(
        "genai_lab.selected_garment_correction.run_isolated_inpaint", isolated)
    monkeypatch.setattr(
        "genai_lab.part_error_correction._inpaint_pipeline_from",
        lambda pipeline: fake_inpaint,
    )
    monkeypatch.setattr(
        "genai_lab.part_error_correction._restore_generation_scales",
        lambda *args: {"status": "restored"},
    )
    monkeypatch.setattr(
        "genai_lab.visual_reference.image_feature",
        lambda *args: np.array([1.0, 0.0]),
    )
    monkeypatch.setattr(
        "genai_lab.selected_garment_correction._similarity",
        lambda *args: 0.75,
    )

    config = {
        "staged_reference_generation": {
            "enabled": True,
            "garment": {
                "enabled": True,
                "feather_radius": 4,
                "target_growth_pixels": 4,
                "soft_boundary_strength": 0.5,
            },
        },
    }
    request = SimpleNamespace(
        prompt="approved prompt",
        negative_prompt="approved negative",
        seed=7,
    )
    approval = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "staged_reference_generation": config["staged_reference_generation"],
        "approved_tags": ["blue_jacket", "long_sleeves"],
        "approved_detail_tags": [],
    }
    pipeline = SimpleNamespace(maybe_free_model_hooks=lambda: None)
    inputs = SimpleNamespace(garment=garment_reference)
    try:
        result, report = correct_selected_garment(
            pipeline,
            generated,
            inputs,
            config,
            request,
            tmp_path,
            approved_run_record=approval,
            output_regions_directory=tmp_path,
        )
        try:
            hard_values = np.asarray(captured["hard"])
            soft_values = np.asarray(captured["soft"])
            assert np.any(hard_values)
            assert not np.any(soft_values[hard_values == 0])
            assert result.getpixel((0, 0)) == (0, 0, 0)
            assert report["hard_paste_used"] is False
            assert report["mask_composite_used"] is False
            assert report["outside_change"]["outside_changed_pixels"] > 0
            assert report["status"] == "corrected_review_required"
            assert Path(report["mask_diagnostic_manifest"]).is_file()
        finally:
            result.close()
            captured["hard"].close()
            captured["soft"].close()
    finally:
        generated.close()
        garment_reference.close()
        foreground.close()
        garment.close()
        identity.close()
        hair.close()
        face.close()
        hard.close()
        soft_boundary.close()
        conditional.close()



