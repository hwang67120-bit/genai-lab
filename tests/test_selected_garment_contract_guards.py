from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from genai_lab.selected_garment_correction import (
    GarmentCorrectionContractError,
    correct_selected_garment,
)


@pytest.mark.parametrize(
    'coverage_status,metrics,expected_error',
    [
        ('UNRESOLVED', None, 'garment_category_unresolved'),
        ('PASS', {'hard_protection_pixels': 20, 'requested_pixels': 20},
         'protection_exceeds_requested_region'),
        ('PASS', {'hard_protection_pixels': 30, 'requested_pixels': 20},
         'protection_exceeds_requested_region'),
        ('PASS', {}, 'protection_exceeds_requested_region'),
    ],
)
def test_garment_contract_guards_stop_before_inpaint_and_release_resources(
    monkeypatch, tmp_path, coverage_status, metrics, expected_error
):
    import genai_lab.selected_garment_correction as correction
    import genai_lab.garment_edit_plan as edit_plan
    import genai_lab.output_coordinate_control as control

    with Image.new('RGB', (16, 16), 'white') as image, Image.new('L', (16, 16), 255) as mask:
        regions = SimpleNamespace(masks={'garment': mask, 'foreground': mask}, close=Mock())
        source_mask = Mock()
        coverage = SimpleNamespace(
            record={'status': coverage_status}, mask=mask,
            growth_envelope=mask, close=Mock(),
        )
        protection = SimpleNamespace(
            hard_protection=mask, soft_boundary_protection=mask,
            conditional_protection=mask, close=Mock(),
        )
        plan = SimpleNamespace(record={'metrics': metrics}, close=Mock())
        projection = Mock(return_value=coverage)
        analyze = Mock(return_value=protection)
        build = Mock(return_value=plan)
        inpaint = Mock(side_effect=AssertionError('inpaint must not be reached'))
        monkeypatch.setattr(correction, '_load_or_redetect_regions', Mock(return_value=regions))
        monkeypatch.setattr('genai_lab.part_error_correction._expanded_mask', Mock(return_value=source_mask))
        monkeypatch.setattr(edit_plan, 'project_target_garment_coverage', projection)
        monkeypatch.setattr(control, 'analyze_output_protection_masks', analyze)
        monkeypatch.setattr(edit_plan, 'build_garment_edit_plan', build)
        monkeypatch.setattr(correction, 'run_isolated_inpaint', inpaint)
        config = {'staged_reference_generation': {'enabled': True, 'garment': {'enabled': True}}}
        request = SimpleNamespace(prompt='approved', negative_prompt='negative', seed=7)
        approval = {
            'prompt': request.prompt, 'negative_prompt': request.negative_prompt,
            'staged_reference_generation': config['staged_reference_generation'],
            'approved_tags': ['jacket'], 'approved_detail_tags': ['long sleeves'],
        }
        with pytest.raises(GarmentCorrectionContractError, match=expected_error) as error:
            correct_selected_garment(
                SimpleNamespace(), image, SimpleNamespace(garment=image), config,
                request, tmp_path, approved_run_record=approval,
            )
        inpaint.assert_not_called()
        source_mask.close.assert_called_once()
        regions.close.assert_called_once()
        coverage.close.assert_called_once()
        assert projection.call_args.args[2] == ('jacket', 'long sleeves')
        if coverage_status == 'UNRESOLVED':
            assert "approved_tags=('jacket', 'long sleeves')" in str(error.value)
            analyze.assert_not_called()
            build.assert_not_called()
            protection.close.assert_not_called()
            plan.close.assert_not_called()
        else:
            assert f"protection={metrics.get('hard_protection_pixels')}" in str(error.value)
            assert f"requested={metrics.get('requested_pixels')}" in str(error.value)
            build.assert_called_once()
            protection.close.assert_called_once()
            plan.close.assert_called_once()
