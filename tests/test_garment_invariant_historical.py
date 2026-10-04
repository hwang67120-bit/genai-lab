"""Replay saved measurements against the actual guard AST; no pipeline/model calls."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from genai_lab.selected_garment_correction import GarmentCorrectionContractError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'genai_lab/selected_garment_correction.py'


def execute_guard(error_name, **values):
    # Execute the unchanged production If node, rather than reimplementing >=.
    tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'correct_selected_garment')
    guards = [n for n in ast.walk(function) if isinstance(n, ast.If)
              and any(isinstance(v, ast.Constant) and isinstance(v.value, str)
                      and v.value.startswith(error_name + ':')
                      for statement in n.body if isinstance(statement, ast.Raise)
                      for v in ast.walk(statement))]
    assert len(guards) == 1
    block = ast.fix_missing_locations(ast.Module(body=[guards[0]], type_ignores=[]))
    exec(compile(block, str(SOURCE), 'exec'),
         {'GarmentCorrectionContractError': GarmentCorrectionContractError, **values})


# Measured sources: outputs/sdxl-local-garment-strength-v2-20260923/cases/<case>/run.json
# or outputs/color-channel-probe-v2-20260923/cases/<case>/run.json.
# JSON pointer for each: /local_refinement/garment_edit_plan/metrics.
@pytest.mark.parametrize('folder,case,protection,requested,blocked', [
    ('sdxl-local-garment-strength-v2-20260923', 'muscular-male', 514975, 490735, True),
    ('sdxl-local-garment-strength-v2-20260923', 'ordinary-female', 23845, 160496, False),
    ('sdxl-local-garment-strength-v2-20260923', 'muscular-female', 29214, 206549, False),
    ('sdxl-local-garment-strength-v2-20260923', 'raccoon', 76477, 590284, False),
    ('color-channel-probe-v2-20260923', 'ordinary-female', 23845, 182817, False),
    ('color-channel-probe-v2-20260923', 'raccoon', 76477, 627615, False),
])
def test_saved_protection_guard(folder, case, protection, requested, blocked):
    path = ROOT / 'outputs' / folder / 'cases' / case / 'run.json'
    if not path.exists():
        pytest.skip('local historical artifact not available: ' + str(path))
    run = json.loads(path.read_text(encoding='utf-8-sig'))
    metrics = run['local_refinement']['garment_edit_plan']['metrics']
    assert metrics['hard_protection_pixels'] == protection
    assert metrics['requested_pixels'] == requested
    assert run['local_refinement']['target_garment_coverage']['status'] == 'REVIEW'
    if blocked:
        with pytest.raises(GarmentCorrectionContractError,
                           match='protection_exceeds_requested_region'):
            execute_guard('protection_exceeds_requested_region', metrics=metrics)
    else:
        execute_guard('protection_exceeds_requested_region', metrics=metrics)


def test_synthetic_equal_protection_is_blocked():
    # Synthetic boundary, NOT a measured pair. Reuse one observed requested area.
    metrics = {'hard_protection_pixels': 160496, 'requested_pixels': 160496}
    with pytest.raises(GarmentCorrectionContractError,
                       match='protection_exceeds_requested_region'):
        execute_guard('protection_exceeds_requested_region', metrics=metrics)


def test_current_coverage_status_vocabulary():
    tree = ast.parse((ROOT / 'genai_lab/garment_edit_plan.py').read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'project_target_garment_coverage')
    statuses = {v.value for n in ast.walk(function) if isinstance(n, ast.Dict)
                for k, v in zip(n.keys, n.values)
                if isinstance(k, ast.Constant) and k.value == 'status'
                and isinstance(v, ast.Constant)}
    assert statuses == {'UNRESOLVED', 'REVIEW'}


@pytest.mark.parametrize('status,blocked', [
    ('UNRESOLVED', True), ('REVIEW', False), ('OK', False), ('PASS', False),
])
def test_synthetic_coverage_status_guard(status, blocked):
    # All records here are synthetic. Only UNRESOLVED/REVIEW are currently emitted;
    # OK/PASS are additional synthetic normal-state examples, not observed states.
    values = {'target_coverage': SimpleNamespace(record={'status': status}),
              'approved_tags': ('jacket',)}
    if blocked:
        with pytest.raises(GarmentCorrectionContractError, match='garment_category_unresolved'):
            execute_guard('garment_category_unresolved', **values)
    else:
        execute_guard('garment_category_unresolved', **values)
